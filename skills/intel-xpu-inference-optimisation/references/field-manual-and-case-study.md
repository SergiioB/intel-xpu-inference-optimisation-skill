# Intel Xe2 Optimization Field Manual: Thinking Model, Attitude, Decision Tree & Full Case Study

This document distills **how to think**, **what attitude to keep**, **how to debug fast**, and **every win, surprise, and clean revert** from pushing Intel Arc Pro B70 (Battlemage Xe2, `bmg_g31`) inference from scratch to production.

---

## 1. The Silicon Mental Model (Think Directly in Xe2 Physics)

Never treat Intel XPU as "CUDA with different names." Reason from the actual Battlemage `bmg_g31` hardware first:

- **Compute & Memory Anchors:** `40` Xe cores (`320` XVEs), `SG=16`, `128 KB` SLM/Xe-core, `256` GRF (`-ze-opt-large-register-file`), **`~521–528 GB/s` sustained GDDR6** (not 608 GB/s marketing), **`~157 TFLOPS` peak BF16 XMX** (`~75–85 TFLOPS` effective), **`367 TOPS` INT8 `s8×s8`**, and **NO native FP4 or FP8 DPAS** (sub-byte/FP8 weights & KV save DRAM bandwidth and unpack to BF16/FP16 in registers).
- **Every single bottleneck falls into exactly ONE of 4 physical regimes:**
  1. **DRAM Bandwidth Bound:** decode weights + KV read bytes divided by `521 GB/s`. Fix by shrinking bytes read (native NVFP4/MXFP8/EXL3/MLX4, FP8/Q8 KV, GQA/multi-token KV reuse, speculative verification) or removing wasted bytes (16B alignment, eliminating FP32 copies).
  2. **XMX Compute Bound:** prefill GEMMs and wide-verify (`M=8`, `24` rows) long-context attention (`206 GFLOPs` at 256K!). SIMT FMA (`~2–3 TFLOPS` effective) CANNOT beat XMX DPAS (`~75 TFLOPS`) here!
  3. **Compiler / Layout / Kernel Pathology:** huge gaps between measured time and physics floor caused by:
     - Triton 1-byte `float8_e4m3fn`/`uint8` `tl.make_tensor_descriptor` breaking 16-bit DPAS layout (`5.4x–11x` slower!).
     - Triton `tl.dot` padding `M=1, G=3` query rows into a `16`-row XMX tile (`81%` wasted -> `4.76x` slower than zero-spill SYCL SIMT!).
     - `icpx 2026.1` `joint_matrix` `get_wi_data` inside an inner K-loop spilling the entire accumulator to private memory (`3x` slower!).
     - Hidden FP32 `lm_head` weight casts (`-42%` decode, `+1.22 GiB` wasted VRAM).
  4. **Scheduler / Allocator / Host Overhead:** vLLM `SlidingWindowManager` over-reserving `40 × mbt` SWA blocks, PyTorch XPU caching allocator fragmenting gigabytes on dynamic per-chunk `torch.empty((n_active, ...))`, display-card KDE/GNOME eating `1.5–4.6 GiB` VRAM, or sequential combine kernels amplifying L2 traffic under XPU graph replay.

---

## 2. Attitude & Problem-Solving Habits (What Actually Produces Breakthroughs)

1. **Relentless, calm, optimistic curiosity — treat every error or slowdown as a clue, never a wall.**
   - When a plugin raises `NotImplementedError("requires --language-model-only")` or rejects `--kv-cache-dtype fp8`, inspect the checkpoint and upstream reference before accepting "unsupported." Checking the 651-line NVIDIA reference revealed the entire 23-layer ViT was blocked by ONE PTX inline-asm line + CUDA FA, and FP8 KV was already handled by the underlying Triton prologue/decode kernels!
   - When FP8 prefill ran `5.4x` slower and then OOM'd at `64K`, staying curious and excited to find the exact cause turned a dead end into an **11x Chunked-LSE breakthrough** (`905 tok/s` at `496K`).
2. **Zero ego — celebrate fast kills and clean reverts just as much as wins.**
   - If an idea looks `3x` faster in a microbenchmark but loses `-10–12%` under live graph replay (Triton Split-KV `80/128`), **revert immediately with zero attachment** and log why.
   - If an idea prices out to `<6%` (`M=4` vs `M=8` CUTLASS NVFP4 = `2.58 ms` out of `38 ms`), **stop before writing code**. Every dead end you kill in 5 seconds saves someone else 5 hours!
3. **Price the physics floor BEFORE writing one line of code.**
   - Compute `active_bytes / 521 GB/s` and decompose one live step into exact milliseconds (`MoE` vs `SWA` vs `Full Attn` vs `Draft` vs `Dense/LM-head`). Never optimize a `1.5 ms` component when a `35 ms` component is sitting right next to it!
4. **Compress every hypothesis into a 5-second standalone probe + exact parity oracle.**
   - Never debug a kernel inside a 90-second vLLM startup loop! Build an AOT `.so` in 8 seconds, check `rel_l2` against reference + microbenchmark in 5 seconds, iterate V1 -> V2 -> V3 in 2 minutes, and ONLY THEN wire it into the live server.
5. **Never trust an isolated microbenchmark alone (`DNR-030`).**
   - Always verify under **live `FULL_DECODE_ONLY` XPU graph replay** with a **same-binary `ENV=0` vs `ENV=1` control**, and prove exact quality/retrieval parity before promoting anything.
6. **Welcome hard skepticism with direct measurement, never defensiveness.**
   - When someone asks *"does FP8 KV noise degrade perplexity at 512K?"*, immediately run matched BF16 vs FP8 held-out perplexity (`4K -> 496K`) + `15%`/`55%` dual-needle recall and publish BOTH the exact numbers AND the honest untested scope.
7. **Study external/community work with genuine respect and surgical decoupling.**
   - Separate *third-party quantization* from *orthogonal kernel/scheduler mechanisms*. Keep official native weights when they win fidelity/prefill/ceiling, and test only the orthogonal runtime ideas that pass your Impact Gate.
8. **Ship small, opt-in, clean, verified increments continuously.**
   - Break `uv` hardlinks before patching site-packages; gate every custom kernel behind an opt-in env flag (`=0` default); check `vram_mm` & `dmesg` after every run; visually inspect every PNG chart and UI screenshot; fold every lesson back into the skill before ending the session.

---

## 3. Master Troubleshooting & Optimization Decision Tree ("If You See X, Do Y")

| Symptom / Observation | Immediate 5-Second Check | Proven Fix |
|---|---|---|
| Decode is `>30%` below `active_bytes / 521 GB/s` | Check `lm_head` dtype/casts, graph capture mode, and per-op ms breakdown | Fix hidden FP32 casts (PR `#69186`), enable `FULL_DECODE_ONLY` graph replay, attack largest ms share |
| `M=1` long-context decode collapses on Triton (`50 ms` @ 256K) | Check `G` query rows per KV head (`G=3` wastes `81%` of `16`-row XMX tile!) | Switch `M=1` to zero-spill AOT `bmg_g31` SYCL SIMT (`4` SGs/WG, contiguous pages, 4-key tiled softmax, `1 KB` SLM E4M3 LUT) -> **`4.76x` kernel / `+71%` live @ 256K** |
| Custom SYCL SIMT is `2–3x` slower than Triton at `M=8` | Compute FLOPs (`24 rows x D x seq_k x 4` = `206 GFLOPs` @ 256K!) | SIMT is compute-bound on `M>=2`; use **hybrid dispatch (`M=1` -> SYCL SIMT, `M>=2` -> XMX DPAS / ESIMD)** |
| SYCL `joint_matrix` slows `3x` after adding in-register scaling | Check for `get_wi_data(sg, acc)` inside the K-loop (`icpx 2026.1`) | Avoid `get_wi_data` inside K-loops (spills accumulator to private memory!); use ESIMD for `M>=2` custom XMX |
| FP8 KV prefill is `5x–11x` slower than BF16 in Triton | Check if `tl.make_tensor_descriptor` loads 1-byte `fp8`/`uint8` | Materialize active pages into **ONE fixed-size `256 MiB` reusable BF16 scratch** + 64K **Chunked-LSE** merge (`11x` faster -> `91%–97%` of BF16 prefill up to `512K`!) |
| `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` at `64K` during chunked prefill | Check for dynamic `torch.empty((n_active, ...))` inside attention | Replace with ONE pre-allocated module-level buffer sliced `buf[:n_active]` (eliminates PyTorch XPU caching-allocator fragmentation!) |
| SWA model max context collapses to `~7K` on vLLM | Inspect `--max-num-batched-tokens` (`4096`/`8192` explodes `40x` SWA pool!) | Lower `mbt` to `2048` (speed), `1024` (2x ctx), or `512` (max KV), OR build a fixed-size SWA device ring |
| Microbenchmark wins `3x` but live decode drops `-10%` | Check combine-kernel traffic & L2 pressure inside captured XPU graph | Revert immediately (`DNR-030`!); never trust isolated 1-layer microbenchmarks without live graph-replay A/B |
| Thinking model returns `content=""` on retrieval test | Inspect `completion_tokens_details.reasoning_tokens` | Raise `max_tokens >= 512` (never `128`/`256`, which `<thinking>` can saturate at 128K!) |
| Plugin refuses Vision (`requires --language-model-only`) | Read upstream NVIDIA `vision.py` & checkpoint `model.visual.*` bytes | Port pure PyTorch XPU ops + `F.scaled_dot_product_attention(..., is_causal=False)` (0 PTX, `29 ms` for 2 images!) |
| Display B70 OOMs on a recipe that fits dedicated card | Check `/sys/class/drm/<card>-*/status` & desktop `drm-resident-vram0` | Prefer dedicated non-display card; on display card cap `gmu <= 1 - (desktop + 1 GiB)/total` & step `ctx` to `ctx_safe` |
| `pkill -f <path>` kills your own shell script | Shell's own command line matched `<path>`! | Filter PIDs explicitly: `ps -eo pid,args \| awk '/pat/ && !/awk/ {print $1}' \| xargs -r kill -TERM` |
| Quantized/FP8 KV promoted on pool size alone | Check `config.json` K/V norms/clips + run held-out PPL & recall | Measure matched BF16 vs FP8 held-out PPL (`4K -> max`) + `15%` oldest / `55%` mid dual-needle recall before shipping |

---

## 4. Complete Annotated Case Study: Inkling 2 Mini on One 32 GB B70 (Chronological)

Every step we actually took in one weekend — including the exact wins, surprises, and clean reverts:

| Step | Question / Wall | 5-Second Diagnosis / Action | Measured Outcome | Decision |
|---|---|---|---|---|
| **0. Intake** | `39.7 tok/s` greedy, text-only, `106K` ctx | Tensor census (`NVFP4` L0–19, `MXFP8` L20–23, BF16 dense/vision, `model_max_length=1M`, L2-clipped K + RMSNormed V); found `vllm-inkling2-xpu` + PR `#69186` | Preserved official native weights (zero conversion!) | **KEEP native** |
| **1. Free Fix** | Hidden FP32 `lm_head` cast | Applied 2-line vLLM PR `#69186` (`torch.mm(..., out_dtype=fp32)`) | Greedy `39.7 -> 56.4 tok/s` (**`+42%`**, `-30%` J/tok, `+1.22 GiB` KV) | **SHIPPED** |
| **2. MoE Kernel** | Triton packed MoE ate `~93 ms` on `M=8` | Built AOT `bmg_g31` CUTLASS Xe2 NVFP4+MXFP8 + 1-WG SLM route-sort + fused SwiGLU (`rel_l2 <= 5e-5`, opt-in `INK2_CUTLASS_MOE=1`) | Card sampling **`80.4 -> 55.3 tok/s`** (`0–104K`, `effort=0.2`: **`99.2`**), prefill **`3,920 -> 2,891 tok/s`**, **`8/8`** tasks PASS | **SHIPPED** |
| **3. SWA Knob** | `mbt=4096` collapsed ctx to `~7K` | Swept `mbt=2048 -> 1024 -> 512` (`u=0.95`) | `163K` DSpark (`160K` real prompt `46.2 tok/s`), **`256K` no-spec (`273.8K` / `308.7K` KV)** | **SHIPPED** |
| **4. Vision** | Plugin raised `NotImplementedError` | Diffed 651-line NVIDIA ViT (`607 MiB` BF16); replaced 1 PTX line + CUDA FA with PyTorch XPU + `F.scaled_dot_product_attention(..., is_causal=False)` | `+0.55 GiB` VRAM, **`29.3 ms`** for 2 images (`cos=0.9994`), **`4/4` live multimodal PASS** (chart OCR `107 tok/s`, counting `170 tok/s`, 2-img) | **SHIPPED** |
| **5. FP8 KV** | `attention.py` blocked `fp8`; then `5.4x` slow prefill & `64K` OOM! | Enabled `fp8_e4m3fn`; 5s probe caught 1-byte Triton descriptor slowdown (`256.6 vs 46.8 ms`) & dynamic `torch.empty` fragmentation -> built **ONE `256 MiB` reusable BF16 scratch + 64K Chunked-LSE** (`lse_max_err=5.7e-6`) | **`11x`** faster FP8 prefill @ 128K+ (`2,701 @ 64K`, `2,195 @ 128K` Vis+DSpark; `905 @ 496K`), **`323K` KV w/ Vision+DSpark** & **`569K` KV (`512K`) no-spec!** | **SHIPPED** |
| **6. 512K Quality** | Skeptic asked: *"Does FP8 noise ruin 512K perplexity?"* | Measured matched BF16 vs FP8 held-out 256-tok technical prose PPL (`4K -> 496K`) + `15%`/`55%` dual-needle recall | `4K–245K`: `-0.48%..+0.41%` (**`+0.07%` @ 245K**, MAE `0.066` nats); **`3.456` @ 384K**, **`3.583` @ 496K**; **`2/2` exact recall @ `495,964` tok!** | **PROVED & PUBLISHED** |
| **7. External Audit & Fast Kills** | Audited external EXL3 serving route; tested Triton Split-KV `80/128` & priced Adaptive DSpark | Live graph-replay A/B on `80/16` regressed `-10–12%` (`73.4 -> 66.0`, `61.0 -> 53.7`); CUTLASS `M=4` vs `M=8` priced at `2.58 ms` (`<6%`) | Kept official native weights; reverted Split-KV `80/128` immediately; skipped Adaptive DSpark cleanly | **CLOSED & REVERTED** |
| **8. SYCL `M=1` Decode** | Triton `M=1` `256K` FP8 took `50.24 ms` (`20.07 tok/s`) | Iterated AOT `bmg_g31` SYCL V1 -> V2 -> V3 (`SingleTokSplitKernel`, 4 SGs/WG, contiguous pages, 4-key tiled softmax, `1 KB` SLM E4M3 LUT); discovered SIMT (`M=1`) vs XMX (`M=8`) split & `get_wi_data` spill | Isolated **`4.76x`** (`50.24 -> 10.56 ms`, `78%` GDDR6 BW); live graph-replayed A/B: **`+19.8%` @ 64K**, **`+39.3%` @ 128K**, **`+71.0%` @ 256K (`20.07 -> 34.32 tok/s`!)**, `4/4` + `128K` recall PASS | **SHIPPED (`INK2_SYCL_DECODE_ATTN=1`)** |

---

## 5. Things I Thought That Proved Wrong (Mental-Model Corrections on Real Xe2 Silicon)

These 8 wrong hypotheses — every one of which sounded 100% logical on paper — are the most valuable part of this skill. Internalize them so you recognize the exact trap in seconds:

1. **Wrong thought: *"Branchless `uint8 -> fp16 -> bf16` bit-shift inside Triton `tl.make_tensor_descriptor` will fix the 5.4x FP8 prefill slowdown."***
   - **What happened:** It got **worse** (`477.7 ms` vs `256.6 ms` at 64K, vs `46.8 ms` BF16)!
   - **Why:** The root cause was not only scalar FP8 conversion — **1-byte `uint8`/`float8` 2D tensor descriptors break Xe2's 16-bit DPAS block layout** before `tl.dot`.
   - **Real fix:** Stop fighting 1-byte tensor descriptors; materialize active pages into BF16 (`47.8 ms` = `5.4x` faster) + 64K Chunked-LSE merge (`46.4 ms` @ 128K = **`11x` faster**).

2. **Wrong thought: *"Allocating a `< 256 MiB` temporary BF16 tensor per layer during chunked prefill can't OOM when `2.5 GiB` is free."***
   - **What happened:** Hit `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` right at `64K`!
   - **Why:** PyTorch XPU's caching allocator caches every distinct dynamic shape (`n_active = 32, 64, 96, ..., 1024`) across chunked-prefill steps simultaneously, fragmenting gigabytes even though only 256 MiB is live at once.
   - **Real fix:** Pre-allocate **ONE fixed-size module-level scratch buffer** and slice `buf[:n_active]` in 128-page chunks -> zero fragmentation, zero OOM all the way to `496K`.

3. **Wrong thought: *"Halving KV bytes with FP8 will automatically make `M=8` Triton decode `20–45%` faster at `64K–128K`."***
   - **What happened:** On stock Triton `attention_triton.py`, FP8 KV decode was **`~12–18%` slower** than BF16 under `104K` (`67.1 -> 43.6` vs `80.4 -> 55.3 tok/s`).
   - **Why:** Triton's scattered 1-byte FP8 pointer loads + software casts outweighed the 2x DRAM savings until either context became huge or a custom SYCL kernel with a `1 KB` SLM E4M3 LUT replaced it (which DID hit `4.76x` on `M=1` at `256K`!).

4. **Wrong thought: *"My isolated 1-layer microbenchmark shows Split-KV `128/16` (and `80/16`) is `1.4x–3.3x` faster than `40/9`, so live decode will fly."***
   - **What happened:** Live graph-replayed decode **regressed `-10%` to `-12%`** (`73.4 -> 66.0` @ 64K, `61.0 -> 53.7` @ 128K), even after isolating target layers and reusing scratch!
   - **Why:** Across 48 visits inside a captured XPU graph, higher split counts inflate sequential combine-kernel work and evict hot weights/KV from L2 cache (`DNR-030`).
   - **Real fix:** Revert immediately with zero ego. Never promote any kernel change without a live same-binary graph-replayed A/B!

5. **Wrong thought: *"Cooperative 8-token SLM KV sharing (`sg_id = tok 0..7`) in SIMT SYCL will crush Triton on `M=8` just like `M=1` did (`4.76x`)."***
   - **What happened:** `M=1` SIMT crushed Triton (`10.56 ms` vs `50.24 ms` = `4.76x`), yet `M=8` SIMT ran **`109.8 ms` vs Triton `40.7 ms`** (`2.7x` slower!).
   - **Why:** **Compute-vs-bandwidth crossover.** `M=8` (`24` rows) at `256K` is **`206 GFLOPs`** — compute-bound on SIMT FP32 FMA (`~2 TFLOPS` effective), whereas Triton `tl.dot` uses hardware XMX DPAS (`~75 TFLOPS`). Conversely, `M=1` (`G=3` rows) wastes `81%` of a `16`-row XMX tile, so SIMT wins `4.76x`!
   - **Real fix:** Check BOTH FLOPs AND bytes before assuming memory-bound; use **hybrid dispatch (`M=1` -> SYCL SIMT, `M>=2` -> XMX DPAS / ESIMD)**.

6. **Wrong thought: *"Scaling SYCL `joint_matrix` accumulators in registers via `get_wi_data(sg, o_mat)` will remove SLM round-trips and speed up `XMX_V2`."***
   - **What happened:** `XMX_V2` got **`3x` slower** (`23.7 ms -> 70.6 ms` at 128K)!
   - **Why:** In `icpx 2026.1`, calling `sycl::ext::oneapi::detail::get_wi_data` inside an inner K-loop forces the compiler to spill the entire `joint_matrix` accumulator to private scratch memory every iteration.

7. **Wrong thought: *"Adaptive DSpark (`0.75` confidence cutoff) gave a big win on an external EXL3 stack, so it'll give us `+15–25%` on CUTLASS Xe2 too."***
   - **What happened:** A 3-second pricing script showed CUTLASS Xe2 NVFP4 (`36` visits) drops only **`2.58 ms`** from `M=8` (`11.31 ms`) to `M=4` (`8.73 ms`) — `< 6%` of a `~38 ms` step, failing the 10% Impact Gate before we wrote a single line of scheduler code.
   - **Real fix:** Never import another engine's lever assuming the same ROI; always price the exact op delta on **your own** kernel first.

8. **Wrong thought: *"My `128K` SYCL retrieval test returned `content=''` — my new SYCL kernel must have broken long-context attention!"***
   - **What happened:** Inspecting `reasoning` showed the model had already found `'SYCL-128K-99412-OK'` on line 1 of its thought trace and spent `489` reasoning tokens (`max_tokens=256` cut it off before emitting `content`). Raising `max_tokens=512` passed `100%` exact!
   - **Real fix:** On thinking models, ALWAYS inspect `reasoning` + `completion_tokens_details` before debugging a kernel, and set `max_tokens >= 512` on long-context retrieval checks.

---

## 6. Operational & Upstream-PR Craft (Small Habits That Prevent Disasters)

- **Always break `uv` hardlinks (`stat -c '%h'`) before editing site-packages:** `uv --link-mode hardlink` shares inodes with the global cache; editing in place silently corrupts other venvs.
- **Never `pkill -f <path>` when your shell command line contains `<path>`:** It kills your own shell mid-script. Filter PIDs via `ps -eo pid,args | awk '/pattern/ && !/awk/ {print $1}' | xargs -r kill -TERM`.
- **Match upstream repository architecture strictly when preparing PRs:** Do NOT push C++ SYCL build files into a pure-Python/Triton wheel repo (`pyproject.toml` with pure `setuptools` and `ruff`). Split upstream contributions into minimal, single-purpose, independently reviewable PRs (e.g. PR #1 Vision, PR #2 FP8 KV), format with the repo's exact `ruff` config, write CPU-runnable unit tests, and run an independent fresh-context reviewer to explicit `VERDICT: OK` before pushing (our 3-round review caught two real edge-case bugs before upstream ever saw the code: silent bound-dropping fallback on `page % 16 != 0`, and `canonicalize_singleton_dim_strides` breaking `_kv_layout` on `cnt==1` short prompts & `kv_heads==1`).
- **Verify every published build script & `.patch` verbatim on a clean upstream checkout:** Dev trees hide missing `-I` paths, missing `-DTORCH_EXTENSION_NAME`, missing `-Xspirv-translator -spirv-ext=...` (required on `icpx 2026.1+`), missing `command -v ocloc` checks, and malformed hand-edited hunk line counts (caught by external contributor testing!). Always test verbatim on a fresh tree and confirm the built `.so` imports cleanly.
- **Check log verbosity before declaring a runtime path broken:** `llama-server` requires `-lv 4` to emit lazy-mmap (`-lzm on`) Engram prefetch INFO lines; an earlier HY4 Flash canary parked solely because default log verbosity hid the proof line.
- **Visually inspect every PNG chart and UI screenshot with `read`:** Humans spot overlapping labels, clipped footers, and stale numbers instantly — never ship a graphic blindly.

---

## 7. The 2-Minute Zero-GPU Multi-GPU SYCL Engine Audit (How We Diagnosed `Strata v0.2.20` HY4-Flash)

When dropped into an unfamiliar multi-file SYCL inference engine (e.g. 16,800 lines across 30+ files) whose measured speed is far below roofline (`251 tok/s` prefill, `18.7 tok/s` decode) or has exit faults, **do not guess or blindly tune CLI flags.** Run these 4 high-signal checks on logs + source code **off-GPU**:

1. **Check VRAM symmetry vs. actual card load (look for stranded GDDR6!):**
   - Read the startup log: if Card 1 (primary, holding trunk + KV + experts) has `~4 GiB` free while Card 0 (peer, holding ONLY experts) has **`13.4 GiB` free** AND `25.3 GB` of experts are spilling to host RAM, immediately grep the VRAM budget flag!
   - One line (`sy_graph.h:149`) reused the primary card's `--vram-experts-gb 17` cap on the peer card too -> adding `--peer-vram-experts-gb 25` moves `+8 GiB` (~2,300 hottest Zipf experts) off slow CPU/RAM straight into 521 GB/s GDDR6.
2. **Diff the Primary-GPU kernel path against the Peer-GPU kernel path (the #1 multi-GPU engine trap!):**
   - Engine authors almost always optimize the primary single-GPU path first (`SG16 XMX 8x16x16 grouped GEMM` for prefill; `SG16 + Q8_1_G64 pre-quant + SLM + dedup` for decode) and leave the secondary peer-GPU path on an older v1 kernel (`REQD_SG32`, ungrouped 1-token GEMV!) or serialized call!
   - Checking `sy_graph.h:1138-1139` + `sy_peer.h:483-621` immediately revealed BOTH: (a) Peer GPU 0 ran **serially after** Primary GPU 1 finished (zero dual-GPU overlap), AND (b) Peer GPU 0 evaluated 4K-token prefill with the **1-token `SG32` decode kernel** (`~128x` redundant weight reads + `656` host-GPU sync barriers per 4K chunk!).
   - **Clean fix pattern:** Reuse the primary GPU's already-proven XMX grouped GEMM and `SG16+preq8+SLM` decode kernels by parameterizing them on `sycl::queue& q` (so zero kernel math is rewritten), launch Peer GPU 0 **asynchronously before** Primary GPU 1, and gate every new path behind a same-binary opt-out env var (`HY4S_PEER_PF_LEGACY=1`, `HY4S_PEER_DEC_SG32=1`) + phase timer (`HY4S_STEP_TIMING=1`)!
3. **Separate exit-time `dmesg` faults from inference-time faults:**
   - When `xe` fires `Engine reset: engine_class=bcs` (blitter copy engine) on BOTH cards **exclusively on process exit** (never during inference), grep `exit|QUIT|release` immediately.
   - Found `sy_serve.h:531` calling `_exit(0)` (`hy4s_exit_fast(0)`) without draining `Q()`, `gx->wait_all()`, `vcache->wait_copy()`, or `peers->release()`, plus `probe_p2p` leaving `ext_oneapi_enable_peer_access` enabled after a cross-context failure! Draining every queue + disabling failed peer access before `_exit(0)` directly targets exit-only `bcs` resets.
4. **Always build and peer-review kernel patches OFF-GPU while another run owns the cards:**
   - Copy into a disjoint `src-opt/` tree (never mutate the measured baseline tree!), compile AOT with `nice -n 19`, generate a clean `git diff` patch, and run a strict read-only fresh-context audit for queue synchronization, device-memory allocation/free pairing, and ascending reduction parity **before** spending a single GPU minute!
   - **Verified payoff on `Strata v0.2.20` HY4-Flash (dual B70):** peer prefill collapsed **`20,607 ms -> 22.7 ms` (`907x` faster, `78.7% -> 0.5%` of prefill)**, cold prefill jumped **`246.6 -> 723.9 / 737.8 tok/s` @ 28K (`2.99x` vs stock, `3.8x` vs `llama.cpp`)**, decode hit **`39.73 tok/s` @ 28K (`dfly2`, `5/5` gate + `2/2` think PASS)**, and GPU teardown faults dropped from **7/7 resets + 50% next-launch hangs -> `0` GPU faults (`5/5` READY on try 1)**! (Also caught a textbook Xe2 bug during R1 -> R2: `launch_gemm_xmx_sg16`'s 8-row `joint_matrix_load` read up to 7 rows past exact-sized `pf_zg`/`pf_hb`, fixed cleanly with `+64` mapped tail rows!)

---

## 8. Deep-Context Competitive Post-Mortem (`64K–262K`): Why Our Native Route Lagged & How to Never Leave `20–60%` on the Table Again

After shipping our `3,920 tok/s` @ 2K / `80.4 tok/s` @ 0K / `512K` FP8 native Inkling 2 Mini route, an external EXL3 serving route beat our **`64K–262K`** serving speeds (`2,187` vs `1,378 tok/s` prefill @ 245K; `61.8–74.5` vs `43.6 tok/s` deep FP8 decode) **even at `180 W` vs `230 W`**. Auditing the underlying kernel and runtime changes proved **none of those deep-context wins required EXL3 quantization** — every single one was an orthogonal kernel or runtime mechanism we had stopped one layer too early on!

### The 5 Specific Mistakes We Made (And Their Exact Fixes)

| What We Did (And Why We Stopped Early) | What We Missed (Root Cause) | Permanent Correction / Proven Fix |
|---|---|---|
| **1. Hit `icpx` `joint_matrix` `get_wi_data` spill (`23.7 -> 70.6 ms`) and fell back to Triton XMX for `M>=2`** | Treated `joint_matrix` as the only way to use XMX in C++. **ESIMD (`WG=1`, `lsc_load_2d` transpose/VNNI + `xmx::dpas<8,8>` + 3-op SIMD FP8->FP16 `((u&127)<<7)\|((u&128)<<8) * 256`)** has ZERO `get_wi_data`, ZERO SLM, ZERO barriers, packs `R=3M` rows cleanly for `M=1..8`, AND beats even our `4.76x` SIMT `M=1` kernel (`1,025 us` vs `1,407 us` @ 256K FP8 = raw `521 GB/s` floor!) | **Never settle for Triton `tl.dot` on custom decode attention when `joint_matrix` spills.** Reach straight for **ESIMD `PagedXmx<M>`** with `-ze-opt-large-register-file`! |
| **2. Celebrated `11x` Triton Chunked-LSE FP8 prefill (`1,378 tok/s` @ 245K) and stopped** | Ignored the mathematical structure of the attention mask: outside `dist < 1024`, full attention has **ZERO relative bias**! Running `<= 8K` bias-free prefix chunks through **native oneDNN flash SDPA** + merging with the `1024`-token Triton biased tail via FP32 LSE (persistent `16 MiB` scratch) eliminates BF16-scratch copy bottlenecks past 64K! | Live same-binary A/B (`INK2_HYBRID_PREFIX_SDPA=1`): **`64K: 2,706 -> 3,050 (+12.7%)`**, **`128K: 2,149 -> 2,576 (+19.9%)`**, **`245K: 1,586 -> 2,024 (+27.6%)`** — beating EXL3 at `64K` & `128K` on official native weights! *(And never run single-shot >32K oneDNN SDPA or per-window allocs!)* |
| **3. Generalized `DNR-030` (Triton Split-KV `80/128` `-12%` regression) into "high splits always lose under graph replay"** | The regression was **Triton's `constexpr` splits + heavyweight sequential combine**, NOT high partitions themselves! ESIMD uses `num_splits` as a **runtime grid dim** + tiny parallel combine + split count in the graph key -> zero re-JIT and **`+3.7–4.5%`** extra decode at 64K–128K! | Always separate *kernel-implementation artifacts* from *hardware-level laws*. Re-test split counts whenever you replace the combine kernel. |
| **4. Assumed BF16 SWA ring would add `+15–25%` decode speed before running a 5-second microbenchmark** | At `M=8` verify, 40 local `1024`-token SWA visits are **launch-bound (`1.94 ms` FP8 vs `1.95 ms` BF16 vs `1.93 ms` ring = `0%` speed delta!)**, while `mbt=4096` costs `+3.31 GiB` KV. | **Always run the 5-second op probe FIRST.** A fixed `(1024+mbt)` SWA ring is a **capacity/`mbt` lever**, not a decode-speed lever. |
| **5. Kept `--no-enable-prefix-caching` permanently off after seeing #53912 corruption** | Never isolated the exact mechanism (`RingPrefix` + `PrefillTail`): hybrid conv/SWA/draft models diverge on cache hits because of missing ring snapshots + **chunk-size-dependent numerics on the final tail** (`d1f14611`). | Snapshot page-aligned frontier + last `<= 32` IDs/hidden/KV rows, replay exact final chunk size, and verify with the `cold -> resend-1 -> resend-2 -> new-salt` SHA-256 gate (`47 s -> 0.70 s` @ 75K!). |

### 5 Mandatory Rules Added to Our Methodology (v3.7.0)

1. **Benchmark & close the gap across the *entire* context ladder (`0K, 64K, 128K, 256K`), never declare victory on `0K–32K` alone:** Compare every depth against BOTH the `521 GB/s` / `75 TFLOPS` roofline AND the fastest public serving curve on the same GPU — regardless of whether the competitor used a different quant.
2. **Decompose every external win into `(quantization delta)` vs `(orthogonal runtime/kernel mechanisms)` on Day 0:** Never dismiss a competitor's headline just because they requantized; immediately inspect their commits (`git log` / patch diff) for orthogonal kernels, attention-mask splits, and scheduler/cache fixes you can port onto official native weights!
3. **ESIMD (`lsc_load_2d` + `xmx::dpas<8,8>`) is the first-class Xe2 weapon for custom attention & low-M decode:** If Triton `tl.dot` wastes tiles or `sycl::joint_matrix` spills on `get_wi_data`, go straight to ESIMD WG=1 with `-ze-opt-large-register-file`.
4. **Upstream-ready engineering from Line 1 (PR #2 review checklist):** Reserve any persistent scratch buffer *before* vLLM's startup `determine_available_memory()` KV sizing, pass scalar batch lengths via CPU metadata (zero per-layer XPU `.item()` syncs!), and track every upstream engine PR in an auto-refreshed live ledger (`tools/sync_upstream_prs.py`).
5. **Respect tight-headroom stability (`~1 GiB` free VRAM):** Under `gmu=0.98`, never allocate transient per-chunk/per-window tensors inside attention loops — they can trigger `bcs` engine faults even before raising OOM. Use one tiny persistent scratch + clean `dmesg` verification on every single run.
