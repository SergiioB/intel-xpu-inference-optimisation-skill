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
| **7. External Audit & Fast Kills** | Audited `Danmoreng` EXL3 repo; tested Triton Split-KV `80/128` & priced Adaptive DSpark | Live graph-replay A/B on `80/16` regressed `-10–12%` (`73.4 -> 66.0`, `61.0 -> 53.7`); CUTLASS `M=4` vs `M=8` priced at `2.58 ms` (`<6%`) | Kept official native weights; reverted Split-KV `80/128` immediately; skipped Adaptive DSpark cleanly | **CLOSED & REVERTED** |
| **8. SYCL `M=1` Decode** | Triton `M=1` `256K` FP8 took `50.24 ms` (`20.07 tok/s`) | Iterated AOT `bmg_g31` SYCL V1 -> V2 -> V3 (`SingleTokSplitKernel`, 4 SGs/WG, contiguous pages, 4-key tiled softmax, `1 KB` SLM E4M3 LUT); discovered SIMT (`M=1`) vs XMX (`M=8`) split & `get_wi_data` spill | Isolated **`4.76x`** (`50.24 -> 10.56 ms`, `78%` GDDR6 BW); live graph-replayed A/B: **`+19.8%` @ 64K**, **`+39.3%` @ 128K**, **`+71.0%` @ 256K (`20.07 -> 34.32 tok/s`!)**, `4/4` + `128K` recall PASS | **SHIPPED (`INK2_SYCL_DECODE_ATTN=1`)** |
