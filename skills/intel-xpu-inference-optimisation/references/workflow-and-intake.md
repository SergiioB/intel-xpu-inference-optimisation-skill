# End-to-End Research & New-Recipe Approach (Zero to Shipped on Intel XPU)

How to approach a brand-new model release, an underperforming recipe, or an open optimization session on Intel Arc Pro B60/B70 (Battlemage Xe2) without wasting GPU hours.

---

## Engineering Mindset & Attitude (What Actually Drives Breakthroughs)

The tools and rooflines only work when paired with the right problem-solving attitude:

1. **Stay positive, curious, and stubbornly constructive — treat every "NotImplementedError" or slowdown as a clue, never a dead end.**
   - When a plugin says `"XPU requires --language-model-only"` or rejects `--kv-cache-dtype fp8`, inspect the checkpoint and upstream reference before accepting "unsupported." On Inkling 2 Mini, the entire 23-layer vision tower was blocked by a single PTX inline-asm line + CUDA FA (`651` lines -> ported to pure PyTorch XPU SDPA in `29.3 ms`), and FP8 KV was already supported by the underlying Triton kernels!
   - When an optimization hits a wall (`5.4x` slower FP8 prefill, then `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` at `64K`), stay calm and excited to find the *why* — isolating the exact root cause (1-byte Triton tensor descriptor + PyTorch XPU caching-allocator fragmentation) turned a failure into an **11x Chunked-LSE breakthrough** (`905 tok/s` at `496K`).
2. **Zero ego — celebrate a fast kill or clean revert just as much as a speedup.**
   - Welcome hard external questions (*"does FP8 noise degrade perplexity at 512K?"*) by running the cleanest direct measurement (`4K -> 496K` held-out PPL + dual-needle recall) rather than arguing from intuition.
   - Study other engineers' work (e.g. `Danmoreng/intel-b70-inkling2-mini-vllm`) with genuine respect — never dismiss it or panic-switch; decouple the artifact from the runtime tricks, keep what beats it, and test what might improve yours.
   - When a promising idea fails live graph-replay verification (Triton Split-KV `80/128` looked `3x` faster in isolation but regressed `-10–12%` end-to-end) or fails the 10% Impact Gate (`M=4` vs `M=8` CUTLASS NVFP4 was only `2.58 ms` = `<6%`), **revert cleanly with zero attachment** and document the lesson so nobody pays that cost twice.
3. **Think in bytes, milliseconds, and 5-second probes — never fly blind.**
   - Always ask: *"Where do the milliseconds actually live?"* and *"What is the cheapest 5-second experiment that proves or kills this hypothesis?"* before writing 500 lines of code.
   - Trust receipts over assumptions: check live `vram_mm`, `dmesg`, exact token counts, numerical parity oracles, and visually inspect every generated chart or UI screenshot.
   - Maintain joyful, steady momentum by shipping small, opt-in, tested increments continuously rather than giant risky rewrites.

---

## Phase 0 — Upstream Reconnaissance & Tensor Census (Zero GPU Time, ~5 min)

Before downloading a single gigabyte, converting weights, or writing code:

1. **Inspect the official model card, `config.json`, and raw tensor headers (`DNR-014`):**
   - Check exact weight dtypes per layer (never infer from repo name — e.g. Inkling 2 Mini mixes NVFP4 in layers 0–19, MXFP8 in layers 20–23, BF16 dense/vision/lm_head).
   - Check architecture levers in `config.json`: MoE vs dense, loop passes, Sliding Window Attention (SWA) vs full-attention ratio, `model_max_length`, K/V pre-storage clipping/normalization (predicts whether unit-scale FP8 KV is safe!), vision tower size (`model.visual.*`), and built-in speculator/drafter weights (`mtp_config`, `dspark`, `dflash`).
   - Copy the **model card's recommended sampling** (`temperature`, `top_p`, `reasoning_effort`) — this is your mandatory public headline sampling.
2. **Search upstream engines, PRs, and plugins first:**
   - Check open PRs and plugin repos across `vllm`, `vllm-xpu-kernels`, `TensorFold`, `exl3xpu`, `llama.cpp`, and `Strata` (e.g. `mgaruccio/vllm-inkling2-xpu`, vLLM PR `#69186`, TensorFold PR `#3586`, llama.cpp `#31125`).
   - Read `references/gotchas.md` (`DNR-001`..`DNR-042`) — never retry a closed class unless its **Revisit when** condition is met.
3. **Run host preflight & price the Xe2 roofline:**
   - Run `python3 scripts/b70_preflight.py`: pick a free **non-display** B70 (`is_display == False`); if forced onto a display-driving card, budget `desktop_vram_mib + 1 GiB` (typically `1.5–4.6 GiB` lost to KDE/GNOME!).
   - Run `python3 scripts/b70_ceiling.py`: compute active bytes/tok, theoretical decode ceiling (`active_bytes / 521 GB/s`), KV bytes/tok (full-attn vs SWA), and prefill TFLOPS ceiling.

### Phase 0.5 — Auditing External Community Repos & Third-Party Quants (Zero GPU Time)

When another engineer publishes an impressive B70 repo or alternate quant (e.g., `Danmoreng/intel-b70-inkling2-mini-vllm` using `EXL3-4bpw` + `trellis-serve` vs our native `NVFP4/MXFP8` `vllm-inkling2-xpu`), **never blindly switch stacks or dismiss the work.** Follow this 5-step decomposition first:

1. **Read primary files & raw JSON before touching the GPU:** Inspect `README.md`, `benchmarks/**/*.md`, `target-parity.json`, `patches/`, and kernel `csrc/` directly via `gh api`. Never import external rates as your own reproduction (`B70-DOCS §19`).
2. **Normalize every metric class & coordinate:** Separate power cap (`180 W` vs `230 W`), sampling (`top_p=0.95/top_k=20` vs `top_p=1.0`), reasoning budget, `mbt`, engine vs client tok/s, and synthetic vs real agent sessions.
3. **Decouple the *artifact/quant* from the *kernel/scheduler mechanisms*:**
   - Ask: *"Are their speedups coming from the third-party quant (`EXL3-4bpw`), or from orthogonal runtime wins we can apply directly to our official native checkpoint?"*
   - Example (`Danmoreng` vs our native Inkling 2 Mini):
     - **Keep our native `NVFP4/MXFP8` artifact** when you want zero double-quantization (EXL3 requantizes official weights with `KL ~0.0052` on 96-tok parity), `+42%` faster short/mid prefill (`3,920` vs `2,753 tok/s`), `512K` (`569K` KV) ceiling, and minimal upstream vLLM 0.34 diff.
     - **Cherry-pick their 4 orthogonal runtime mechanisms** onto our native route: (a) **Adaptive DSpark** (`0.75` confidence early-stop up to `k=7`), (b) **Dedicated fixed-size SWA ring** (decouples SWA VRAM from `mbt=4096/6144`), (c) **128-partition SYCL FP8 decode attention** (`65–77 tok/s` @ `196K–232K` vs Triton FP8 `43.6 tok/s` @ `128K`), and (d) **Hybrid oneDNN prefix SDPA** for keys outside the `1024`-token relative-bias band (`2,157 tok/s` @ `128K` even at `180 W`).
4. **Check quality & failure honesty:** Look for real end-to-end task evaluations (e.g. blind SWE-bench repairs, `compaction.keepRecentTokens=4096` for 240K Pi runs) AND parity/reuse caveats (e.g. prefix-reuse retrieval misses, KL vs official checkpoint).
5. **Run the Impact Gate before reproducing:** Only download/benchmark an external quant on GPU if it can beat your adopted route on BOTH quality and your target regime (`>= 10%`); otherwise, port the orthogonal kernel/scheduler wins onto your native checkpoint!

### Phase 0.6 — Engine Release Examination & Multi-Track Parallel Orchestration

When a new engine release drops (e.g. `TensorFold v1.3.4`, `Strata v0.2.20`, `llama.cpp` PRs, `compute-runtime 26.40+`), examine it thoroughly **off-GPU** before burning a single GPU second:

1. **Audit the exact commit delta (never trust release-note headlines alone):**
   - Check whether a new kernel is **default ON vs opt-in**, whether its speedup applies to **plain (`serial`) decode, drafted decode, or both** (e.g. TensorFold `#3960` `attn_merge` `+40% @ 121K` boosts plain decode only while drafted is level), and whether old flags/envs were removed (`XPU_ATTN_PREF` dropped in `v1.3.4`).
   - Check compiler/driver compatibility against your live `ocloc` / `icpx` / `compute-runtime` version (e.g. `-cl-intel-128-GRF-per-thread` rejected by `ocloc 26.05.1`; `-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,...` needed by `icpx 2026.1+`).
   - Check **silent behavioral overrides** (does the engine force `thinking=none`, alter chat templates, require a specific pack format like `MLX 4/32`, or need `-lv 4` verbosity to emit lazy-mmap proof logs?).
   - Run a **continuous driver/firmware radar** alongside engine releases: `intel/compute-runtime` (ULLS latency, peer-import, `unitrace` multi-card fixes), `oneapi-src/level-zero`, Linux `xe` KMD (`min_freq` BMG P2P floor), and GuC firmware (`70.65+` `#948` reset fixes).
2. **Parallelize cleanly with ONE GPU owner + off-GPU tracks:**
   - **Hard rule:** NEVER run two GPU benchmarks concurrently on the same host.
   - **Winning pattern:** Assign **one exclusive GPU owner** (live dual-B70 bring-up & benchmark) while running **1–2 zero-GPU parallel tracks**: (a) off-GPU AOT release build (`nice -19 -j 4`) + source audit + ready-to-run preflight-guarded benchmark script, (b) primary-source Breakthrough Radar, or (c) independent read-only upstream PR code reviews.
3. **Fresh-Tree Script & Patch Verification Gate (before publishing any build script or `.patch`):**
   - Never assume a build script or `.patch` works just because it ran in your dev tree — dev trees silently hide missing `-I` include paths, missing `-DTORCH_EXTENSION_NAME`, missing `-spirv-ext` flags, missing `ocloc` checks, or hand-edited patch hunk counts!
   - Test every published `build-*.sh` and `.patch` **verbatim** against a clean upstream checkout and verify `python -c "import _ext"` + idempotent second run before publishing.

---

## Phase 1 — Establish One Clean Correctness Foothold (~15 min)

1. **Pick the highest-fidelity native engine first** (`references/engines-and-topology.md` & `references/quantization.md`):
   - Native NVFP4/MXFP8 safetensors -> `vLLM XPU` (preserve native weights before trying GPTQ/GGUF!).
   - Dense 20B–32B fast interactive -> `TensorFold` (MLX 4-bit + DFlash2) or `exl3xpu` (EXL3 4bpw + fp8 KV for 245K+).
   - MoE / broad GGUF -> `llama.cpp SYCL` (`>= 2026-10-29`, `q8_0/q8_0` KV) or `Strata` (16B-aligned packs).
2. **Stand up ONE clean baseline with graph replay ON:**
   - `vLLM XPU`: `-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY`, `--no-enable-prefix-caching` on packed/speculative stacks, short `VLLM_RPC_BASE_PATH`, `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`, `ZE_AFFINITY_MASK=<gpu>`.
   - Verify: clean startup, exact KV pool tokens, 1 short real prompt (reasoning & tool parser working), and clean `SIGTERM` teardown restoring baseline VRAM (`32,615 MiB`) and clean `dmesg`.

---

## Phase 2 — Hypothesis-Driven Bottleneck Decomposition (Never Guess!)

Compare your baseline decode & prefill rates against your Phase 0 roofline. If the gap is `> 15%`, decompose **where the milliseconds actually go** in strict ROI order:

### Step 2.1 — Kill Trivial Host / Dtype / Launch Bugs First (Minutes, +20%–50% ROI)
- Audit `lm_head`, norms, router ops, and logits processors for accidental FP32 weight casts or per-step CPU syncs.
- Example: vLLM PR `#69186` fixed a 2-line FP32 `lm_head` cast -> **+42% decode, -30% J/tok, +1.22 GiB KV** before writing a single GPU kernel!

### Step 2.2 — Sweep Scheduler, Memory & Power Knobs Next
- **SWA context unlock:** Sweep `--max-num-batched-tokens` (`2048` for speed ladders, `1024`/`512` to cut SWA reservations 2x–3x; never `4096`/`8192`).
- **Power cap regime:** Check `150 W` for MoE models (can beat `230 W`!) vs `180–230 W` for dense/looped models.
- **Speculative decoding:** Measure target `step_ms`, draft `step_ms`, and mean accepted length `a` separately (`tok/s = a / step_ms`); sweep `k` and check greedy parity (`drafted == serial` vs altered hashes).

### Step 2.3 — Unlock Missing Features by Diffing Against Upstream NVIDIA/Reference Code
- If a plugin refuses `--kv-cache-dtype fp8` or Vision (`--language-model-only`), read the upstream NVIDIA implementation sitting next to it!
- **Vision Tower port pattern:** Replace CUDA PTX inline asm and CUDA FA2/3/4 with vectorized PyTorch XPU ops + native `F.scaled_dot_product_attention(..., is_causal=False)` -> zero JIT stall, `29.3 ms` for 2 images across 23 BF16 ViT layers, verified vs CPU reference (`cos = 0.9994`)!
- **FP8 KV enablement pattern:** View `uint8` storage as `torch.float8_e4m3fn` and verify Triton XPU load/store in a 2-second micro-test first!

### Step 2.4 — Microbenchmark & Parity-Test Every Kernel Change in 5 Seconds BEFORE Full-Server Runs
- Apply the **Impact Gate (`>= 10%` removable wall-time share)** before writing SYCL or Triton code.
- Write a **5-second standalone op microbenchmark** on the exact hot shape (`M=8` verify, `M=2048` prefill @ 64K/128K):
  - This caught the **5.4x–11x Triton XPU 1-byte FP8 `tl.make_tensor_descriptor` slowdown** (`256.6 ms` vs `46.8 ms` BF16 @ 64K; `~510 ms` @ 128K) in 5 seconds instead of guessing!
  - And it proved the **Chunked-LSE 256 MiB reusable BF16 scratch fix** (`46.4 ms` @ 128K = **11x faster**, `lse_max_err = 5.7e-6`, zero PyTorch XPU caching-allocator fragmentation)!
- Always verify **exact numerical parity** (exhaustive bit patterns, `rel_l2 <= 5e-5` vs FP32/CPU oracle) before touching the live server.
- Always **break `uv` hardlinks** before patching site-packages, and **gate custom kernels behind an opt-in env flag** (default `0` = stock).

---

## Phase 3 — Full Qualification Across the Entire Context Window

Never stop at a short smoke or a raw KV-pool printout:

1. **Speed ladder (model-card sampling, `n >= 3` seeds, zero prefix cache):** Measure `0, 8K, 32K, 64K, 128K, max` with unique entropy-first prompts (`prefix_cache_hits_total == 0.0`). Count both `delta.reasoning` and `delta.content`. Keep greedy (`temp=0`) strictly as a separate diagnostic table.
2. **Separate capacity probes:** Keep lower-`mbt` max-context configs (`mbt=1024/512`) in their own capacity table, proven with real completed prompts.
3. **Quality, Perplexity & Retrieval Gates:**
   - Deterministic task suite (`8+` checks including `"14" not "30"`, JSON, code trace, multi-turn correction).
   - **For quantized/FP8 KV:** Measure **matched BF16 vs FP8 held-out natural-prose perplexity** (`4K -> max_ctx` via `/v1/completions` `echo=True, logprobs=1`) + **`15%` oldest / `55%` mid dual-needle exact recall** up to max context (use `max_tokens >= 384` on thinking models so reasoning never starves `content`!).
   - **For Vision:** Live real-image OCR (e.g. reading numbers off a chart PNG), synthetic visual counting, multi-image comparison, and pure-text regression check.
4. **Hardware Cleanliness Proof:** Verify zero stray processes (`ps` filtered via `awk`, never `pkill -f` matching your own shell!), `vram_mm` restored to baseline, unchanged power caps, and clean `dmesg`.

---

## Phase 4 — Packaging, Visuals & Release Checklist

1. **Cookbook:** Sanitize all private paths/PCI IDs, commit raw JSON evidence & patches first, pin commit SHA in `data/benchmarks.v1.json` (`benchmark` vs `capability`), regenerate catalog & site feeds, pass all 12 `scripts/check-repo.py` gates, merge with `--merge`.
2. **Launcher & Portal (`b70-launcher` + `xecores.com`):** Add clean `ctx`/`ctx_safe`/`ctx_max`, `kv_options`, `vision` toggle, display-GPU VRAM adaptation, unit tests, deterministic tarball verification (`packaging/release.py --verify`), and live `post-deploy-check.mjs` (7/7 PASS).
3. **Charts & Communication:** Render 2x anti-aliased PNGs, visually inspect every PNG with `read` for zero overlap/clipping, state exact scope and honest limits, draft human no-slop posts (never auto-post), and fold every new discovery back into `SESSIONS.md` and this skill!
