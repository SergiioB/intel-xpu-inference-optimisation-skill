# Battlemage Xe2 (`bmg_g31`) GPU Kernel Engineering Guide

Only write or modify GPU kernels after (1) config/engine levers plateau, (2) profiling proves a removable wall-time share that clears `>= 10%` end-to-end, and (3) your correctness lane (Class 1 / Class 2 / Class 3) is frozen in `PLAN.md`.

---

## 1. What Wins on Intel Xe2 (`bmg_g31`)

1. **AOT compilation with large GRF mode:**
   ```bash
   icpx -fsycl -fsycl-targets=spir64_gen \
     -Xs "-device bmg_g31 -options -ze-opt-large-register-file" \
     -O3 -std=c++20 -fPIC -shared \
     -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED
   ```
   - `128 KB SLM` per Xe-core; `256` GRF (`-ze-opt-large-register-file`) prevents catastrophic scratch-memory spills on CUTLASS XMX tiles.
   - Always compile against pinned `intel/sycl-tla` + `vllm-xpu-kernels` headers so `cuda_runtime_api.h` is never pulled in.
2. **16-byte aligned weight layouts:** Xe2 splits misaligned 16-byte global loads (`2.3x–4.7x` measured throughput delta in Strata). Repack unaligned formats (`IQ3_XXS` 98 B, `IQ4_XS` 136 B, `Q8_0` 34 B) or use planar register layouts (stride-3 register regions turn into slow per-element moves!).
3. **CUTLASS Xe2 grouped GEMM + 1-WG SLM route-sort + fused activation (`M=8..2048`):**
   - On Inkling 2 Mini, native NVFP4 (`e2m1`+`e4m3`/16, branchless exact E4M3 subnormal+normal scale decode, `rel_l2 <= 5e-5` vs FP32) CUTLASS Xe2 + `vllm-xpu-kernels` MXFP8 CUTLASS Xe2 + 1-WG SLM route-sort + fused softcapped SwiGLU cut `M=8` verify 44-visit MoE time from ~93 ms to ~25 ms and lifted end-to-end card-sampling decode **+40.5% (`57.2 -> 80.4 tok/s`)** and cold prefill **+23.6% (`3,172 -> 3,920 tok/s`)**.
   - Keep a safe fallback to PyTorch `argsort`/`scatter_add` when route count exceeds the 1-WG SLM sort ceiling.
4. **Fused multi-token MoE GEMV (`M=2..8`, e.g. `mul_mat_vec_q_moe_mt`):** Prevents speculative verification from degrading into `k` serial GEMVs or padded `BM=16` prefill tiles.
5. **Prefix-attention DPAS without wide SLM staging (`XPU_ATTN_PREF=256`):** Splits long-context prefix attention cleanly across Xe2 DPAS units (`90 tok/s @ 32K`, `51 tok/s @ 131K` on TensorFold 27B).
6. **Peer-store dual-card collectives without remote atomics:** Write partial buffers directly across PCIe (`~19.9 us` for 160 KiB) and synchronize with a lightweight flag — never cross-queue `memcpy` events (`DNR-032`) or remote atomics (absent on B70 PCIe).
7. **FP8 KV prefill via ONE reusable 256 MiB BF16 scratch + Chunked-LSE merge (`5.4x–11x` speedup up to 512K):** On Intel Triton XPU, 1-byte `float8_e4m3fn` / `uint8` `tl.make_tensor_descriptor` loads break Xe2's 16-bit DPAS block layout (`256.6 ms` inline FP8 vs `46.8 ms` BF16 at 64K; `~510 ms` at 128K!). Materializing active pages into ONE fixed-size module-level BF16 scratch (`<= 256 MiB`, filled in 128-page slices) and — above 64K — looping over 64K key windows (`k_lo_tok` / `k_hi_tok`) with exact FP32 online FlashAttention LSE merging (`lse_max_err = 5.7e-6`) runs native BF16 2D-block-load attention at **`46.4 ms` (`11x` faster at 128K, `91%–97%` of BF16 prefill all the way to 512K: `1,822 / 1,378 / 1,055 / 905 tok/s` at `128K / 245K / 384K / 496K`!)**. Covers 100% of SWA layers (`~12 MiB`) in 1 pass. Never allocate dynamic `torch.empty((n_active, ...))` per prefill chunk or PyTorch XPU's caching allocator fragments multi-GiB and hits `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`!
8. **Porting ViT Vision Towers to Intel XPU without PTX/CUDA:** Replace CUDA PTX `cvt.rn.bf16.f32` inline asm and FA2/FA3/FA4 `flash_attn_varlen_func(causal=False)` with vectorized PyTorch XPU ops + native `F.scaled_dot_product_attention(..., is_causal=False)` per image slice — zero Triton JIT stall, `29.3 ms` for 2 images across 23 BF16 ViT layers (`cos = 0.9994` vs CPU BF16, `+0.55 GiB` VRAM, 4/4 live multimodal PASS).
9. **ESIMD `PagedXmx<M>` (`M=1..8`, BF16 & FP8) Supersedes Both Triton `tl.dot` AND SIMT `M=1` Decode Attention:**
   - **Why Triton `tl.dot` and `icpx` `joint_matrix` both fail long-context decode:** Triton `tl.dot` pads `M=1, G=3` into `16` rows (`81%` wasted -> `50.24 ms` @ 256K FP8) and suffers sequential combine/L2 overhead on high `constexpr` splits (`DNR-030`); meanwhile `icpx 2026.1` `sycl::ext::oneapi::detail::get_wi_data(sg, acc)` inside a K-loop spills the entire `joint_matrix` accumulator (`23.7 -> 70.6 ms` = `3x` regression!).
   - **Why ESIMD `PagedXmx<M>` (`workgroup=1`, zero SLM, zero subgroups, zero barriers) wins across ALL `M=1..8`:**
     - Grid = `(num_splits, KV_H, ceil(G*M / 8))`: packs `R = G*M` query rows directly into `8`-row XMX tiles (`M=1` = 1 tile, `M=8` = 3 tiles).
     - Hardware `lsc_load_2d<u32,4,16,transpose>` for K and `lsc_load_2d<u8/u16,16,16,vnni>` for V — zero cross-lane movement, zero `get_wi_data`, 100% GRF under `-ze-opt-large-register-file`.
     - **3-op branchless SIMD FP8 E4M3 -> FP16 dequant (zero LUT!):** `((u & 127) << 7) | ((u & 128) << 8)` then `× 256.0f` — beats even our `4.76x` SYCL SIMT `M=1` SLM-LUT kernel (`1,025 us` vs `1,407 us` @ 256K FP8 = right on the `~8.2 ms` `521 GB/s` floor!) AND keeps `M=8` (`206 GFLOPs` @ 256K) on hardware `xmx::dpas<8,8>`!
     - **Runtime split grid + small parallel combine (`EsimdCombineGated`) + fixed per-window splits (`SWA=16, FULL=128`):** dodges `DNR-030` (zero re-JIT, zero sequential combine stall — live split sweep `{32, 64, 96, 128}` stayed within `±3%` with `128` winning deep context!).
     - **Live verified on Card 1 (`INK2_ESIMD_DECODE_ATTN=1`, Vision + FP8 KV + DSpark `k=7`, `ctx=262144`, `FULL_DECODE_ONLY`):** `24/24` kernel parity PASS (`rel_l2 = 2.43–2.63e-3`, `max_abs <= 2.4e-4`), `2.46x / 1.71x / 1.90x` full 48-visit layer mix vs Triton, live decode **`54.1 -> 68.0` (+12% step/s @ 64K), `59.6 -> 76.1` (+19% @ 128K), `37.6 -> 56.6 tok/s` (+33% @ 245K)**, and **`-22%` to `-34%` J/token** (`6.12 -> 4.06 J/tok` @ 245K), `391`/`14`/JSON/`128K` needle/Vision OCR PASS, `0` faults!
   - **Historical SIMT `M=1` baseline (still valid when ESIMD is unavailable):** Zero-spill 4-SG/WG SIMT `SingleTokSplitKernel` hit **`4.76x`** (`50.24 -> 10.56 ms`) and lifted live graph-replayed `M=1` decode **`+19.8%` @ 64K, `+39.3%` @ 128K, `+71.0%` @ 256K (`20.07 -> 34.32 tok/s`)**, while proving `M=8` SIMT is compute-bound (`109.8 ms` vs `40.7 ms` XMX).
10. **Chunked (`<= 8K`) oneDNN Prefix SDPA + Triton Relative-Bias Tail (`+12.7%` @ 64K, `+19.9%` @ 128K, `+27.6%` @ 245K LIVE!):**
    - Outside a model's finite relative-bias band (`dist >= 1024`), full attention has **zero relative bias**. Running the bias-free prefix in **`<= 8K`-token chunks** through native oneDNN flash SDPA (with ONE persistent `16 MiB` scratch + reshape gather, zero per-window allocations!) and merging with the `1024`-token Triton relative-bias tail via FP32 LSE jumped live native Inkling FP8+Vision+DSpark cold prefill **`2,706 -> 3,050 tok/s` (`+12.7%` @ 64K)**, **`2,149 -> 2,576 tok/s` (`+19.9%` @ 128K)**, and **`1,586 -> 2,024 tok/s` (`+27.6%` @ 245K)** — with zero `dmesg` faults and `128K` exact needle + Vision OCR PASS!
    - **Critical oneDNN SDPA trap:** Single-shot oneDNN SDPA degrades superlinearly past `~32K` windows (slower than Triton!); **chunking to `<= 8K` windows with a tiny persistent scratch is mandatory.**
    - **SWA ring is a capacity/`mbt` lever, NOT a decode-speed lever:** At `M=8` verify, 40 local `1024`-token SWA visits are **launch-bound (`1.94 ms` FP8 vs `1.95 ms` BF16 vs `1.93 ms` ring = `0%` speed delta)**. Build a fixed `(1024 + mbt)` SWA ring to unlock `mbt=4096` at `256K` (`+3.31 GiB` KV saved), not to speed up decode!
11. **Exact-Tail Single-Slot Prefix Caching (`RingPrefix` + `PrefillTail`, `INK2_RING_PREFIX_CACHE=1` — `49.6 s -> 0.21 s` exact resend & `0.47 s` new-turn @ `128K`, `0.35–0.38 s` every turn in a 5-turn `75K` session!):**
    - Prefix-cache divergence on hybrid conv/SWA/speculative models (vLLM `#53912` class — reproduced live on flag-off stock vLLM diverging at token 12–72 with AND without DSpark!) comes from missing recurrent/ring snapshots AND **chunk-size-dependent numerics on the final tail**.
    - Cap hits to a hit-aligned frontier (`RingPrefix`), snapshot conv/KV rows (`slack + W - 1` rows so all short-conv taps at any admitted boundary stay covered!) + last `slack` token IDs/hidden rows (`PrefillTail`), replay exact tail outputs, fail closed on unreachable hits, and verify with the **`cold -> resend-1 -> resend-2 -> +new-turn -> new-salt`** exact SHA-256 + multi-turn gate (**live verified on Card 1 across previously-vetoed `mod 128 = 44..68` lengths:** `18K: 5.35 -> 0.05 s` resend / `0.18–0.23 s` new-turn; `75K: 25.4 -> 0.14 s` resend / `0.33 s` new-turn; `128K: 49.6 -> 0.21 s` resend / `0.47 s` new-turn — `105x–236x` TTFT! **5-turn growing `75K` session:** every turn hit `cached=75,008` in **`0.35–0.38 s`** and Turn 4 recalled needles from BOTH Turn 0 and Turn 2!).
    - **4 mandatory vLLM-0.34 XPU invariants for hybrid prefix caching:**
      1. **Size `slack` from `cache_hit_alignment_tokens = LCM(group_block_sizes)`, NOT `block_size` alone!** On hybrid conv/SWA/full-attn models, vLLM aligns prefix hits to `hit_tokens = LCM(16, 128, 64) = 128` (`hash_tokens = 16`). A hardcoded `slack = 32` only admits `prompt mod 128 ∈ [9, 32]` (**~19% hit rate — 81% missed!**). Set **`slack = max(32, hit_tokens + 8)` (= `136`)** and scan `hit_tokens`-aligned boundaries so **~100% of prompt lengths & growing multi-turn turns hit**!
      2. **Multimodal models null `input_ids` (`model_runner.py`):** set `requires_raw_input_tokens = True` (opt-in behind the flag!) so raw `input_ids` flow alongside `inputs_embeds`, and skip tail capture when `IMAGE_PAD_ID` is present.
      3. **Speculative decode re-prefills (`DSpark`):** scope the fail-closed coverage guard to a fresh request's first forward via block-table page fingerprint (`block_table[0,0]`) and keep `ring_high` monotonic (`max`).
      4. **Wider state window (`slack + W - 1`):** capture `W - 1` extra cache rows ahead of the `slack` token-ID tail so short-conv kernel taps at the hit boundary never read stale slots.
12. **Always gate custom kernels behind an opt-in env flag** (default `0` = exact stock behavior), **reserve persistent scratches before vLLM startup KV sizing**, **pass CPU batch metadata instead of per-layer `.item()` device syncs**, and test **under graph replay** (`FULL_DECODE_ONLY`), never only eager mode!

---

## 2. What FAILS on Intel Xe2 (Closed Kernel Classes — Do Not Waste Time)

| Closed Pattern | Measured Outcome on B70 | Root Cause |
|---|---|---|
| **Persistent Megakernels (`ADR-0009`)** | Closed by architecture audit | No safe global cross-workgroup barriers across all 32 Xe-cores; persistent occupancy conflicts with 256-GRF XMX tiles; decode is bandwidth/issue-bound, not launch-bound once graphs are ON |
| **Wide 256-lane WG SLM K/V staging (`DNR-018`)** | **5x SLOWER** (`46.6 -> 9.0 tok/s` at 32K) | Register pressure (`6 Q heads x 16 lanes` per subgroup @ 256 GRF) spills to scratch |
| **Fusing tiny kernels under graph replay** | `0%` end-to-end gain | XPU graphs already amortize launch overhead; wall time lives inside the large GEMM/GEMV/attention kernels |
| **Software pipelining / manual LSC prefetch in decode GEMV** | `M=16` `39 -> 86 ms` (slower) or noise | Register spills destroy occupancy on Xe2 |
| **Lookup-table (LUT) dequant codebooks** | **4x slower** than ALU bit-unpack | SLM bank conflicts & latency lose to register ALU bit-ops on Xe2 |
| **Subgroup-shuffle probe trellis without DPAS (`DNR-029`)** | `28.7x` (`M=1`) to `62x` (`M=8`) slower than DPAS INT4 | 4 shuffles + 32-bit mul without DPAS starves XMX |
| **Blind microbench-driven GEMM swaps (`DNR-030`)** | `-1.1%` to `-6.1%` end-to-end (`XPU_GU_WIDE64=1`) | Isolated kernel time did not shrink the critical path of the full speculative round |
| **2-card `unitrace --device-timing` (`DNR-022`, `DNR-028`)** | Hangs, segfaults, `xe` Engine memory CAT error | Cross-device timestamped event wait bug on `compute-runtime 26.05` (use `--host-timing` or per-rank profilers) |

---

## 3. Device-Side Graph-Safe Grouped MoE Prefill & Fail-Closed Admission (`vllm-hf-int4-g16-xpu v0.2.0`, v3.9.0)

- **Pair vs grouped MoE crossover (`432` quiet-host cooled samples, `M = 1..512`):**
  - Keep the low-overhead pair dispatch path for small `M` (`M = 1..16`; at `M=16`, Qwen eager grouped dipped to `0.903x` pair).
  - Switch to **device-side histogram -> prefix scan -> token scatter -> grouped W4A16 GEMM** (zero host sync, zero host sort, XPU-graph-safe, bit-exact vs pair!) from `M >= 32` onward: **`3.41x` eager / `3.52x` graph** faster on `Gemma-4-26B` (`128E`) and **`3.27x` / `3.46x`** on `Qwen3.6-35B` (`256E`) at `M=512`.
  - Across `M = 8..64`, packed 4-bit beats stock BF16 by **`1.48x–2.06x` eager** and **`2.18x–2.73x` graph replay**. At `M=512`, BF16 still beats W4A16 packed (`3.97` vs `7.17 ms` Gemma; `3.36` vs `4.59 ms` Qwen) until DPAS Step B grouped MoE prefill lands.
- **Mandatory fail-closed Xe2 2D block-I/O & MoE admission invariants:**
  1. Validate **effective q-pack base 64-byte alignment**, legal 2D block width/pitch/surface geometry, and `N >= 16`, `N % 16 == 0` (safe fallback under `auto`, hard fail under forced native).
  2. Reject empty expert banks (`E == 0`), validate `int64` routing IDs before narrowing to `int32`, and assert `processed == admitted` before first forward so missing expert banks can never silently fall through to uninitialized BF16 weights.
  3. Never encode raw-bank provenance as a spoofable string prefix (e.g. `"block_sparse_moe:"`) — use a typed key `tuple[bool, str]` (`is_raw_bank, member_name`) and cover it with a negative spoofing unit test!

---

## 4. Mandatory Kernel Verification Checklist

Before running a full benchmark with any new/modified Xe2 kernel:
1. **Unit parity vs exact FP32/FP64 reference** across every row count `M` served (`M = 1, 2, 4, 8, 16, 64, 512, 2048`) AND edge-case scales (including E4M3 subnormals, zero rows, masked padding tokens).
2. **Graph capture & replay check:** Verify the kernel runs identically inside `FULL_DECODE_ONLY` XPU graph replay vs eager mode (no host-side dynamic allocations or broken pointer captures!).
3. **Full-round attribution:** Interleaved `control -> candidate -> control` on the actual engine of record.
4. **Correctness lane + scenario suite:** Exact-answer battery + multi-depth retrieval + `dmesg` clean + VRAM restored to baseline.
