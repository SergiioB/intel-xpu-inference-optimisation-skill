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
9. **Hybrid oneDNN Prefix SDPA + Relative-Bias Tail & Dedicated SWA Device Rings (from external source review of `Danmoreng/intel-b70-inkling2-mini-vllm`):**
   - Outside a model's finite relative-bias band (`dist >= 1024`), full attention has zero relative bias — older prefix keys can run on native oneDNN SDPA and merge with the short relative-bias causal tail via FP32 LSE (`2,157 tok/s` @ 128K at 180 W).
   - Storing local SWA layers in a dedicated fixed-size `1024`-token device ring completely decouples SWA VRAM from vLLM's `--max-num-batched-tokens`, enabling `mbt=4096/6144` alongside `262K` context + Vision + DSpark.
   - Using `128` KV partitions in native SYCL FP8 decode attention avoids Triton FP8 decode bottlenecks at `128K–256K` (`65–77 tok/s` vs `43.6 tok/s`), and **Adaptive DSpark** (0.75 confidence cutoff up to `k=7`) cuts wasted verify rounds on low-confidence steps.
10. **Always gate custom kernels behind an opt-in env flag** (default `0` = exact stock behavior) and test **under graph replay** (`FULL_DECODE_ONLY`), never only eager mode!

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

## 3. Mandatory Kernel Verification Checklist

Before running a full benchmark with any new/modified Xe2 kernel:
1. **Unit parity vs exact FP32/FP64 reference** across every row count `M` served (`M = 1, 2, 4, 8, 16, 64, 512, 2048`) AND edge-case scales (including E4M3 subnormals, zero rows, masked padding tokens).
2. **Graph capture & replay check:** Verify the kernel runs identically inside `FULL_DECODE_ONLY` XPU graph replay vs eager mode (no host-side dynamic allocations or broken pointer captures!).
3. **Full-round attribution:** Interleaved `control -> candidate -> control` on the actual engine of record.
4. **Correctness lane + scenario suite:** Exact-answer battery + multi-depth retrieval + `dmesg` clean + VRAM restored to baseline.
