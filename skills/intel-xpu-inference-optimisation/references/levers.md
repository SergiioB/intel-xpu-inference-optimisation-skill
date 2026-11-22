# Proven Intel Arc Pro B60/B70 Levers & Workload Regimes (Ordered by Measured ROI)

Every number below comes from controlled B70 measurements recorded across the Intel Arc Pro B70 Inference Cookbook (`v1.15.0`) and B70-DOCS campaigns (2026-08 through 2026-11). Use them as ranked priors, never unverified transfers to a different model.

## 1. Engine & Format Selection Matrix (By Workload Regime)

| Model Class / Regime | Proven Best Route on B70 | Measured Speed & Context | Key Caveat / Boundary |
|---|---|---|---|
| **Dense ~27B, 1x B70, lossless `0–131K`** (`Qwen3.8-27B`) | **TensorFold XPU** MLX 4-bit + DFlash2 + `XPU_ATTN_PREF=256` prefix-attn kernel | **127 / 115 / 90 / 51 tok/s** code (`0 / 8K / 32K / 131K`), **`drafted == serial`**, warm follow-ups `0.3–1.1 s` via `--prompt-cache-gib` | Prefix-attn kernel lives on fork tag; check text parity if strict upstream token-identity is required |
| **Dense ~27B, 2x B70 single stream** (`Qwen3.8-27B`) | **TensorFold `--tp 2`** (upstream PR `#3586` peer-store collectives, no atomics) | **165.5 / 143.0 / 117.5 / 76.8 tok/s** (`1.25x–1.5x` vs 1 card, `drafted == serial`); **250K bf16 KV @ `63.3 tok/s`** | Never use stock vLLM oneCCL TP2 (`DNR-004`, `-45%..-71%`); use independent servers for multi-user |
| **Dense ~27B, 1x B70 `131K–262K` native** (`Qwen3.8-27B`) | **exl3xpu** EXL3 trellis 4bpw + fp8 KV (`287K` KV pool) | **63.25 tok/s** w/ MTP3 (`245,583`-tok real prompt completed at card sampling); C16 aggregate `426 tok/s`; int8 prefill + oneDNN SDPA | **MTP3 alters 27% of greedy outputs (`DNR-017`)** — turn MTP off for exact one-shot code. Dense only (`DNR-011`). Never use GPTQ/AutoRound W4A16 on Qwen3.8-27B (`DNR-009`, answers `30` not `14`) |
| **Looped Native NVFP4/MXFP8 MoE** (`Inkling 2 Mini 26.4B`) | **vLLM XPU** + `vllm-inkling2-xpu` + PR `#69186` (+ opt-in CUTLASS Xe2 `INK2_CUTLASS_MOE=1` + DSpark `k=7`) | **80.4–55.3 tok/s** (`0–104K` card sampling; `effort=0.2` **99.2 tok/s**), **3,920–2,891 tok/s** cold prefill, **8/8 PASS**; **163K DSpark / 256K (`308.7K` KV) no-spec** via SWA `mbt=1024/512` | Never convert native NVFP4 to GPTQ/EXL3 (`DNR-035`, `DNR-041`); DSpark alters greedy hashes |
| **Mid Hybrid Mamba/MoE** (`Nemotron-3.5-Lightning-30B-A3B`) | **vLLM XPU** GPTQ-INT4 G64 + DFlash `n=7` + grouped-topk patch + SSU B8/W4 tuning | **186.6 tok/s** (`p2048/g128`, 150 W) | Always send via chat template — raw `/v1/completions` loops `14/18` vs chat `0/18` (`DNR-016`) |
| **Large MoE 74B–125B** (`Bolt-74B`, `Flash-Next 125B-A6B`, `Ornith-122B`) | **Strata / XeStrata SYCL** aligned `IQ3_XXS`/`IQ3_S` + MTP (`--spec 4 --spec-min-p 0.5`), or **llama.cpp SYCL** (`GGML_SYCL_GROUPED_MAX_ROWS=0`, `mul_mat_vec_q_moe_mt`), or **vLLM TP2 AutoRound** (`patch_xpu_moe_topk_pad.py`) | Flash-Next Strata 1x B70: **48.03 tok/s** decode / **989 tok/s** 8K prefill (`2.22x` decode / `5.2x` prefill vs dual-split llama.cpp); Ornith-122B 2x B70: `28.1 tok/s` / `700 tok/s` pp2048 | **Residency before streaming**; check per-tensor GGUF census (`DNR-014`) and 16-byte block alignment |
| **Dense 65B / Vision GGUF** (`Ripple-65B`, `Muse-Glimmer-30B`) | **llama.cpp SYCL** built with `GGML_SYCL_F16=ON` + AOT `bmg_g31` + `q8_0/q8_0` KV | Ripple-65B 2x B70 `-sm tensor`: **720 tok/s** pp2048 (`3.3x` vs F32), **14.5 tok/s** decode, 11-turn 118K `0.0%` sag | Requires post-2026-10-29 `#31125` build for fast `q8_0`/`q4_0` KV at 64K (`DNR-019`) |

## 2. Ranked Cross-Engine Levers (Cheapest / Highest ROI First)

### Tier 1 — Architecture & Config Levers (Zero Custom Kernels)
1. **Rebase before tuning:** Check upstream PRs/commits first (e.g. llama.cpp `#31125` SYCL flash-attn gave `1.8–2.1x` at 64K; vLLM `#69186` gave `+39–42%`; TensorFold `#3586` gave `1.25–1.5x` TP2).
2. **Preserve native packed quantization:** Serve native NVFP4/MXFP8 directly before converting to GPTQ/GGUF (`DNR-009`, `DNR-040`, `DNR-041`).
3. **Eliminate hidden FP32 weight casts & host round-trips:** vLLM PR `#69186` (`torch.mm(..., out_dtype=torch.float32)`) delivered **+42.1% short / +38.7% 8K decode, -30% J/tok, +1.22 GiB KV** in 2 lines.
4. **Enable graph replay & persist JIT caches:** `-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY` on vLLM XPU; `GGML_SYCL_ENABLE_GRAPH=1` on llama.cpp SYCL (`+34.1%` short / `+26.6%` 8K on Inkling, 5/5 greedy identical). Keep Intel NEO direct submission ON (`DNR-020`: turning it off costs `-6%` 1-card, `-23%` 2-card).
5. **Tune SWA `--max-num-batched-tokens`:** On SWA-heavy vLLM models, `mbt=4096/8192` collapses context to `~7K`, while `mbt=1024`/`512` unlocks `163K` DSpark and `256K` (`308.7K` KV) no-spec.
6. **Match KV quantization to kernel coverage:**
   - `llama.cpp SYCL` (`>= 2026-10-29`, `#31125`): `q8_0/q8_0` (`22.3 tok/s`) or `q4_0/q4_0` (`23.9 tok/s`) beats `q5_0/q4_1` (`10.5 tok/s`) by **1.8x–2.1x** at 64K (`DNR-019`).
   - `exl3xpu`: `fp8` KV unlocks `262K` on one B70 (`287K` pool).
   - `TensorFold`: `bf16` KV for `0–131K` (or 2x B70 `250K`); `int8` KV for 1-card `250K` (`DNR-023`).
7. **Build `llama.cpp SYCL` with `GGML_SYCL_F16=ON` + AOT `bmg_g31`:** Lifted Ripple-65B pp2048 from `218 -> 720 tok/s` (**3.3x**). Keep `GGML_SYCL_ENABLE_OPT=1` (`DNR-037`: `=0` loses `-29%`) and `SYCL_CACHE_PERSISTENT=0` (`DNR-031`).
8. **Dual-B70 placement:** Two independent single-card servers for multi-user load; peer-store TP2 without atomics (`TensorFold --tp 2` PR `#3586`, `llama.cpp` `GGML_SYCL_TP_P2P=2`) for single-stream latency. Never stock vLLM oneCCL TP2 (`DNR-004`).

### Tier 2 — Speculative Decoding & Prompt Caching
9. **Speculative decoding (`DFlash2`, `DSpark`, `MTP`):**
   - Prefer lossless verifiers (`TensorFold` DFlash2: `drafted == serial`).
   - Measure `step_ms`, accepted length `a`, and per-position survival separately.
   - Pad TopK cleanly when hardware/kernel requires it (Bolt 74B TopK `9 -> 10` with zero router weight, `patch_xpu_moe_topk_pad.py`).
   - Remember draft-side quantization is an independent axis (`+39–49%` when separately quantizable).
10. **Prefix / prompt caching (engine-specific!):**
    - `exl3xpu`: measured `5.4x` cache win (requires zero-hit-delta + leading-entropy guard on cold cells).
    - `TensorFold`: `--prompt-cache-gib` sized to one state gives `0.3–1.1 s` warm follow-ups up to `131K`.
    - `vLLM XPU` W4A16/speculative stacks: **keep OFF (`--no-enable-prefix-caching`)** until multi-turn corruption checks pass.

### Tier 3 — Custom AOT `bmg_g31` Xe2 Kernels (After Config Levers Plateau)
11. **16-byte aligned weight repacking:** Repacking unaligned blocks (`IQ3_XXS` 98 B, `IQ4_XS` 136 B) to 16-byte aligned layouts yields **2.3x–4.7x** memory throughput on Xe2.
12. **Native NVFP4 + MXFP8 CUTLASS Xe2 grouped GEMM + 1-WG SLM route-sort + fused SwiGLU:**
    - Compiled AOT for `bmg_g31` with `-ze-opt-large-register-file` against `intel/sycl-tla` + `vllm-xpu-kernels` headers.
    - Lifted Inkling 2 Mini + DSpark `k=7` by **+40.5% card-sampling decode (`57.2 -> 80.4 tok/s`), +33–36% greedy (`71.6/67.3 -> 95.1/91.7 tok/s`), +23.6% cold prefill (`3,172 -> 3,920 tok/s`)**, `rel_l2 <= 5e-5` vs FP32, **8/8 PASS**.
13. **Fused multi-token MoE verify (`mul_mat_vec_q_moe_mt`, `M=2..8`):** Essential for MTP/DSpark MoE verify so `M=4..8` does not degrade to `k` serial GEMVs or padded `BM=16` prefill tiles.
14. **Prefix-attention DPAS kernel (`XPU_ATTN_PREF=256`):** Lifted TensorFold Qwen3.8-27B long-context decode to `90 tok/s @ 32K` and `51 tok/s @ 131K` (avoid naive 256-lane SLM K/V staging which spilled registers `-5x`, `DNR-018`).
15. **True `s8 x s8` INT8 prefill GEMMs (`367 TOPS`):** The only 2x prefill compute lever on Xe2 (`exl3xpu` int8 prefill + oneDNN SDPA). Keep decode bit-exact or lossless and label prefill precision explicitly.
