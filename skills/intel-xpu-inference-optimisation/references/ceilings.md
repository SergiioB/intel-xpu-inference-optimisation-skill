# Intel Arc Pro B60/B70 (Battlemage Xe2) Theoretical Ceilings & Silicon Datapaths

Always compute the Intel Xe2 silicon roofline before running multi-hour sweeps or writing GPU kernels. Spec-sheet peaks overstate real throughput; use measured B70 datapath and bandwidth anchors from the cookbook & B70-DOCS campaigns.

## 1. Intel Xe2 (`bmg_g31`) Silicon Datapaths & Measured Anchors

| Hardware Path (per Xe-core per clock) | Peak Rate | Practical Role on Arc Pro B60/B70 |
|---|---:|---|
| **XMX FP16 / BF16 (`DPAS`)** | 2,048 ops (~157 TFLOPS) | Where every current W4A16/W8A16 decode & prefill lands after dequant (~75–85 TFLOPS practical ~50% MFU at 230 W) |
| **XMX INT8 (`s8 x s8`)** | 4,096 ops (**367 TOPS**) | **Only fully exposed 2x prefill compute path on Xe2** (used by `exl3xpu` int8 prefill; unused by most stock paths) |
| **XMX INT4 / INT2 (`int x int`)** | 8,192 ops | Hardware-real 2x per instruction via ESIMD only; requires 4-bit activations |
| **FP8 / MXFP4 / NVFP4** | **No DPAS on Xe2** | Bandwidth-saving byte-storage formats only; unpacked in registers to FP16/BF16 |
| **DP4A (vector)** | 1,024 ops | `llama.cpp` SYCL MMVQ decode with `Q8_1` activations (`M <= 8`) |

| Memory & Interconnect Fact | Measured / Real Anchor | Notes |
|---|---|---|
| Pin GDDR6 bandwidth (B70 32 GB) | 608 GB/s spec -> **521–528 GB/s** sustained copy | Use **525 GB/s** for practical decode rooflines; `~532–549 GB/s` achieved by TensorFold MLX INT4 (`87–90%` of pin), `493 GB/s` aligned `Q6_K` (Strata), `381 GB/s` EXL3 `M=4` verify (`63%`) |
| Visible VRAM (`vram_mm`) | ~32,615–32,656 MiB per B70 | Keep `>= 1,500–2,048 MiB` free on C1; `< 500 MiB` hard abort; never allocate beyond `free - 4 GiB` in diagnostics |
| **16-byte block alignment** | **2.3x–4.7x** measured impact (Strata) | Xe2 splits misaligned 16-byte loads! `IQ4_XS` (136 B), `IQ3_XXS` (98 B), `Q8_0` (34 B) are NOT 16-byte aligned; aligned native packs win big |
| SLM & GRF per Xe-core | **128 KB SLM**; `128` or `256` GRF (`-ze-opt-large-register-file`), SG `16/32` | Always compile AOT CUTLASS Xe2 with `-Xs "-device bmg_g31 -options -ze-opt-large-register-file"` |
| Dual-card PCIe (x8/x8) | **11.1–11.9 GB/s copy-only P2P, NO peer atomics** | Stock oneCCL TP2 loses `-45%..-71%`; peer-store collectives without atomics (`TensorFold --tp 2` PR `#3586`, `GGML_SYCL_TP_P2P=2`) win `1.25x–1.5x` (`~19.9 us` per 160 KiB exchange) |

*(For Arc Pro B60 24 GiB, scale VRAM to 24 GiB and measure device-copy bandwidth live before pricing.)*

## 2. Decode Ceiling (Memory-Bandwidth Bound)

Bytes read from GDDR6 per generated token at context length `L`:

```text
B_token(L) = B_dense_visits
           + B_active_experts_visits
           + B_lm_head
           + B_kv_read(L)
           + B_recurrent_state
```

Watch four common counting errors across real B70 model families:
1. **Looped architectures (`Inkling 2 Mini`):** count *visits*, not stored layers (24 stored layers, layers 6–17 run 3 passes -> **48 visits**, `5.4B` effective active vs `3.4B` physical).
2. **Large-vocabulary `lm_head`:** `vocab=201K–248K` in BF16 is `0.93–1.2 GB` (`15–20%` of the entire decode step!). Recasting it to FP32 per step reads/writes an extra `1.72 GB` (why vLLM PR `#69186` gained `+39–42%`).
3. **Hybrid / SWA / Mamba / GDN KV reads:** only full-attention visits grow linearly with `L` (`32–38 KiB/tok` fp8 on Qwen3.8-27B — co-dominant with weights past `~100K`; `32 KiB/tok` BF16 on Inkling's 8 full visits while 40 SWA visits cap at `1,024`).
4. **MoE residency vs PCIe streaming (`Qwen3.8-Flash-Next 125B-A6B`, `Ornith-1.5-122B`):** compare total expert bytes to available 1-card or 2-card VRAM first. A PCIe miss stream at `11.5 GB/s` costs a third of a Flash-Next round (`residency before streaming`).

```text
decode_ceiling_tok_s(L) = BW_sustained_B_s / B_token(L)
```

### Speculative Decoding Ceiling (`DFlash` / `DSpark` / `MTP`)

With draft length `k` and mean accepted tokens per verify step `a` (`1 <= a <= k + 1`):

```text
t_verify = max(B_token(L) / BW_sustained,  (k + 1) * FLOPs_per_tok / Compute_practical)
t_step   = t_draft(k) + t_verify
tok_s    = a / t_step
```

Key empirical rules:
- **Verify kernel must scale cleanly at `M = k + 1`:** Replacing Triton `M=8` MoE verify with native NVFP4/MXFP8 CUTLASS Xe2 cut Inkling's 44-visit MoE verify from ~93 ms to ~25 ms (`57.2 -> 80.4 tok/s` card sampling, `71.6 -> 95.1` greedy).
- **Drafting fails when undrafted decode is launch-bound:** On unpatched llama.cpp SYCL Inkling (running at only 15% of byte ceiling), DSpark `n=3` *lost* `-31%` (`DNR-034`).
- **Draft-side quantization is independent of target:** On quantized dense targets, external drafters trained on BF16 hidden states lose (`DFlash2` on quantized 27B: `25 vs 51 tok/s`); use native MTP/DFlash trained on the target's actual representation.

## 3. Cold Prefill Ceiling (Compute Bound)

```text
FLOPs_prefill(L) = 2 * active_params_per_tok * L
                 + 2 * n_full_attn_visits * attn_hidden * L^2
                 + 2 * n_swa_visits * attn_hidden * L * min(L, window)
```

Divide by practical BF16 XMX (`~78 TFLOPS`) — or `~180 TFLOPS` only when **true `s8 x s8` INT8 prefill GEMMs** are active (`exl3xpu` int8 prefill). If a requested prefill target exceeds `~50–65%` of raw BF16 XMX peak on 1x B70 (e.g. `10,000 tok/s` on a `5.4B`-active model requires `108 TFLOPS` sustained across every op), state early that it requires 2x B70 or INT8 `s8 x s8` prefill.

## 4. KV Cache & vLLM SWA `--max-num-batched-tokens` Ceiling

- **Full-attention KV bytes/token:** `2 * N_full * H_kv * D * b_kv` (`b_kv = 2` BF16, `1` FP8/Q8_0, `~0.56` Q4_0).
- **vLLM SWA in-flight reservation:** `SlidingWindowManager` scales SWA block reservation with `--max-num-batched-tokens` across all SWA visits. On Inkling (40 SWA visits):
  - `mbt=4096/8192` -> SWA demands `10.2–17.4 GiB`, collapsing max context to `~7K`.
  - `mbt=2048` -> `106K` DSpark / `200K` no-spec.
  - `mbt=1024, u=0.95` -> **`163K` DSpark (`194.1K` KV)** / **`256K` no-spec (`273.8K` KV)**.
  - `mbt=512, u=0.95` -> **`308,696` KV tokens** no-spec.
- **Dense 27B at 250K–262K:** On 1x B70, `exl3xpu` fp8 KV holds `287K` KV tokens (`245K` real prompt); TensorFold fits `250K` on 1 card with `int8` KV (`20.7 tok/s`, no prompt cache left over — `DNR-023`) or `250K` with `bf16` KV on 2 cards `--tp 2` (`63.3 tok/s`).

Run `python3 skills/intel-xpu-inference-optimisation/scripts/b70_ceiling.py --spec examples/<model>.json` on the worked examples (`inkling2-mini-b70.json`, `qwen38-27b-dense-b70.json`, `nemotron35-30a3-b70.json`).
