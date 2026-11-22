# KV Cache Quantization, Long-Context Capacity & Session Stability (Intel Arc Pro B60/B70)

Long-context serving on a 24–32 GB Intel Arc Pro B60/B70 is governed by three separate physics walls: **KV quality (K vs V asymmetry)**, **KV kernel coverage**, and **allocator reservation (SWA `mbt` & hybrid state slots)**.

---

## 1. Why K Needs More Precision Than V (`softmax(Q * K^T / sqrt(d_k)) * V`)

- **K (Keys)** determine *which tokens receive attention* — quantization noise in K corrupts attention routing and compounds across layers.
- **V (Values)** are *weighted-averaged* across tokens — quantization noise averages out cleanly.
- **Rule:** Spend bits on K, save bits on V.

### Measured llama.cpp KV KL-Divergence Table (`#23470` + B70 Validation)

| K Cache | V Cache | Mean KL-Divergence | Same Top-p % | VRAM vs FP16 | B70 Verdict |
|---|---|---:|---:|---:|---|
| `f16` / `bf16` | `f16` / `bf16` | `0.000` | `100%` | `100%` | Reference baseline |
| **`q8_0`** | **`q8_0`** | **`~0.003`** | `~97%` | `~75%` | **DEFAULT on `llama.cpp >= 2026-10-29` (`#31125`):** 2.1x faster at 64K (`22.3` vs `10.5 tok/s`) AND near-lossless! |
| **`q8_0`** | **`q4_1`** | **`~0.003`** | `~97%` | **`~50%`** | **Best VRAM-to-quality trade on pre-`#31125` builds** (near-lossless, half VRAM) |
| `q8_0` | `q4_0` | `~0.0048` | `~96.7%` | `~44%` | Near-lossless, slightly tighter VRAM |
| `q5_0` / `q5_1` | `q4_1` | `~0.006–0.008` | `~93%` | `~44%` | Acceptable quality, **but bypasses `#31125` DPAS flash-attn (`10.5 tok/s` at 64K, `DNR-019`)** |
| **`q4_0`** | *any* | **`~5.5`** | **`~11.6%`** | `~25%` | **CATASTROPHIC QUALITY CLIFF — NEVER USE `q4_0` FOR K!** Output is destroyed regardless of V precision (`q4_0 K + f16 V = 5.49` KLD). Even though `#31125` decodes `q4_0/q4_0` at `23.9 tok/s`, it is prohibited as a default! |

---

## 2. KV Scales with Attention Geometry, NOT Total Parameters

A huge architectural insight measured on B70 (`q8_0 K / q4_1 V`):

| Context | Dense 27B KV | MoE 35B-A3B KV | Ratio |
|---|---:|---:|---:|
| `32K` | `~1.7 GB` | `~0.5 GB` | **3.8x smaller** |
| `128K` | `~6.96 GB` | `~1.85 GB` | **3.8x smaller** |
| `256K` | `~13.6 GB` | `~3.6 GB` | **3.8x smaller** |
| `512K` | `~27.2 GB` | `~7.2 GB` | **3.8x smaller** |

**Consequence:** A 25 GB MoE model with small attention/hybrid-GDN layers can fit **256K–512K** on one 32 GB B70, while a 16 GB dense 27B model hits the wall around **256K** (`exl3xpu` fp8 KV = `287K` pool, `245K` real prompt; `TensorFold` bf16 KV = `131K` on 1 card / `250K` on 2 cards `--tp 2`, or `250K` on 1 card with `int8` KV leaving no room for a prompt cache, `DNR-023`).

---

## 3. Engine-Specific KV & SWA Tuning

### vLLM XPU (Sliding Window Attention `mbt` Knob)
- vLLM's `SlidingWindowManager` reserves SWA in-flight blocks proportional to `--max-num-batched-tokens` across **every SWA visit** (40 visits in Inkling 2 Mini).
- **Measured on Inkling 2 Mini (1x B70, BF16 vs FP8 E4M3 KV):**
  - `BF16 KV`, `mbt=4096 / 8192`: SWA demands `10.2–17.4 GiB` -> max context collapses to **`~7K`**.
  - `BF16 KV`, `mbt=2048`: **`81.9K`** (`84.8K` KV w/ Vision+DSpark), **`106,496`** DSpark (`u=0.93`), **`200,000`** no-spec (`u=0.95`).
  - `BF16 KV`, `mbt=1024, u=0.95`: **`163,840`** DSpark (`194.1K` KV) and **`262,144` (256K)** no-spec (`273.8K` KV; **`308,696`** KV at `mbt=512`).
  - **`FP8 E4M3 KV` (`--kv-cache-dtype fp8`)** halves BOTH full-attention KV (`32,768 -> 16,384 B/tok`) AND all 40 SWA reservations:
    - `mbt=2048, u=0.92` w/ DSpark + Vision: **`177,637` KV (`163K` ctx, `2.1x` vs BF16)** — OCR `107 tok/s`, **64K exact retrieval PASS** (`68.0 tok/s`).
    - `mbt=1024, u=0.95` w/ DSpark + Vision: **`323,439` KV (`256K` ctx, `1.23x`)** WITH Vision Tower + DSpark `k=7` BOTH ON!
    - `mbt=1024, u=0.95` no-spec text: **`569,497` KV (`512K` ctx, `1.09x`)** on ONE 32 GB B70!
  - **FP8 KV Speed & Long-Context Quality on Xe2 (Measured 4K -> 496K!):**
    - Inline FP8 `tl.make_tensor_descriptor` loads in Triton XPU are **5.4x–11x slower** at prefill (`256.6 ms` vs `46.8 ms` BF16 at 64K; `~510 ms` at 128K). Casting active pages into **ONE fixed-size reusable `256 MiB` BF16 scratch + exact FP32 online FlashAttention LSE chunking across 64K key windows** recovers **91%–97% of native BF16 prefill speed ALL THE WAY TO 512K** (`1,822 tok/s` @ 128K, `1,378 tok/s` @ 245K, `1,055 tok/s` @ 384K, **`905 tok/s` @ 496K**, `46.4 ms` vs `510 ms` at 128K = **11x speedup**!).
    - **Measured BF16 vs FP8 Held-Out Perplexity & 496K Dual-Needle Recall (Inkling 2 Mini, `model_max_length=1M`, L2-clipped K + RMSNormed V):**
      - `4K`–`245K` matched vs BF16: **`-0.48%` to `+0.41%`** (`+0.07%` at `245,760` tokens: `3.4536` vs `3.4512`, token logprob MAE `0.066` nats — **does not grow with depth!**).
      - `384K` (`393,216`): **`3.4557`** (`+0.06%` vs `245K`); `496K` (`496,000`): **`3.5833`** (`+3.7%` vs `245K`, zero collapse).
      - **Dual-needle exact retrieval (`15%` oldest KV + `55%` mid KV):** **`2/2` EXACT PASS at BOTH `245,726` tokens (`181.8 s`) AND `495,964` tokens (`542.7 s`)!**
    - Decode `<= 104K` stays `~12–18%` faster on BF16 (`80.4->55.3` vs `67.1->43.6 tok/s`); use BF16 KV for peak `<=104K` speed and FP8 KV for `163K–512K` or Vision + DSpark!
- **Hybrid Mamba / GDN models:** State slots and block alignment cap concurrency, not KV tokens. Keep exact block alignment (e.g. `1,600` vs rounded `2,048` lifted C16 by `+17%`).

### `llama.cpp SYCL`
- Always pass `-fa on` for quantized KV (except models with additive attention bias `kq_b != nullptr` like Inkling, where llama.cpp disables FA automatically, `DNR-036`).
- On builds `>= 2026-10-29` (`#31125`): default to `-ctk q8_0 -ctv q8_0` (`22.3 tok/s` at 64K vs `10.5` for `q5_0/q4_1`).

### `exl3xpu` & `TensorFold XPU`
- **`exl3xpu`:** `fp8` KV only (`bf16` rejected); `287K` KV pool on 27B 4bpw; measured **`5.4x` prefix-cache win** (requires zero-hit-delta & leading-entropy guard for cold cells).
- **`TensorFold`:** `bf16` KV default (`131K` on 1 card, `250K` on 2 cards `--tp 2` at `63.3 tok/s`); `int8` KV for 1-card `250K` (`20.7 tok/s`). Size `--prompt-cache-gib` to hold one full state for **`0.3–1.1 s` warm follow-ups up to 131K**.

---

## 4. Prefix Caching & 11-Turn Session Stability

- **Prefix caching policy (never assume safe!):**
  - `vLLM XPU` W4A16 / speculative stacks: **OFF (`--no-enable-prefix-caching`)** until upstream `#53912`-class cache corruption is fixed and proven clean with a multi-turn corruption check.
  - `exl3xpu` & `TensorFold`: **ON** for interactive chat; explicitly declared and verified `0` cache-hit delta for cold prefill benchmarks.
- **One decode cell is NOT a long-context verdict (`SESSION-STABILITY.md`):** Run a multi-turn session trace (e.g. 11 turns seeded at `118K–131K`) and measure decode sag across turns (`Ripple-65B` 2-card 118K: `0.0%` sag; `Qwen3.8-27B` `q8_0` KV 131K: `-0.5%` sag; cached TTFT `0.6–9 s`). Flag any memory growth (e.g. OpenVINO INT4 `>= 196K` `CL_OUT_OF_RESOURCES`).
