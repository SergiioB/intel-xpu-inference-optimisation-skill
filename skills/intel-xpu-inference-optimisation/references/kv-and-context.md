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
- **Measured on Inkling 2 Mini (1x B70, BF16 KV):**
  - `mbt=4096 / 8192`: SWA demands `10.2–17.4 GiB` -> max context collapses to **`~7K`**.
  - `mbt=2048`: **`106,496`** DSpark (`u=0.93`) / **`200,000`** no-spec (`u=0.95`) — best prefill speed.
  - `mbt=1024, u=0.95`: unlocks **`163,840`** DSpark (`194.1K` KV, `160K` real prompt @ `46.2 tok/s`) and **`262,144` (256K)** no-spec (`273.8K` KV, `256,000` real prompt @ `23.1 tok/s`).
  - `mbt=512, u=0.95`: expands KV pool to **`308,696` tokens** no-spec.
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
