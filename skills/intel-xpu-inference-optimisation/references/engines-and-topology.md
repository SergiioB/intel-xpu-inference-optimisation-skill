# Intel Arc Pro B60/B70: 6-Engine Matrix, Power Sweet Spots & Dual-Card Topology

Choose the right engine, power envelope, and card placement BEFORE tuning flags or writing kernels.

---

## 1. Six-Engine Capability & Status Matrix on B70

| Engine | Best Fit on B70 | Supported Quant / KV | Strengths | Hard Limits / Caveats |
|---|---|---|---|---|
| **vLLM XPU** | Native `NVFP4/MXFP8` plugins (`Inkling 2 Mini`), `GPTQ-INT4` / `AutoRound` MoE (`Nemotron`, `Bolt-74B`), continuous batching (`Cn`) | Native NVFP4/MXFP8 (plugin), GPTQ/AutoRound W4A16, FP8, `moe_wna16`; BF16/FP8 KV | OpenAI API + `/metrics` counters, `FULL_DECODE_ONLY` graphs, DSpark/DFlash/MTP, SWA `mbt` tuning, custom CUTLASS Xe2 plugins | Keep `--no-enable-prefix-caching` on W4A16/spec stacks; `v0.31.0` AutoRound crashes with AVX-512 SIGILL on Zen 3 (`DNR-027`); needs `VLLM_RPC_BASE_PATH` under `/tmp` |
| **TensorFold XPU** | Lossless single- & dual-card dense 27B (`Qwen3.8-27B`), fast `--prompt-cache-gib` warm turns | MLX affine 4-bit `g64`, GPTQ `g128`; BF16 & INT8 KV | Streams **532–549 GB/s** (`87–90%` pin!), lossless DFlash2 (`drafted == serial`), peer-store **`--tp 2` (`1.25x–1.5x`, PR `#3586`)**, `0.3–1.1 s` warm 131K turns | Source build (Zig + `ocloc`); refuses NVFP4/FP8; fork prefix-attn kernel alters upstream tokens |
| **exl3xpu** | Dense 27B at **131K–262K** on ONE card (`245K` real prompt), high `C16` aggregate (`426 tok/s`) | EXL3 trellis (4bpw, mixed-K SC5); `fp8` KV only | `287K` KV pool on 1 card, `s8 x s8` int8 prefill + oneDNN SDPA, `5.4x` prefix-cache win | **Dense only (`DNR-011`)**; requires isolated dockerd; MTP3 alters 27% of greedy outputs (`DNR-017`); greedy repeats ~78% (`n=5` needed); EXL3 kernel probes operator-stopped |
| **llama.cpp SYCL** | GGUF workhorse (`Ripple-65B`, `Muse-Glimmer-30B`, `Ornith-122B`, vision `mmproj`, wide arch coverage) | All GGUF types; `q8_0/q8_0` (`#31125` fast FA), `q8_0/q4_1`, `f16` KV | `GGML_SYCL_F16=ON` (`3.3x` prefill), `GGML_SYCL_ENABLE_GRAPH=1`, local `GGML_SYCL_TP_P2P=2` peer-store allreduce, `llama-bench` synthetic baseline | Always set `SYCL_CACHE_PERSISTENT=0` on DPC++ 2026.1 (`DNR-031`) and `GGML_SYCL_ENABLE_OPT=1` (`DNR-037`); additive attention bias disables `-fa on` (`DNR-036`); `q4_0` K destroys quality |
| **Strata / XeStrata SYCL** | Large MoE (`Qwen3.8-Flash-Next 125B-A6B`) with 16-byte aligned native packs + expert cache | Aligned `IQ3_XXS` / `IQ3_S` packs + MTP; `int8` KV | **2.3x–4.7x** from 16B alignment (`48.03 tok/s` decode / `989 tok/s` 8K prefill on 1x B70 for 125B-A6B!) | Strict per-model geometry & tensor-type support (`DNR-012`); check tensor census first (`DNR-014`) |
| **OpenVINO GenAI** | Clean `INT4-GDN8` dense alternative (`<= 128K`) | NNCF INT4/INT8; INT8/INT4 KV | Clean exact-answer battery, grafted MTP5 reached `79 tok/s` @ 512 | `CL_OUT_OF_RESOURCES` at `>= 196K` (`#4483`); draft off `>= 98K`; not primary serving stack |

*(Note: `Cascadia` PP2 retains ~100% of 1-card speed across 2 cards with no speedup (`DNR-006`); `SGLang XPU` remains blocked/unverified on discrete B70 until an upstream XPU build passes B70 gates.)*

---

## 2. Power Sweet Spots: MoE and Dense Scale in OPPOSITE Directions (`POWER-SWEET-SPOTS.md`)

Never blindly max out the power cap — and **never change power caps secretly during a benchmark** (record as-found `power1_cap` and restore it at cleanup!).

| Workload Class | 150 W | 180 W | 230 W (Stock) | Operational Recommendation |
|---|---:|---:|---:|---|
| **MoE (e.g. 35B-A3B vLLM MTP)** | **125.7 tok/s** (~58 C) | ~125 tok/s | 115.4 tok/s (**-8.2% slower!**) | **150 W sweet spot:** MoE self-limits ~140 W; higher cap adds heat noise without speed. |
| **Dense 27B (llama.cpp Q4_K_M / EXL3 4bpw)** | 18–22 tok/s / `49.2` EXL3 (71 C) | **~21–24 tok/s** (`0.155 t/s/W`) | **23–26 tok/s / `63.25` EXL3 (+18–30%)** (79 C) | **180 W sustained** efficiency sweet spot; **230 W** for burst / peak benchmarks (watch 79 C throttling!). |

- **Sensors:** Read `power1_cap` (microwatts), `energy1_input` (microjoules), and **`temp2_input`** (package temp in millidegrees — `temp1_input` does not exist on B70!). Always resolve `hwmon` via PCI BDF, never a remembered `hwmonN` number.

---

## 3. Dual-B70 Topology Rules (`ADR-0008`, `ADR-0014`, `DNR-004/005/006/021/032`)

On PCIe 4.0 x8/x8 dual-B70 hosts:
- **Measured P2P bandwidth:** `11.1–11.9 GB/s` checksum-clean direct peer copies; **NO hardware peer atomics**.
- **Default for > 1 user (always wins aggregate):** Run **two independent single-card servers** pinned with `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1` + `ZE_AFFINITY_MASK=0` / `1` (2x aggregate throughput, zero collective overhead, independent fault domains).
- **Single-stream speed beyond 1-card ceiling:**
  - **WORKS (`1.25x–1.5x`):** Peer-store collectives **without remote atomics** — `TensorFold --tp 2` (upstream PR `#3586`: `165.5 / 143.0 / 117.5 / 76.8 tok/s` on Qwen3.8-27B, `drafted == serial`, `~19.9 us` per 160 KiB exchange) and patched `llama.cpp` `GGML_SYCL_TP_P2P=2` (`32.18` vs `28.28` tg128; NEVER mode 1 `GGML_SYCL_TP_P2P=1` which leaks 224K cross-queue events and segfaults, `DNR-032`).
  - **CLOSED / DO NOT USE FOR SINGLE-STREAM SPEED:** Stock vLLM TP2 / oneCCL (`-45%..-71%`, `DNR-004`), `llama.cpp` RPC split (`54%` of 1 card, `DNR-005`), `Cascadia` PP2 (`0%` gain, `DNR-006`), and host-staged second-card MoE caches (`33.4 vs 48.0 tok/s`).
  - **Dual-card capacity split (`-sm tensor` / `--layer-split`):** Appropriate when a model does not fit on 1 card (`Ripple-65B`, `Bolt-74B` AutoRound TP2, `Ornith-122B`, `Qwen3.8-27B` 250K bf16 KV). Mount `/dev/dri/by-path` for only the selected cards inside Docker so oneCCL never opens unselected cards (`DNR-027`).
