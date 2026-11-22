# intel-xpu-inference-optimisation-skill

An open-standard agent skill for **hill-climbing local LLM serving speed, maximum context, and energy efficiency on Intel Arc Pro B60/B70 (Battlemage Xe2, `bmg_g31`) and Intel XPU GPUs** across **`vLLM XPU`**, **`TensorFold XPU`**, **`exl3xpu`**, **`llama.cpp SYCL`**, and **`Strata / XeStrata`**.

Distilled from months of measured campaigns across **8+ model families** and **42 closed arms (`DNR-001`–`DNR-042`)** in the [Intel Arc Pro B70 Inference Cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) and B70-DOCS research program, inspired by `sybil-solutions/inference-optimisation-skill` and specialized end-to-end for Intel Xe2 silicon, Level Zero, SYCL, Triton, and CUTLASS Xe2.

## Measured B70 Wins Distilled Into This Skill (32 GB Arc Pro B70, 150–230 W)

| Model Family / Engine | Lever | Measured Before → After |
|---|---|---|
| **Qwen3.8-27B (Dense 27B)**, 1× B70, TensorFold XPU | MLX 4-bit + DFlash2 + `XPU_ATTN_PREF=256` prefix-attn kernel + `--prompt-cache-gib` | **127 / 115 / 90 / 51 tok/s** code (`0 / 8K / 32K / 131K`), **`drafted == serial`**, warm follow-ups `0.3–1.1 s` up to 131K |
| **Qwen3.8-27B**, 2× B70, TensorFold `--tp 2` | Peer-store collectives without remote atomics (PR `#3586`) | **165.5 / 143.0 / 117.5 / 76.8 tok/s** (`1.25×–1.5×` vs 1 card, `drafted == serial`); **250K bf16 KV @ 63.3 tok/s** |
| **Qwen3.8-27B**, 1× B70, `exl3xpu` | EXL3 4bpw + fp8 KV (`287K` pool) + int8 prefill + oneDNN SDPA | **245,583-tok real prompt** at model-card sampling (`63.25 tok/s` w/ MTP3; `426 tok/s` C16 aggregate; MTP off for exact one-shot code) |
| **Inkling 2 Mini (26.4B Native NVFP4/MXFP8 + Vision)**, 1× B70, vLLM XPU | PR `#69186` FP32 `lm_head` + AOT `bmg_g31` CUTLASS Xe2 + DSpark `k=7` + XPU 23-layer ViT + FP8 KV (`256 MiB` reusable BF16 scratch) | Greedy `39.7 → 95.1 tok/s` (`2.4×`); card-sampling **80.4 → 55.3 tok/s** (`0–104K`, `effort=0.2` **99.2**); **3,920 tok/s** prefill; **8/8 text + 4/4 vision PASS** (`29.3 ms` ViT, OCR `107 tok/s`); **569K KV (512K) / 323K KV w/ Vision+DSpark** |
| **Nemotron-3.5-Lightning-30B-A3B**, 1× B70, vLLM XPU | GPTQ-INT4 G64 + DFlash `n=7` + grouped-topk patch + SSU B8/W4 tuning | **186.6 tok/s** (`p2048/g128`, 150 W) |
| **Qwen3.8-Flash-Next (125B-A6B)**, 1× B70, Strata SYCL | 16-byte aligned `IQ3_XXS` native pack + MTP (`--spec 4 --spec-min-p 0.5`) | **48.03 tok/s** decode / **989 tok/s** 8K prefill (**2.22×** decode / **5.2×** prefill vs dual-split llama.cpp) |
| **Qwen3.8-27B**, 1× B70, llama.cpp SYCL (`>= 2026-10-29`) | `#31125` DPAS flash-attn with `q8_0`/`q4_0` KV + `GGML_SYCL_TP_P2P=2` | 64K decode: `10.5` (`q5_0/q4_1`) → **22.3 / 23.9 tok/s** (**1.8×–2.1×**); 2-card peer-store allreduce `28.3 → 32.2` tg128 |
| **Ripple-65B Q4_K_M**, 2× B70, llama.cpp SYCL | `GGML_SYCL_F16=ON` + AOT `bmg_g31` | `pp2048` prefill: `218` → **720 tok/s** (**3.3×**), 11-turn 118K session `0.0%` sag |

## Repository layout

```text
PROMPT.md                                                   # Standalone prompt for any model/agent
skills/intel-xpu-inference-optimisation/
├── SKILL.md                                                # Agent skill entry point
├── PROMPT.md                                               # Copy of standalone prompt
├── references/
│   ├── ceilings.md                                         # Xe2 silicon datapaths (XMX FP16/INT8/INT4, 16B align), looped/MoE/SWA & P2P ceilings
│   ├── quantization.md                                     # 6 quant axes, format showdown (NVFP4/EXL3/MLX/AutoRound/GGUF/IQ/FP8), drafter economics
│   ├── kv-and-context.md                                   # K/V KL-divergence table (q8_0 vs q4_0 K cliff), 3.8x dense-vs-MoE KV, SWA mbt, 11-turn sag
│   ├── engines-and-topology.md                             # 6-engine capability matrix, 150W MoE vs 180W/230W Dense power sweet spots, dual-B70 TP2
│   ├── kernels-xe2.md                                      # AOT bmg_g31 CUTLASS/SYCL/ESIMD patterns, 16B alignment, why megakernels/SLM staging fail
│   ├── levers.md                                           # Workload regime matrix + 15 ranked B70 levers across all 5 engines & 8+ models
│   ├── gotchas.md                                          # All 42 closed failure classes (DNR-001..042) & Intel XPU/SYCL/driver/profiling traps
│   ├── methodology.md                                      # 3-class correctness-lane framework, card sampling, energy, VRAM & retrieval gates
│   └── publishing.md                                       # Pinning, opt-in gating, sanitization, cookbook CI & 2x PIL result graphics
├── scripts/
│   ├── b70_preflight.py                                    # Read-only Intel GPU / VRAM / power / RAM preflight
│   └── b70_ceiling.py                                      # Stdlib-only Intel Xe2 roofline & SWA-KV calculator
└── examples/
    ├── inkling2-mini-b70.json                              # Looped native NVFP4/MXFP8 MoE + SWA + DSpark
    ├── qwen38-27b-dense-b70.json                           # Dense 27B 4bpw + fp8 KV + MTP3 / DFlash2
    └── nemotron35-30a3-b70.json                            # Hybrid Mamba2/MoE 30B-A3B + DFlash n7
```

## Quick start

### Install as a Devin CLI skill (global)
```bash
git clone https://github.com/SergiioB/intel-xpu-inference-optimisation-skill.git
mkdir -p ~/.config/devin/skills
cp -r intel-xpu-inference-optimisation-skill/skills/intel-xpu-inference-optimisation ~/.config/devin/skills/
```

### Run the included stdlib-only helpers
```bash
# Read-only Intel XPU host preflight (discovers Intel GPUs by PCI BDF, never remembered hwmon indices)
python3 skills/intel-xpu-inference-optimisation/scripts/b70_preflight.py

# Compute Intel Arc Pro B70 decode, prefill, speculative, and SWA-mbt ceilings
python3 skills/intel-xpu-inference-optimisation/scripts/b70_ceiling.py \
  --spec skills/intel-xpu-inference-optimisation/examples/qwen38-27b-dense-b70.json
```

## Companion repositories

- **Recipes, patches, and benchmark catalog:** [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
- **XeCores / B70 Launcher:** [SergiioB/b70-launcher](https://github.com/SergiioB/b70-launcher)

## License

MIT — see [LICENSE](LICENSE).