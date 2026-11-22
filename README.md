# intel-xpu-inference-optimisation-skill

An open-standard agent skill for **hill-climbing local LLM serving speed, maximum context, and energy efficiency on Intel Arc Pro B60/B70 (Battlemage Xe2, `bmg_g31`) and Intel XPU GPUs** across `vLLM XPU`, `llama.cpp SYCL`, `TensorFold XPU`, and `exl3xpu`.

Distilled from measured campaigns in the [Intel Arc Pro B70 Inference Cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook), inspired by the workflow discipline of `sybil-solutions/inference-optimisation-skill` and specialized for Intel Xe2 hardware, Level Zero, SYCL, Triton, and CUTLASS Xe2.

## What this workflow has achieved on Intel Arc Pro B70 (32 GB, 230 W)

| Model / Engine | Lever | Measured Before → After |
|---|---|---|
| **Inkling 2 Mini (26.4B native NVFP4/MXFP8)**, 1× B70, vLLM XPU | PR `#69186` FP32 `lm_head` fast path (`torch.mm(..., out_dtype=fp32)`) | Greedy decode `0 / 8K`: `39.7 / 37.5` → **56.4 / 52.0 tok/s** (`+42% / +39%`, 3/3 byte-identical at 8K), `+1.22 GiB` KV |
| **Inkling 2 Mini**, 1× B70, vLLM XPU + DSpark `k=7` | Custom AOT `bmg_g31` CUTLASS Xe2 native NVFP4+MXFP8 grouped GEMM + SLM route-sort + fused SwiGLU (`INK2_CUTLASS_MOE=1`) | Realistic card-sampling decode `0 / 64K / 104K`: `57.2 / 50.1 / 45.0` → **80.4 / 64.0 / 55.3 tok/s** (`effort=0.2`: **99.2 tok/s**); cold prefill `2K`: `3,172` → **3,920 tok/s**; **8/8 quality/retrieval PASS** |
| **Inkling 2 Mini**, 1× B70, vLLM XPU | SWA `--max-num-batched-tokens` tuning (`2048 → 1024 / 512`, `u=0.95`) | Max proven context: `106K` → **163K (`160K` real prompt @ 46.2 tok/s)** with DSpark; `200K` → **256K (`256K` real prompt @ 23.1 tok/s)** and **308.7K KV pool** no-spec |
| **Qwen3.8-27B**, 1× B70, llama.cpp SYCL (`>= 2026-10-29`) | `q8_0` / `q4_0` KV with SYCL flash-attention (`#31125`) | 64K decode: `10.5` (`q5_0/q4_1`) → **22.3 / 23.9 tok/s** (`1.8×–2.1×`) |
| **Ripple-65B Q4_K_M**, 2× B70, llama.cpp SYCL | `GGML_SYCL_F16=ON` + AOT `bmg_g31` | `pp2048` prefill: `218` → **720 tok/s** (`3.3×`) |

## Repository layout

```text
PROMPT.md                                                   # Standalone prompt for any model/agent
skills/intel-xpu-inference-optimisation/
├── SKILL.md                                                # Agent skill entry point
├── PROMPT.md                                               # Copy of standalone prompt
├── references/
│   ├── ceilings.md                                         # Xe2 roofline, looped MoE, SWA mbt & spec math
│   ├── levers.md                                           # Proven B70/Xe2 levers ordered by ROI
│   ├── gotchas.md                                          # Level Zero, SYCL, Triton, vLLM XPU traps
│   ├── methodology.md                                      # Interleaved A/B, card sampling, energy, VRAM guards
│   └── publishing.md                                       # Pinning, opt-in gating, sanitization, CI & charts
├── scripts/
│   ├── b70_preflight.py                                    # Read-only Intel GPU / VRAM / power / RAM preflight
│   └── b70_ceiling.py                                      # Stdlib-only Intel Xe2 roofline & SWA-KV calculator
└── examples/
    └── inkling2-mini-b70.json                              # Worked B70 Inkling 2 Mini ceiling spec
```

## Quick start

### Install as a Devin CLI skill (global)
```bash
git clone https://github.com/SergiioB/intel-xpu-inference-optimisation-skill.git
mkdir -p ~/.config/devin/skills
cp -r intel-xpu-inference-optimisation-skill/skills/intel-xpu-inference-optimisation ~/.config/devin/skills/
```

### Install for any Agent Skills-compatible client (`SKILL.md`)
Copy `skills/intel-xpu-inference-optimisation/` into your client's skills directory, or feed [`PROMPT.md`](PROMPT.md) directly into your agent.

### Run the included stdlib-only helpers
```bash
# Read-only Intel XPU host preflight
python3 skills/intel-xpu-inference-optimisation/scripts/b70_preflight.py

# Compute Intel Arc Pro B70 decode, prefill, speculative, and SWA-mbt ceilings
python3 skills/intel-xpu-inference-optimisation/scripts/b70_ceiling.py \
  --spec skills/intel-xpu-inference-optimisation/examples/inkling2-mini-b70.json
```

## Companion repositories

- **Recipes, patches, and benchmark catalog:** [SergiioB/intel-arc-pro-b70-inference-cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook)
- **XeCores / B70 Launcher:** [SergiioB/b70-launcher](https://github.com/SergiioB/b70-launcher)

## License

MIT — see [LICENSE](LICENSE).
