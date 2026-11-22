# Quantization & Drafters on Intel Arc Pro B60/B70 (Battlemage Xe2)

There is no single "best quant" across Intel XPU engines and model architectures. Quantization controls four independent outcomes — **kernel dispatch**, **task quality**, **VRAM/context capacity**, and **decode throughput** — across **six completely independent axes**:

1. Target routed/dense weights
2. Draft model body
3. Draft LM head
4. Target verify LM head
5. Activations (BF16/FP16 vs INT8 `s8 x s8`)
6. KV cache (see `references/kv-and-context.md`)

**A verdict on one axis NEVER transfers to another.**

---

## 1. Format Showdown on Battlemage Xe2 (`bmg_g31`)

Because Xe2 has **no native FP4 or FP8 DPAS hardware** (only FP16/BF16 XMX `2048 ops`, INT8 `s8 x s8` XMX `4096 ops / 367 TOPS`, INT4 ESIMD `8192 ops`, and DP4A `1024 ops`), every W4A16/W8A16 format saves GDDR6 bandwidth and unpacks to FP16/BF16 in registers.

| Format | Primary Engine(s) | Measured B70 Behavior | Strengths & Traps |
|---|---|---|---|
| **Native NVFP4 / MXFP8** | `vLLM XPU` (model plugins + opt-in CUTLASS Xe2) | Inkling 2 Mini: `80.4->55.3 tok/s` (`0–104K`, DSpark), `3,920 tok/s` prefill, **8/8 PASS** | **Never double-quantize native NVFP4/MXFP8** to GPTQ, EXL3, Q8_0, or BF16 GGUF (`DNR-035`, `DNR-040`, `DNR-041`). Preserve native storage and write/use native Xe2 dequant+DPAS kernels. |
| **EXL3 Trellis (4bpw / mixed-K SC5)** | `exl3xpu` (isolated dockerd) | Qwen3.8-27B: `63.25 tok/s` w/ MTP3 (`230 W`), **245K real prompt** (`287K` fp8 KV pool), clean 6/6 exact battery | **Dense only** (`DNR-011`: `exl3xpu` has no `FusedMoE` quant method). Passes the in-head arithmetic canary (`"14"`) where symmetric W4A16 failed! MTP3 alters 27% of greedy outputs (`DNR-017`). |
| **MLX Affine 4-bit (`g64`)** | `TensorFold XPU` | Qwen3.8-27B: **127 / 115 / 90 / 51 tok/s** (`0 / 8K / 32K / 131K`), **532–549 GB/s** (`87–90%` of 608 GB/s pin!), **`drafted == serial`** | Asymmetric affine `g64` preserves dense 27B quality AND streams near pin bandwidth. Repacking MLX affine -> symmetric GPTQ is lossy dequant/requant (`Decision 3`). |
| **AutoRound / GPTQ INT4 (sym `G128` / `G64`, W4A16)** | `vLLM XPU` | Nemotron-30B-A3B: **186.6 tok/s** (`G64` + DFlash); Ornith-35B-A3B (`ADR-0007`); Bolt-74B TP2 | **CRITICAL QUALITY TRAP (`DNR-009`):** On dense `Qwen3.8-27B`, every symmetric INT4 W4A16 export (GPTQModel + 2x AutoRound) answered `"30"` instead of `"14"` on in-head arithmetic while passing coherence & PPL! Always run exact-answer batteries first. **Converter rule (`ADR-0007`):** AutoRound (signSGD) tuned `30,720/30,720` projections with **0% RTN fallback** vs GPTQModel's `10.4–24.8%` untuned (better Δlogprob `0.2006 vs 0.2025`, top-1 `91.4% vs 90.5%`, KL `0.277 vs 0.319`). Note `DNR-027`: `vllm v0.31.0` AutoRound library crashes with AVX-512 SIGILL on Zen 3 CPUs. |
| **GGUF `Q4_K_M` / `UD-Q4_K_XL`** | `llama.cpp SYCL` | Same-source Qwen3.8-27B showdown: `Q4_K_M` **20.01 tok/s** (~336 GB/s) vs `Q5_K_M` `16.97` vs `Q8_0` `13.62` (~396 GB/s eff.) | `Q4_K_M` is the llama.cpp workhorse. **Blanket mixed-precision "rescue" of GDN/attn tensors loses on BOTH axes** (`+10.7%` bytes -> `-5.8%` decode, zero measurable quality gain). **Never trust filename labels (`DNR-014`):** Unsloth UD GGUFs contain mixed per-tensor types not in the filename; read tensor headers and fail closed! |
| **GGUF `IQ3_XXS` / `IQ3_S` / `IQ4_XS` (16B-aligned packs)** | `Strata / XeStrata SYCL`, `llama.cpp SYCL` | Flash-Next 125B-A6B Strata 1x B70: **48.03 tok/s** decode / `989 tok/s` 8K prefill (`2.22x` decode vs llama.cpp) | Raw GGUF `IQ3_XXS` (98 B), `IQ4_XS` (136 B), and `Q8_0` (34 B) are **not 16-byte aligned** — Xe2 splits unaligned loads (`2.3x–4.7x` speedup from 16-byte aligned native repacking in Strata!). |
| **Block FP8 (W8A16)** | `vLLM XPU` | Qwen3.8-27B-FP8 TP2: `48.08 tok/s` research route | Needs 2 cards for 27B; no native FP8 DPAS on Xe2. |
| **OpenVINO NNCF INT4 / INT8** | `OpenVINO GenAI` | Qwen3.8-27B `INT4-GDN8`: clean battery, `28.5 tok/s @ 150W` (up to `79 tok/s` w/ grafted MTP5) | Fails `CL_OUT_OF_RESOURCES` at `>= 196K` (`openvino.genai#4483`). |

---

## 2. Converting a Model Yourself (AutoRound W4A16 Baseline, `ADR-0007`)

Only convert when no high-quality native packed artifact (`NVFP4/MXFP8`, `EXL3`, `MLX 4-bit`, `GGUF`) fits your target engine:
1. **Default converter:** **AutoRound (signSGD)** with mixed-domain calibration (code, reasoning, tool calls, technical prose, general instruction). GPTQModel is fallback only when autograd/host RAM blocks AutoRound.
2. **Scope order:** Quantize routed MoE experts first. Keep attention, routers, shared experts, embeddings, `lm_head`, vision encoders, and MTP/draft tensors at source BF16 until each additional scope passes its own quality & speed gate.
3. **Never substitute `G64` and `G128` interchangeably** without verifying exact XPU kernel dispatch (`moe_wna16` / `LinearBase` / `ParallelLMHead`).
4. **Record in conversion manifest:** sample count, token count, seed, source revision, excluded tensors, and every quantizer RTN fallback.

---

## 3. Critical Quality Evaluation Rules for Quantization

1. **NEVER rank formats by `llama-perplexity` on hybrid linear-attention (GDN) models (`llama.cpp#28879`):** On `qwen3_5`/GDN architectures, full F16 scored *worse* than `Q4_K_M` on 7/8 chunks and PPL grew with chunk length due to recurrent-state accumulation. Require **per-position logit parity (KL / top-1 agreement) against F16/BF16**, plus exact-answer batteries!
2. **Exact-answer canaries catch silent quant corruption:** Coherence, replay, and perplexity all passed the fluent wrong `"30"` answer on Qwen3.8-27B W4A16 (`DNR-009`). Always include exact in-head arithmetic, code state-trace, JSON schema, and multi-depth needle retrieval.
3. **Activation quantization (`W4A8` vs `s8 x s8` INT8 prefill):** `W4A8` prefill is closed (`<= 1.27x` attribution ceiling + historical quality divergence). True `s8 x s8` INT8 prefill (`367 TOPS`, e.g. `exl3xpu` int8 prefill) is a valid labelled lossy prefill trade when decode stays bit-exact/lossless.

---

## 4. Quantization × Drafter Economics (`DFlash` / `DSpark` / `MTP`)

Measured on B70 (`QUANTIZATION-AND-DRAFTERS.md`):
1. **Drafter acceptance depends on matching the target's quantized hidden states:** A heavy BF16-trained external drafter (`DFlash2`, 5 layers, `3.85 GB`) running against a quantized GPTQ-INT4 27B target collapsed to `~2.0` accepted tokens vs `~3.5` for the native 1-layer BF16 MTP head (`25.2 vs 50.8 tok/s` — **2x slower**!).
2. **Cycle-cost asymmetry:** At `C1` on one B70, a native 1-layer MTP head shares embeddings/`lm_head` (~near-zero marginal bytes), whereas a multi-layer external drafter streams gigabytes per cycle and needs `>= 1.5x–2x` the native head's acceptance just to break even.
3. **Where block drafters DO win:**
   - **Native bundled drafters** trained with the target (`Inkling` DSpark `k=7`: `80.4–99.2 tok/s` w/ CUTLASS Xe2; `Nemotron` DFlash `n=7`: `186.6 tok/s`).
   - **Matched high-fidelity target formats** (`TensorFold` MLX 4-bit + DFlash2: `127 tok/s` lossless `drafted == serial`).
4. **Draft-head quantization & vocab pruning:** Quantizing the native MTP head separately (`Draft INT4 overlay`) saved `+39.6% / +48.6%` on the GPTQ stack; pruning the draft LM head vocabulary leaves target verification lossless while cutting draft head bytes up to 8x.
5. **Sweep draft depth `k` per workload:** Per-position acceptance decays sharply past positions 4–5 on natural text (`~0.1–0.3`); depth `k=4` beat `k=6` by `10–20%` on dense 27B. Always report both accepted tokens/round AND per-position marginal survival.
