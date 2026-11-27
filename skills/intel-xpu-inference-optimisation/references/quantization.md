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


---

## 5. 12-Family Quantization Study, `HF-INT4-G16` (`/1` & `/2`), `BiasCorr` & Ultra-Low-Bit Laws (v3.9.0)

Full 12-family empirical study (`Qwen3.8-27B`, `Gemma-4-26B-A4B`, `Qwen3.6-35B-A3B`, `Gemma-4-E4B`, `Granite-4.0-h-tiny`, `Qwen3-8B`, `OLMo-2-7B`, `OLMoE-1B-7B`, `Llama-3.2-3B`, `Phi-4-mini`, `Qwen2.5-1.5B`, `SmolLM2-1.7B`; 49 sanitized receipts, Cookbook `v1.24.0` + `vllm-hf-int4-g16-xpu v0.2.0`) proved the following reusable engineering laws:

### 5.1 12-Family Data-Free Format Hierarchy & Trellis vs Scalar Boundary
- **`Had128NoPerm + HF-INT4-G16` (`4.25–4.26 bpw`):** Symmetric INT4 weights with progressive group-16 scales (`W_hat = D_row * (q * u_g)`, `q in [-8..+7]`, `u_g in [1..16]`, so `q * u_g in [-128..+112] subset INT8` natively — zero zero-point plane, exact `INT8` XMX DPAS, safe `INT32` accumulator bound for `K <= 65,536`).
- **12-family sweep result (data-free, causal next-token KL vs BF16, `pos 0` excluded):**
  - Beats `MLX-4bit` (`asym G64`, `4.50 bpw`) in **`12/12`** tested model families.
  - Beats `SymINT4-G128 RTN` in **`11/11`** families with that arm, and beats shipped calibrated `GPTQ-Int4-sym-G128` on `Qwen3.8-27B` (`0.0904` vs `0.1757` combined KL; code KL `0.149` data-free / `0.129` with BiasCorr vs `0.302` GPTQ — less than half).
  - **Weight sensitivity (`V` vs `K`):** `v_proj` quantization hurts more than `k_proj` in **`11/12`** families (only `Qwen2.5-1.5B` reversed).
  - **Format-by-rotation interaction:** Contiguous block Hadamard rotation (`Had128NoPerm`, zero online permutation scatter, `24.3x` faster than naive online gather) strongly helps uniform INT4, while hurting logarithmic `NVFP4-4/6`.
  - **Trellis (`TFQ3-4b`) vs scalar (`HF-INT4-G16`) boundary:** Trellis coding (`4.00 bpw`) still wins pure rate-distortion per bit (`0.0253 / 0.1104` vs `0.0318 / 0.1480` on matched 400 matrices). Scalar `HF-INT4-G16` wins on simple live Xe2 serving (`W4A8-INT8` prefill + `W4A16` decode from one packed buffer) and unmodified BF16 embeddings/`lm_head`.

### 5.2 Format Contracts: `hf_int4_g16/1` (Dense `Had128`) vs `hf_int4_g16/2` (MoE Uniform `Had64`)
- **`hf_int4_g16/1`:** Dense `Had128` contract (`K % 128 == 0`). Backward-compatible byte-for-byte between `v0.1.0` and `v0.2.0` (`308/308` kernel outputs bit-identical SHA-256).
- **`hf_int4_g16/2`:** Uniform `Had64` contract for MoE and modern architectures whose expert projections satisfy `K % 64 == 0` but NOT `K % 128 == 0` (e.g., `Gemma-4-26B` `K = 704, 2112`).
- **Rule:** Never mix `Had128` and `Had64` blocks ad hoc inside one checkpoint without enforcing an explicit `had_block_rule` at load/admission time. Uniform `Had64` passed both 3-check and 4-check OOD gates on Gemma-4 offline, while the earlier mixed `Had128/Had64` layout failed OOD chat for both calibration variants.
- **Live serving trade-offs on 1x B70 (`vllm/vllm-openai-xpu:v0.31.0`):**
  - Dense `Qwen3.8-27B`: `1.28x` faster 8K prefill (`2,364` vs `1,849 tok/s`), `1.26x` faster 32K prefill (`2,054` vs `1,634 tok/s`), `-10.7%` energy at 32K, flat 329-item live battery (`287` vs `283`, `p=0.48`), trading off `10% to 12%` slower single-user decode. Under `Exp 22` (`FP8` KV, `8K -> 128K`), faster prefill wins full cold turns in wall clock AND energy from `32K` through `128K` (see `kv-and-context.md` §5).
  - Multimodal MoE `Gemma-4-26B-A4B-it` (`14.64 GiB` weights, `Exp 21` quiet host `230 W / 2,800 MHz`, `52–55 C`, `n=3`, model-card sampling): **`84.76 tok/s`** (51 tok, `1.987 J/tok`), **`70.80 tok/s`** (2K, `1,252 tok/s` prefill), **`55.60 tok/s`** (8K), `8/8` canaries + vision PASS, live vs offline WT2 PPL `25.983` vs `25.982`.

### 5.3 `BiasCorr` (`b = -ΔW · E[x]`): Scaling Laws, Ultra-Low-Bit Split & Mandatory Negative Controls
- **Mechanism & Pos0 exclusion:** Weight rounding shifts per-channel output means by `(W - W_hat) * E[x]`. Always exclude token position `0` (`X[:, 1:, :]`, attention-sink outlier; including `pos 0` degraded `NVFP4` OOD KL by `+29%`). Fuse FP32 bias into the GEMM/SwiGLU/GeGLU epilogue immediately after DPAS before FP16 rounding — never ship a standalone bias kernel (`-9.5%` tax -> **`<= +0.30%` TTFT overhead `0–128K`** on quiet-host TensorFold patch v5, with **`11/11` `drafted == serial`** on GPU0 and `--tp 2`).
- **All-496-linear `mean_x` cosine mechanism:** Post-RMSNorm input projections have universal means across domains (`cos = 0.913–0.931`), whereas post-SwiGLU output projections (`mlp.down_proj`, `cos = 0.153–0.191`) are strongly domain-specific. Calibrate on disjoint general prose + code (and chat-formatted mix for instruct models) with logged token-ID SHA-256 and `0/0` 6-gram overlap.
- **MoE `shared_only` scaling law:** On 128E–256E MoEs (`Gemma-4-26B`, `Qwen3.6-35B`), a **`2.0–2.54 MiB` `shared_only`** sidecar matches a `122.5 MiB` `all_experts` sidecar (`48x` smaller; `-36.7% / -22.2% / -29.4%` WT2/Code/OOD on `Qwen3.6-35B`) because routed experts see too few tokens under small calibration sets.
- **Mandatory 4-check gate ON SERVED GPU KERNELS (never trust simulated math alone!):** Require (1) WikiText-2 KL 95% block-bootstrap CI `< 0`, (2) Code KL CI `< 0`, (3) held-out OOD chat KL CI `< 0` for instruct models, and (4) reversed-sign `-b` control strictly worse. **`Exp 21` proof:** converter `--cal-chat-template` default passed simulated math on Code (`[-0.05385, -0.00723]`) yet missed on actual served GPU kernels (`[-0.04884, +0.00073]`) -> rejected (`publishable: false`). Always gate on the exact served kernel path!
- **Regime split across bit widths (`Exp 18` & `Exp 20`):**
  - **4-bit & 3-bit:** Cuts KL drift (`-14%` to `-37%` across Qwen3.8 / Qwen3.6 / Gemma-4; transfers zero-shot to Spanish Wikipedia `-17.1% / -19.0%`, Don Quixote `-12.7% / -22.6%`, and `2,048 / 4,096` context `-14.5%` to `-20.6%`), while **total binary task scores stay flat** (at 3-bit `1,024`-token cap, plain GSM8K scores `58/60` vs `57/60` corrected, `p=1.0`, exposing an earlier 384-cap `41 -> 48` reading as a truncation artifact!).
  - **2-bit (`2.26 bpw` `had_int2_g16`):** Genuine statistically decisive task rescue — a single `14.89 MiB` open-loop `BC_open` vector lifted `Qwen3.8-27B` from **`31 -> 93/160` (`72/10`, `p = 1.0e-12`)**, GSM8K **`8 -> 50/60` (`42/0`, `p = 4.5e-13`)**, untruncated GSM8K `6/17 -> 15/17` (`p = 0.0039`), and WT2 PPL `65.2 -> 20.3` (still well below 3-bit `122/160`). Simple open-loop `BC_open` beat sequential closed-loop `BC_closed` in **`6/6`** ultra-low-bit formats.
- **Mandatory Negative Controls (NEVER apply `BiasCorr` blindly!):** Open-loop `BiasCorr` actively **HURTS** models whose weights already compensate for quantization or use post-norm / incompatible scaling:
  - Calibrated `GPTQ-Int4` (`+5.7%` WT2 KL)
  - Trained/QAT ternary `prism-ml/Ternary-Bonsai-2-27B` (`0.653 -> 1.507` weight-diff, `2.112` open act, `9.317` closed act — note Bonsai-2 itself crushes PTQ ternary `2.689` and corrected 2-bit `1.138`, while trailing corrected 3-bit `0.271`)
  - Post-norm `OLMo-2-7B` (`+92%` to `+211%`), `Llama-3.2-3B` (`+17%` to `+43%`), `SmolLM2-1.7B` (`+18%` to `+51%`), `Phi-4-mini` (`+3%` to `+24%`), and `Granite-4.0-h-tiny` `HF-G16` arm.

### 5.4 Cross-Hardware Generality Boundary (Portable Math vs Xe2-Specific Speed)
- **100% Universal (NVIDIA, AMD, Intel, Apple MLX, CPU/NPU):** Pos0-excluded `BiasCorr` + fused epilogue (`~0%` cost), `shared_only` MoE sidecars, 2-bit open-loop rescue, `HF-INT4-G16` quality-per-bit, `V > K` sensitivity (`11/12`), and plain rotated 3-bit KV beating 1-bit QJL (`3.13x`).
- **Uniquely Valuable on Xe2 (Arc Pro B70), Ampere (SM80), RDNA3 & INT8-Only NPUs/CPUs:** Zero-unpack `W4A8-INT8` unlocks the *only* `2x` hardware prefill multiplier on silicon that has `2x` INT8 matrix throughput and **no native FP8/FP4 matrix units**.
### 5.5 `HF-INT4-G16/1` Mathematical Bounds, Killed Dead Ends, Full-$H$ GPTQ & Weights-vs-vLLM Ranking (v3.10.0)

Deep mathematical and real-shard verification of `Had128NoPerm + HF-INT4-G16` (`4.2603 bpw` with `BiasCorr`, `4.2551 bpw` without; `q in [-8..+7]`, `u in {1..16}`, `125` unique integer levels in `[-128..+112]`, all `125` realized on `Qwen3.8-27B`) proved that **the V1 bit container already sits within `0.01 dB` of its weight-MSE optimum** — meaning both the largest quality gains and the largest speed gains live **inside the existing V1 format**, not in inventing a new format:

#### Proven Format Theorems & Real-Weight Bounds
- **Gaussian equivalence after `Had128NoPerm` (`[MEASURED]` + `[THEOREM]`):** Real `Qwen3.8-27B` weights quantize to **`22.12–22.14 dB`** after `Had128`, matching i.i.d. $\mathcal{N}(0,1)$ noise (`22.11–22.13 dB`) under the exact same quantizer. Because any orthogonal rotation of an isotropic Gaussian remains Gaussian, **no larger, global, complex, or learned rotation can materially improve W4A16 weight MSE**.
- **Uniquely forced integer scale set (`[THEOREM]`):** Native Xe2 INT8 DPAS requires $q \cdot u \in [-128, 127]$. With $q \in [-8, 7]$, $-8u \ge -128 \implies u \le 16$. The 16 integers $\{1..16\}$ are the **only** distinct positive integers that fit; a non-uniform scale mapping can only exist as a full 256-byte 2D int8 LUT `T[u][q]` (`+0.30 dB` weight SQNR in 6 Lloyd iterations — Format V3 trade-off).
- **Row-coupling `u` panic falsified on real shards (`[MEASURED]`):** Even on `mlp.down_proj` (`K=17,408`, 1,088 groups/row), `u_mean = 9.40–9.44`, `u <= 3` occurs in only **`0.009%`** of groups (`0.0014%` of total error), and the entire cost of 4-bit row-coupled `u` vs ideal continuous G16 scales is just **`0.37–0.39 dB`** (whereas ideal continuous G32 / FP8-G32 loses **`-0.76 to -0.81 dB`**).
- **True INT32 accumulator bound (`[THEOREM]`):** With $|a| \le 127$ and $|q \cdot u| \le 128$, worst-case dot product is $16{,}256 K < 2^{31}-1$ for **`K <= 132,104`** (not `17,408` or `65,536`).
- **Why Hadamard blinds diagonal weighting (`[THEOREM]`):** Hadamard rotation flattens the **diagonal** of the activation Hessian $H = X^T X$ toward $\operatorname{tr}(H)/K$ while preserving **100% of its eigenvalues**. That explains why diagonal-weighted `ActMSE-CD` (`Exp 15C`) gained almost nothing, whereas **full-$H$ GPTQ/LDLQ** retains the entire eigenvalue spectrum.

#### 7 Attractive Dead Ends Killed (Do Not Spend GPU Hours)

| Proposed Idea | Verdict | Evidence / Proof |
|---|---|---|
| Bigger / global / `R1` / SpinQuant / FlatQuant rotation for W4A16 | **KILLED** | `-0.12` to `+0.005 dB` weights, `<= +0.14 dB` held-out outputs (`[MEASURED]` + Gaussian invariance `[THEOREM]`). Only helps W4A8 activation crest (`+2.0% / +4.5%` KL ceiling — low ROI). |
| Complex unitary / DFT / FFT / phase-butterfly rotation | **KILLED** | Welch / $\ell_\infty$ optimality `[THEOREM]`: real Sylvester $H_B$ already achieves global minimum coherence $1/\sqrt{B}$; real embedding $\Phi(U)$ doubles storage without lowering coherence. |
| Per-head `R2` (V–O) rotation on gated-attention / GDN models | **KILLED** | Models with `attn_output_gate=true` apply elementwise $\sigma(\text{gate})$ right before `o_proj`, breaking $R_2$ commutativity (`[THEOREM]` + `[SOURCE-VERIFIED]`). |
| Optimized sign vectors $s^* \in \{-1,+1\}^K$ | **KILLED** | Layer MSE spread is `0.237%` (`~0.005 dB` ceiling `[MEASURED]`). |
| Harder `d_row` clipping or joint `(d, u, q)` coordinate descent | **KILLED** | Already uses 6 clipped ratios (`0.76..1.06`); extending to `0.40` gains `+0.006–0.014 dB`; `Exp 15C` `MSECD` KL unchanged (`0.03193` vs `0.03196`). |
| `FP8-E4M3` G32 scales | **KILLED** | Even ideal continuous G32 scales lose **`-0.80 dB`** vs G16 (`[MEASURED]`). |
| "Centered-Hessian GPTQ + BiasCorr is orthogonal" | **FALSIFIED OOD** | In-sample gain is `<= 0.06 dB`; **out-of-sample, uncentered GPTQ beats or ties centered+BC in `11/12` held-out cells** (up to `+1.17 dB`) because `BiasCorr` is a constant vector while uncentered GPTQ suppresses per-token mean scaling error $\Delta W \hat{\mu} a_t$ (`[THEOREM]` + `[MEASURED]`). |

#### Highest-ROI Quality Lever (Zero Format Change, Zero Runtime Cost): Full-$H$ Uncentered GPTQ Inside V1
- **Algorithm:** Keep exact V1 container (`d_row` from RTN, `u` lazily selected at each group start, `q in [-8,7]` placed sequentially via full-$H$ GPTQ error feedback with uncentered $H$, sequential closed-loop inputs).
- **Measured L0 signal (`[MEASURED]`):** On held-out WikiText-2 + disjoint Python stdlib with `65,536` calibration tokens, `L0 in_proj_qkv` and `in_proj_z` gained **`+3.41 to +4.28 dB`** output SNR over `RTN + BiasCorr` (projected **`-25% to -45%` 64L KL** `[HYPOTHESIS-UNTESTED]`, enough to challenge or beat `TFQ3-4b` data-free inside V1!).
- **Mandatory sample-size law (`n >> K`):** At `1,024` calibration tokens on `mlp.down_proj` (`K=17,408`), GPTQ overfits catastrophically (`44.7 dB` train SNR, **`-0.17 to -1.39 dB` held-out**). Use **`>= 256K` mixed calibration tokens** and always gate on a cheap 2-layer pilot (`L3` + `L40` `down_proj`, pass gate `>= +1.0 dB` held-out SNR over `RTN+BiasCorr`) before paying for a full 64-layer conversion.
- **Interaction with `BiasCorr`:** After uncentered GPTQ, mean error drops to `0.2–0.3%` of total error and `BiasCorr` adds `<= 0.01 dB` per layer; re-run the 4-check served-kernel `BiasCorr` gate after GPTQ lands rather than assuming they stack.

#### Deep-Layer GPU1 Verification of Full-$H$ Uncentered GPTQ Inside V1 (`Exp 19-Pilot` & `Exp 19-Pilot-B`, v3.11.0)
Measured on GPU1 (`Qwen3.8-27B` BF16 streamed across `L0..L40`, exact `xe2_dpas_native_v1` pack/unpack round-trip `max_abs == 0.0` on every tensor, `261,632` pos0-excluded tokens with `0/0` 6-gram overlap against frozen WT2 & Code eval windows):
- **Deep-layer gains hold across all 17 tested projections (`L0`, `L3`, `L39`, `L40`) — zero regressions, zero format/kernel changes:**
  - `L0` (`261K`, `damp=0.01`): `+4.12` to **`+4.86 dB`** across all 5 projections (`down_proj +4.86 dB`, `gate_proj +4.71 dB`, `in_proj_qkv/z +4.52 dB`, `out_proj +4.12 dB`).
  - `L3` (first full-attention): `self_attn.o_proj` **`+5.45 dB`** (`~3.5x` lower MSE!), `q_proj` **`+2.85 dB`**, `mlp.gate_proj` **`+2.14 dB`**, `mlp.down_proj` (`K=17,408`) **`+1.19 dB`** (`+1.25 WT2 / +1.13 Code` at `damp=0.35`).
  - `L39` (deep full-attention): `self_attn.o_proj` **`+3.35 dB`** (`20.41 -> 23.76 dB`), `q_proj` **`+2.42 dB`**, `mlp.gate_proj` **`+1.39 dB`**, `mlp.down_proj` **`+0.69 dB`** (at `damp=0.05`).
  - `L40` (deep GatedDeltaNet, full-alphabet stdlib): `in_proj_qkv` **`+2.26 dB`**, `out_proj` **`+1.64 dB`**, `mlp.gate_proj` **`+1.52 dB`**, `mlp.down_proj` **`+0.86 dB`** (`+1.22 WT2 / +0.50 Code` at `damp=0.35`).
- **Empirical `n >> K` overfitting curve on `mlp.down_proj` (`K=17,408`):**
  - `L3.down_proj`: `16K` = `-2.12 dB` -> `65K` = `-0.79 dB` -> `261K (d=0.01)` = `+0.53 dB` -> `261K (d=0.05)` = `+0.78 dB` -> `261K (d=0.20)` = `+1.11 dB` -> **`261K (d=0.35)` = `+1.19 dB`**.
  - `L40.down_proj`: `16K` = `-2.47 dB` -> `65K` = `-0.22 dB` -> `261K (d=0.01)` = `+0.48 dB` -> `261K (d=0.05)` = `+0.69 dB` -> **`261K (d=0.35)` = `+0.86 dB`**.
- **Per-Projection Marchenko–Pastur Shrinkage Damping Rule (do NOT use one global `damp=0.01`):**
  - Dense RMSNorm-fed attention inputs (`q/k/v/o`, `in_proj_qkv/z/a/b`, `out_proj`, `K=5120–6144`): **`damp = 0.05`** (preserves `+2.2` to `+5.45 dB` without deep-layer overfitting).
  - `mlp.gate_proj` / `up_proj` (`K=5120`): **`damp = 0.10`** (`+1.50` to `+2.14 dB`).
  - Sparse-SwiGLU `mlp.down_proj` (`K=17,408`, lower effective active-neuron sample count): **`damp = 0.35`** (`+0.3` to `+0.66 dB` better held-out than `damp=0.01`!).
- **Un-normalized SwiGLU Hessian energy imbalance (`[MEASURED]`):** After RMSNorm (`attn_in`, `attn_out`), `tr(H_prose) / tr(H_code)` is `0.92–1.01`, but directly after un-normalized SwiGLU (`mlp.down_proj`), `tr_prose / tr_code` swings from **`2.86x`** at `L3` (silently weighting unweighted `H` 74% prose / 26% code; trace-balancing boosted L3 Code by `+0.39 dB`) to **`0.75x`** at `L40`. Always audit domain Hessian trace ratios on mixed calibration corpora.
- **Three confirmed V1-GPTQ implementation invariants:**
  1. **Uncentered GPTQ absorbs `BiasCorr`:** across all 17 tested projections, adding post-GPTQ `BiasCorr` changed held-out SNR by **`0.00 to +0.01 dB`**.
  2. **`d_row` from RTN strictly beats unclipped `amax/112`:** across all 17 projections, keeping RTN's `d_row` beat `amax/112` by **`+0.07 to +0.10 dB`** with saturation fraction `< 0.0008`.
  3. **Xe2 Hybrid CPU/XPU GPTQ Execution:** accumulate `H = X_rot^T X_rot` on XPU in FP32 (flushing every 8 mini-batches to CPU FP64), compute `torch.linalg.cholesky` **once per shared input group** in CPU FP64 (`1.1 s` for `K=5120`, `31.3 s` for `K=17,408` — XPU falls back to CPU anyway), then run the `bs=128` GPTQ block update on XPU in FP32 (`< 2 s`). Only **4 Choleskys/layer** cover all 7–8 decoder linears (`~53 min` for full 64-layer sequential closed-loop 27B conversion).

#### Master Decision Ranking: Weights/Converter vs. vLLM/Kernel Upgrades (All Inside V1)

| Domain | Rank | Change | Status & Expected Impact | Pass / Kill Gate |
|---|---:|---|---|---|
| **vLLM / Kernel** | **1** | **Dual-zebin GRF split:** compile decode/sk4/actq at **128 GRF**, tall prefill at **256 GRF** | `[HYPOTHESIS]` **`+1.0 tok/s` (`C=1`), `+1.0–1.5 tok/s/req` (`C=4/8`)**; fixes 1.25-wave EU stall on `N=5120` | **Bit-exact** greedy SHA-256; check `m4` spill report |
| **vLLM / Kernel** | **2** | **Small-M DPAS decode (`M=1..32` W4A16-on-DPAS or Digit-DPAS) + real `m3` + bisect `exp8a` `M=4` regression** | `[HYPOTHESIS]` **`C=4`: `+45–50%` (`20 -> ~29–30 tok/s/req`)**, **`C=8`: `+22–26%` (`21.4 -> ~26–27`)**, halves `M=3`, kills `M=4->5` precision cliff, unlocks **MTP** | Flat `M=1..32` curve + full KL & exact-answer parity gate (`DNR-009`) |
| **Weights / Converter** | **3** | **Full-$H$ Uncentered GPTQ into V1 (`>= 256K` cal tokens, pinned corpus)** | `[MEASURED]` `+3.4–4.3 dB` L0 SNR; `[HYPOTHESIS]` **`-25% to -45%` 64L KL** (0 runtime cost) | Pilot `L3/L40 down_proj` `>= +1.0 dB` held-out; 64L DF KL `<= 0.0253` WT2 / `<= 0.1104` Code; kill if `< 10%` KL win |
| **vLLM / Kernel** | **4** | **Fix `v0.2.0` default `w4a16 -> Triton` fallback at `M>=5`, log all fallbacks, fix false `K<=17408` guard** | `[SOURCE-VERIFIED]` Hygiene bugfixes preventing silent multi-× slowdowns | Zero silent Triton fallbacks under `HF_INT4_G16_BACKEND=native` |
| **vLLM / Kernel** | **5** | **Single-pass in-register `actq` (`+2%` prefill) + `A8` pre-expanded int8 prefill diagnostic (`16x` A-tile re-read)** | `[HYPOTHESIS]` Up to **`+28–30%` prefill (`1.28x -> ~1.7–1.8x` vs GPTQ)** if A-tile re-reads are the limiter | If pre-expanded int8 hits `>= 250 TOPS`, fix unpack; if `~170 TOPS`, add SLM-shared A / 32-col tile |
| **Weights (V3 opt.)** | **6** | **2D int8 LUT `T[u][q]` (`256 B/tensor`)** or **per-projection `BiasCorr` shrinkage** | `[MEASURED]` `+0.30 dB` (`T[u][q]`) or `-2% to -4%` KL (shrinkage — only if GPTQ is not adopted) | Kill `T[u][q]` if `M=1` decode regresses `> 2%` or KL win `< 3%` |
