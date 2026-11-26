# Measurement Methodology, Correctness Lanes & Guardrails (Intel Arc Pro B60/B70)

Most apparent speedups or regressions in local LLM tuning are observer bugs, warm-cache leakage, mixed metric classes, or silently relaxed correctness gates. Every publishable B70 measurement obeys these rules.

## 1. Metric Definitions (Never Mix Classes — `DNR-015`)

| Metric | Exact Definition | Never Confuse With |
|---|---|---|
| **Cold prefill rate (`cold_input_tok_s`)** | `actual_prompt_tokens / TTFT_s` on a unique entropy-first prompt (`>= 64` chars leading entropy) with verified `prefix_cache_hits_total` delta `== 0.0` | Engine-native `pp` rate or warm/cache-hit TTFT |
| **Post-first-token decode (`decode_tok_s`)** | `(completion_tokens - 1) / (t_end - t_first_token)` counted with tokenizer/server `usage`, NOT SSE chunks (speculation emits `~3` tok/chunk — the stale `24.6` EXL3 figure was chunk rate!) | SSE chunk rate or `e2e_tok_s` |
| **Decode energy (`decode_j_per_tok`)** | `(energy_end_J - energy_start_J) / (completion_tokens - 1)` from hardware `energy1_input` counter over monotonic seconds | Configured TDP (`230 W`) divided by tok/s |
| **Concurrent throughput (`cN`)** | Total completion tokens across all `N` concurrent streams divided by wall time from first start to last finish | Single-stream `C1` speed (`153 tok/s` was `C16`!) |

## 2. Three-Class Correctness-Lane Framework (Freeze BEFORE Inspecting Outputs!)

Declare the change class in `PLAN.md` **before** running the candidate. Never promote Class 1 -> Class 2 to salvage a failed run, nor demote Class 2 -> Class 1 to kill a near-miss:

| Class | Definition | Terminal Gate | On Divergence |
|---|---|---|---|
| **Class 1 — Bit-identical by construction** | Accessor/layout/precision-preserving rewrites | **Output token IDs `sha256` AND full layer-norm `sha256`** | **REJECT.** No semantic appeal. |
| **Class 2 — Same math, different association / libm** | `sycl::exp` vs `std::exp`, FMA contraction, sum order, fp16 round-trip, CUTLASS tile order | **FP64/FP32 kernel oracle** (`rel-Fro <= 1e-3` or `rel_l2 <= 5e-5`, divergence ratio `<= 10`) + **zero answer-carrying-span token flips** (including special tokens like `<|im_end|> -> <|im_start|>`) + deterministic replay + `>= 1,024`-token coverage | Adjudicated by kernel oracle + answer-carrying span; never gate on accumulated layer-N scalar RMS alone (`RL-222`). |
| **Class 3 — Approximate / precision reduction** | Quantization, INT8 prefill, dropped terms | Separate predeclared quality objective (exact-answer battery + task suite), stricter than Class 2 | Not eligible for Class-2 lane. |

**Libm-vs-logic control:** If recomputing one op the host's way restores norm identity, libm/reassociation is the whole difference; if it does not, there is a logic bug — stop before running the full battery.

## 3. Sampling, Depth & Reproducibility Discipline

- **Public headline speed:** Use the **model card's realistic sampling** (`temperature`, `top_p`, default `reasoning_effort`, fixed output budget e.g. `2,048` tokens, `n >= 3` seeds) across all reachable standard depths (`0`, `8k`, `32k`, `64k`, `128k`, `max`). Label unreachable depths explicitly (`does not fit (largest: N)`).
- **Greedy diagnostics (`temperature=0`, `top_p=1`):** Exactness, attribution, and speculation parity only — never headline greedy speed as realistic serving speed.
- **Reproducibility is a selection criterion:** Verify whether an engine repeats greedy outputs deterministically (`exl3xpu` repeated 78% of items across runs and needed `n=5` to see a 1-item delta).
- **Thinking models (`--reasoning-parser`):** Capture **both** `delta.reasoning` and `delta.content` (plus `delta.reasoning_content`). Record `empty_answer` explicitly; pair high-effort cells that saturate `max_tokens` inside `<think>` with finished-answer cells (e.g. `effort=0.2`).
- **Capacity probes vs speed ladders:** Different `--max-num-batched-tokens` (`1024`/`512` vs `2048`) or output lengths (`128` vs `2048`) belong in separate capacity tables, never merged into the speed ladder.

## 4. Noise Control, Energy & Hardware Safety

1. **Measure on the engine of record:** Always benchmark a candidate against the current fastest adopted route, not a slow fallback arm.
2. **One variable per arm, interleaved controls:** `control -> candidate -> control`, `n >= 3` pairs for screen, `n >= 5` fresh pairs for confirmation. Discard one same-shape warmup after cold JIT.
3. **Thermal & power discipline:** Record as-found `power1_cap` (never alter caps to fake a pass) and restore them at cleanup. Gate non-display cards `<= 55 C`, display cards against their own idle (`<= 68 C`, `DNR-024`), and declare the sustained thermal regime.
4. **Live VRAM tiers (`vram_mm`):**
   - `< 500 MiB`: hard abort.
   - `500–1,023 MiB`: isolated capacity observation only.
   - `1,024–2,047 MiB`: staged C1 / max-context profile.
   - `>= 2,048 MiB`: preferred C1 performance.
   - `>= 3,072 MiB`: concurrency.
   - Diagnostics/arena probes: never allocate beyond `free - 4 GiB` and never stress the display card.
5. **Post-run cleanliness proof:** Verify `min_vram_mib`, `vram_after_mib` restored to baseline, `caps_after_uw` unchanged, and `dmesg_ok` (zero `xe` resets or PCIe AER drops).
6. **Mandatory quality & retrieval battery:** Always include exact in-head arithmetic (the `"14" not "30"` canary that caught `DNR-009`), structured JSON, code state-trace, boundary logic, multi-turn correction, and multi-depth needle retrieval (`1.4K`, `8.5K`, `33.8K`, `135K+`). Six canaries prove only smoke health, never general quality.

## 5. Long-Context Quantized / FP8 KV Quality Gate

Never ship a 256K–512K quantized KV pool on capacity alone — verify that quantization noise does NOT accumulate across the full context window:

1. **Matched BF16 vs quantized KV held-out perplexity:** Build a strictly held-out natural technical prose evaluation slice (`256` tokens, excluded from the prefix corpus) conditioned on `4K, 32K, 64K, 128K, 245K` (and ultra-deep `384K, 496K` beyond BF16's VRAM ceiling). Score via `/v1/completions` with `echo=True, logprobs=1, max_tokens=1, temperature=0` and report `NLL`, `PPL`, `ΔPPL (%)`, and per-token `logprob MAE`.
2. **Dual-needle recall spanning oldest + middle KV:** Bury two unique keys at **`~15%` depth (oldest KV entries)** and **`~55%` depth (middle KV)** inside real prose/code up to the maximum usable context (`245K` and `496K`) and verify **`2/2` exact recall**. For reasoning models, set `max_tokens >= 384` (never `128`, which can saturate `<thinking>` and return empty `content=""`).
3. **Check architecture-level KV bounds:** Check whether the model normalizes/clips K & V before KV storage (e.g., Inkling 2 L2-clips K and RMSNorms V right before KV write, which kept FP8 KV within `-0.48%..+0.41%` PPL of BF16 through `245K`, `3.58` PPL at `496K`, and `2/2` exact recall at `495,964` tokens). Never generalize that result blindly to standard unclipped transformers without re-measuring!


## 6. Adversarial Forensic Review Posture & 13-Point Audit Checklist (v3.9.0)

Before declaring any quantization study, kernel patch, or benchmark ready for publication or upstream PR, run an independent adversarial review (or self-audit) with **this exact forensic attitude** — distilled from 10 rounds of cross-model peer review on the B70 27B quantization study:

### Forensic Review Attitude
1. **Treat every prose claim as guilty until verified against raw JSON receipts, tensor headers, and live `file:line` source.** Never trust a previous pass's summary table.
2. **Zero cheerleading, zero softening.** Lead with the exact verdict and a numbered table of concrete defects (`file:line`, receipt key, severity `HIGH/MEDIUM/LOW`).
3. **Tag provenance strictly and write `unexplained` when a cause is unmeasured.** Separate `[THEOREM]`, `[LIVE]`, `[HISTORICAL]`, `[INFERRED]`, and `[UNTESTED]`. Never invent a plausible story for an anomalous dip (e.g., `mlp_down M=512` `0.92x` stayed `unexplained` even after 8 kernel variants failed to beat oneDNN).
4. **Hunt two-variable confounds relentlessly.** If a prototype changed two things at once (e.g. separate bias kernel + disabling fused `ngen_swiglu`, or `bf16` bias on W4A16 vs `fp32` bias on W4A8), reject attribution immediately.
5. **Demand adversarial null controls, not just positive wins:**
   - **Sign-flipped control (`-b`):** must catastrophically *worsen* KL (verified: `+84.0%` WT2, `+68.7%` Code, DNR drops `8/8 -> 5/8`).
   - **Matched-L2 random-direction control (`>= 3` seeds):** must *worsen* KL (verified: `+28.5–35.0%`).
   - **IEEE `-0.0f` zero-bias path control:** must match `NULL` **`100%` SHA-256 bit-for-bit** across every depth and prompt.
   - **Disjoint calibration corpora:** log exact corpus file list, token-ID SHA-256, and measured 6-gram overlap (`0/0`).
6. **Audit silent fail-open paths and whole-repo blast radius.** Grep every caller of every modified kernel across other model families, reference verifiers, quantization loaders, and argument binders.

### 13 Real Failure Classes Caught by Adversarial Audit (Check Every Time)

| # | Failure Class Caught | Mandatory Prevention Rule |
|---|---|---|
| 1 | **Silent fail-open sidecar load** (printed "loaded" before checking if any key matched) | Track every loaded key (`BiasTracker`); fail closed (`rc != 0`) on any unused/missing key + add a negative test. |
| 2 | **Reference oracle silently dropping new math** (`fp64_ref` overwrote output without bias) | Update FP64/FP32 verification oracles in the exact same patch as the GPU kernel. |
| 3 | **Unsupported quant paths silently ignoring opt-in state** (`W4A8`/`W4A4`/`GPTQ`) | Assert `bias == null` or return an explicit error on every path that does not implement the feature. |
| 4 | **Stale OpenCL kernel args on trailing signature change** (`Dev.launch` sets args by index) | Enforce `CL_KERNEL_NUM_ARGS` arity assertions (`kernelArgs`) at kernel init time. |
| 5 | **Cross-family blast radius** (shared zebins/macros broke another model family's callers) | Grep the entire repo for every touched kernel/macro symbol and update all callers (e.g. Gemma 4). |
| 6 | **Two-variable prototype confound** (`add_bias_f16` also disabled fused `ngen_swiglu`) | Fuse into the epilogue so only one variable changes; never publish prototype speed as feature cost. |
| 7 | **Proxy-vs-shipped checkpoint gap** (re-quantized proxy showed `-12.9%` Code KL while shipped checkpoint was a `-2.77%` null) | Never transfer proxy CIs onto shipped checkpoints; re-measure on the exact shipped artifact. |
| 8 | **Rank-1 DC Mean Trap in uncentered SQNR** (`91–95%` energy in a constant DC vector inflated total SQNR to `34 dB`) | Retire uncentered SQNR; require centered SQNR + full-depth `KL(P_ref || P_Q)` + `blk32` 95% CIs. |
| 9 | **Clock drift & impossible timer algebra** (`2050 vs 2750 MHz` across sessions; `core + FWHT > e2e` by `38–76 us`) | Log `act_mhz`, `temp`, `power_w` per cell; run alternating `b1 -> cand -> b2`; reject runs where `parts > total`. |
| 10 | **Marginal vs wall-clock cost confusion** (`+10.5 s` in-harness vs `229 s` real standalone export) | Report both marginal compute and end-to-end wall-clock artifact generation time. |
| 11 | **Constant/citation drift** (`456` B580 vs `608 GB/s` B70; `0.004` vs `0.0051 bpw`; wrong DNR ID) | Recompute every constant directly from hardware specs, safetensors headers, and ledger files. |
| 12 | **Calibration provenance & shingle overlap** (unpinned 2.6 KB docs slice overlapped eval prompt) | Pin calibration corpus with token-ID SHA-256 and verified 6-gram overlap; use disjoint general text/code. |
| 13 | **Local host hacks & private leaks in diffs** (`build/xpu.zig` compiler workaround, `/data/...`, PCI IDs) | Keep host workarounds out of feature patches; scrub all private paths/PCI IDs before publishing. |
| 14 | **Token-cap truncation artifacts** (`384`-tok GSM8K looked like `41 -> 48`, vanished at `1,024` tok `58 -> 57`) | Always run math/reasoning checks at `>= 1,024` tokens and report untruncated subsets separately. |
| 15 | **Simulated-math vs served-kernel gating gap** (`Exp 21` passed simulated math, failed served kernel by `+0.00073`) | Pre-register that the actual served GPU kernels decide pass/fail; never ship a sidecar gated only on simulated math. |
| 16 | **Decode-only metric distortion** (judging a `-10%` decode format without `TTFT + decode` turn time) | Report end-to-end turn wall clock (`TTFT + reply`), `J/turn`, and break-even output length across `8K -> 128K`, and separate cold vs warm prefix-cached regimes. |
| 17 | **Overstating internal AI audits as peer review** | Explicitly disclose single-author single-workstation scope; never call AI-agent audits "human peer review", and never rewrite frozen receipt evidence to satisfy prose-hygiene filters. |
