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
