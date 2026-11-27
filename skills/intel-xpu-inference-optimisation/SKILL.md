---
name: intel-xpu-inference-optimisation
description: >
  Hill-climb local LLM serving speed and context capacity on Intel Arc Pro B60/B70
  (Battlemage Xe2, bmg_g31) and Intel XPU GPUs across vLLM XPU, TensorFold XPU,
  exl3xpu, llama.cpp SYCL, and Strata/XeStrata: single-user and concurrent decode
  tok/s, cold prefill tok/s and TTFT, speculative decoding (DFlash2, DSpark, MTP),
  KV cache and SWA max-context tuning, 16-byte aligned packs, peer-store TP2, AOT
  CUTLASS Xe2 kernels, and energy (J/token) under live VRAM, thermal, and 3-class
  correctness gates. Computes Intel Xe2 silicon rooflines first, checks closed arms
  (DNR-001..DNR-042), runs noise-controlled single-variable A/B tests, and qualifies
  every winner with realistic model-card sampling and exact retrieval before publishing.
---

# Intel XPU Inference Optimisation (Hill-Climbing on Arc Pro B60/B70 & Xe2)

You systematically hill-climb local LLM serving speed, maximum context, and energy efficiency on Intel Arc / Battlemage Xe2 GPUs (`bmg_g31` and related XPU devices), backed by roofline arithmetic, a positive ego-free engineering mindset, and reproducible evidence.

**Attitude:** Treat every `NotImplementedError`, OOM, or unexpected slowdown with calm curiosity as a clue to isolate in a 5-second probe — never give up early, never attach ego to a candidate that loses an A/B (revert fast and log why), welcome hard external critiques by measuring them directly, and ship small, verified, opt-in wins continuously.

## Core Loop

```
preflight & DNR check -> roofline ceiling -> baseline -> pick highest (gap x likelihood) lever
       ^                                                                          |
       +--- update ledger + PLAN.md <--- accept/reject (controls + quality) <-----+
```

Read only the reference file needed for the current step:
- `references/field-manual-and-case-study.md` — **Complete Xe2 Mental Model, Attitude, Decision Tree & Annotated Case Study:** how to think in Battlemage Xe2 physics (4 regimes), 8 winning habits, "If You See X -> Do Y" troubleshooting table, full Inkling 2 Mini journey (`39.7` -> `80.4/99.2` + Vision + `512K` FP8 KV + `11x` Chunked-LSE + `+13–28%` chunked oneDNN prefix SDPA + ESIMD `PagedXmx<M>`), 2-min multi-GPU SYCL audit (Strata `907x` peer prefill), and the `64K–262K` deep-context competitive post-mortem!
- `references/workflow-and-intake.md` — **Start here for a new model or open optimization session:** full 5-phase approach from Day-0 upstream/tensor reconnaissance -> external-repo audit -> roofline -> baseline & 5-second op decomposition -> opt-in kernel/feature iteration -> 512K quality gates -> shipping.
- `references/ceilings.md` — Intel Xe2 silicon datapaths (`FP16/BF16`, `INT8 s8xs8 367 TOPS`, `INT4 ESIMD`, `DP4A`, 16B alignment), looped/MoE active-byte math, SWA `mbt` KV formula, and speculative ceilings.
- `references/quantization.md` — Complete Intel Xe2 quantization & drafter economics guide (12-family data-free hierarchy, `HF-INT4-G16/1` (`Had128`) & `/2` (uniform `Had64` MoE), trellis `TFQ3` vs scalar trade-off, `BiasCorr` 4-check served-kernel OOD gate, `2.0–2.54 MiB` `shared_only` MoE law, 2-bit `31 -> 93/160` rescue vs 3-bit 1,024-cap null, mandatory negative controls, **plus v3.10.0 V1 mathematical bounds, 7 killed format/rotation dead ends, full-$H$ closed-loop uncentered GPTQ inside exact V1 (`Exp 19-Full`, v3.12.0: **64L WT2 KL `-39.32%` (`0.01643`), Code KL `-30.44%` (`0.08972`)**, `p=1.0`, **beats trellis `TFQ3-4b` DF by `-35% / -19%`** with 0 format/kernel change & no `BiasCorr` needed via `0.05/0.10/0.35` per-projection shrinkage), and the Weights-vs-vLLM ROI decision matrix**).
- `references/kv-and-context.md` — KV cache K/V asymmetry & KL-divergence table, plain Hadamard-rotated 3-bit/4-bit KV vs TurboQuant 1-bit QJL (`3.13x / 2.65x` lower KL, `256K` projected in `20.5 GiB`), `Exp 22` live `8K -> 128K` end-to-end turn-latency & break-even output scaling, FP8 E4M3 KV 512K Chunked-LSE (`+0.07%` PPL @ 245K), and 11-turn session stability.
- `references/engines-and-topology.md` — 6-engine capability matrix (`vLLM XPU`, `TensorFold XPU`, `exl3xpu`, `llama.cpp SYCL`, `Strata/XeStrata`, `OpenVINO GenAI`), MoE (`150 W`) vs Dense (`180 W / 230 W`) power sweet spots, and dual-B70 topology rules.
- `references/kernels-xe2.md` — Battlemage Xe2 (`bmg_g31`) GPU kernel engineering guide: **dual-zebin 128-GRF decode vs 256-GRF prefill occupancy law**, small-M DPAS W4A16 & Digit-DPAS (`M=1..32`, killing `M=4` FMA collapse & `M=4->5` cliff), prefill A-tile `16x` re-read diagnostic, AOT `icpx`, 16B/64B alignment, CUTLASS Xe2 grouped GEMM + device-side graph-safe grouped MoE prefill (`3.41x–3.52x` @ `M=512`), fail-closed MoE admission invariants, Chunked-LSE FP8 prefill (`11x`), pure-PyTorch XPU ViT porting, peer-store collectives, and closed Xe2 kernel patterns.
- `references/levers.md` — Workload regime matrix + 15 proven Intel XPU levers ordered by measured ROI across 12+ model families.
- `references/gotchas.md` — All 42+ closed arms (`DNR-001`..`DNR-042`) & hard-won Level Zero, SYCL, Triton, vLLM, MoE memory-cap, 384-token truncation, served-kernel gating, and quiet-host monitor traps.
- `references/methodology.md` — 3-class correctness-lane framework, noise-controlled interleaved A/B, realistic card sampling, cold-cache & 512K KV quality gates, end-to-end turn latency vs decode-only evaluation, single-author validation disclosure, the **Adversarial Forensic Review Posture & 24-Point Audit Checklist**, and the **Multi-Model Adversarial Swarm & Real-Weight Falsification Protocol (v3.10.0)**.
- `references/publishing.md` — Pinning, opt-in kernel gating, sanitization, `data/benchmarks.v1.json` schema, 12-gate CI, and 2x anti-aliased result graphics.
- `scripts/b70_preflight.py` & `scripts/b70_ceiling.py` — Stdlib-only host preflight and roofline calculator (`examples/`: `inkling2-mini-b70.json`, `qwen38-27b-dense-b70.json`, `nemotron35-30a3-b70.json`).

---

## Step 0: Preflight & Closed-Arm Gate (Zero GPU Waste)

1. **Read closed arms first.** In this cookbook, read `docs/DO-NOT-REPEAT.md` and `docs/WHAT-WORKED.md`. Never re-run a closed arm unless its **Revisit when** condition is met.
2. **Run read-only preflight:**
   ```bash
   python3 skills/intel-xpu-inference-optimisation/scripts/b70_preflight.py
   ```
   Verify:
   - Target non-display card PCI ID, live free VRAM (`vram_mm`), as-found power cap (never alter caps to fake a benchmark), and package temperature (`<= 55 C` admission).
   - No competing inference server or container on the target GPU.
   - Safe host `MemAvailable` (`>= 20 GiB` remaining under any memory-capped job) and root/data disk headroom (`>= 512 MiB` root reserve).
3. **Inspect the actual checkpoint tensors.** Read `config.json` and safetensors/GGUF tensor headers (`DNR-014`). Never infer weight formats from repo names.
4. **Declare constraints upfront:** target card count (1x vs 2x), context goal, realistic model-card sampling (`temperature`, `top_p`, `reasoning_effort`), and whether lossy/approximate changes are allowed.

---

## Step 1: Price the Intel Xe2 Roofline First

Run `scripts/b70_ceiling.py` before writing code or launching long sweeps (`references/ceilings.md`):

- **Use measured sustained B70 constants, not marketing sheets:**
  - Sustained GDDR6 bandwidth: **~521–528 GB/s** (not 608 GB/s peak).
  - Practical BF16 XMX compute: **~50% of ~157 TFLOPS** (~75–85 TFLOPS effective) on large GEMMs.
  - **No native FP4 or FP8 DPAS on Xe2 Battlemage:** NVFP4 / MXFP8 / INT4 / EXL3 weights save memory bandwidth and unpack to BF16/FP16 in registers (`W4A16`/`W8A16`).
- **Decode ceiling (memory-bound):** `active_bytes_per_token / sustained_BW`. Include dense weights, routed experts (`top_k / n_experts`), loop passes (e.g. Inkling runs 24 stored layers across 48 visits), `lm_head`, and KV bytes at target depth.
- **Speculative ceiling:** divides target bytes by mean accepted length `a`, plus `k` draft steps (draft `lm_head` and verify batch size `M = k+1` often dominate).
- **Prefill ceiling (compute-bound):** `2 * active_params_per_token + attention_flops(context)` divided by practical XMX TFLOPS.
- **KV & SWA ceiling:** separate full-attention layers from Sliding Window Attention (SWA) layers. On vLLM, SWA in-flight block reservation scales with `--max-num-batched-tokens` across every local-attention visit.
- **Impact gate:** before any job > 1 hour or any custom kernel, check whether the best-case removable wall-time share beats the current adopted route by `>= 10%`. If not, stop and document the arithmetic.

---

## Step 2: Non-Negotiable Discipline

1. **Freeze your correctness lane BEFORE running (`references/methodology.md`):**
   - **Class 1 (bit-identical):** token IDs `sha256` + full layer-norm `sha256`; any divergence = REJECT.
   - **Class 2 (different libm/reassociation/tile order):** FP64/FP32 kernel oracle + zero answer-carrying-span flips (including special tokens) + deterministic replay.
   - **Class 3 (approximate/quantization):** predeclared exact-answer + task quality objective.
2. **One variable per arm, measured on the engine of record.** Never combine engine, quant, patch, and drafter without isolating attribution, and never benchmark a speedup only against a slow fallback arm.
3. **Capacity and correctness before speed.**
   - VRAM admission tiers: `< 500 MiB` hard abort; `500–1023 MiB` isolated observation only; `1024–2047 MiB` staged C1; `>= 2048 MiB` preferred C1 performance; `>= 3072 MiB` concurrency.
   - Always check exact-answer canaries (e.g. the `"14" not "30"` in-head arithmetic test that caught `DNR-009` on symmetric W4A16 27B!).
4. **Separate realistic card sampling from greedy diagnostics.**
   - Headline speed tables use the model card's sampling (`temperature`, `top_p`, default `reasoning_effort`, `n >= 3` seeds) across the full reachable context window (`0`, `8K`, `32K`, `64K`, `128K`, `max`).
   - Count tokens with tokenizer/server `usage`, NEVER SSE chunks (`DNR-015`).
5. **Never fake cold prefill or misread display-card thermals.** Use unique entropy-first prefixes (`>= 64` chars) + `prefix_cache_hits_total == 0.0`. Gate non-display B70 `<= 55 C`, display B70 against its own idle (`<= 68 C`, `DNR-024`), and never profile 2-card B70 with `unitrace --device-timing` (`DNR-028`).
6. **Never force-kill a server mid-prefill.** Drain/abort requests, send `SIGTERM`, wait for exit, and verify baseline VRAM and clean `dmesg`.
7. **Gate custom kernels behind an opt-in switch** (default `0` = exact stock) and log every closed arm with a concrete **Revisit when** condition.

---

## Step 3: Proven Intel XPU Lever Order (Highest ROI First)

See `references/levers.md` for measured B70 deltas and citations:

1. **Preserve native checkpoint quantization.** If a model ships native NVFP4/MXFP8 (e.g. Inkling 2 Mini), run it directly in vLLM XPU before converting to GPTQ/INT4/GGUF (`DNR-009`, `DNR-041`).
2. **Eliminate hidden FP32 weight casts & host syncs.** Audit `lm_head` and norm/router ops first — vLLM PR `#69186` (`torch.mm(..., out_dtype=torch.float32)`) delivered **+39–42% decode, -30% J/tok, and +1.22 GiB KV** in 2 lines.
3. **Enable decode graph replay.** Always use `-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY` on vLLM XPU (never `--enforce-eager` for speed claims) and `GGML_SYCL_ENABLE_GRAPH=1` on llama.cpp SYCL. Persist `TRITON_CACHE_DIR`.
4. **Tune `--max-num-batched-tokens` for SWA context capacity.** On SWA-heavy models in vLLM, `mbt=4096/8192` collapses max context, while `mbt=1024` or `512` can unlock **2x–3x larger context** (Inkling: `106K -> 163K` DSpark, `200K -> 256K / 308.7K KV` no-spec). Report capacity profiles separately from `mbt=2048` speed ladders.
5. **Speculative decoding (`DSpark`, `MTP`, `DFlash`).** Measure per-step verify cost and accepted length `a` separately (`tok/s = a / step_ms`), check greedy parity (`drafted == serial` vs altered hashes), and sweep draft length `k` per depth and concurrency.
6. **Engine-specific quick wins:**
   - **llama.cpp SYCL:** build with `GGML_SYCL_F16=ON` (`3.3x` prefill), keep `GGML_SYCL_ENABLE_OPT=1`, set `SYCL_CACHE_PERSISTENT=0` on DPC++ 2026.1, use `q8_0/q4_0` KV on >= 2026-10-29 builds (`1.8–2.1x` at 64K).
   - **Dual-B70:** default to two independent single-card servers for multi-user throughput; use peer-store TP2 without remote atomics (not stock oneCCL TP2) when optimizing single-stream latency.
7. **Custom AOT `bmg_g31` CUTLASS Xe2 kernels (after config levers plateau).**
   - Compile with `icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg_g31 -options -ze-opt-large-register-file"` against pinned `intel/sycl-tla` + `vllm-xpu-kernels` headers.
   - Fuse SLM route-sort + grouped GEMM + activation (e.g. softcapped SwiGLU) for `M=8`–`2048` verify/prefill batches; verify numerical parity vs exact FP32 reference (`rel_l2 <= 5e-5`) across normal and subnormal scales before end-to-end runs.

---

## Step 4: Qualification & Publication Gate

Before promoting any result (`references/publishing.md`):
1. **Deterministic quality & long-context retrieval suite:** math, structured JSON, state-trace code, multi-turn correction, and multi-depth needle retrieval (e.g. `1.4K`, `8.5K`, `33.8K`, `135K`).
2. **Finished-answer check:** if default high reasoning effort saturates `max_tokens`, verify non-empty final answers at lower effort (e.g. `effort=0.2`) or higher budget.
3. **Full hardware cleanliness:** `dmesg` clean of GPU faults/resets, VRAM restored to baseline, power caps unchanged.
4. **Sanitized, commit-pinned public evidence:** zero private host paths, PCI IDs, credentials, or multi-MB binaries in git. Separate `kind: "benchmark"` records from `kind: "capability"` records in `data/benchmarks.v1.json`, run `python3 scripts/check-repo.py` (all 12 gates PASS), and merge with `--merge` so pinned evidence commits remain in history.
