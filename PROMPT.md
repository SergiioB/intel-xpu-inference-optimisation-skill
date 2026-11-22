# Intel XPU Inference Optimisation — Standalone Agent Prompt

Use this prompt with any coding agent or LLM when optimising a local LLM server on an Intel Arc Pro B60/B70 (Battlemage Xe2, `bmg_g31`) or Intel XPU host.

---

You are optimising local LLM inference on Intel Arc / Battlemage Xe2 GPUs (`vLLM XPU`, `llama.cpp SYCL`, `TensorFold XPU`, `exl3xpu`, `Strata`, or `OpenVINO`). Follow this end-to-end research & optimization workflow strictly:

1. **Phase 0 — Upstream reconnaissance, tensor census & closed-arm check (zero GPU time):** Inspect `config.json`, raw safetensors/GGUF tensor headers (`DNR-014`: exact per-layer dtypes, MoE vs dense, SWA ratio, `model_max_length`, K/V pre-storage clipping/norms, vision tower, drafter), and upstream PRs/plugins before converting weights or writing code. Read `docs/DO-NOT-REPEAT.md` (`DNR-001`..`DNR-042`) and `docs/WHAT-WORKED.md`.
2. **Preflight before touching GPUs:** Run `python3 skills/intel-xpu-inference-optimisation/scripts/b70_preflight.py`. Prefer a free non-display B70 (`is_display == False`); on display cards budget `desktop_vram_mib + 1 GiB` (`1.5–4.6 GiB`). Verify free VRAM (`vram_mm`), as-found power caps (never alter caps to fake results), package temp (`<= 55 C` non-display / `<= 68 C` display), `MemAvailable`, disk reserves, and zero competing workloads.
3. **Compute the Xe2 roofline first:** Run `python3 skills/intel-xpu-inference-optimisation/scripts/b70_ceiling.py` (`~521–528 GB/s` sustained BW, `~50%` of `157 TFLOPS` BF16 XMX, **no native FP4/FP8 DPAS**). Only invest > 1 hour or write custom kernels when the removable wall-time share can beat the adopted route by `>= 10%`.
4. **Baseline + hypothesis-driven decomposition in ROI order (never guess!):**
   - Get ONE clean baseline running with graph replay ON (`FULL_DECODE_ONLY` / `GGML_SYCL_ENABLE_GRAPH=1`), then decompose where the step ms live using 5-second isolated op microbenchmarks.
   - Preserve native NVFP4/MXFP8 checkpoints before converting to GPTQ/GGUF.
   - Eliminate hidden FP32 weight casts (`lm_head`, e.g. vLLM PR `#69186` `+42%`).
   - Tune `--max-num-batched-tokens` on SWA models (`2048` speed ladder; `1024`/`512` capacity; never `4096`/`8192`) and power caps (`150 W` MoE vs `180–230 W` dense).
   - Unlock missing plugin features (pure-PyTorch XPU ViT via `F.scaled_dot_product_attention`, FP8 KV via ONE fixed-size `256 MiB` reusable BF16 scratch + 64K Chunked-LSE merge — `11x` faster than 1-byte Triton tensor descriptors and avoids PyTorch XPU caching-allocator fragmentation!).
   - Evaluate speculation (`DSpark`, `MTP`, `DFlash2`) with separate `step_ms`, acceptance length `a`, and greedy parity checks.
   - Build opt-in AOT `bmg_g31` CUTLASS Xe2 kernels (`sycl-tla` + `icpx -ze-opt-large-register-file`, default env `0` = stock) only after config levers plateau, verified against FP32 oracle (`rel_l2 <= 5e-5`). Always break `uv` hardlinks before patching site-packages.
5. **Measure honestly & qualify the whole context window:**
   - One variable per arm with same-session controls. Headline speeds use model-card sampling (`n >= 3` seeds, full reachable context ladder, unique entropy-first cold prefixes with `prefix_cache_hits_total == 0.0`); keep greedy as an exactness diagnostic only.
   - Count both `delta.reasoning` and `delta.content`; use `max_tokens >= 384` on thinking models for retrieval tests.
   - For quantized/FP8 KV, prove held-out natural-prose perplexity vs BF16 across `4K -> max_ctx` + `15%` oldest / `55%` mid dual-needle exact recall (and live multimodal OCR/counting if vision is enabled).
   - Record hardware energy (`energy1_input`), live min VRAM, clean `dmesg`, and clean `SIGTERM` teardown (kill by exact PID via `awk`, never `pkill -f` matching your own shell!).
6. **Package & ship cleanly:** Sanitize private paths/PCI IDs, pin evidence commits in `data/benchmarks.v1.json` (`benchmark` vs `capability`), pass 12/12 `scripts/check-repo.py` gates, merge with `--merge`, visually inspect any 2x PNG charts, and fold new lessons back into `SESSIONS.md` and this skill.
