# Intel Arc Pro B60/B70 & XPU Gotchas & Closed Arms (`DNR-001`..`DNR-042`)

Hard-won traps across `vLLM XPU`, `TensorFold XPU`, `exl3xpu`, `llama.cpp SYCL`, `Strata/XeStrata`, Linux `xe` driver, and Intel `compute-runtime`. Read before debugging a crash, OOM, quality failure, or closed optimisation idea.

## 1. Quality & Quantization Traps (Check Before Speed!)

- **Symmetric W4A16 (GPTQ / AutoRound) quality failure on dense 27B (`DNR-009`):** On Qwen3.8-27B, every symmetric INT4 W4A16 export answered `"30"` instead of `"14"` on an in-head arithmetic canary while passing coherence, replay, and perplexity! Always run an exact-answer battery first.
- **Double-quantizing native NVFP4/MXFP8 checkpoints (`DNR-035`, `DNR-040`, `DNR-041`):** Re-encoding 4-bit NVFP4 experts into GPTQ INT4, EXL3, Q8_0, or BF16 GGUF adds error or bloats bytes with zero accuracy gain (no higher-precision expert source exists).
- **Never trust filename quant labels (`DNR-014`):** Unsloth UD GGUFs mix per-tensor quant types (e.g. `IQ1_S` inside an `IQ3_XXS` file). Always read the safetensors/GGUF tensor-type census and fail closed on unknown types.
- **Raw `/v1/completions` prompt-format loops (`DNR-016`):** Nemotron-3.5-Lightning looped `14/18` via raw `/v1/completions` vs `0/18` through `/v1/chat/completions` with its shipped chat template. Check prompt formatting before blaming weights.
- **Speculative decoding altering greedy output (`DNR-017`, `DNR-034`):** `exl3xpu` MTP3 altered 27% of greedy outputs on Qwen3.8-27B (and dropped working JS pages from `9/9` to `5–6/9`); Inkling DSpark alters greedy hashes vs no-spec. Never claim `drafted == serial` without a direct hash check.
- **Tiny reassociation / fused-kernel KL drift (`DNR-039`, `RL-222`):** Inkling llama.cpp patch `0002` boosted prefill `2.4x` but had mean KL `0.00112` (failing `<= 1e-4`) and diverged greedy on `4/5` prompts. Never promote an unverified fast kernel.

## 2. vLLM XPU & Level Zero Runtime Traps

- **Linux 107-byte Unix socket limit (`VLLM_RPC_BASE_PATH`):** Deep working dirs make ZeroMQ IPC paths exceed 107 bytes and crash before load. Always set `VLLM_RPC_BASE_PATH` to a short `/tmp/...` directory.
- **`uv` hardlink cache corruption:** Break hardlinks (`cp f f.orig && rm f && cp f.orig f`) before patching any file inside a `uv`-created venv.
- **Piecewise graph capture fails on Intel XPU:** Use `-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY`. Piecewise capture or dynamic draft lengths fail with `UR_RESULT_ERROR_OUT_OF_RESOURCES`.
- **vLLM XPU `v0.31.0` AutoRound AVX-512 SIGILL (`DNR-027`):** The shipped AutoRound XPU library uses AVX-512 host instructions and crashes with `trap invalid opcode` on non-AVX-512 CPUs (e.g. AMD Zen 3 Ryzen 7 5700X3D). Also mount `/dev/dri/by-path` inside Docker containers or oneCCL fails `onecclCommInitRank` with code 11.
- **Prefix caching corruption on vLLM XPU W4A16 / speculative stacks:** Keep `--no-enable-prefix-caching` until multi-turn corruption checks pass (corruption was reproduced even on zero-hit cold requests).
- **Streaming reasoning field (`delta.reasoning` vs `delta.reasoning_content`):** On vLLM >= 0.34 with `--reasoning-parser`, thinking tokens stream in `delta.reasoning`. Capture both `delta.reasoning` and `delta.content`.
- **SWA `--max-num-batched-tokens` collapse:** On SWA-heavy models (40 SWA visits in Inkling), `mbt=4096/8192` collapses max context to `~7K`. Use `mbt=2048` (speed ladder) or `mbt=1024/512` (max context).
- **Missing PR `#69186` FP32 `lm_head` fast path (`DNR-042`):** Costs `-30%` decode speed and `-1.22 GiB` KV cache per step.

## 3. SYCL, CUTLASS Xe2, TensorFold & llama.cpp Traps

- **Missing `-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED`:** Causes `sycl-tla` headers to fail looking for `cuda_runtime_api.h`.
- **Missing `-ze-opt-large-register-file`:** Causes Xe2 CUTLASS tiles to spill registers to scratch memory (`2x–3x` slowdown).
- **E4M3 subnormal scales in NVFP4:** Test both normal and subnormal scales (`exp == 0`) against exact FP32 reference (`rel_l2 <= 5e-5`).
- **`SYCL_CACHE_PERSISTENT=1` on DPC++ 2026.1 (`DNR-031`):** `SIGSEGV` in `PersistentDeviceCodeCache::getItemFromDisc` on `llama.cpp SYCL`. Always set `SYCL_CACHE_PERSISTENT=0`.
- **`GGML_SYCL_ENABLE_OPT=0` (`DNR-037`):** Disables reordered weight layout and drops decode `-29%`.
- **`q5_0` K / `q4_1` V KV on post-2026-10-29 `llama.cpp` (`DNR-019`):** Bypasses `#31125` DPAS flash-attn (`10.5` vs `22.3–23.9 tok/s` at 64K); default to `q8_0/q8_0`.
- **Additive attention bias disables llama.cpp flash-attention (`DNR-036`):** Models with `kq_b != nullptr` (Inkling relative-position bias) cannot use `-fa on` in current llama.cpp.
- **Disabling Intel NEO direct submission (`DNR-020`):** `EnableDirectSubmission=0` turns every launch into a kernel ioctl (`-6%` 1-card, `-23%` 2-card).
- **SLM K/V staging in wide 256-lane WG (`DNR-018`):** Spilled registers and slowed TensorFold tree verify **5x** (`46.6 -> 9.0 tok/s` at 32K).
- **Blindly substituting wide GEMM kernels (`DNR-030`):** `XPU_GU_WIDE64=1` looked faster in isolation but *lost* `-1.1%..-1.9%` end-to-end on TensorFold `--tp 2`. Always validate on the full round.
- **Cross-queue peer `memcpy` events (`DNR-032`):** `GGML_SYCL_TP_P2P=1` leaked 224K events and segfaulted (`exit 139`); use mode 2 (peer stores + flag).

## 4. Hardware, Profiling, Multi-GPU & Operations Traps

- **Never profile 2-card B70 with `unitrace --device-timing` or unpatched `XPU_PROFILE=1` (`DNR-022`, `DNR-028`):** On `compute-runtime 26.05.37020.3`, cross-device event waits cause hangs, segfaults, and `xe` "Engine memory CAT error" resets on both cards! Use `--host-timing` or per-rank profilers (`EnableImplicitConvertionToCounterBasedEvents=0` or runtime `>= 26.40`).
- **Never use one 55 C cooldown gate when one B70 drives the desktop (`DNR-024`):** Display B70 idles at `~65–66 C`. Gate each card against its own idle baseline (e.g. non-display `<= 55 C`, display `<= 68 C`) and declare the sustained thermal regime.
- **Never identify a GPU by a remembered `hwmon` index:** Always resolve `hwmon` and `vram_mm` from PCI BDF (`/sys/bus/pci/devices/0000:XX:00.0/...`) and pin `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`.
- **Dynamic per-chunk scratch allocations fragment PyTorch XPU caching allocator:** Calling `torch.empty((n_active, ...))` across chunked-prefill steps caches dozens of distinct shapes simultaneously and hits `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` at 64K even when each buffer is `< 256 MiB`. Always slice a single pre-allocated fixed-size scratch buffer (`buf[:n_active]`).
- **`icpx 2026.1` `joint_matrix` `get_wi_data` accumulator spill (`3x` slowdown):** Indexing `sycl::ext::oneapi::detail::get_wi_data(sg, o_mat)` inside an inner K-loop to apply per-row softmax scaling forces `icpx` to spill the `joint_matrix` accumulator to private memory (`23.7 ms -> 70.6 ms` @ 128K). Use ESIMD for custom `M>=2` XMX attention, and use zero-spill SIMT for `M=1` (`4.76x` faster than Triton `tl.dot` @ 256K FP8, `+71%` end-to-end!).
- **8-row `joint_matrix_load` tail-padding page faults (`EngineClass: 5 ccs` `0x…aa0000`):** XMX `8x16` `joint_matrix_load` tiles read 8 rows unconditionally from `m0 = pbase + sg_m * 8`, reading up to 7 rows past `pbase + ntok` when `ntok % 8 != 0`. Allocating dynamic grouped-MoE activation buffers (`pf_zg`/`pf_hb`) to *exact* `P` rows triggers intermittent `ccs` page faults whenever `P * K * 2` lands within 21 KiB of a GPU page boundary. Always reserve `>= +64` mapped tail rows (`alloc_p = max(P + 128, P + P/4 + 128)`, `cap_p = alloc_p - 64`)!
- **Never call `_exit(0)` with un-drained Level Zero / SYCL copy or peer queues:** Abrupt `_exit(0)` yanks DRM FDs while `bcs` copy/peer queues are active, firing `Fault response: Unsuccessful -EINVAL` + `Engine reset: engine_class=bcs` on both B70 cards and hanging ~50% of subsequent launches. Drain `Q()`, copy queues, and peer queues (`release()` + `disable_peer_access`) before `_exit(0)` -> `0` GPU faults and `100%` clean consecutive launches!
- **Never trust isolated Split-KV microbenchmarks under XPU graph replay (`DNR-030`):** Increasing Triton decode `num_splits` from `40/9` to `80/16` or `128/16` looked `1.4x–3.3x` faster in isolated 1-layer microbenchmarks, yet regressed live graph-replayed Inkling FP8+Vision+DSpark decode by **`-10%` to `-12%`** (`73.4 -> 66.0` @ 64K, `61.0 -> 53.7` @ 128K) due to combine-kernel sequential reduction & L2 pressure across 48 visits! Keep `40/9` on Triton decode; only native SYCL fused-reduction decode kernels benefit from `128` partitions.
- **Price `M=1..8` MoE verify cost before building Adaptive DSpark:** On CUTLASS Xe2 NVFP4 (`36` visits), `M=4` vs `M=8` only differs by **`2.58 ms`** (`8.73` vs `11.31 ms` out of `~38 ms` = `<6%` ceiling), failing the 10% Impact Gate — whereas engines with slower wide-verify MoE kernels see bigger Adaptive DSpark gains.
- **Triton XPU 1-byte FP8 `tl.make_tensor_descriptor` 5.4x–11x slowdown:** Direct `float8_e4m3fn` or `uint8` 2D block loads into `tl.dot` break Xe2's 16-bit DPAS layout (`256.6 ms` vs `46.8 ms` BF16 at 64K; `~510 ms` at 128K). Cast active pages into ONE reusable 256 MiB BF16 scratch + Chunked-LSE merge across 64K key windows (`46.4 ms` at 128K = `11x` faster!).
- **`echo=True` prompt-logprob scoring at 256K–512K:** `/v1/completions` with `echo=True, logprobs=1` returns the full prompt text in JSON and runs `lm_head` over all prompt tokens — budget ~3–9 min/point at `245K`–`496K` and never print raw `choices[0].text` to stdout.
- **Reasoning-model `max_tokens=128` empty retrieval trap:** Even at `reasoning_effort=0.2`, a thinking model can spend 128 tokens inside `<thinking>` on a long-context retrieval prompt and return `content=""`. Always use `max_tokens >= 384` for retrieval tests.
- **Never `pkill -f <path>` from a shell whose own command line contains `<path>`:** `pkill -f` kills its own parent shell and aborts the command! Always filter PIDs explicitly with `ps -eo pid,args | awk '/pattern/ && !/awk/ {print $1}' | xargs -r kill -TERM`.
- **Display-driving B70 desktop VRAM overhead (`1.5–4.6 GiB`):** A B70 driving a KDE/GNOME desktop routinely loses `1.5–4.6 GiB` of VRAM to desktop clients (`drm-resident-vram0` deduped per `drm-client-id`) and idles at `~65 C` (`DNR-024`). Detect connected displays via `/sys/class/drm/<card>-*/status == "connected"`, prefer a dedicated non-display card by default, and cap `gpu_memory_utilization <= 1 - (desktop_vram + 1 GiB) / total_vram` + step default `ctx` down to `ctx_safe` when serving on a display card!
- **AIB B70 subsystem IDs:** Board-partner B70s (ASRock `1849:6025`, Sparkle, Gunnir) share PCI ID `8086:e223` but NOT Intel's reference subsystem `8086:1701`. Match on `8086:e223`.
- **Forced kills mid-prefill cause `xe` CCS resets:** Drain requests, send `SIGTERM`, wait for exit, verify baseline VRAM and clean `dmesg`. Check `dmesg` for PCIe AER drops before blaming code.
- **`systemd-run` traps (`DNR-025`, `DNR-026`):** Pass `--expand-environment=no` (otherwise `$$` expands to `$`), use `runuser -u <user> --` to preserve `render`/`video`/`docker` groups, and verify binaries exist and are newer than sources before queuing runs.
- **Closed multi-card shortcuts (`DNR-004`, `DNR-005`, `DNR-006`, `DNR-010`, `DNR-011`, `DNR-012`):** Stock vLLM oneCCL TP2 (`-45..-71%`), llama.cpp RPC (`54%`), Cascadia PP2 (`0%`), host-staged second-card MoE caches (`33.4 vs 48.0 tok/s`), and storage-tier NVMe streaming at 5 GB/s are all closed.
