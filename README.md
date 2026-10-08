# fukuro-llama-cpp-turboquant

![Primary: self-hosted](https://img.shields.io/badge/primary-self--hosted-blue)
[![Mirror: GitHub](https://img.shields.io/badge/mirror-github.com-lightgrey)](https://github.com/fukuro-kun/fukuro-llama-cpp-turboquant)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

> **Fork-Lineage:** `ggml-org/llama.cpp` → `TheTom/llama-cpp-turboquant` (TurboQuant KV/Weight compression) → `AtomicBot-ai/atomic-llama-cpp-turboquant` (Gemma 4 MTP, Qwen NextN, UDT quantisation) → **fukuro** (self-hosted primary remote; GitHub mirror, Codeberg legacy).
> Details, Remotes und Feature-Vergleich: **[FORKS.md](FORKS.md)**

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Spekulative Decodierung via MTP-/NextN-Heads: `--spec-type draft-mtp` laedt einen Multi-Token-Prediction-Assistant — eingebettet im Target-GGUF (NextN-Layer) oder als separate `-md`/`--spec-draft-model`-GGUF. Durchsatzgewinne sind modell-, quant- und backend-spezifisch; fruehere Messungen an Gemma 4 und Qwen 3.6 bleiben unten und in [MTP.md](MTP.md)/[NEXTN.md](NEXTN.md) als historische Belege erhalten. Pro-Modell-Validierung (Sidecar-Shape + Tokenizer) erforderlich. Aktuelle Flag-/Architektur-Referenz: [docs/speculative.md](docs/speculative.md).**
- **Qwen 3.6 kombinierte `_MTP.gguf`-Quants:** fuer NextN-faehige Targets liegen die empfohlenen vorgefertigten Quants in der **[AtomicChat — Qwen 3.6 UDT](https://huggingface.co/collections/AtomicChat/qwen-36-udt-atomicchat-6a0481f5cc5a057c07759176)**-Collection ([27B](https://huggingface.co/AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF) · [35B-A3B](https://huggingface.co/AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF)) — gebaut mit dem oeffentlichen Unsloth-MTP-Imatrix plus Fork-Masken, die NextN-/MTP-Tensoren auf `Q8_0` pinnen (Draft-Akzeptanz erhalten) und Attention-Q/K auf `Q6_K` anheben (passt zu TurboQuant3-KV); Upstream-Quellen funktionieren ebenfalls: [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) / [`unsloth/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF).
- **TurboQuant KV cache & weights: WHT-rotated low-bit quantization with backend-native kernels (Metal `TurboFlash`, CUDA, Vulkan, HIP). Use `-ctk turbo3 -ctv turbo3` for ~4.3× KV compression, or quantize weights to `TQ4_1S`/`TQ3_1S`. See [Compression below](#turboquant--kv-cache--weight-compression).**
- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- **Dieser Fork — Multimodal x spekulativ:** eine fruehere Implementierung erlaubte `mtp`/`nextn`/`eagle3` neben `--mmproj` (Text-Turns mit Draft, Bild-Turns fielen auf Target-only zurueck). Diese Whitelist existiert im aktuellen Code nicht mehr — die Kombination ist ein **offenes Gate** ohne Kompatibilitaets- oder Speedup-Zusicherung bis zur Neuverifikation. Siehe [docs/speculative.md](./docs/speculative.md).
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)
- **This fork — Vulkan & CUDA stability work:** `Q3_K`/`Q6_K` block-load shaders (+57 % tg128 / +78 % tg128 on Intel BMG), iq1 shared-memory reduction for `mul_mm`, host-memory `shared_mutex` optimisation for multi-stream, and CUDA KV-Cache pre-reservation for FlashAttention (reduces OOM risk). Siehe Cherry-Pick-Tabelle in [FORKS.md §5.4](FORKS.md) und `docs/fork/` fuer die Benchmark-Zahlen.
- **Dieser Fork — Vulkan-APU-GPU-Hang-Fix:** `nodes_per_submit=10` fuer UMA-Geraete verhindert `amdgpu.lockup_timeout`-GPU-Ring-Resets auf AMD-APUs. Historischer Befund vom Juni 2026 (**nicht** erneut verifiziert): ohne den Fix hingen AMD-APUs (RADV PHOENIX, Renoir u.a.) bei Kontexten >=186k und Prompts >=16384; der damalige Verweis auf das offene Upstream-Issue [#21724](https://github.com/ggml-org/llama.cpp/issues/21724) und die Aussage, kein anderer Fork/Upstream habe den Fix, wurden seitdem nicht erneut geprueft. Die 262k/~22-t/s-Zahlen sind eine **historische, modellspezifische Messung** (AMD 760M, Juni 2026), kein universelles Versprechen. Der Fix erkennt UMA-Geraete ueber `ctx->device->uma`, keine Nutzerkonfiguration noetig. Siehe [docs/fork/archive/rca/2026-06-21_VULKAN_GPU_HANG_ROOT_CAUSE.md](docs/fork/archive/rca/2026-06-21_VULKAN_GPU_HANG_ROOT_CAUSE.md).
- **This fork — DiffusionGemma (experimental):** block text-diffusion MoE on Gemma-4-backbone with entropy-bound decoder. Forward pass and decoding loop functional; Self-Conditioning blocked (SC tensors missing in GGUFs). Siehe Feature-Matrix in [FORKS.md §6](FORKS.md).
- **Dieser Fork — DOX-Framework:** registrierte `AGENTS.md`-Vertragshierarchie (Root + Domaenen-Children unter `src/`, `ggml/`, `common/`, `tools/`, `tests/`, `scripts/`, `docs/`) fuer reproduzierbare Agenten-Arbeit — Verzeichnisse ohne eigenen Vertrag bleiben parent-gesteuert. Siehe [AGENTS.md](AGENTS.md) und [docs/DOX_AUDIT.md](docs/DOX_AUDIT.md).

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

## Credits & Lineage

This fork stands on the shoulders of several projects and people:

- **[@ggerganov](https://github.com/ggerganov) and the ggml team** — for the original [llama.cpp](https://github.com/ggml-org/llama.cpp) inference engine and the [ggml](https://github.com/ggml-org/ggml) tensor library that powers everything.
- **[@TheTom](https://github.com/TheTom)** — for the original [TurboQuant](https://github.com/TheTom/llama-cpp-turboquant) design: WHT-rotated KV/weight quantization, the reference kernels, and the relentless backend ports (Metal TurboFlash, CUDA, Vulkan, HIP). None of the compression work in this fork would exist without that foundation.
- **[@AtomicBot-ai](https://github.com/AtomicBot-ai)** — for Gemma 4 MTP speculative decoding, Qwen 3.x NextN shared-model drafting, the UDT quantisation masks, and the extensive documentation (MTP.md, NEXTN.md, speculative-decoding matrix benches).

## Spekulative Decodierung (MTP / NextN) — aktueller Stand

Dieser Fork enthaelt **Multi-Token-Prediction**-Spekulation ueber den generischen
Draft-Pfad: `--spec-type draft-mtp` mit einem Assistant-/Draft-Modell ueber
`-md`/`--spec-draft-model`, oder eingebetteten MTP-/NextN-Layern im Target-GGUF
(der Server baut dann einen `LLAMA_CONTEXT_TYPE_MTP`-Kontext auf demselben
geladenen Modell — siehe `tools/server/server-context.cpp`).

> **Namens-Hinweis.** Die in historischen Dokumenten gezeigten alten
> Flags/Typen — `--spec-type mtp`, `--spec-type nextn`, `--mtp-head`,
> `--draft-block-size`, `--draft-max`/`--draft-min` und APIs wie
> `llama_decode_mtp_async` / `llama_set_nextn` — existieren im aktuellen Code
> **nicht mehr**. Zu verwenden sind `--spec-type draft-mtp`, `-md`,
> `--spec-draft-n-max`/`--spec-draft-n-min`.
> Referenz: **[docs/speculative.md](docs/speculative.md)**.

**Readiness-Gate.** Spekulative Gewinne sind pro Modell, Quant und Backend
unterschiedlich — es gibt keinen universellen Speedup. Vor Verlass auf ein
Paar die Kompatibilitaets-Pruefungen (Sidecar-Tensor-Shape + Tokenizer) und
einen kontrollierten Target-vs-Draft-Benchmark mit identischen
Prompts/Tokenbudgets durchfuehren. Die historischen Matrix-Ergebnisse unten
(Gemma 4, Qwen 3.6) sind datierte Belege, keine aktuellen Zusicherungen.

### Gemma 4 MTP (historische Implementierungs-Zusammenfassung)

Der urspruengliche Gemma-4-MTP-Pfad paarte `gemma4`-Targets mit dem offiziellen
**`gemma4_assistant`**-Draft-Head. Zentrale Eigenschaften dieser Implementierung
(inzwischen durch den generischen `draft-mtp`-Pfad abgeloest):

- **+30-50 % Short-Prompt-Durchsatz** auf Gemma 4 26B-A4B / 31B im
  Matrix-Bench (`f16` KV); Akzeptanzrate ~85-88 % auf Dense-Targets
  (historisch, M4 Max / Metal, Mai 2026).
- **Async-Pipeline (Depth-2)**, die Draft-Compute mit der
  Post-Accept-Buchfuehrung ueberlappte (APIs inzwischen entfernt; siehe
  `MTP.md`, historisch).
- **Centroid LM head** for Edge variants (E2B / E4B); dense tied head for
  26B-A4B / 31B.

### Pre-built assistant GGUFs

Recommended quantization is **`Q4_K_M`** (throughput is identical to F16 on
this assistant size — bandwidth, not weight precision, dominates — while
footprint is ~4× lower). Also published: `Q4_K_S`, `Q5_K_M`, `Q8_0`, `F16`.

> [AtomicChat / Gemma 4 Assistant GGUF collection](https://huggingface.co/collections/AtomicChat/gemma-4-assistant-gguf)

| Target model | Assistant (MTP head) GGUF |
|---|---|
| Gemma 4 E2B | [`AtomicChat/gemma-4-E2B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-E2B-it-assistant-GGUF) |
| Gemma 4 E4B | [`AtomicChat/gemma-4-E4B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) |
| Gemma 4 26B-A4B | [`AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF) |
| Gemma 4 12B | [google/gemma-4-12B-it-assistant](https://huggingface.co/google/gemma-4-12B-it-assistant) (self-convert via this fork) |
| Gemma 4 31B | [`AtomicChat/gemma-4-31B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-31B-it-assistant-GGUF) |

### Quick start

```bash
# Manuelle Ausfuehrung — aktuelle Flags (fuer jedes validierte Target/Assistant-Paar).
llama-server \
  -m /path/to/gemma-4-target.gguf \
  -md /path/to/gemma-4-assistant-Q4_K_M.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 --spec-draft-n-min 0 \
  -c 16384 \
  -ngl 99 -ngld 99 \
  -fa on \
  --host 127.0.0.1 --port 8080
```

> **Fork-Hinweis (Legacy):** die Repo-Helper-Skripte stammen aus der alten
> `--draft`/MTP-Generation und sind **nicht** gegen die aktuellen Flags
> geprueft — sie koennen entfernte Flags uebergeben. Bis zur Reparatur
> gelten sie als unverifiziert; Status und Regeln siehe
> [`scripts/AGENTS.md`](scripts/AGENTS.md) ("Legacy-Helper"), aktuelle
> Flags in [`docs/speculative.md`](docs/speculative.md).

```bash
# Dense-Targets.
scripts/run-gemma4-mtp-server.sh         # 26B-A4B
scripts/run-gemma4-31b-mtp-server.sh     # 31B

# Edge-/Centroid-Head-Targets — MTP_PRESET=throughput|lift|balanced|quality.
MTP_PRESET=throughput scripts/run-gemma4-e4b-mtp-server.sh
MTP_PRESET=throughput scripts/run-gemma4-e2b-mtp-server.sh
```

### Bench-Snapshot (HISTORISCH — MacBook Pro M4 Max, 40-Core-GPU, 48 GB, Metal, Single-Slot, Mai 2026)

Median-tps ueber 3 Laeufe mit Q4_K_M-Assistant-Heads auf dem alten `mtp`-Pfad
(`--draft-block-size 3`; E4B `MTP_PRESET=throughput`, `B = 2`). Datierte,
modellspezifische Belege — keine aktuelle Zusicherung. Siehe
`.scratch/bench-logs/gemma-matrix-fullrun-20260512-224705.md`.

| model | mode | n=128 tps | n=512 tps | accept@128 | accept@512 |
|---|---|---:|---:|---:|---:|
| gemma-E4B | f16-base    | 90.3  | 89.0  | — | — |
| gemma-E4B | f16-mtp     | **94.3** | 86.0  | 80.0 % | 64.5 % |
| gemma-E4B | turbo3-mtp  | **67.8** | **64.5** | 82.6 % | 72.3 % |
| gemma-26B | f16-base    | 83.6  | 82.7  | — | — |
| gemma-26B | **f16-mtp** | **110.8** | 75.7 | 84.0 % | 67.9 % |
| gemma-26B | turbo3-mtp  | **80.5**  | **69.2** | 84.9 % | 66.1 % |
| gemma-31B | f16-base    | 19.4  | 17.5  | — | — |
| gemma-31B | **f16-mtp** | **21.2** | **18.5** | 88.0 % | 74.4 % |
| gemma-31B | turbo3-mtp  | **19.4**  | **16.3** | 88.0 % | 70.7 % |

### Schalter (aktuell)

- `--spec-draft-n-max` / `--spec-draft-n-min` — Draft-Token-Grenzen pro Runde.
- `-md` / `--spec-draft-model` — Pfad zur Assistant-/Draft-GGUF.
- `-ngld`, `-ctkd`, `-ctvd` — GPU-Layer und KV-Cache-Typen des Drafts.

Historische Schalter (`--draft-block-size`, `--mtp-head`,
`LLAMA_MTP_SKIP_STREAK_THRESHOLD`, `LLAMA_PIPELINE_DEPTH2`,
`LLAMA_MTP_ACC_TRACE`) sind in **[MTP.md](MTP.md)** (HISTORISCH)
dokumentiert und existieren im aktuellen Code nicht mehr. Nutzer-Flags:
[docs/speculative.md](docs/speculative.md).

## Qwen 3.6 NextN — Shared-Model-Draft (historische Zusammenfassung)

Qwen-3.6-Targets (`qwen35` Dense und `qwen35moe` MoE) enthalten die
NextN-/MTP-Hilfskopf-Gewichte **in der kombinierten `*_MTP.gguf`**. Auf dem
aktuellen `draft-mtp`-Pfad fordert der Server ohne explizites `-md`
`llama_model_n_layer_nextn(model) > 0` und baut einen zweiten
`LLAMA_CONTEXT_TYPE_MTP`-Kontext auf demselben geladenen Modell.

Eigenschaften der urspruenglichen Implementierung (vollstaendige
Beschreibung in `NEXTN.md`, HISTORISCH):

- **+28-36 % Durchsatz** auf Qwen 3.6 35B-A3B MoE und **+5-7 %** auf 27B
  Dense — historische, modellspezifische Messungen (M4 Max / Metal,
  Mai 2026), keine aktuellen Zusicherungen.
- Draft-KV nur fuer den NextN-Layer dimensioniert; Pre-Norm-Hidden-States
  fliessen vom Target ueber
  `llama_set_embeddings_nextn`/`llama_get_embeddings_nextn_ith`
  (`src/llama-ext.h` — fork-internes Staging-API, nicht stabil).
- **Kombinierbar mit TurboQuant3-KV** (`-ctk turbo3 -ctv turbo3`) — auf
  MoE-Targets war die Kombination in der gemessenen Konfiguration der
  empfohlene Default.

### Pre-built model GGUFs

**Recommended:** the AtomicChat **UDT** (UD-Turbo) collection — drop-in combined `_MTP.gguf` quants tuned for this fork. One repo per model, 5 quants each (Q3 / **Q4** / Q5 / Q6 / Q8 `_K_XL`), plus the `mmproj` for vision and the original Unsloth imatrix re-hosted for reproducibility:

| Target | Recommended (AtomicChat UDT) | Upstream baseline (Unsloth) |
|---|---|---|
| Qwen 3.6 35B-A3B (MoE) | [`AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF`](https://huggingface.co/AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF) | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) |
| Qwen 3.6 27B (dense)   | [`AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF`](https://huggingface.co/AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF)         | [`unsloth/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)         |

What makes UDT different from a vanilla `llama-quantize -imatrix` run:

- **MTP-aware imatrix** — calibrated by Unsloth with the NextN head active (we re-host their public [`imatrix_unsloth.gguf_file`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/blob/main/imatrix_unsloth.gguf_file) so you can reproduce or re-mix on top of it).
- **NextN-preserve mask** — every `blk.*.nextn.*` and `mtp.*` tensor pinned to `Q8_0`. Tiny size cost (~10 MiB), keeps draft acceptance high.
- **TurboQuant3-friendly mask** — `attn_q` / `attn_k` bumped to `Q6_K` so the file pairs cleanly with `-ctk turbo3 -ctv turbo3`.
- **Kombinierte `_MTP.gguf`** — Target und NextN-Head in einer Datei. Fuer den aktuellen Shared-Model-Pfad `--spec-type draft-mtp` ohne `-md` verwenden. Explizites `-md` laedt ein separates Draft-Modell, auch bei identischem Dateipfad.
- **Apache-2.0**, full attribution: Qwen team (weights), Unsloth (imatrix + BF16 sources), @TheTom (TurboQuant), AtomicChat (UDT masks + packaging).

Collection: [AtomicChat — Qwen 3.6 UDT](https://huggingface.co/collections/AtomicChat/qwen-36-udt-atomicchat-6a0481f5cc5a057c07759176). Full recipe & runbook: [docs/qwen-udt/RUNBOOK.md](docs/qwen-udt/RUNBOOK.md). Mask files: [`scripts/quantize-masks/qwen36-ud-{base,v1-nextn,v2-turbo3,v3-combined}.txt`](scripts/quantize-masks).

### Quick start

```bash
# Kombinierte _MTP.gguf (eingebetteter NextN-Head): kein -md noetig — der
# Server erkennt die eingebetteten Layer und baut den MTP-Kontext auf demselben Modell.
llama-server \
  -m /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL_MTP.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 --spec-draft-n-min 1 \
  -c 8192 -ngl 99 -ctk turbo3 -ctv turbo3 -fa on \
  --host 127.0.0.1 --port 8080
```

> **Fork-Hinweis (Legacy):** die Repo-Helper-Skripte sind **nicht** gegen
> die aktuellen Flags geprueft und gelten bis zur CLI-Reparatur als
> unverifiziert — siehe [`scripts/AGENTS.md`](scripts/AGENTS.md) und
> [`docs/speculative.md`](docs/speculative.md).

```bash
scripts/run-qwen36-27b-nextn-server.sh        # Qwen 3.6 27B dense
scripts/run-qwen36-35ba3b-nextn-server.sh     # Qwen 3.6 35B-A3B MoE
```

Wird der NextN-Head als separate **`_mtp`-GGUF** ausgeliefert, ist er als
Fallback weiterhin nutzbar — `-md` auf diese Datei zeigen; der Server laedt
sie dann als Draft-Modell.

### Bench-Snapshot (HISTORISCH — MacBook Pro M4 Max, 40-Core-GPU, 48 GB, Metal, Single-Slot, Mai 2026)

Median-tps ueber 3 Laeufe (alter `nextn`-Pfad, `--draft-max 2 --draft-min 1`),
Single-Slot, Shared-Target/Draft-Modell. Datierte Belege — keine aktuelle
Zusicherung. Siehe `.scratch/bench-logs/qwen-matrix-fullrun-20260512-222625.md`.

| model | mode | n=128 tps | n=512 tps | accept@128 | accept@512 |
|---|---|---:|---:|---:|---:|
| qwen-27B dense   | f16-base       | 21.3 | 20.8 | — | — |
| qwen-27B dense   | f16-nextn      | **22.9** | **21.6** | 93.9 % | 85.1 % |
| qwen-27B dense   | turbo3-base    | 19.7 | 18.7 | — | — |
| qwen-27B dense   | turbo3-nextn   | **20.8** | **19.7** | 85.5 % | 78.7 % |
| qwen-35B-A3B MoE | f16-base       | 70.1 | 69.6 | — | — |
| qwen-35B-A3B MoE | **f16-nextn**  | **95.2** | **89.1** | 88.2 % | 78.7 % |
| qwen-35B-A3B MoE | turbo3-base    | 61.8 | 62.0 | — | — |
| qwen-35B-A3B MoE | **turbo3-nextn** | **82.7** | **77.2** | 82.9 % | 80.6 % |

### Schalter (aktuell)

- `--spec-type draft-mtp` — aktiviert den MTP-/NextN-Draft-Pfad.
- `-md` / `--spec-draft-model` — eigenstaendige Draft-GGUF (weglassen, wenn
  das Target-GGUF die NextN-Layer bereits enthaelt).
- `--spec-draft-n-max` / `--spec-draft-n-min` — Draft-Token-Grenzen pro Runde.

Die vollstaendige Architektur (Graph-Dispatch, KV-only-NextN-Dimensionierung,
Hidden-State-Transfer, Performance-Trade-offs und die
27B-Dense-Compute-Bound-Analyse) steht in **[NEXTN.md](NEXTN.md)**
(HISTORISCH); aktuelle CLI-Flags: [docs/speculative.md](docs/speculative.md).

## TurboQuant — KV cache & weight compression

> **Credits.** TurboQuant in this fork is built on top of the absolutely
> awesome work by **[@TheTom](https://github.com/TheTom)** in
> [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant).
> Huge thanks for the original WHT-rotated quantization design, the reference
> kernels, and the relentless backend ports — none of this would exist
> without that project. ❤️

This fork (`atomic-llama-cpp-turboquant`) packages **TurboQuant** as a family
of WHT-rotated low-bit quantization formats with backend-native kernels. They
target two distinct memory-traffic problems:

- **KV cache compression** — `TURBO2_0` / `TURBO3_0` / `TURBO4_0` (2/3/4-bit,
  WHT + PolarQuant). Selected at runtime via `-ctk` / `-ctv`.
- **Model weight compression** — `TQ3_1S` / `TQ4_1S` (3/4-bit, WHT-rotated
  Lloyd-Max with `block_size = 32`). Selected at quantize time as a
  `--type` for `llama-quantize`.

### KV cache types (`-ctk` / `-ctv`)

| Type | Bits | Compression vs F16 | Notes |
|---|---:|---:|---|
| `turbo2` | 2 | ~7.5× | maximum compression, intended for large-context budgets |
| `turbo3` | 3 | ~5.1× | **recommended default**; Metal `TurboFlash` decode kernel |
| `turbo4` | 4 | ~3.8× | highest accuracy of the family, safest fallback |

Typical invocation with full GPU offload + Flash-Attention:

```bash
llama-server -m model.gguf -c 32768 -ngl 99 \
  -ctk turbo3 -ctv turbo3 -fa on
```

Pair with `--cache-reuse N` and a long `-c` to see the practical KV-budget
win — TurboQuant typically shifts the OOM ceiling on Apple Silicon /
discrete GPUs by 3-6× at the same context length.

### Weight quantization types (`llama-quantize`)

| Type | Bits | Block size | Notes |
|---|---:|---:|---|
| `TQ3_1S` | 3 | 32 | 8-level Lloyd-Max + WHT rotation |
| `TQ4_1S` | 4 | 32 | 16-level Lloyd-Max + WHT rotation; fused Metal/Vulkan MUL_MAT_VEC kernels |

```bash
# Convert / re-quantize an F16/F32 GGUF to TQ4_1S.
llama-quantize model-f16.gguf model-tq4_1s.gguf TQ4_1S
```

`TQ4_1S` typically delivers ~25-35 % size reduction vs Q8_0 with single-digit-%
PPL deltas; on bandwidth-bound models / GPUs it can also be faster than Q8_0
because of the lighter memory traffic.

### Backend support

| Backend | KV `turbo2` / `turbo3` / `turbo4` | Weights `TQ3_1S` / `TQ4_1S` |
|---|---|---|
| Metal (Apple Silicon) | yes; `TurboFlash` flash-attn decode kernel for `turbo3` (off-by-default on Apple10 — see PR #91) | yes (V2.1 fused kernels) |
| CUDA (NVIDIA) | `turbo3` / `turbo4` (full); `turbo2` via reference path | `TQ4_1S` MUL_MAT_VEC |
| Vulkan | `turbo3` + `turbo4` KV (FA + SET_ROWS); `turbo2` not yet | `TQ4_1S` (specialised MUL_MAT_VEC, SET_ROWS, CPY) |
| HIP / ROCm | `turbo3` KV; F16-K + TURBO-V mixed dispatch | reference |
| CPU | reference (correctness, not throughput) | reference |

For combining TurboQuant KV with **Gemma 4 MTP speculative decoding**, see
[MTP.md §11-12](MTP.md). The matrix bench shows that the combo
(`turbo3` KV + MTP) is the right pick when the target model is bandwidth-bound
(e.g. Gemma 4 31B), and that f16-KV + MTP wins when the target is
compute-bound (e.g. Gemma 4 26B-A4B on M4 Max).

For **Qwen 3.6 NextN speculative decoding** on top of TurboQuant3 KV, see
[NEXTN.md §7](NEXTN.md). The matrix bench shows that `turbo3` KV + NextN
is the recommended default on the MoE target (Qwen 3.6 35B-A3B,
**+24-36 % tps** over the `turbo3-base` baseline at single-slot), and lifts
the dense Qwen 3.6 27B by ~5 % on top of `turbo3-base` despite the model
being draft-compute-bound.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-7](https://huggingface.co/collections/shoumenchougou/rwkv7-gxx-gguf)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [BonzAI App](https://apps.apple.com/us/app/bonzai-your-local-ai-agent/id6752847988) (proprietary)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LlamaLib](https://github.com/undreamai/LlamaLib) (Apache-2.0)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from Hugging Face Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
- [LLMKube](https://github.com/defilantech/llmkube) - Kubernetes operator for llama.cpp with multi-GPU and Apple Silicon Metal
  support"
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. The `MODEL_ENDPOINT` must point to a Hugging Face compatible API endpoint.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/cli)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>MTP-Spekulation mit einer Assistant-GGUF aktivieren (dieser Fork)</summary>

    Ein Target mit einem kompatiblen MTP-/Assistant-Drafter ueber den
    generischen Draft-Pfad paaren (`--spec-type draft-mtp` + `-md`).
    Sidecar-Tensor-Shape und Tokenizer-Kompatibilitaet muessen pro
    Modellpaar validiert werden — siehe
    [docs/speculative.md](docs/speculative.md).

    Pre-built assistant GGUFs (recommended **`Q4_K_M`** / `Q4_K_S` for best
    speed/quality) are published in the [AtomicChat / Gemma 4 Assistant GGUF
    collection](https://huggingface.co/collections/AtomicChat/gemma-4-assistant-gguf):

    | Target model | Assistant (MTP head) GGUF |
    |---|---|
    | Gemma 4 E2B | [`AtomicChat/gemma-4-E2B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-E2B-it-assistant-GGUF) |
    | Gemma 4 E4B | [`AtomicChat/gemma-4-E4B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) |
    | Gemma 4 26B-A4B | [`AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF) |
    | Gemma 4 12B | [google/gemma-4-12B-it-assistant](https://huggingface.co/google/gemma-4-12B-it-assistant) (self-convert via this fork) |
    | Gemma 4 31B | [`AtomicChat/gemma-4-31B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-31B-it-assistant-GGUF) |

    ```bash
    # Manuelle Ausfuehrung — aktuelle Flags.
    llama-server \
      -m   /path/to/gemma-4-target.gguf \
      -md  /path/to/gemma-4-assistant-Q4_K_M.gguf \
      --spec-type draft-mtp \
      --spec-draft-n-max 3 --spec-draft-n-min 0 \
      -c 16384 \
      -ngl 99 -ngld 99 \
      -fa on \
      --host 127.0.0.1 --port 8080
    ```

    **Fork-Hinweis (Legacy):** die Repo-Helper-Skripte sind **nicht** gegen
    die aktuellen Flags geprueft — aeltere Helper koennen entfernte Flags
    uebergeben. Bis zur Reparatur gelten sie als unverifiziert; siehe
    `scripts/AGENTS.md` ("Legacy-Helper") und
    [`docs/speculative.md`](docs/speculative.md).

    ```bash
    # Dense-Targets.
    scripts/run-gemma4-mtp-server.sh           # 26B-A4B
    scripts/run-gemma4-31b-mtp-server.sh       # 31B

    # Edge-/Centroid-Head-Targets.
    scripts/run-gemma4-e4b-mtp-server.sh
    scripts/run-gemma4-e2b-mtp-server.sh
    ```

    Architektur und Benchmark-Historie (alte Flag-Namen) stehen in
    [MTP.md](MTP.md) (HISTORISCH). Aktuelle Flags:
    [docs/speculative.md](docs/speculative.md).

    </details>

- <details>
    <summary>Enable Qwen 3.6 NextN speculative decoding (this fork)</summary>

    Bei **Qwen-3.6**-kombinierten `*_MTP.gguf`-Checkpoints (der offizielle
    Qwen-Converter packt die NextN-Hilfskopf-Gewichte in dieselbe Datei wie
    das Target) **`--spec-type draft-mtp`** ohne `-md` uebergeben — der
    Server fordert `llama_model_n_layer_nextn(model) > 0` und baut einen
    zweiten `LLAMA_CONTEXT_TYPE_MTP`-Kontext auf dem bereits geladenen
    Target-Modell. Kombinierbar mit TurboQuant3-KV
    (`-ctk turbo3 -ctv turbo3`); fruehere Messungen auf Qwen 3.6 35B-A3B MoE
    zeigten zweistellige t/s-Gewinne — historischer, modellspezifischer
    Beleg (siehe `NEXTN.md`), keine Zusicherung.

    Pre-built combined `_MTP.gguf` quants (recommended **`Q4_K_XL`**,
    matches the matrix bench cells):

    | Target | Combined `_MTP.gguf` |
    |---|---|
    | Qwen 3.6 35B-A3B (MoE) — AtomicChat UDT | [`AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF`](https://huggingface.co/AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF) |
    | Qwen 3.6 27B (dense)   — AtomicChat UDT | [`AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF`](https://huggingface.co/AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF) |
    | Qwen 3.6 35B-A3B (MoE) — Unsloth        | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) |
    | Qwen 3.6 27B (dense)   — Unsloth        | [`unsloth/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) |

    ```bash
    llama-server \
      -m /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL_MTP.gguf \
      --spec-type draft-mtp \
      --spec-draft-n-max 2 --spec-draft-n-min 1 \
      -c 8192 -ngl 99 -ctk turbo3 -ctv turbo3 -fa on \
      --host 127.0.0.1 --port 8080
    ```

    **Fork-Hinweis (Legacy):** die Repo-Helper-Skripte sind **nicht** gegen
    die aktuellen Flags geprueft und gelten bis zur CLI-Reparatur als
    unverifiziert — siehe `scripts/AGENTS.md` und
    [`docs/speculative.md`](docs/speculative.md).

    ```bash
    scripts/run-qwen36-27b-nextn-server.sh        # Qwen 3.6 27B dense
    scripts/run-qwen36-35ba3b-nextn-server.sh     # Qwen 3.6 35B-A3B MoE
    ```

    Eigenstaendige `*_mtp`-GGUFs (separates Draft-Artefakt) sind als
    Fallback weiterhin nutzbar — `-md` auf diese Datei zeigen.

    Die vollstaendige Architektur, KV-only-NextN-Dimensionierung,
    Hidden-State-Transfer und der Matrix-Bench (inkl. der
    27B-Dense-Compute-Bound-Analyse) stehen in [NEXTN.md](NEXTN.md)
    (HISTORISCH). Aktuelle CLI-Flags:
    [docs/speculative.md](docs/speculative.md).

    </details>

- <details>
    <summary>Enable TurboQuant KV cache compression (this fork)</summary>

    Use a TurboQuant KV-cache type for both K and V — typically with
    Flash-Attention enabled — to cut KV memory traffic and footprint at
    long contexts. Recommended default is **`turbo3`** (3-bit, ~4.3× vs F16,
    accelerated by `TurboFlash` on Metal and dedicated kernels on
    CUDA / Vulkan / HIP).

    ```bash
    # ~4.3x KV compression vs F16, full GPU offload, Flash-Attn on.
    llama-server -m model.gguf -c 32768 \
      -ngl 99 -ctk turbo3 -ctv turbo3 -fa on
    ```

    Pick a stronger compression preset by stepping the bit-width:

    ```bash
    -ctk turbo2 -ctv turbo2   # 2-bit KV, ~7.5x vs F16 (highest compression)
    -ctk turbo3 -ctv turbo3   # 3-bit KV, ~5.1x  (default sweet spot)
    -ctk turbo4 -ctv turbo4   # 4-bit KV, ~3.8x  (highest accuracy / fallback)
    ```

    See the longer write-up [above](#turboquant--kv-cache--weight-compression)
    for weight quantization (`TQ4_1S` / `TQ3_1S`) and the per-backend support
    matrix.

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
