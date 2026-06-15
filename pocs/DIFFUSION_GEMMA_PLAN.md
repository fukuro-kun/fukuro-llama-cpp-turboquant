# Gemma4-Diffusion Integration — Analyse & Plan

> KI-Soloarbeit, durchgefuehrt 2026-06-15 ab 04:00 Uhr.
> PR: https://github.com/ggml-org/llama.cpp/pull/24423 (DiffusionGemma)
> Basis: Unser Fork (fukuro-llama-cpp-turboquant) mit TurboQuant + MTP + NextN

---

## PR-Uebersicht

| Eigenschaft | Wert |
|-------------|------|
| **Autor** | danielhanchen (Unsloth) |
| **Status** | Draft (16 Commits, "Large PR" laut Bot) |
| **Groesse** | 3693 Zeilen, 26 Dateien |
| **Basis** | ggml-org/llama.cpp:master |
| **Ziel** | DiffusionGemma 26B-A4B Unterstuetzung |

### Neue Dateien (15)

| Datei | Zweck | Riskant? |
|-------|-------|----------|
| `conversion/diffusion_gemma.py` | HF->GGUF Konvertierung fuer DiffusionGemma | Nein (neu) |
| `examples/diffusion-gemma-eval/` | Evaluierungs-Tool | Nein (neu) |
| `examples/diffusion-gemma-server/` | HTTP-Server fuer DiffusionGemma | Nein (neu) |
| `examples/diffusion-gemma-server/diffusion-gemma-visual-server.cpp` | Visueller Server | Nein (neu) |
| `ggml/src/ggml-cuda/diffusion-sampling.cu` | CUDA-Kernel fuer Diffusion-Sampling | **Mittel** (CUDA) |
| `ggml/src/ggml-cuda/diffusion-sampling.cuh` | CUDA-Header | **Mittel** (CUDA) |
| `src/models/diffusion-gemma.cpp` | Modell-Graph fuer DiffusionGemma | Nein (neu) |

### Geaenderte Dateien (11) — Konfliktpotential

| Datei | Aenderung | Konflikt mit Fork? | Risiko |
|-------|-----------|-------------------|--------|
| `common/arg.cpp` | Neue `--diffusion-*` Flags | **Ja** — wir haben MTP-Flags | **Mittel** |
| `common/common.h` | `common_params_diffusion` erweitert | **Ja** — Struktur erweitert | Gering |
| `examples/CMakeLists.txt` | Neue Subdirs | Nein | Gering |
| `examples/diffusion/diffusion-cli.cpp` | Erweitert fuer DiffusionGemma | Nein (existiert) | Gering |
| `examples/diffusion/diffusion.cpp` | Entropy-bound decoder | Nein (existiert) | Gering |
| `examples/diffusion/diffusion.h` | Neue Parameter | Nein (existiert) | Gering |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Diffusion-Registrierung | **Ja** — Vulkan-Turbo3! | **HOCH** |
| `gguf-py/gguf/constants.py` | `DIFFUSION_GEMMA` Arch/Tensor | Gering | Gering |
| `gguf-py/gguf/gguf_writer.py` | `add_diffusion_*` Methoden | Gering | Gering |
| `include/llama.h` | Neue C-API Funktionen | **Ja** — API-Erweiterung | Gering |
| `src/llama-arch.cpp` | `LLM_ARCH_DIFFUSION_GEMMA` | **Ja** — Arch-Tabelle | **Mittel** |
| `src/llama-arch.h` | `LLM_ARCH_DIFFUSION_GEMMA`, `llm_arch_is_diffusion` | **Ja** — Enum erweitert | Gering |
| `src/llama-model.cpp` | Modell-Loading fuer DiffusionGemma | **Ja** — MTP-Loading | **Mittel** |
| `src/llama-model.h` | Header-Erweiterung | Gering | Gering |
| `src/models/gemma4-common.h` | Gemma4-Shared-Code erweitert | **Ja** — Unser Gemma4! | **Mittel** |
| `src/models/models.h` | Modell-Dispatcher | Gering | Gering |
| `tests/test-llama-archs.cpp` | Arch-Test | Gering | Gering |

---

## Kritische Konflikte (Detail)

### 1. `common/arg.cpp` — CLI-Argumente (Mittel)

Wir haben in `common/arg.cpp` Fork-spezifische Flags:
- `--mtp-head` / `--spec-type mtp`
- `--draft-block-size`
- `--gpu-layers-draft`

Der PR fuegt hinzu:
- `--diffusion-blocks`
- `--diffusion-visual-progress`
- `--diffusion-visual-interval`
- `--diffusion-eb` (entropy-bound)

**Loesung:** Die neuen Flags koennen einfach neben den existierenden eingefuegt werden. Keine Ueberschneidung in Namen.

### 2. `src/llama-arch.cpp/.h` — Architektur-Tabelle (Mittel)

Wir haben:
- `LLM_ARCH_GEMMA4` (65)
- `LLM_ARCH_GEMMA4_ASSISTANT` (66)
- `LLM_ARCH_GEMMA4_MTP` (67)

Der PR fuegt ein:
- `LLM_ARCH_DIFFUSION_GEMMA` — muss nach `GEMMA4_MTP` eingefuegt werden

**Loesung:** Enum-Wert am Ende anhaengen. `gguf-py/gguf/constants.py` synchronisieren.

### 3. `src/models/gemma4-common.h` — Gemma4 Shared Code (Mittel)

Dies ist der **wichtigste Konflikt**. Der PR erweitert den gemeinsamen Gemma4-Code fuer Diffusion-Unterstuetzung. Da unser Fork bereits Gemma4 mit MTP hat, muss geprueft werden, ob die Diffusion-Erweiterungen mit unserem Gemma4-Code kompatibel sind.

**Loesung:** `diff` unseres `gemma4-common.h` gegen upstream `gemma4-common.h` vor dem Merge.

### 4. `ggml/src/ggml-cuda/ggml-cuda.cu` — CUDA-Backend (HOCH)

Wir haben **Vulkan-Turbo3** Aenderungen in diesem File (532 Commits laut BRANCHES.md). Der PR fuegt Diffusion-Registrierung hinzu.

**Loesung:** Sehr sorgfaeltig pruefen. Die Diffusion-Registrierung ist klein (wenige Zeilen), aber an der richtigen Stelle einzufuegen ist kritisch.

### 5. `src/llama-model.cpp` — Modell-Loading (Mittel)

Wir haben MTP-Loading-Logik (`llama_model_load_mtp_from_file`). Der PR erweitert das Modell-Loading fuer DiffusionGemma-Spezifika (canvas_length, entropy-bound, etc.).

**Loesung:** Die Diffusion-Logik ist orthogonal zum MTP-Loading. Vorsichtig einfuegen.

---

## Nicht-riskante Dateien (einfach zu mergen)

| Datei | Warum einfach |
|-------|--------------|
| `conversion/diffusion_gemma.py` | Neu, keine Konflikte |
| `examples/diffusion-gemma-*/` | Neu, keine Konflikte |
| `ggml/src/ggml-cuda/diffusion-sampling.cu/.cuh` | Neu, nur neuer CUDA-Kernel |
| `src/models/diffusion-gemma.cpp` | Neu, eigene Modell-Datei |
| `tests/test-llama-archs.cpp` | Nur neuer Test-Eintrag |

---

## Empfohlener Merge-Plan (Option A — manuell)

### Phase 1: Vorbereitung (15 Min)
1. `git fetch upstream` (ggml-org/llama.cpp)
2. Unser `gemma4-common.h` mit upstream `gemma4-common.h` vergleichen
3. Unser `ggml-cuda.cu` mit upstream `ggml-cuda.cu` vergleichen

### Phase 2: Nicht-riskante Dateien (30 Min)
4. Neue Dateien kopieren (diffusion-gemma.cpp, diffusion-sampling.cu, etc.)
5. Neue Examples/CMakeLists.txt Eintraege
6. `gguf-py` Erweiterungen

### Phase 3: Riskante Dateien (60 Min)
7. `src/llama-arch.cpp/.h` — DiffusionG Arch einfuegen
8. `src/models/gemma4-common.h` — Sorgfaeltig mergen
9. `src/llama-model.cpp/.h` — Diffusion-Loading einfuegen
10. `common/arg.cpp` — Diffusion-Flags einfuegen
11. `ggml/src/ggml-cuda/ggml-cuda.cu` — Registrierung einfuegen

### Phase 4: Build & Test (30-120 Min)
12. `cmake -B build -DLLAMA_CUDA=ON`
13. `cmake --build build -j$(nproc)`
14. Modell download: `unsloth/diffusiongemma-26B-A4B-it-GGUF`
15. Test: `./build/bin/llama-diffusion-cli -cnv -n 128`

---

## Modell-Info

| Eigenschaft | Wert |
|-------------|------|
| HF Repo | `unsloth/diffusiongemma-26B-A4B-it-GGUF` |
| Quantisierung | Q8_0 (ca. 28GB) oder Q4_K_M (ca. 16GB) |
| VRAM fuer Q4_K_M | ~16GB |
| Auf unseren Hosts | **uranus** (32GB) ideal; **hydra/styx** (8GB) zu klein fuer 26B |

**Hinweis:** hydra und styx haben nur 8GB VRAM — zu wenig fuer DiffusionGemma 26B selbst in Q4_K_M. Test muss auf uranus (wenn online) oder mit CPU-Offload.

---

## Entscheidungen (von fukuro bestaetigt)

| Frage | Entscheidung |
|-------|-------------|
| `LLAMA_EXAMPLE_DIFFUSION` | ✅ Neues Flag hinzufuegen |
| `examples/diffusion-gemma-*` | ✅ Nach `tools/` verschieben (unser Muster) |
| Test-Strategie | ✅ Erster Test mit CPU-Offload auf hydra |
| Konvertierungsskript | ✅ Sofort uebernehmen (`conversion/diffusion_gemma.py`) |
| GGUF-Modell | ✅ Vorgefertigtes GGUF von Unsloth herunterladen (keine Safetensors) |

### Herunterladen des Modells

```bash
# Via HuggingFace Hub (skill huggingface-cli verfuegbar)
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF \
    --local-dir modelle/diffusiongemma-26B-A4B-it-GGUF \
    --include "*Q4_K_M*"
```

**Hinweis:** Q4_K_M ≈ 16 GB Download. Bei CPU-Offload auf hydra wird der Test langsam sein, aber er zeigt, ob der Merge funktioniert.

## Offene Fragen fuer morgen

- Keine — alle Entscheidungen getroffen. Merge kann beginnen.
