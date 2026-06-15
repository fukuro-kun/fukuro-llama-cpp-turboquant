# DiffusionGemma Integration — Masterplan (Option C: Monolithische Portierung)

> Stand: KI-Soloarbeit 2026-06-15, 05:00 Uhr
> Ausgangslage: PR #24423 (DiffusionGemma) in unseren Fork mergen
> Architektur-Problem: Unser Fork hat keine `llama_model_base`-Hierarchie (upstream Feature)
> Loesung: Monolithische Portierung (Graph-Builder-Klasse + Case-Statements)

---

## Analyse: Warum ist die Portierung machbar?

### Erkenntnis 1: Der Graph-Builder ist eigenstaendig

`src/models/diffusion-gemma.cpp` definiert eine **eigenstaendige Graph-Builder-Klasse**, aehnlich wie `llm_build_gemma4_iswa` oder `llm_build_gemma3`. Sie:
- Verwendet `gemma4-common.h` (bereits extrahiert)
- Definiert `class llm_graph_input_attn_diffusion` (region-aware mask)
- Definiert `class llm_build_diffusion_gemma` (Haupt-Graph-Builder)
- Benoetigt **keine** `llama_model_base`-Hierarchie

### Erkenntnis 2: Unser Fork verwendet genau dieses Muster

In `src/llama-model.cpp` bei Zeile 9259:
```cpp
case LLM_ARCH_GEMMA4:
    llm = std::make_unique<llm_build_gemma4_iswa>(*this, params);
```

Das ist das **exakt gleiche Muster** — `std::make_unique<Klasse>(*this, params)`.

### Erkenntnis 3: Nur 4 Stellen in `llama-model.cpp` muessen erweitert werden

| Stelle | Funktion | Zeile (ca.) | Was hinzufuegen |
|--------|----------|-------------|-----------------|
| A | `load_hparams()` | ~1660 | `case LLM_ARCH_DIFFUSION_GEMMA:` mit Diffusion-spezifischen HParams |
| B | `load_tensors()` | ~4800 | `case LLM_ARCH_DIFFUSION_GEMMA:` mit Tensor-Mapping |
| C | `build_graph()` | ~9260 | `case LLM_ARCH_DIFFUSION_GEMMA:` mit `std::make_unique<llm_build_diffusion_gemma>` |
| D | (weiterer Switch) | ~9870 | `case LLM_ARCH_DIFFUSION_GEMMA:` (wahrscheinlich fuer RoPE oder Sampling) |

---

## Option C: Detaillierter Plan

### Phase 1: Graph-Builder vorbereiten (60 Min)

#### 1.1 `src/models/diffusion-gemma.cpp` anpassen

**Problem:** Die Datei verwendet `dynamic_cast<llama_model_diffusion_gemma*>` aus dem Diff.
**Loesung:** Ersetzen durch direkten Zugriff auf `model` (monolithisch).

**Aenderungen in `diffusion-gemma.cpp`:**
- Entferne alle `dynamic_cast<const llama_model_diffusion_gemma *>(model)`
- Ersetze durch direkten Zugriff: `model.hparams.canvas_length`, `model.sc_enabled`, etc.
- Die `llama_model` Struct in unserem Fork enthaelt bereits die Felder (dank Subagent Merge)

#### 1.2 `src/models/models.h` erweitern

```cpp
// Forward declaration (wie alle anderen Graph-Builder)
struct llm_build_diffusion_gemma;
```

#### 1.3 `src/CMakeLists.txt` eintragen

```cmake
# In src/CMakeLists.txt, unter models/:
set(LLAMA_SOURCES ...
    models/diffusion-gemma.cpp
    ...
)
```

---

### Phase 2: `llama-model.cpp` erweitern (90 Min)

#### 2.1 `load_hparams()` — ca. Zeile 1660

```cpp
case LLM_ARCH_DIFFUSION_GEMMA:
    {
        // Gemma-4 HParams (identisch zu GEMMA4)
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);
        // ... (wie GEMMA4)

        // Diffusion-spezifische HParams
        ml.get_key(LLM_KV_DIFFUSION_CANVAS_LENGTH, hparams.canvas_length, false);
        ml.get_key(LLM_KV_DIFFUSION_EB_MAX_STEPS, hparams.eb_max_steps, false);
        ml.get_key(LLM_KV_DIFFUSION_EB_T_MIN, hparams.eb_t_min, false);
        ml.get_key(LLM_KV_DIFFUSION_EB_T_MAX, hparams.eb_t_max, false);
        ml.get_key(LLM_KV_DIFFUSION_EB_ENTROPY_BOUND, hparams.eb_entropy_bound, false);
        ml.get_key(LLM_KV_DIFFUSION_EB_STABILITY, hparams.eb_stability, false);
        ml.get_key(LLM_KV_DIFFUSION_EB_CONFIDENCE, hparams.eb_confidence, false);

        type = LLM_TYPE_UNKNOWN;
    } break;
```

#### 2.2 `load_tensors()` — ca. Zeile 4800

```cpp
case LLM_ARCH_DIFFUSION_GEMMA:
    {
        // Identisch zu GEMMA4, aber mit enc_out_scale
        // (Code kann kopiert werden von GEMMA4-Case, dann enc_out_scale hinzufuegen)
        // ...
    } break;
```

#### 2.3 `build_graph()` — ca. Zeile 9260

```cpp
case LLM_ARCH_DIFFUSION_GEMMA:
    {
        llm = std::make_unique<llm_build_diffusion_gemma>(*this, params);
    } break;
```

#### 2.4 Weiterer Switch — ca. Zeile 9870

Pruefen, was dort passiert (wahrscheinlich RoPE-Typ oder Sampling), und entsprechenden Case hinzufuegen.

---

### Phase 3: Build & Test (60-120 Min)

#### 3.1 CMake-Konfiguration

```bash
cd ~/git/fukuro-llama-cpp-turboquant
cmake -B build -DLLAMA_CUDA=ON -DLLAMA_NATIVE=ON
```

#### 3.2 Build

```bash
cmake --build build -j$(nproc)
```

**Erwartete Fehler und Fixes:**
- `undefined reference to llm_build_diffusion_gemma` → `src/CMakeLists.txt` pruefen
- `unknown member canvas_length` → `llama_model.h` pruefen (dank Subagent Merge sollte es da sein)
- `missing include` → `src/models/models.h` pruefen

#### 3.3 Test

```bash
# CPU-Offload auf hydra (8GB VRAM reicht nicht fuer 26B)
./build/bin/llama-diffusion-cli \
  -m modelle/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -ngl 0 -cnv -n 128 \
  --ctx-size 4096
```

**Erwartung:** Langsam (CPU-only), aber sollte ohne Crash laufen.

---

## Risiko-Einschaetzung

| Risiko | Wahrscheinlichkeit | Auswirkung | Mitigation |
|--------|-------------------|------------|------------|
| `llama_model` hat nicht alle Diffusion-Felder | Gering | Build-Fehler | Subagent hat Felder bereits in `llama-model.h` eingefuegt |
| `gemma4-common.h` inkompatibel mit unserem Gemma4 | Mittel | Falsche Ausgaben | Vergleichen mit `src/models/gemma4-iswa.cpp` |
| Graph-Builder hat CUDA-spezifische Aufrufe | Gering | Build-Fehler auf non-CUDA | `ggml/src/ggml-cuda/diffusion-sampling.cu` ist bereits kopiert |
| `llm_graph_input_attn_diffusion` funktioniert nicht | Mittel | Falsche Attention | Testen mit einfachem Prompt |
| `examples/diffusion/` existiert nicht in unserem Fork | Hoch | `llama-diffusion-cli` fehlt | In `tools/` kopieren (bereits geschehen) |

---

## Alternative: Option D (Minimaler Merge)

Falls Option C scheitert, kann man auch **nur die Konvertierung und Grundstruktur** behalten:
- `conversion/diffusion_gemma.py` ✅
- `gguf-py` Erweiterungen ✅
- Arch-Definitionen ✅
- **Ohne** Graph-Builder (DiffusionGemma laeuft dann nicht, aber die Infrastruktur ist da)

Das ist der **sicherste Fallback**.

---

## Empfehlung

**Option C ist machbar** — der Blocker war kleiner als gedacht. Die Graph-Builder-Klasse ist eigenstaendig und passt in unsere monolithische Struktur.

**Geschaetzte Gesamtzeit:** 4-5 Stunden (inkl. Debugging)

**Empfohlene Reihenfolge:**
1. `src/CMakeLists.txt` eintragen
2. `src/models/models.h` Forward declaration
3. `src/llama-model.cpp` 4 Cases einfuegen
4. `src/models/diffusion-gemma.cpp` anpassen (dynamic_cast entfernen)
5. Build testen
6. Debuggen

---

## Anhang: Was bereits erledigt ist

| Aufgabe | Status | Wer |
|---------|--------|-----|
| `src/llama-arch.h/.cpp` — Arch/Tensor/Name Mapping | ✅ Fertig | Subagent |
| `src/llama-model.h/.cpp` — Partial (C-API, HParams-Stub) | ✅ Fertig | Subagent |
| `common/arg.cpp/.h` — CLI-Flags | ✅ Fertig | Subagent |
| `include/llama.h` — C-API Erweiterungen | ✅ Fertig | Subagent |
| `gguf-py/gguf/constants.py/.gguf_writer.py` | ✅ Fertig | Subagent |
| `tests/test-llama-archs.cpp` | ✅ Fertig | Subagent |
| Neue Dateien kopieren (diffusion-gemma.cpp, CUDA, Examples) | ✅ Fertig | Subagent |
| `gemma4-common.h` extrahiert | ✅ Fertig | KI |
| `tools/CMakeLists.txt` aktualisiert | ✅ Fertig | KI |
| Q4_K_M Modell heruntergeladen | ✅ Fertig | KI |
| `src/models/diffusion-gemma.cpp` anpassen (dynamic_cast → monolithisch) | ⏳ Offen | User/Morgen |
| `src/models/models.h` Forward declaration | ⏳ Offen | User/Morgen |
| `src/llama-model.cpp` 4 Cases einfuegen | ⏳ Offen | User/Morgen |
| `src/CMakeLists.txt` eintragen | ⏳ Offen | User/Morgen |
| Build & Test | ⏳ Offen | User/Morgen |
