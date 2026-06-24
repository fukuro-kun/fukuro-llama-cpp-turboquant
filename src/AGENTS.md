# src/ — AGENTS.md (Child-DOX)

**Zweck:** C++ Core der Inference Engine. Modell-Architekturen, Graphen-Building, KV-Cache Management, Kontext/Scheduler, Sampling und Quantisierung. Beherbergt alle Modell-Implementierungen.

**Eigentuemer:** fukuro + KI-Agent

**Geltungsbereich:** Alle Pfade unter `src/`.

---

## Lokale Vertraege

### Code-Stil

- C++17, konsistent mit Upstream llama.cpp.
- Keine `using namespace std;` in Headern.
- Praeferierte Dateiendung: `.cpp` / `.h`.

### Fork-spezifische Dateien

Diese Dateien enthalten Fork-spezifische Logik und duerfen bei Upstream-Syncs besonders geprueft werden:

| Datei | Zweck |
|-------|-------|
| `llama-quant.cpp` | TurboQuant KV-/Weight-Quantisierungs-Pfade |
| `models/gemma4-assistant.cpp` | Gemma 4 Multi-Token Prediction (MTP) |
| `models/qwen3next.cpp` | Qwen 3.x NextN spekulative Decodierung (Shared-Model-Draft) |
| `models/qwen35.cpp` | Qwen 3.5 Dense-Modell |
| `models/qwen35moe.cpp` | Qwen 3.5 MoE-Modell |
| `models/diffusion-gemma.cpp` | DiffusionGemma-Integration |

### Wichtige Konfiguration in `gemma4-assistant.cpp`

- `n_embd_inp_impl` und `n_embd_out_impl` **müssen** auf `n_embd_backbone` gesetzt werden (nicht auf `n_embd`). Der Assistant hat ein eigenes `n_embd` (z.B. 1024), aber Input/Output-Projektionen (`nextn_proj_pre/post`) arbeiten mit `n_embd_backbone` (z.B. 3840). Wenn diese nicht korrekt gesetzt werden, wird der `pending_h`-Puffer in `speculative.cpp` zu klein dimensioniert → Backbone-Hidden-State wird trunciert → 0% MTP-Akzeptanz.
- `llama_kv_cache_iswa_context` muss `get_turbo_innerq_scale_inv()` (und `get_turbo_rot_forward/inverse`) überschreiben und an `ctx_base` delegieren, da `build_attn_mha` über `this->mctx` (den ISWA-Wrapper) auf diese Methoden zugreift.

### Modell-Implementierungen

- `src/models/` enthaelt >100 Modell-Dateien (eine pro Architektur).
- Neue Modelle erfordern Eintrag in `llama-arch.cpp` (Arch-Tabelle) und eigene `.cpp` unter `models/`.

---

## Arbeitsanleitung

### Kernkomponenten — Wo was lebt

| Komponente | Primaere Dateien | Was dort passiert |
|------------|------------------|-------------------|
| Architekturen | `llama-arch.cpp`, `llama-arch.h` | Modell-Arch-Registry, Layer-Mappings, Tensor-Namen |
| Graph-Building | `llama-graph.cpp`, `llama-graph.h` | Compute-Graphen fuer Forward-Pass, Node-Scheduling |
| KV-Cache | `llama-kv-cache.cpp`, `llama-kv-cache.h`, `llama-kv-cache-iswa.cpp` | Cache-Allokation, Attention-Store, In-Place-SWA |
| Kontext / Scheduler | `llama-context.cpp`, `llama-context.h` | Batch-Processing, Token-Einbettung, MTP-Integration |
| Sampling | `llama-sampler.cpp`, `llama-sampler.h` | Top-K/Top-P/Temperature, Grammatik, Repeat-Penalty |
| Modell-Loading | `llama-model.cpp`, `llama-model.h` | GGUF-Parsing, Tensor-Mapping, Gewichts-Initialisierung |
| Quantisierung | `llama-quant.cpp` | TurboQuant-Quantisierungs-Logik |

### Aenderungen an Modell-Architekturen

1. `llama-arch.cpp`: Arch-Tabelle und HParams erweitern.
2. `models/<arch>.cpp`: Forward-Graph und Layer-Logik implementieren.
3. `llama-graph.cpp`: Falls neue GGML-Ops benoetigt.
4. Build testen: `cmake --build build -j$(nproc)`.
5. `tests/test-models.cpp` oder `./build/bin/llama-cli` mit betroffenem Modell pruefen.

### Aenderungen am KV-Cache

- KV-Cache-Aenderungen betreffen nahezu alle Modell-Dateien (Attention-Pfade).
- `llama-kv-cache-iswa.cpp` ist separat fuer In-Place-Sliding-Window Attention.
- TurboQuant-KV-Pfade kreuzen sich mit `ggml/src/ggml-turbo-quant.c`.

---

## Verifikation

- [ ] Build erfolgreich (`cmake --build build -j$(nproc)`)
- [ ] Bei Arch-Aenderung: `llama-cli` laedt betroffenes Modell ohne Crash
- [ ] Bei KV-Cache-Aenderung: `./build/bin/test-kv-cache` (falls vorhanden) oder manueller Test mit Prompt-Wiederholung
- [ ] Fork-spezifische Dateien nach Upstream-Sync auf Konsistenz geprueft

---

## Child-DOX-Index

| Pfad | Zweck | Status |
|------|-------|--------|
| `src/models/` | >100 Modell-Implementierungen (eine Datei pro Architektur) | [x] Aktiv |

*Hinweis: `src/models/` benoetigt kein weiteres Child-DOX pro Datei. Ein einzelnes Child-AGENTS.md fuer `src/models/` genuegt, um Modell-Implementierungs-Richtlinien und den Datei-Index zentral zu halten.*
