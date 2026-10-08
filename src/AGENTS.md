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
| `models/qwen4exp.cpp` | Qwen4Exp: QSA, GDN, PLE, Hyper-Connections, MoE |
| `models/muse-glimmer.cpp` | Muse Glimmer: SWA, QK-norm, attention gate |
| `models/laguna.cpp` | Laguna: sigmoid-MoE, shared expert, hybrid SWA |
| `models/nanbeige.cpp` | Nanbeige: looped transformer, n_loops-Expansion |
| `models/hy-v3.cpp` | Hy-v3: sigmoid-MoE, NextN/MTP draft head |
| `models/qwen3tts.cpp` | Qwen3-TTS: erbt von qwen3vl |
| `models/pockettts.cpp` | PocketTTS: CALM text backbone |
| `models/k2-horizon.cpp` | K2-Horizon: MoVA-Attention, grouped RMS-Norm, eigener Tokenizer-Dispatch (`unicode_regex_split_custom_k2_horizon` in `unicode.cpp`; Regex-String in `llama-vocab.cpp` ist Dispatch-Schluessel — beide muessen synchron bleiben). Enthaelt zudem den K2-MTP-Pfad: Sidecar-Tensor-Loading (`blk.48.nextn.*`, kein Tokenizer/Embeddings im Sidecar — `token_embd`/`output` kommen vom Basismodell), eigener MTP-Forward-Graph (Embedding-First/Hidden-Second-Concat, GRMS, softplus_{ln2}-Gate, RoPE-NEOX, Position i+1, `shared_head_norm`-Chain-State) |
| `llama-memory-hybrid-idx.cpp` | Qwen4Exp QSA-Indexer-Hybrid-Memory |

### Wichtige Konfiguration in `gemma4-assistant.cpp`

- `n_embd_inp_impl` und `n_embd_out_impl` **müssen** auf `n_embd_backbone` gesetzt werden (nicht auf `n_embd`). Der Assistant hat ein eigenes `n_embd` (z.B. 1024), aber Input/Output-Projektionen (`nextn_proj_pre/post`) arbeiten mit `n_embd_backbone` (z.B. 3840). Wenn diese nicht korrekt gesetzt werden, wird der `pending_h`-Puffer in `speculative.cpp` zu klein dimensioniert → Backbone-Hidden-State wird trunciert → 0% MTP-Akzeptanz.
- `llama_kv_cache_iswa_context` muss `get_turbo_innerq_scale_inv()` (und `get_turbo_rot_forward/inverse`) überschreiben und an `ctx_base` delegieren, da `build_attn_mha` über `this->mctx` (den ISWA-Wrapper) auf diese Methoden zugreift.

### Wichtige Konfiguration in `qwen4exp.cpp` / Hybrid-Memory

- **HC-Breite:** Qwen4Exp nutzt Hyper-Connections; die relevante Hidden-State-Breite fuer den MTP-Draft ist die Ausgangsbreite (z.B. 10240 = 4×2560), nicht `n_embd_inp()`. Der Draft-Kontext (`LLAMA_CONTEXT_TYPE_MTP`) allokiert Batch-Buffers mit `n_embd_out()`.
- **Fullstate-Restore:** der Whole-Context-Restore fuehrt den globalen Cache-Clear **einmal ausserhalb der Stream-Schleife** aus (nicht erneut pro Stream) — andernfalls werden frueher restaurierte Streams wieder gewiped.
- **Hybrid-Indexer (`llama-memory-hybrid-idx.cpp`):** der Restore spiegelt das Attention-Slot-Layout (`state_read_sinfo` liefert das Layout); `init_batch` respektiert den `n_rs_seq`-Vertrag der Basisklasse.
- **`n_rs_seq`:** der recurrent-State-Split wird vom Vertrag respektiert, aber `qwen4exp` klemmt `n_rs_seq` derzeit auf 0 (`llama-context.cpp`; nur qwen35/moe erlauben den Pfad). Konsequenz: die obige Hybrid-Indexer-Split-Korrektur ist unter `n_rs_seq = 0` **inaktiv** und kann den getesteten q8_0-Spec-Fehler nicht erklaeren. Ob der Clamp selbst zur Degradation beitraegt, ist **nicht** als Root Cause ausgeschlossen — er wurde nur nicht als Erklaerung des init_batch-Befunds benoetigt.

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
5. `./build/bin/llama-cli` mit betroffenem Modell pruefen (ein `tests/test-models.cpp` existiert nicht).

### Aenderungen am KV-Cache

- KV-Cache-Aenderungen betreffen nahezu alle Modell-Dateien (Attention-Pfade).
- `llama-kv-cache-iswa.cpp` ist separat fuer In-Place-Sliding-Window Attention.
- TurboQuant-KV-Pfade kreuzen sich mit `ggml/src/ggml-turbo-quant.c`.

---

## Verifikation

- [ ] Build erfolgreich (`cmake --build build -j$(nproc)`)
- [ ] Bei Arch-Aenderung: `llama-cli` laedt betroffenes Modell ohne Crash
- [ ] Bei KV-Cache-Aenderung: `./build/bin/test-save-load-state`, `test-state-restore-fragmented` oder `test-recurrent-state-rollback` (registriert, LABEL `model` — benoetigen die Modell-Fixture) oder manueller Test mit Prompt-Wiederholung
- [ ] Fork-spezifische Dateien nach Upstream-Sync auf Konsistenz geprueft

---

## Child-DOX-Index

| Pfad | Zweck | Status |
|------|-------|--------|
| `src/models/` | >100 Modell-Implementierungen (eine Datei pro Architektur) | [~] Kein eigenes DOX — durch dieses Dokument abgedeckt |

*Hinweis: `src/models/` benoetigt kein weiteres Child-DOX pro Datei; die Modell-Implementierungs-Richtlinien stehen oben unter "Fork-spezifische Dateien" und "Modell-Implementierungen". Ein eigenes `src/models/AGENTS.md` erst bei eigenstaendiger Vertragsdomaene anlegen.*
