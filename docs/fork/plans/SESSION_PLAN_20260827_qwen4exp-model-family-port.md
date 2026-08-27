# SESSION_PLAN: #99 Qwen4Exp und LAN-relevante Modellfamilien

**Erstellt:** 2026-08-27  
**Status:** ☐ Geplant — Review-Loop abgeschlossen, Korrekturen eingearbeitet, noch keine Code-Aenderungen
**ROADMAP-Item:** #99  
**Quellstand:** Unsloth-qwen4exp-Snapshot `035e22731` auf gemeinsamer llama.cpp-Basis `4fc4ec554`  
**Ziel:** Selektiver Port von Qwen4Exp sowie Muse Glimmer, Laguna, Nanbeige, Hy-v3, Qwen3-TTS und PocketTTS in den TurboQuant-Fork

## 1. Zielbild

Der TurboQuant-Fork soll sieben neue Modellarchitekturen laden, konvertieren und korrekt ausfuehren koennen, ohne die Fork-spezifischen Funktionen zu beschaedigen:

1. `qwen4exp` — primaeres Ziel; QSA, GDN, PLE, Hyper-Connections und MoE.
2. `muse-glimmer` — textuell und multimodal nutzbares 30B-Modell.
3. `laguna` — MoE-Coding-Modellfamilie; insbesondere die Consumer-taugliche XS-Variante.
4. `nanbeige` — kleines Looped-Transformer-Modell fuer Edge-/Agent-Workloads.
5. `hy_v3` — MoE mit bestehender NextN-/MTP-Infrastruktur.
6. `qwen3tts` — Text-Backbone und anschliessend vollstaendige TTS-Pipeline.
7. `pockettts` — Text-/Conditioning-Backbone und anschliessend vollstaendige TTS-Pipeline.

Die Integration erfolgt als **selektiver Symbol-Port**. Der grosse Unsloth-Commit wird nicht cherry-gepickt, da er weitere, fuer das LAN nicht priorisierte Architekturen und fremde Infrastruktur enthaelt.

## 2. Verbindliche Entscheidungen

### 2.1 In Scope

- C++-Architektur-Registry, Hparams, Tensor-Mappings und Modell-Factory fuer alle sieben Architekturen.
- Alle sieben Modellimplementierungen unter `src/models/`.
- GGUF-Python-Registry, Writer-Funktionen und Converter fuer alle sieben Architekturen.
- `llama_memory_hybrid_idx` fuer Qwen4Exp-QSA.
- Vollstaendige Text-Inferenz fuer alle sieben Architekturen.
- Multimodale Muse-Glimmer-Pipeline, sofern die Referenzdateien ohne fremde Architektur-Abhaengigkeiten portierbar sind.
- End-to-End-Audio fuer Qwen3-TTS und PocketTTS als eigene, nachgelagerte Phase.
- CUDA-, Vulkan- und CPU-Build-Kompatibilitaet; Laufzeittests passend zur verfuegbaren Hardware.
- Korrektheitsvergleich von Qwen4Exp gegen den funktionierenden isolierten Referenz-Build.

### 2.2 Nicht in Scope

- Kein direkter Cherry-Pick von `035e22731`.
- Keine Produktionsintegration in InferenzQuelle waehrend dieses Plans.
- Kein Umbau des bestehenden MoE-Caches.
- Kein thecodacus Host-Pinning/Prefetch-Merge im Rahmen dieses Ports.
- Keine Portierung von `bailingmoe3`, `kimi-k3`, `minimax-01`, `minimax-m3`, `dots3note` oder `granite-switch`.
- Keine Portierung der nur dafuer benoetigten Infrastruktur:
  - `llama-kv-cache-dsv4.*`
  - `llama-kv-cache-msa.*`
  - `llama-kv-cache-dsa-iswa.*`
  - `lightning-indexer.cu`
  - `dsv4-hc.cu`
- Keine Uebernahme allgemeiner `mmid.cu`-/`topk-moe.cu`-Optimierungen ohne separaten Benchmark und Review.
- Kein Anheben von `LLAMA_MAX_EXPERTS` auf 1024: Qwen4Exp nutzt 512, Laguna 256 und Hy-v3 192 Experten; der bestehende Wert 512 reicht.

### 2.3 Spaeter separat bewerten

- `granite-switch`: technisch interessant, aber vorerst keine ausreichend reife GGUF-/llama.cpp-Nutzung fuer den LAN-Betrieb. Separates ROADMAP-Item nach verfuegbarem Testmodell.
- Dynamischer TurboQuant-MoE-Cache versus statisches Profil-/Slot-Caching: erst nach korrekter Qwen4Exp-Baseline benchmarken.
- Nicht-funktional notwendige fused Top-K-/MMID-Optimierungen.

## 3. Quelllage und Portierungsstrategie

### 3.1 Gemeinsame Basis

- Beide Entwicklungszweige enthalten llama.cpp-Commit `4fc4ec554` als gemeinsame Basis.
- Der TurboQuant-Fork liegt mit eigenen TurboQuant-, MTP-/NextN-, Vulkan-, Expected-Attention- und Betriebsanpassungen darueber.
- Der isolierte Referenzzweig enthaelt zusaetzlich thecodacus-Commits und den Unsloth-Architekturstand `035e22731`.
- Deshalb werden neue Dateien aus dem gepinnten Referenzstand uebernommen, Shared-Dateien aber **symbolweise** in den aktuellen Fork integriert.

### 3.2 Merge-Regeln

1. Fork-Versionen von Shared-Dateien bleiben die Ausgangsbasis.
2. Nur Symbole uebernehmen, die von einer der sieben Zielarchitekturen referenziert werden.
3. Keine zusammenhaengenden Diff-Bloecke blind kopieren; fremde Kimi-/DeepSeek-/MiniMax-/Dots-Symbole bleiben draussen.
4. Bestehende Fork-Felder und Pfade muessen erhalten bleiben:
   - TurboQuant Rotation und InnerQ.
   - Expected Attention.
   - Gemma-4 Assistant/MTP.
   - Qwen NextN.
   - DiffusionGemma.
   - UBBoost und `n_ubatch_prefill`.
   - Intel Hybrid-Attention.
   - Vulkan-TurboQuant.
   - MoE Expert Cache.
5. Nach jeder Integrationswelle muss der Fork kompilieren. Keine sieben Architekturen auf einmal bis zum ersten Build aufstauen.

## 4. Zielartefakte

### 4.1 Neue Kern-Dateien

- `src/models/qwen4exp.cpp`
- `src/models/muse-glimmer.cpp`
- `src/models/laguna.cpp`
- `src/models/nanbeige.cpp`
- `src/models/hy-v3.cpp`
- `src/models/qwen3tts.cpp`
- `src/models/pockettts.cpp`
- `src/llama-memory-hybrid-idx.cpp`
- `src/llama-memory-hybrid-idx.h`

### 4.2 Neue Converter

- `conversion/qwen4exp.py`
- `conversion/muse_glimmer.py`
- `conversion/laguna.py`
- `conversion/nanbeige.py`
- `conversion/qwen3tts.py`
- `conversion/pockettts.py`
- Hy-v3 nutzt den bestehenden Hunyuan-Converter als Basis (`HYV3Model` in `conversion/hunyuan.py`), registriert fuer HF-Klasse `HYV3ForCausalLM`. Benoetigt zusaetzlich `MODEL_ARCH.HY_V3`-Enum, Tensor-Liste und Registrierung in `conversion/__init__.py`. Hy-v3 verwendet SIGMOID-Gating (nicht SQRT_SOFTPLUS).

### 4.3 Shared-Dateien mit erwarteten Aenderungen

- `src/llama-arch.h`
- `src/llama-arch.cpp`
- `src/llama-hparams.h`
- `src/llama-hparams.cpp`
- `src/llama-model.h`
- `src/llama-model.cpp`
- `src/llama-cparams.h` nur wenn von den Zielpfaden tatsaechlich benoetigte Felder fehlen.
- `src/llama-context.cpp` fuer Graph-Node-Budget ausschliesslich fuer NANBEIGE (`n_tokens * 40`-Zweig). QWEN4EXP, HY_V3, LAGUNA, MUSE_GLIMMER, QWEN3TTS, POCKETTTS benoetigen kein explizites Budget (Default-Zweig wie in Referenz).
- `src/llama-model-saver.cpp` fuer Qwen4Exp-Indexer-Metadaten-Roundtrip (6 Indexer-KV-Keys) und `supports_arch`-Blacklist-Eintraege fuer LAGUNA und MUSE_GLIMMER (`return false`).
- `src/models/models.h`
- `src/CMakeLists.txt` fuer `llama-memory-hybrid-idx.cpp`.
- `gguf-py/gguf/constants.py`
- `gguf-py/gguf/gguf_writer.py`
- `gguf-py/gguf/tensor_mapping.py` nur fuer wirklich neue HF-Tensornamen.
- `conversion/__init__.py`
- `convert_hf_to_gguf_update.py` fuer Laguna.

### 4.4 Nachgelagerte Modalitaets-Dateien

Muse Glimmer:
- Referenzimplementierung unter `tools/mtmd/models/`.
- Modell-Template nur dann uebernehmen, wenn es fuer den getesteten Checkpoint erforderlich ist.

Qwen3-TTS:
- `tools/mtmd/models/qwen3tts-gen.cpp`
- `tools/mtmd/models/qwen3tts-spkenc.cpp`

PocketTTS:
- `tools/mtmd/models/pockettts-gen.cpp`
- `tools/mtmd/models/pockettts-spkenc.cpp`
- `tools/mtmd/models/pockettts-seanet.cpp`

Die Audio-Writer-Keys und MTMD-Registrierungen werden erst in dieser Phase aufgenommen, nicht vorsorglich in den Kern-Port.

## 5. Phasenplan

## Phase 0 — Reproduzierbare Ausgangslage

**Ziel:** Eine belastbare Baseline vor dem ersten Port.

- [ ] Aktuellen Working Tree pruefen; vorhandene fremde/untracked Dateien nicht anfassen.
- [ ] Neuen Feature-Branch `feature/qwen4exp-model-family-port` erstellen.
- [ ] Referenz-Snapshot `035e22731` und gemeinsame Basis `4fc4ec554` dokumentieren.
- [ ] Referenz-Build und Referenzmodell unveraendert lassen.
- [ ] Referenz-GGUF-Provenienz dokumentieren: UD-IQ4_XS-GGUF von `unsloth/Qwen3.8-Flash-Next-GGUF`, Pfad auf Uranus-Shared-Model-Dir, Shard-Anzahl.
- [x] Pruefen, ob `llama_model_qwen3vl` bereits im Fork existiert — **BESTAETIGT:** `src/models/qwen3vl.cpp` und `LLM_ARCH_QWEN3VL` sind im Fork vorhanden. Qwen3-TTS ist nicht blockiert.
- [ ] Aktuellen TurboQuant-Fork mit CUDA bauen.
- [ ] Zusaetzlich CPU- und Vulkan-Baseline-Builds dokumentieren (Fork nutzt Vulkan produktiv auf Mars/Venus/Intel).
- [ ] CMake-Cache, CUDA-Linkage und Device-Erkennung gemaess Root-AGENTS verifizieren.
- [ ] Bestehende gezielte Tests fuer KV-Cache, Modell-Laden, GGUF und MoE ausfuehren und Baseline-Failures festhalten.
- [ ] Vorhandene Produktionsskripte und Services nicht aendern.

**Gate 0:** Branch sauber, CUDA-Build gruen, bekannte Baseline dokumentiert.

## Phase 1 — Minimaler GGUF-/Architektur-Vertrag

**Ziel:** Nur die Metadaten- und Tensor-Sprache schaffen, die die sieben Modelle wirklich verwenden.

### C++ Registry

- [ ] Sieben `LLM_ARCH_*`-Enums und Namen hinzufuegen.
- [ ] Nur die von Qwen4Exp benoetigten Hyper-Connection-, PLE-, QSA- und Compressor-KV-Keys hinzufuegen.
- [ ] Nanbeige-Keys `num_loops` und `skip_loop_final_norm` hinzufuegen.
- [ ] ~~Laguna-/Hy-v3-Gating-Funktion `SQRT_SOFTPLUS` hinzufuegen.~~ **STREICHEN:** Verifiziert, dass sowohl Laguna als auch Hy-v3 SIGMOID verwenden. `SQRT_SOFTPLUS` ist bereits im Codebase fuer DeepSeek4/DFlash vorhanden und wird von keiner Zielarchitektur benoetigt.
- [ ] Pre-existing Bug nachziehen: `SOFTMAX_WEIGHT` (C++ Enum-Wert 3) fehlt in Python `ExpertGatingFuncType` und im String-Mapping `llama_expert_gating_func_name`. Natuerlicher Zeitpunkt fuer den Fix.
- [ ] Nur die von Zielmodellen referenzierten Tensor-Enums, Namen und Tensor-Infos hinzufuegen.
- [ ] Keine Kimi-/DeepSeek-/MiniMax-/Dots-/Granite-Switch-Tensoren uebernehmen, nur weil sie im selben Referenz-Diffblock stehen.
- [ ] Architektur-Eigenschaften in `llama-arch.cpp` explizit setzen:
  - `LLM_ARCH_QWEN4EXP` in `llm_arch_is_hybrid()` aufnehmen (sonst wird `llama_memory_hybrid` ohne Indexer instanziiert).
  - `LLM_ARCH_QWEN3TTS` nicht in `llm_arch_is_hybrid`/`llm_arch_is_recurrent` (dichtes Attention-Modell, erbt Qwen3VL-Pfad).
  - `llm_arch_supports_sm_tensor` fuer alle sieben Architekturen pruefen und korrekt setzen (sonst crasht Tensor-Split).
  - RoPE-Zuordnungen pro Modell explizit aus der Referenz uebernehmen.

### GGUF Python

- [ ] Sieben `MODEL_ARCH`-Enums und Namen hinzufuegen.
- [ ] Nur benoetigte `Keys`, `MODEL_TENSOR`-Werte und `MODEL_TENSORS`-Listen hinzufuegen.
- [ ] Passende `GGUFWriter.add_*`-Methoden implementieren.
- [ ] Fork-spezifische DiffusionGemma-Writer-Methoden unveraendert erhalten.
- [ ] Converter registrieren und Alias-Kollisionen testen.
- [ ] HF-Klassennamen fuer alle sieben Architekturen in `conversion/__init__.py` (`TEXT_MODEL_MAP`) eintragen:
  - `Qwen4ExpForCausalLM` → `qwen4exp`
  - `MuseGlimmerForCausalLM` → `muse_glimmer`
  - `LagunaForCausalLM` → `laguna`
  - `NanbeigeForCausalLM` → `nanbeige`
  - `HYV3ForCausalLM` → `hunyuan` (bestehender Converter, `HYV3Model`-Klasse)
  - `Qwen3TTSForConditionalGeneration` → `qwen3tts`
  - `PocketTTSForConditionalGeneration` → `pockettts`
  - Exakte HF-Klassennamen vor Implementierung gegen Referenz-Converter verifizieren.

### Tests

- [ ] Python-Importtest fuer `gguf` und `conversion`.
- [ ] Registry-Roundtriptest: Architekturname → Enum → Name.
- [ ] Metadaten-Roundtrip fuer Qwen4Exp und Nanbeige.
- [ ] C++-Test `tests/test-llama-archs.cpp` um neue Arch-Namen ergaenzen.
- [ ] Laguna-Chat-Template-Tests aus Referenz (`tests/test-chat-auto-parser.cpp`, 11 Laguna-spezifische Tests) portieren.
- [ ] Keine Konvertierung eines grossen Modells in dieser Phase erzwingen; kleine synthetische Metadaten-Fixures bevorzugen.

**Gate 1:** Registry kompiliert und Python-Module importieren; keine fremden Architektur-Symbole eingeschleppt.

## Phase 2 — Qwen4Exp Kernintegration

**Ziel:** Das bereits vorhandene GGUF laden und deterministisch gegen den isolierten Referenz-Build vergleichen.

### Modell und Hparams

- [ ] `qwen4exp.cpp` aus dem gepinnten Referenzstand uebernehmen.
- [ ] `llama_model_qwen4exp` in `models.h` deklarieren und in der Factory registrieren.
- [ ] `llama-model.cpp` Factory: `case LLM_ARCH_QWEN4EXP: return new llama_model_qwen4exp(params);`
- [ ] Qwen4Exp-Hparams minimal in `llama-hparams.*` integrieren:
  - Hyper-Connections.
  - PLE/N-Gram-Konfiguration.
  - QSA-Indexer-Konfiguration.
  - Output-Gruppen/LoRA-Rank.
  - Compressor-Ratios und RoPE-Basis.
- [ ] Nur die Qwen4Exp-Layer-Felder in `llama_layer` (`src/llama-model.h`) aufnehmen.
- [ ] `src/llama-model.cpp` Tensor-Laden fuer neue Layer-Felder ergaenzen.
- [ ] `src/llama-model-saver.cpp`: 6 Indexer-KV-Writes uebernehmen (`ATTENTION_INDEXER_HEAD_COUNT/KEY_LENGTH/TOP_K/BLOCK_SIZE/LOCAL_BLOCKS/TYPES`).
- [ ] `TENSOR_ALLOW_RESHAPE` ist ein neues Symbol aus dem Unsloth-Snapshot (0 Matches im aktuellen Fork). Nur uebernehmen, wenn ein Qwen4Exp-Callsite es benoetigt; Verhalten durch Loader-Test absichern.

### QSA Memory

- [ ] `llama_memory_hybrid_idx` und `llama_memory_hybrid_idx_context` uebernehmen.
  - Vererbung: `llama_memory_hybrid_idx` erbt von `llama_memory_hybrid`; `llama_memory_hybrid_idx_context` erbt von `llama_memory_hybrid_context`.
  - Drei Subcontexts: `attn` (geerbt), `recr` (geerbt), `idx` (neu, `get_idx()` liefert `const llama_kv_cache_context *`).
  - `qwen4exp.cpp` nutzt `static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx)` fuer Downcast.
- [ ] `llama-model.cpp` Memory-Factory: `needs_mem_idx = (arch == LLM_ARCH_QWEN4EXP)` und `filter_idx`-Callback nur fuer QWEN4EXP mit `indexer_head_size > 0`. Konstruktion von `new llama_memory_hybrid_idx(...)` im `needs_mem_idx`-Pfad.
- [ ] Slot-Info-Garantie: `init_batch()` kopiert `heads_attn` nach `heads_idx` (Referenz-Zeile 113-115). Getrennt driftende Zellzuordnung ist unzulaessig; Assertion in `qwen4exp.cpp:284` verifiziert `mctx_idx->get_n_kv() == mctx->get_attn()->get_n_kv()`.
- [ ] **KEINE** `llama-context.cpp`-Cast-Erweiterung noetig: Vererbung reicht. Der vorhandene `build_inp_mem_hybrid()`-Cast in `llama-graph.cpp` funktioniert unverändert, da `hybrid_idx_context` IS-A `hybrid_context`.
- [ ] TurboQuant-Methoden: Vererbung allein genuegt. `llama_memory_hybrid_context` delegiert bereits `get_turbo_rot_forward()`, `get_turbo_rot_inverse()`, `get_turbo_innerq_scale_inv()` an `ctx_attn`. `hybrid_idx_context` erbt diese automatisch. Keine explizite Delegation erforderlich.
- [ ] Expected-Attention- und Intel-Hybrid-Attention-Code nicht ersetzen.

### Converter

- [ ] `conversion/qwen4exp.py` integrieren.
- [ ] Text- und Conditional-Generation-Klassen registrieren.
- [ ] ~~Vision-Teil nur gegen bereits vorhandene Qwen3-VL-Bausteine verdrahten; keine fremden Vision-Backends kopieren.~~ **STREICHEN:** Qwen4Exp hat keine Vision-Komponente (nur QSA, GDN, PLE, Hyper-Connections, MoE). Dies war ein Copy-Paste-Rest.

### Korrektheit

- [ ] Modell-Metadaten und alle vorhandenen GGUF-Shards (typisch 1–3, je nach Quantisierung) gleichzeitig laden.
- [ ] Tensoranzahl, Tensorformen, Layerzahl, Expert-Anzahl und PLE-Groesse mit Referenz vergleichen.
- [ ] Deterministischen Kurzprompt mit identischem Seed/Sampling gegen Referenz ausfuehren. Vergleich auf gemeinsame KV-Typen (z.B. `f16`) beschraenken; der Referenz-Branch hat kein TurboQuant (`turbo3`/`turbo4`). TurboQuant-KV separat als Fork-Optimierung evaluieren.
- [ ] 1K-, 8K- und mindestens ein laengeres Kontext-Szenario pruefen.
- [ ] QSA dense/sparse Umschaltung und Indexer-Cache ueber mehrere Eval-Aufrufe testen.
- [ ] Cache-Clear, Sequenz-Freigabe und erneute Inferenz testen.
- [ ] MoE-Cache waehrend dieser Phase deaktiviert lassen, um die Architektur-Baseline nicht zu vermischen.

**Gate 2:** Qwen4Exp laedt, erzeugt plausible und gegen die Referenz reproduzierbare Ausgabe, QSA-Indexer driftet nicht, Sanitizer/Assertions bleiben ruhig.

## Phase 3 — Leichte Textarchitekturen, einzeln integriert

**Ziel:** Jede Architektur isoliert hinzufuegen; nach jeder Architektur Build und Smoke-Test.

### 3A Muse Glimmer

- [ ] Modelldatei und Converter integrieren.
- [ ] SWA-Pattern, QK-Norm, Attention-Gate und Logit-Softcap pruefen.
- [ ] Text-only GGUF laden und deterministischen Smoke-Test ausfuehren.
- [ ] Multimodale Pipeline noch nicht voraussetzen.

### 3B Laguna

- [ ] Modelldatei und Converter integrieren.
- [ ] Sigmoid-/Sqrt-Softplus-Routing, Shared Expert, Score-Correction-Bias und SWA/Global-Layer pruefen.
- [ ] MoE-Fallback ohne CUDA-MoE-Cache testen.
- [ ] Consumer-tauglichen Checkpoint als spaeteren Laufzeittest vormerken.

### 3C Nanbeige

- [ ] Modelldatei und Converter integrieren.
- [ ] Physische Layer versus `n_loops`-expandierte Ausfuehrung pruefen.
- [ ] Final-Norm-Verhalten pro Loop testen.
- [ ] Graph-Node-Budget in `llama-context.cpp`: NANBEIGE zum `n_tokens * 40`-Zweig hinzufuegen (Referenz Zeile 2306). Fork hat derzeit nur `QWEN3NEXT, KIMI_LINEAR, QWEN35, QWEN35MOE`.

### 3D Hy-v3

- [ ] Modelldatei (`hy-v3.cpp`, 394 Zeilen) und `llama_model_hy_v3 : public llama_model_base` in `models.h` registrieren.
- [ ] Converter: `HYV3Model` in `conversion/hunyuan.py` (nicht `HunYuanMoEModel`), registriert fuer `HYV3ForCausalLM`. Hy-v3 verwendet SIGMOID-Gating.
- [ ] `MODEL_ARCH.HY_V3`-Enum, Tensor-Liste (27 Tensor-Typen inkl. `NEXTN_*`) und Registrierung in `conversion/__init__.py`.
- [ ] NextN-/MTP-Layerfilter in die bestehende Fork-Logik integrieren (`n_layer_nextn`, `graph_mtp`).
- [ ] Target-only zuerst testen, danach optional vorhandenen NextN-Head.
- [ ] Keine neue MTP-Infrastruktur parallel zur bestehenden einfuehren.

### 3E Qwen3-TTS Backbone

- [ ] Drei-Zeilen-Erbmodell auf `llama_model_qwen3vl` integrieren (bewusst korrekt: Converter faltet Text-Projection-MLP in Embedding, haengt Codec-Vocab an `TOKEN_EMBD`, schreibt `codec_head` als `OUTPUT`).
- [ ] **Blocker geprueft:** Fork enthaelt `llama_model_qwen3vl` bereits (`src/models/qwen3vl.cpp`, `LLM_ARCH_QWEN3VL`). Qwen3-TTS ist nicht blockiert.
- [ ] Talker-Converter (`Qwen3TTSTalkerModel` fuer `Qwen3TTSForConditionalGeneration`) und Architektur-Aliase registrieren.
- [ ] Konvertierungstest: `TOKEN_EMBD`-Shape = `[n_embd, text_vocab + codec_vocab]`, `OUTPUT`-Shape = `[n_embd, codec_vocab]`.
- [ ] Vocab-Roundtrip: Codec-Token-IDs (`codec_bos_id`, `codec_eos_token_id`, `codec_language_*_id`) korrekt.
- [ ] Suppress-Tokens-Test: GGUF-Metadaten enthalten korrekte Suppress-Liste.
- [ ] Text-/Code-Token-Forward-Pass ohne Audio-Decoder testen.

### 3F PocketTTS Backbone

- [ ] Modelldatei und Converter integrieren.
- [ ] Tied Output und fehlenden klassischen `lm_head` korrekt behandeln.
- [ ] Text-/Conditioning-Forward-Pass ohne Flow-/Mimi-Decoder testen.

**Sub-Gates 3A–3F:** Nach jeder Architektur: kompiliert, laedt (falls Checkpoint verfuegbar), keine Regression in bereits gruenen Architekturen. Ein Fehler in 3A kaskadiert nicht in 3B–3F.

**Gate 3:** Alle sechs Architekturen kompilieren; jede verfuegbare kleine/Consumer-taugliche Architektur laedt; fehlende Enterprise-Checkpoints werden als nicht laufzeitvalidiert markiert, nicht als erfolgreich behauptet.

## Phase 4 — Multimodal und Audio End-to-End

**Ziel:** Muse Glimmer, Qwen3-TTS und PocketTTS nicht bei einem formal ladbaren Backbone stehen lassen.

### Muse Glimmer

- [ ] MTMD-Modell registrieren: `tools/mtmd/models/muse-glimmer.cpp`, `tools/mtmd/models/models.h` (`clip_graph_*`-Klasse), `tools/mtmd/CMakeLists.txt`, `tools/mtmd/clip.cpp` Dispatch-Switch.
- [ ] `gguf-py/gguf/constants.py`: `VISION_PROJECTOR_TYPE` fuer Muse Glimmer. `gguf-py/gguf/gguf_writer.py`: `add_*`-Methoden.
- [ ] Vorhandene Vision-Bausteine wiederverwenden; keine fremden Vision-Backends kopieren.
- [ ] Text-only Regressionstest nach MTMD-Integration wiederholen.
- [ ] Bild→Text-Smoke-Test mit kleinem reproduzierbarem Bild.

### Qwen3-TTS

- [ ] Code-Predictor und Speaker-Encoder integrieren: `tools/mtmd/models/qwen3tts-gen.cpp`, `tools/mtmd/models/qwen3tts-spkenc.cpp`, `tools/mtmd/models/models.h`, `tools/mtmd/CMakeLists.txt`, `tools/mtmd/clip.cpp` Dispatch.
- [ ] `gguf-py/gguf/constants.py`: `AUDIO_PROJECTOR_TYPE` fuer Qwen3-TTS (`QWEN3TTS_SPKENC`, `QWEN3TTS_GEN`). `gguf-py/gguf/gguf_writer.py`: Audio-Writer-Methoden.
- [ ] Audio-spezifische GGUF-Keys und Writer-Funktionen minimal hinzufuegen.
- [ ] Text→Codes, Codes→Waveform und Speaker-Conditioning getrennt testen.
- [ ] End-to-End-WAV erzeugen und strukturell validieren; Audioqualitaet separat pruefen.

### PocketTTS

- [ ] Flow-Matching-Generator, Speaker-Encoder und SeaNet/Mimi-Decoder integrieren: `tools/mtmd/models/pockettts-gen.cpp`, `tools/mtmd/models/pockettts-spkenc.cpp`, `tools/mtmd/models/pockettts-seanet.cpp`, `tools/mtmd/models/models.h`, `tools/mtmd/CMakeLists.txt`, `tools/mtmd/clip.cpp` Dispatch.
- [ ] `gguf-py/gguf/constants.py`: `AUDIO_PROJECTOR_TYPE` fuer PocketTTS (`POCKETTTS_SPKENC`, `POCKETTTS_GEN`). `gguf-py/gguf/gguf_writer.py`: Audio-Writer-Methoden.
- [ ] MTMD-Registrierung und Audio-Writer-Keys hinzufuegen.
- [ ] Conditioning, Flow-Schritte und Waveform-Ausgabe getrennt testen.
- [ ] End-to-End-WAV erzeugen und strukturell validieren; Audioqualitaet separat pruefen.

**Gate 4:** Muse Glimmer verarbeitet Bild+Text; beide TTS-Pipelines erzeugen gueltige Audioartefakte. Falls Referenzgewichte fehlen, bleibt die jeweilige Modalitaet explizit als blockiert markiert.

## Phase 5 — Build- und Backend-Matrix

**Ziel:** Keine Architektur durch einen CUDA-only Erfolg als allgemein integriert deklarieren.

### Dev-Builds

- [ ] CUDA-Release-Build; Backend-Flags, Linkage und Devices pruefen.
- [ ] CPU-Build fuer generische Graph-/Loader-Kompatibilitaet.
- [ ] Vulkan-Release-Build mindestens kompilieren und Shader-Generierung pruefen.

### Laufzeittests

- [ ] Dual-Ada-Testsystem: Qwen4Exp und groessere CUDA-Modelle (z.B. Gemma-4-26B-A4B QAT als CUDA-Regression).
- [ ] Ampere-Dev-System: nur Builds und kleine CPU-/GPU-Smokes gemaess GPU-Nutzungsregeln.
- [ ] Pascal-System (8GB VRAM): Nanbeige und kleine TTS-Backbones; Qwen4Exp mit 512 Experten passt nicht in 8GB VRAM. Keine FA-Annahme (Pascal hat keine Tensor Cores).
- [ ] RDNA3-Vulkan-System: Nanbeige, kleine TTS-Modelle und optional Muse Glimmer, sofern Speicher reicht.
- [ ] Intel-Vulkan-System: nur kleine Modelle; Intel-Hybrid-Attention auf Regression pruefen.

### Pflichtchecks je Backend

- [ ] Modell laden und ersten Token erzeugen.
- [ ] Prefill und Decode getrennt erfassen.
- [ ] Cache-Clear und zweiter Request.
- [ ] Kein stiller CPU-Fallback beim GPU-Test.
- [ ] Keine neuen Warnungen, Assertions oder NaNs.

**Gate 5:** CUDA-Build und Qwen4Exp-Laufzeit gruen; CPU/Vulkan mindestens kompilieren; kleine Modelle auf geeigneten Backends verifiziert oder mit konkreter Blockade dokumentiert.

## Phase 6 — Regression und Fork-Vertraeglichkeit

**Ziel:** Bestehende Fork-Funktionen nach dem Shared-Code-Merge beweisbar erhalten.

- [ ] Gemma-4 Hauptmodell laedt mit TurboQuant-KV.
- [ ] Gemma-4 Assistant/MTP kompiliert und vorhandener Smoke-Test bleibt gruen.
- [ ] Qwen3 NextN kompiliert und vorhandener Smoke-Test bleibt gruen.
- [ ] DiffusionGemma-Registry/Writer bleiben importierbar.
- [ ] Expected Attention kompiliert und KV-Cache-Methoden bleiben vorhanden.
- [ ] UBBoost-Felder und Prefill-Ubatch-Pfad bleiben erhalten.
- [ ] Intel Hybrid-Attention-Codepfad bleibt unveraendert aktivierbar.
- [ ] Vulkan turbo3/turbo4 Shader-Generierung bleibt gruen.
- [ ] Bestehender MoE-Cache kompiliert, bleibt aber fuer Korrektheitstests deaktiviert.
- [ ] C++-Tests `test-turbo-quant` und `test-expected-attention` laufen gruen.
- [ ] Qwen4Exp-spezifischer KV-/Memory-Test: Slot-Info-Drift zwischen Attention und Indexer, Indexer-Replay ueber mehrere Eval-Aufrufe.
- [ ] `LLAMA_MAX_EXPERTS` bleibt 512, sofern kein Zielcheckpoint nachweislich mehr benoetigt.

**Gate 6:** Keine Regression in den genannten Fork-Vertraegen.

## Phase 7 — Qwen4Exp Leistungsbaseline

**Ziel:** Erst die korrekte neue Engine gegen den isolierten Referenz-Build bewerten.

### Vergleichskonfiguration

- Gleiches GGUF, gleiche Shards, gleicher Kontext, gleiche GPU-Auswahl, gleiche Sampling-Parameter.
- KV-Typen: Vergleich auf gemeinsame Typen (`f16`) beschraenken. Der Referenz-Branch hat kein TurboQuant (`turbo3`/`turbo4`); TurboQuant-KV separat als Fork-Optimierung evaluieren.
- MoE-Cache aus.
- thecodacus-Pinning/Prefetch aus, sofern im Vergleichsbinary vorhanden.
- Warmup und mindestens drei Messwiederholungen.

### Messpunkte

- [ ] Startzeit und Ladezeit.
- [ ] CPU-RSS und GPU-Modellbuffer pro Device.
- [ ] Prefill bei kurzen, 1K-, 2K-, 3K- und 8K-Prompts.
- [ ] Decode mit mindestens 200–500 Ausgabetokens.
- [ ] Stabilitaet bei wiederholten Requests.
- [ ] QSA-Indexer-/Hybrid-Memory-Verhalten bei wachsendem Kontext.
- [ ] Output-/Logit-Korrektheit gegen Referenz.

**Gate 7:** Der TurboQuant-Port ist funktional mindestens gleichwertig zum isolierten Referenz-Build; Leistungsabweichungen sind erklaert und reproduzierbar.

## Phase 8 — Dokumentation und Merge-Vorbereitung

- [ ] `FORKS.md`: neue Modellfamilien und Quellprovenienz dokumentieren.
- [ ] `CHANGELOG.md`: erst nach implementiertem, verifiziertem Stand aktualisieren.
- [ ] `src/AGENTS.md`: neue fork-spezifische Modell-Dateien und QSA-Memory aufnehmen.
- [ ] Root-`AGENTS.md`: Zweck nur aktualisieren, wenn Support tatsaechlich integriert ist.
- [ ] `docs/fork/ROADMAP.md`: #99 phasengenau aktualisieren.
- [ ] Oeffentliche Doku frei von Hostnamen, internen Pfaden und privaten Details halten.
- [ ] Test- und Benchmark-Ergebnisse in die vorgesehene Benchmark-Dokumentation/Trilium-Note eintragen, nicht in Modell-Info-Notes.
- [ ] Feature-Branch in logisch getrennten Commits halten; kein einzelner Riesencommit.

**Gate 8:** Code, Tests, DOX und Betriebsdoku sind konsistent; Merge-Review kann beginnen.

## Phase 9 — Spaeterer MoE-Cache-Vergleich, nicht Merge-Blocker

Diese Phase beginnt erst nach Gate 8 und ist kein Bestandteil der Architektur-Korrektheit.

Vergleich:

1. Baseline ohne Cache.
2. TurboQuant dynamischer LRU-Cache.
3. TurboQuant mit persistiertem Hot-Set und deaktivierter laufender Eviction/Backfill, falls sich ein statischer Modus ohne Codeumbau konfigurieren laesst.
4. Falls erforderlich: separater echter statischer Profil-/Slot-Prototyp nach dem in der Trilium-Notiz beschriebenen Prinzip.

Zu messen:

- Decode-Rate und Latenzverteilung.
- Hit-Rate pro Layer und Expert-Rolle.
- H2D-Bytes und Insert-/Eviction-Rate.
- CPU/GPU-Overlap.
- Warm- versus Cold-Start.
- Chat-, Coding- und gemischte Prompts.
- Nutzen bei verschiedenen VRAM-Budgets.

Entscheidungsregel: Der dynamische Cache wird nicht als fortschrittlicher angenommen. Er bleibt nur aktiv, wenn er gegen die statische Alternative messbar gewinnt oder mindestens gleich schnell bei besserer Robustheit ist.

## 6. Review-Loop

Jede komplexe Phase endet mit zwei unabhaengigen Reviews, maximal zwei parallel:

1. `review-swe`: Korrektheit, Call-Graph, Speicher-/Lifetime-Fehler, Build-Risiken.
2. `review-glm`: Architektur-Konsistenz, Fork-Regressionen, Scope-Leaks und Testluecken.

Review-Gates:

- Nach Phase 1: Registry/GGUF-Vertrag.
- Nach Phase 2: Qwen4Exp und `hybrid_idx`.
- Nach Phase 4: MTMD-/Audio-Pipeline.
- Nach Phase 6: Gesamtregression.
- Vor Merge: finaler Diff-Review gegen Scope und Sicherheitsregeln.

Jeder Befund wird entweder behoben oder mit begruendeter Entscheidung im Plan dokumentiert. Kein Produktionsmerge bei offenem P0/P1-Befund.

## 7. Commit-Struktur

Vorgesehene kleine, reviewbare Commits:

1. `feat: GGUF-Registry fuer neue Modellfamilien`
2. `feat: Qwen4Exp Hparams und Tensorvertrag`
3. `feat: Qwen4Exp QSA-Hybrid-Memory`
4. `feat: Qwen4Exp Modell und Converter`
5. `feat: Muse Glimmer und Laguna Modell-Support`
6. `feat: Nanbeige und Hy-v3 Modell-Support`
7. `feat: Qwen3-TTS und PocketTTS Backbones`
8. `feat: Muse-Glimmer Multimodal-Pipeline`
9. `feat: Qwen3-TTS Audio-Pipeline`
10. `feat: PocketTTS Audio-Pipeline`
11. `test: Modellfamilien und Fork-Regressionen`
12. `docs: Qwen4Exp und Modellfamilien dokumentieren`

Die genaue Aufteilung darf waehrend der Implementierung kleiner werden; Architektur-Registry, QSA-Memory und Audio-Pipelines sollen nicht in einem unreviewbaren Commit vermischt werden.

## 8. Abbruch- und Rollback-Kriterien

- Shared-Code-Merge entfernt oder umgeht Fork-spezifische TurboQuant-/MTP-/EA-Felder.
- Qwen4Exp-Output weicht bei identischen Bedingungen unerklärt von der Referenz ab.
- QSA-Indexer und Attention-Cache verwenden unterschiedliche Slot-/Cell-Zuordnungen.
- Wiederholbare NaNs, Assertions, Speicherfehler oder Cache-Korruption.
- Eine Zusatzarchitektur erzwingt die Portierung der ausgeschlossenen Enterprise-Infrastruktur.
- Audio-/Vision-Pipeline benoetigt einen wesentlich groesseren, ungetesteten MTMD-Refactor.

Bei einem Abbruch bleibt der letzte gruene Commit erhalten. Die betroffene Architektur wird auf einen separaten Feature-Branch oder ein spaeteres ROADMAP-Item verschoben; bereits gruene, unabhaengige Architekturen werden nicht mit verworfen.

## 9. Definition of Done

Der Plan ist abgeschlossen, wenn:

- [ ] Alle sieben Architektur-IDs und Converter sauber registriert sind.
- [ ] Qwen4Exp das vorhandene UD-IQ4_XS-GGUF im TurboQuant-Fork korrekt laedt und ausfuehrt.
- [ ] Qwen4Exp gegen den isolierten Referenz-Build funktional verglichen wurde.
- [ ] Muse Glimmer, Laguna und Nanbeige mit realen GGUFs oder klar dokumentierten Modellverfuegbarkeits-Blockaden getestet wurden.
- [ ] Hy-v3 kompiliert und der bestehende NextN-/MTP-Pfad nicht regressiert.
- [ ] Qwen3-TTS und PocketTTS mindestens ihre Backbones laden; End-to-End-Audio ist gruen oder mit fehlenden Referenzartefakten konkret blockiert.
- [ ] CUDA gruen und CPU/Vulkan mindestens build-validiert sind.
- [ ] Bestehende TurboQuant-, MTP-/NextN-, DiffusionGemma-, Expected-Attention-, UBBoost- und Intel-Hybrid-Attention-Pfade nicht regressiert sind.
- [ ] Alle Review-Gates geschlossen sind.
- [ ] DOX, FORKS, CHANGELOG und ROADMAP dem tatsaechlichen Implementierungsstand entsprechen.
- [ ] Keine Produktionskonfiguration ohne separate Freigabe veraendert wurde.

## 10. Review-Verlauf

### Review-Runde 1 (zwei unabhaengige Subagents)

- **Review-SWE (5136c11a):** 4 P0, 9 P1, 5 P2. Hauptschauplaetz: QSA-Memory-Verdrahtung unterspezifiziert, `llama-context.cpp` dynamic_cast-Kette, Qwen3-TTS-Vererbung, Converter-/MTMD-/Arch-Registry-Dateien fehlen.
- **Review-GLM (c677e33b):** 0 P0, 5 P1, 6 P2. Hauptschauplaetz: `SQRT_SOFTPLUS`-Integrationsflaeche, `llama-graph.cpp` fehlt, Qwen4Exp Graph-Node-Budget, `llm_arch_is_hybrid`-Registrierung.

### Verifizierung gegen Referenzstand `035e22731` (explore-exec Subagent)

Strittige Befunde wurden direkt gegen den funktionierenden Referenzcode verifiziert:

| Befund | Ergebnis | Korrektur |
|--------|----------|-----------|
| `hybrid_idx` Vererbung | VERIFIZIERT | Erbt von `hybrid`; drei Subcontexts (attn/recr/idx) |
| `llama-context.cpp` dynamic_cast | WIDERLEGT | Nicht noetig; Vererbung reicht |
| Turbo-Delegation | VERIFIZIERT | Vererbung allein genuegt |
| Qwen3-TTS erbt von `qwen3vl` | VERIFIZIERT | Bewusst korrekt; Fork muss `qwen3vl` enthalten |
| Hy-v3 `sqrt_softplus` | WIDERLEGT | Beide (Hy-v3 + Laguna) verwenden SIGMOID |
| `SQRT_SOFTPLUS`-Integrationsflaeche | WIDERLEGT | Streichen; weder Laguna noch Hy-v3 nutzen es |
| Qwen4Exp Graph-Node-Budget | WIDERLEGT | QWEN4EXP nicht in Budget-Liste; Default-Zweig |
| Nanbeige Graph-Node-Budget | VERIFIZIERT | NANBEIGE in `n_tokens*40`-Liste; muss in Fork |
| `llm_arch_is_hybrid`-Registrierung | VERIFIZIERT | Muss fuer QWEN4EXP hinzugefuegt werden |
| `llama-model-saver.cpp` | TEILWEISE | QWEN4EXP: 6 Indexer-Keys; LAGUNA+MUSE_GLIMMER: Blacklist |
| Laguna-Chat-Template-Tests | VERIFIZIERT | 11 Tests in Referenz; zu portieren |
| MTMD-Registry | VERIFIZIERT | Nur Qwen3TTS/PocketTTS/MuseGlimmer (Phase 4) |

### Eingearbeitete Korrekturen

- P0: `llama-model.cpp` Factory/Memory-Verdrahtung spezifiziert; QSA-Indexer-Design detailliert; `llama-context.cpp`-Cast-Erweiterung gestrichen (Vererbung reicht); Turbo-Delegation auf Vererbung vereinfacht.
- P1: `SQRT_SOFTPLUS` komplett gestrichen (WIDERLEGT); `llm_arch_is_hybrid`/`supports_sm_tensor` explizit; Converter-HF-Klassennamen aufgelistet; `llama_layer`+Saver spezifiziert; Sub-Gates 3A–3F; MTMD-Dateien explizit; C++-Tests erweitert; Referenzvergleich auf `f16` beschraenkt.
- P2: GGUF-Provenienz in Phase 0; Vulkan/CPU-Baseline; Pascal-VRAM explizit; `TENSOR_ALLOW_RESHAPE` als neu markiert; Vision-Bezug in Phase 2 gestrichen; `SOFTMAX_WEIGHT` pre-existing Bug; GGUF-Shards praezisiert; konkretes CUDA-Regessionsmodell.
