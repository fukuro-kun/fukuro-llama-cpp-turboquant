# CHANGELOG — fukuro-llama-cpp-turboquant

> **HISTORISCH.** Historischer Port-/Verbesserungsbericht (bis ca. Ende
> September 2026). Die **aktive** Chronologie ist `../../CHANGELOG.md`
> (Repository-Root); Eintraege hier werden nicht fortgeschrieben.

Alle nennenswerten Änderungen am Fork. Format: Datum → Commit → Beschreibung.

---

## 2026-08-27 — Qwen4Exp-Modellfamilien-Port (#99)

**Session:** InferenzQuelle-Verwaltungssession (Verantwortlich: InferenzQuelle-Agent)
**Branch:** `feature/qwen4exp-model-family-port`
**Quelle:** Unsloth-qwen4exp-Snapshot `035e22731` auf llama.cpp-Basis `4fc4ec554`
**Plan:** `docs/fork/plans/SESSION_PLAN_20260827_qwen4exp-model-family-port.md`

### Zusammenfassung

Port von 7 neuen Modellarchitekturen in den TurboQuant-Fork als selektiver
Symbol-Port (kein Cherry-Pick des großen Unsloth-Commits). Qwen4Exp als
Hauptziel mit vollständiger Kernintegration. 6 weitere Architekturen als
Text-Backbone. Muse Glimmer Vision als vollständiger MTMD-Port.

### Commits (16 Commits, +4721 Zeilen, 37 Dateien)

#### Phase 0: Planung & Registry

| Commit | Beschreibung |
|--------|--------------|
| `601ab32cc` | docs: Qwen4Exp Modellfamilien-Port Plan + ROADMAP #99 |
| `cc58fb8c3` | arch: registry for qwen4exp + 6 model family architectures |
| `cb969302a` | hparams: add qwen4exp hparams (PLE, HC, indexer, rope_pattern) |
| `5c483c6fb` | model.h: add qwen4exp layer fields (HC, PLE, indexer) |
| `82e202d2b` | python: registry for qwen4exp + 6 model family architectures |

#### Phase 1-2: Qwen4Exp Kernintegration

| Commit | Beschreibung |
|--------|--------------|
| `9107fc086` | feat: Qwen4Exp Kernintegration — Model, Graph, Memory-Hybrid-Idx |
| `63d6bbfb1` | fix: Qwen4Exp PLE-Layers-Laden korrigiert + get_arr Template-Instanziierung |
| `a37837b15` | fix: PLE-Arrays via get_arr laden statt get_key_or_arr mit Längenvalidierung |
| `1872700a9` | fix: get_arr unterstützt UINT64-Arrays für Qwen4Exp PLE-Metadaten |
| `15ce98a05` | fix: PLE-Token-Embedding-Shape aus GGUF lesen statt zu berechnen |
| `2dbc9faa3` | fix: Qwen4Exp verwendet llama_memory_hybrid_idx statt llama_memory_hybrid |
| `6a4c2a986` | fix: PLE n-gram-Hash korrigiert (XOR-Summe über alle Tokens) + & → && |

#### Phase 3: 6 weitere Architekturen (Text-Backbone)

| Commit | Beschreibung |
|--------|--------------|
| `4b344651c` | feat: Muse Glimmer, Laguna, Nanbeige, Hy-v3, Qwen3-TTS, PocketTTS Backbones |

#### Phase 4a: Muse Glimmer Vision Port

| Commit | Beschreibung |
|--------|--------------|
| `4479f00fb` | feat: Muse Glimmer Vision Port — ViT mit sparse window attention + pixel-shuffle |

#### Phase 5-8: Build, Regression, Doku

| Commit | Beschreibung |
|--------|--------------|
| `1da023915` | docs: Phase 5-7 Status aktualisiert — Build-Matrix, Regression, Baseline |
| `4a2f40c50` | docs: src/AGENTS.md um neue Modell-Dateien ergänzt |
| `778fb3c72` | fix: CPU get_rows unterstützt TurboQuant-Typen (TURBO2_0/3_0/4_0) |

### Was portiert wurde

**Qwen4Exp (vollständig):**
- Architektur-Registry, Hparams, Layer-Fields, Modell-Factory
- QSA (Query Semantic Attention), GDN, PLE (Persistent Layer Embeddings)
- Hyper-Connections, MoE mit 512 Experten
- `llama_memory_hybrid_idx` für QSA-Memory
- GGUF-Python-Converter und Writer
- 7 Bugfixes für PLE-Laden, n-gram-Hash, Memory-Typ, UINT64-Arrays

**6 weitere Architekturen (Text-Backbone):**
- Muse Glimmer, Laguna, Nanbeige, Hy-v3, Qwen3-TTS, PocketTTS
- Jeweils: Architektur-Registry, Hparams, Modell-Factory, Graph
- GGUF-Python-Registry

**Muse Glimmer Vision (vollständiger MTMD-Port):**
- `clip_graph_muse_glimmer` — 50-Layer ViT mit 2D RoPE, sparse window attention
- `PROJECTOR_TYPE_MUSE_GLIMMER` — Enum, String, 8 Switch-Cases
- `build_vit_opts` erweitert: `attn_mask_layers`, `callback_layer_out`, `skip_post_ln`
- `mtmd_image_preprocessor_muse_glimmer` — Aspect-Ratio-preserving Grid-Size-Resize
- `RESIZE_ALGO_LANCZOS` — Lanczos-3 Filter (sinc×sinc, radius 3.0) aktiviert
- Token-Zählung nutzt `params.n_merge` statt hartkodiert `/2`
- Divisibilitäts-Assert für `grid_w/grid_h % n_merge`

**Bugfix (TurboQuant-spezifisch):**
- CPU `get_rows` unterstützt TurboQuant-Typen `TURBO2_0`, `TURBO3_0`, `TURBO4_0`
  (war zuvor nur für Q4_0/Q8_0 implementiert, brachte Qwen4Exp-Tests zum Absturz)

### Build-Status

- Build: Grün (CUDA, Vulkan, CPU)
- Tests: 614/614 grün (turbo-quant, expected-attention + 612 weitere)
- MTMD-Target: Gebaut
- Review: review-swe für Muse Glimmer durchgeführt, P0+P1 gefixt

### Was NICHT portiert wurde (bewusst)

- **Kein Cherry-Pick von `035e22731`** — selektiver Symbol-Port statt großem Merge
- **Keine MTMD-Audio-Infrastruktur** — `llama-kv-cache-dsv4.*`, `llama-kv-cache-msa.*`,
  `lightning-indexer.cu`, `dsv4-hc.cu` nicht portiert (nur für Audio-Modelle benötigt)
- **Keine Qwen3-TTS/PocketTTS Audio-Generierung** — nur Text-Backbone portiert;
  vollständige Audio-Pipeline als separate Evaluations-Phase geplant
- **Keine `bailingmoe3`, `kimi-k3`, `minimax-01`, `minimax-m3`, `dots3note`,
  `granite-switch`** — nicht LAN-relevant
- **Kein `LLAMA_MAX_EXPERTS`-Raise auf 1024** — 512 reicht für alle portierten Modelle
- **Keine thecodacus Host-Pinning/Prefetch-Merge** — separater Plan
- **Keine MoE-Cache-Optimierung** — erst nach Qwen4Exp-GPU-Baseline benchmarken

### Offen (Phase 4b, 9)

- **Phase 4b: TTS-Entscheidung** — Qwen3-TTS/PocketTTS standalone evaluieren
  (Probe-Hören, Vergleich mit xtts-api und Pandora), dann entscheiden ob
  Audio-Pipeline in Fork integriert wird. Handoff:
  `docs/HANDOFF-tts-evaluation.md` (lokal, unversioniert)
- **Phase 9: MoE-Cache-Vergleich** — dynamischer TurboQuant-MoE-Cache vs
  statisches Profil-/Slot-Caching. Nicht Merge-blockierend.

---

## 2026-08-22 und früher

Siehe `git log` für frühere Commits. Letzter Commit vor dieser Session:
`2c9725c3d` (2026-08-22) — docs: Prefetch-Env-Vars in Skripten kommentiert.
