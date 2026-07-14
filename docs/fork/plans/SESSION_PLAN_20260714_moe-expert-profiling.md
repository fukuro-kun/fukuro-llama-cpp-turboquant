# SESSION_PLAN: #62 MoE Expert Profiling & REAP Pruning

**Erstellt:** 2026-07-14
**Status:** ✅ Implementiert und verifiziert
**ROADMAP-Item:** #62 MoE Expert Profiling & REAP Pruning (Tier 2, 1-2 Tage)
**Quelle:** PR #20454 (srossitto79, OPEN, +1248 Zeilen, 10 Dateien)
**Paper:** REAP — arXiv:2510.13999 (Cerebras Research)

## Session-Ziel

Portiere den REAP Expert Profiler (C++ Tool) und die GGUF-Pruning-Skripte (Python) von PR #20454 in den Fork. Das Tool sammelt REAP-Saliency-Scores wahrend der Inferenz und kann GGUF-Dateien prunen (Experten mit niedrigstem Score entfernen). Komplementar zu unserer MoE-Freq-Tracking-Infrastruktur aus #40.

## Architektur

### REAP Score (Eq. 9 des Papers)
```
REAP(j) = mean_{t : j in topk(t)} [ g_j(t) * ||f_j(t)||_2 ]
```
- `g_j(t)` = Router gate weight (post-softmax) fur Experte j bei Token t
- `f_j(t)` = Expert FFN output (pre-weighting, ffn_moe_down)
- `||f_j(t)||_2` = L2-Norm des Expert-Outputs

### Implementierung
- **C++ Profiler** (`tools/expert-profile/`): Nutzt `ggml_backend_eval_callback` um drei Tensoren pro MoE-Layer abzufangen:
  - `ffn_moe_topk-{il}` [n_expert_used, n_tokens] I32 — welche Experten selektiert
  - `ffn_moe_weights-{il}` [1, n_expert_used, n_tokens] F32 — Gate-Weights
  - `ffn_moe_down-{il}` [n_embd, n_expert_used, n_tokens] F32 — Expert-Outputs
- **Python Pruner** (`tools/moe-pruning/`): `gguf_prune.py` schneidet Expert-Achsen aus GGUF
- **JSON Output**: Per-Layer REAP-Scores, Activation-Counts, Gate-Weights, EAN-Scores

### Fork-Kompatibilitat (verifiziert)
- `cb_eval` callback: Vorhanden (include/llama.h:365, src/llama-context.cpp:89)
- `ffn_moe_topk/weights/down` Tensor-Namen: Vorhanden (src/llama-graph.cpp:1572,1598,1778)
- `tools/CMakeLists.txt`: Eintrag hinzufugen
- Build-System: Standard `add_executable` + `target_link_libraries`

## Schritte

| # | Status | Aufgabe | Verifikation |
|---|--------|---------|--------------|
| 1 | ✅ | Patch heruntergeladen und analysiert | /tmp/pr20454.patch vorhanden |
| 2 | ✅ | `tools/expert-profile/` erstellen (CMakeLists.txt + expert-profile.cpp) | Build grun |
| 3 | ✅ | `tools/moe-pruning/` erstellen (Python-Skripte) | Skripte ausfuhrbar |
| 4 | ✅ | `tools/CMakeLists.txt` erweitern | `add_subdirectory(expert-profile)` |
| 5 | ✅ | Build auf Hydra (CPU-only) | `llama-expert-profile` binary vorhanden |
| 6 | ✅ | Smoke-Test auf Styx mit 26B A4B MoE-Modell | JSON-Output mit REAP-Scores |
| 7 | ✅ | Build auf Styx (CUDA) | Binary vorhanden |
| 8 | ✅ | Profilierung auf Styx mit 26B QAT | expert_stats.json mit 30 Layern |
| 9 | ✅ | Code-Review via review-swe | Keine P0/P1 Issues |
| 10 | ✅ | ROADMAP/CHANGELOG/TTT aktualisieren + commit + push | Dokumentation aktuell |

## Verifikations-Strategie

1. **Build**: `cmake --build build -j$(nproc)` — muss ohne Fehler durchlaufen
2. **Smoke-Test**: `llama-expert-profile -m <kleines-moe-modell> --jsonl <sample> --output /tmp/test.json`
3. **Output-Validierung**: JSON muss pro Layer `reap`, `activation_counts`, `ean_mean` enthalten
4. **Vergleich mit #40**: REAP-Scores sollten ahnliche Hot/Cold-Pattern zeigen wie MoE-Freq-Tracking

## Defaults

- Bei Unklarheiten im Patch: PR-Code als Referenz, nicht raten
- Python-Skripte 1:1 ubernommen (keine Fork-Modifikationen notig)
- C++ Profiler: Falls Fork-spezifische Anpassungen notig ( TurboQuant, Gemma 4), minimal halten

## Offene Fragen

Keine — der PR ist self-contained und beruhrt keine existierenden Fork-Dateien außer `tools/CMakeLists.txt`.
