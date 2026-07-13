# SESSION_PLAN: Buffer-Usage Fix für MoE Expert-Weights

**Datum:** 2026-07-13
**Status:** ☐ offen
**Roadmap-Item:** #37 (Voraussetzung), #13, #14 (freigeschaltet)
**Meilenstein:** M3 (MoE-Offloading v2)

## Session-Ziel

Fix den Root Cause Bug aus #37: MoE Expert-Weights haben `GGML_BACKEND_BUFFER_USAGE_COMPUTE` (2) statt `WEIGHTS` (1), was den `op_offload`-Mechanismus blockiert. Dadurch wird MUL_MAT_ID komplett auf CPU ausgeführt — thecodacus Prefetch und Expert-Cache sind inaktiv.

Nach dem Fix sollten Expert-Weights WEIGHTS-Usage haben, `op_offload` MUL_MAT_ID auf GPU offloaden, und der Cross-Backend-Copy-Pfad aktiviert werden.

## Entscheidungen

| Frage | Entscheidung | Begründung |
|-------|-------------|------------|
| Buffer-Usage Fix wo anwenden? | In llama-model.cpp nach alloc_ctx_tensors | Minimal-invasiv, alle Buffer bekommen WEIGHTS |
| op_offload aktivieren? | Ja, via Env-Var oder Flag | Default-off für Sicherheit, explizite Aktivierung |
| thecodacus Prefetch testen? | Ja, nach Fix | Sollte dann automatisch funktionieren |

## Schritte

1. ☐ Root Cause: Warum haben Expert-Weights COMPUTE statt WEIGHTS? (Subagent)
2. ☐ Fix: Buffer-Usage korrekt setzen für Expert-Weights
3. ☐ Verifikation: op_offload aktiviert MUL_MAT_ID auf GPU
4. ☐ Benchmark: Styx Vorher/Nachher
5. ☐ Doku + Commit

## Verifikations-Strategie

| Schritt | Metrik | Vorher | Nachher (erwartet) |
|---------|--------|--------|-------------------|
| Buffer-Usage | Debug-Print usage-Wert | 2 (COMPUTE) | 1 (WEIGHTS) |
| op_offload | MUL_MAT_ID auf GPU? | Nein (CPU) | Ja (GPU) |
| thecodacus Prefetch | Prefetch aktiv? | Nein | Ja |
| Expert-Cache | Cache-Hits > 0? | 0 | > 0 |
| tg128 Performance | t/s auf Styx | 27.4 t/s | > 27.4 t/s (TBD) |

## Recherche-Strategien

| Problemtyp | Recherche-Strategie |
|------------|---------------------|
| Buffer-Usage Setzung | Subagent: Code-Pfad tensor_buft_overrides → alloc → set_usage |
| op_offload Verhalten | Subagent: Code-Analyse ggml-backend.cpp op_offload |
| Build-Fehler | Inkrementell zurückbauen, git diff prüfen |
| Performance-Regression | Benchmark Vorher/Nachher auf Styx |

## Recherche-Fallbacks

| Block | Mögliches Problem | Recherche-Fallback |
|-------|-------------------|-------------------|
| Buffer-Usage Fix | Andere Code-Pfade brechen | git grep nach BUFFER_USAGE_WEIGHTS, alle Stellen prüfen |
| op_offload | MUL_MAT_ID nicht auf GPU | Debug-Print backend_id für MUL_MAT_ID |
| thecodacus Prefetch | Prefetch-Buffer-Allokation schlägt fehl | Log-Output, OOM prüfen |
| Benchmark | Kein Speedup | Cache-Hit-Rate prüfen, Prefetch-Statistiken |

## Defaults

Bei unklaren Entscheidungen:
1. Minimal-invasiver Fix bevorzugen (set_usage nach alloc)
2. op_offload default-off, explizite Aktivierung via Env-Var
3. Wenn Fix zu invasiv → dokumentieren und postponed

## Offene Fragen

- Warum hat der thecodacus-Patch diesen Bug nicht bemerkt? (Vermutlich auf System mit WEIGHTS-Usage getestet)
- Gibt es Modelle wo Expert-Weights korrekt WEIGHTS-Usage haben? (Ohne tensor_buft_overrides?)
