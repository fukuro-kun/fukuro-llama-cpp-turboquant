# SESSION_PLAN: Buffer-Usage Fix für MoE Expert-Weights

**Datum:** 2026-07-13
**Status:** ☐ offen
**Roadmap-Item:** #37 (Voraussetzung), #13, #14 (freigeschaltet)
**Meilenstein:** M3 (MoE-Offloading v2)

## Session-Ziel

Fix den Root Cause Bug aus #37: thecodacus Prefetch und Expert-Cache funktionieren nicht auf Styx. Ursprüngliche Hypothese: Expert-Weights haben COMPUTE statt WEIGHTS Buffer-Usage. Tatsächlicher Root Cause: Selektiver Kopierpfad prüft nur nodes[0], aber bei Gemma-4 ist nodes[0] = SCALE nicht MUL_MAT_ID.

## Ergebnis

**Status: Abgeschlossen — thecodacus Prefetch funktioniert (+28-31% PP), Expert-Cache postponed**

### Root Cause geklärt

1. **Expert-Weights haben korrekt WEIGHTS-Usage (1)** — Der Debug-Output mit `usage=2` (COMPUTE) bezog sich auf einen anderen Buffer (Compute-Staging-Buffer im ersten Split), nicht auf die Expert-Weight-Buffer.

2. **Selektiver Kopierpfad war broken** — Der Code prüfte nur `split->graph.nodes[0]` auf `MUL_MAT_ID`. Bei Gemma-4 ist die erste Node im Split `GGML_OP_SCALE` (32), nicht `MUL_MAT_ID` (30). Fix: Suche im gesamten Split-Graph nach dem MUL_MAT_ID-Node.

3. **op_offload Default-Threshold=32 verhindert GPU-Offload bei Decode** — `GGML_OP_OFFLOAD_MIN_BATCH=1` aktiviert GPU-Offload, aber -37% Performance auf Pascal (Copy-Overhead > CPU-Compute). Default Threshold=32 ist optimal für Pascal.

### thecodacus Prefetch: +28-31% PP

| Test | Ohne Prefetch | Mit Prefetch | Delta |
|------|--------------|--------------|-------|
| pp512 | 421 t/s | 540 t/s | **+28%** |
| pp2048 | 394 t/s | 517 t/s | **+31%** |
| pp8192 | 311 t/s | 338 t/s | **+8.5%** |
| tg8 | 23-27 t/s | 20-25 t/s | -10-15% (VRAM-Verbrauch) |

Prefetch ist bereits im Styx-Server-Script aktiviert (`GGML_SCHED_PREFETCH_EXPERTS=1`).

### Expert-Cache: Postponed

- Cache-Hit-Rate = 0% weil Expert-Weights den selektiven Kopierpfad nicht durchlaufen
- Expert-Level-Cache im selektiven Kopierpfad crasht (Buffer-Recycling-Konflikt)
- Postponed bis Buffer-Pinning oder Per-Expert Tensor-Splitting

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
