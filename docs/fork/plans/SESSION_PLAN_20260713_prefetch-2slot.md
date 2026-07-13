# SESSION_PLAN 2026-07-13 — thecodacus Prefetch 2-Slot + M3 Abschluss

**Datum:** 2026-07-13 (Solo-Session, fortgesetzt)
**Status:** ✅ Abgeschlossen

## Session-Ziel

thecodacus Expert Prefetch reparieren, tg-Regression eliminieren, M3 Meilenstein abschließen.

## Entscheidungen

| Frage | Entscheidung | Begründung |
|-------|-------------|------------|
| Prefetch Slot-Count konfigurierbar machen? | Ja, via `GGML_SCHED_PREFETCH_SLOTS` | Ermöglicht Tuning ohne Code-Änderung |
| Wie viele Slots optimal? | 2 (Sweet-Spot) | 1: kein Overlap (-11.9% pp), 2: +28.9% pp +2.1% tg, 3: -35% tg auf Mars |
| PipeShard implementieren? | Nein (❌) | 4-12 Wochen Aufwand, marginaler Benefit auf Pascal/RDNA3 |
| Expert-Cache weiter verfolgen? | Postponed | Nur mit op_offload relevant (-37% auf Pascal), Buffer-Recycling-Crash |
| Mars-Server mit Prefetch? | Ja, 2-Slot | +8.8% pp, +3.8% tg — reiner Win |

## Schritte

1. ✅ Prefetch Debug-Prints entfernt, Batch-Size-Check korrigiert
2. ✅ `GGML_SCHED_PREFETCH_SLOTS` env var hinzugefügt
3. ✅ Styx Slot-Count Benchmark: 1/2/3 Slots vs baseline
4. ✅ Mars Slot-Count Benchmark: 1/2/3 Slots vs baseline
5. ✅ 2-Slot als Sweet-Spot identifiziert (reiner Win auf beiden Systemen)
6. ✅ Beide Server-Scripts auf 2-Slot umgestellt
7. ✅ PipeShard #15 evaluiert → ❌ (Subagent-Recherche)
8. ✅ M3 Meilenstein als abgeschlossen markiert
9. ✅ ROADMAP, CHANGELOG, SNAPSHOT aktualisiert
10. ✅ Code-Review (keine P0/P1, 3 P2 gefixt)
11. ✅ Monatliche Recherche gestartet (4 parallele Subagents)

## Verifikations-Strategie

| Test | System | Vorher | Nachher | Delta |
|------|--------|--------|---------|-------|
| pp512 | Styx | 423.47 | 546.03 | +28.9% |
| pp2048 | Styx | 378.61 | 515.53 | +36.2% |
| pp8192 | Styx | 311.76 | 426.16 | +36.7% |
| tg8 | Styx | 24.91 | 25.44 | +2.1% |
| pp512 | Mars | 169.39 | 184.26 | +8.8% |
| tg128 | Mars | 16.12 | 16.73 | +3.8% |

## Offene Fragen

- Expert-Cache Crash: Buffer-Recycling-Konflikt, benötigt Buffer-Pinning (postponed)
- Prefetch auf Hydra (Ampere): nicht getestet (GPU-Verbot auf Hydra)
- Prefetch auf Uranus (Ada): nicht getestet (kein MoE-Offload-Setup)
