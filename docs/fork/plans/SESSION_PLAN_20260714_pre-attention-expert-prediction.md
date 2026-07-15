# SESSION_PLAN — #65 Pre-Attention Expert Prediction + #87 Cross-Layer Gate

**Datum:** 2026-07-14 | **Item:** #65 (arXiv:2511.10676) + #87 (arXiv:2502.12224) | **Systeme:** Styx (CUDA, MoE-Offload), Mars (Vulkan, Referenz)

## Status

- **#65 Pre-Attention Expert Prediction:** ⏭️ Verschoben — Blocker: Training-Pipeline, CC BY-NC-SA Lizenz, fehlender Expert-Cache-Manager
- **#87 Cross-Layer Gate Expert Prediction (Fate):** ❌ Evaluiert — Expert-Selection-Overlap 6.6% (nahe Random-Baseline 6.25%). Nicht viable für 128-Experten MoE. Siehe `docs/fork/2026-07-15_CROSS_LAYER_GATE_EVAL.md`.

## Session-Ziel

~~Implementiere Pre-Attention Expert Prediction für Gemma-4 26B-A4B MoE auf Styx.~~

**Aktualisiert (2026-07-15):** Evaluiere #87 Fate Cross-Layer Gate als Fallback für #65. Ergebnis: ❌ nicht viable für Gemma-4 26B-A4B.

## Entscheidungen (autonom, User nicht verfügbar)

| Frage | Entscheidung | Begründung |
|-------|-------------|------------|
| Welches Item? | #65 Pre-Attention Expert Prediction | Styx's Haupt-Bottleneck ist MoE-Offload. Direkte Adresse. |
| Uranus aus? | Ja, E4B-Endpoint läuft | Styx+Mars suffizient für MoE-Tests |
| Paper vs PR? | Paper (arXiv:2511.10676), kein PR | Reine Forschung, keine Vorlage |
| Testmodell? | Gemma-4 26B-A4B QAT Q4_K_XL | Produktiv-Modell auf Styx |
| Benchmark? | tg128 + pp512 mit/ohne Prediction | Direkter Vergleich mit Baseline |

## Schritte

| # | Status | Beschreibung | Verifikation |
|---|--------|-------------|--------------|
| 1 | ☐ | Paper-Recherche (arXiv:2511.10676) — Algorithmus verstehen | Subagent-Report |
| 2 | ☐ | Codebase-Analyse — MoE-Pfad, Gate-Computation, Prefetch-Infra | Subagent-Report |
| 3 | ☐ | Architektur-Assessment — Ist Pre-Attention Prediction im Fork möglich? | Graph-Build-Order verifiziert |
| 4 | ☐ | Implementierung: Predictor (2 linear functions) | Build grün auf Hydra |
| 5 | ☐ | Implementierung: Integration in MoE-Graph | Build grün, kein Crash |
| 6 | ☐ | Benchmark auf Styx: tg128 + pp512 mit/ohne Prediction | Metriken-Vergleich |
| 7 | ☐ | Benchmark auf Mars: Referenz (voller Offload) | Metriken-Vergleich |
| 8 | ☐ | ROADMAP + CHANGELOG + TTT aktualisieren | Doku aktuell |
| 9 | ☐ | Code-Review + Commit + Push | Review ship-ready |

## Verifikations-Strategie

| Metrik | Vorher (Baseline) | Nachher (mit Prediction) | Ziel |
|--------|-------------------|--------------------------|------|
| tg128 (Styx) | ~26.65 t/s (QAT, 8k) | ? | >26.65 t/s (mindestens neutral) |
| pp512 (Styx) | ~25.7 t/s (QAT, 224k) | ? | >25.7 t/s |
| Expert Hit Rate | ~37.8% (aus #61 Benchmark) | ? | >50% |
| Prediction Accuracy | — | ? | >90% (Paper: 93-97%) |
| Prefetch Overhead | 0ms | ? | <5ms pro Layer |

## Recherche-Strategien

| Problemtyp | Recherche-Strategie |
|------------|---------------------|
| Paper-Algorithmus unklar | arXiv-MCP + Exa semantische Suche |
| MoE-Graph-Build-Order unklar | grep + read in src/llama-graph.cpp |
| Gate-Computation Position | grep nach "ffn_gate_inp" + "mul_mat_id" |
| Prefetch-Infra | grep nach "PREFETCH_EXPERTS" + "SCHED_PREFETCH" |
| Tiefer Architektur-Blocker | coding-research Skill nach 3 Versuchen |

## Recherche-Fallbacks

| Block | Mögliches Problem | Recherche-Fallback |
|-------|-------------------|-------------------|
| Paper nicht auf arXiv | arXiv-ID falsch | Exa semantische Suche + Google Scholar |
| Algorithmus braucht Training | Fine-Tuning nötig | Abbrechen → ❌ mit Begründung |
| Gate-Position unklar | Graph-Build verschachtelt | Subagent für src/llama-graph.cpp Deep-Dive |
| Predictor zu langsam | Overhead > Benefit | Threshold-basiertes Skip, nur bei hohem Kontext |
| Styx-Crash | OOM durch Predictor-Gewichte | Predictor auf CPU, nur Gewichte auf GPU |

## Defaults

1. **Pragmatisch:** Falls Paper Training erfordert → ❌, stattdessen #87 Cross-Layer Gate evaluieren
2. **Reversibel:** Predictor als Compile-Flag (`LLAMA_MOE_PREDICT=1`), default OFF
3. **Dokumentiert:** Alle Erkenntnisse in CHANGELOG + ROADMAP

## Offene Fragen

- Braucht der Predictor Training? (Paper-Recherche läuft)
- Ist die Gate-Computation vor oder nach Attention? (Codebase-Analyse läuft)
- Funktioniert der Ansatz mit Gemma-4's A4B-Architektur (4 active experts von 128)?
