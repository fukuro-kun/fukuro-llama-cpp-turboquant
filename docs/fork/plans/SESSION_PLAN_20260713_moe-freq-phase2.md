# SESSION_PLAN: MoE Expert Frequency Phase 2 — Frequency-Guided Offloading

**Erstellt:** 2026-07-13 (Solo-Session)
**Status:** In Arbeit

## Session-Ziel

Erweitere das MoE Expert Frequency Tracking (Phase 1) um:
1. Export der Frequency-Daten als JSON für Offline-Analyse
2. Import von Frequency-Daten zur datengetriebenen Layer-Offloading-Auswahl
3. Statt "erste N Layer auf CPU" → "N kälteste Layer auf CPU" basierend auf Frequency-Daten

## Entscheidungen

| Frage | Entscheidung | Begründung |
|-------|-------------|------------|
| Per-Expert-Platzierung? | Nein — erfordert Tensor-Splitting (3D→2D) | Zu tiefer Eingriff für diese Session |
| Frequency-Export-Format? | JSON | Standard, leicht zu lesen und zu importieren |
| Auto-Placement oder manuell? | Import-basiert (2-Phasen-Ansatz) | 1. Warmup mit Tracking → Export, 2. Reload mit Import |
| Welche Metrik für "kalt"? | Entropy der Expert-Verteilung | Niedrige Entropy = konzentrierte Auswahl = hot Layer → GPU |

## Schritte

- [x] Phase 1: Expert Frequency Tracking (implementiert + reviewed)
- [x] Code-Review Fixes (GroveMoE, allow_reuse, cleanups)
- [ ] 2a: JSON Export der Frequency-Daten (`LLAMA_MOE_FREQ_EXPORT=<file>`)
- [ ] 2b: JSON Import + frequency-guided Layer-Auswahl (`LLAMA_MOE_FREQ_IMPORT=<file>`)
- [ ] 2c: Benchmark auf Styx: Standard vs. Frequency-Guided Offloading
- [ ] 2d: ROADMAP + CHANGELOG aktualisieren

## Verifikations-Strategie

| Schritt | Metrik | Vergleich |
|---------|--------|-----------|
| 2a | JSON-File existiert, enthält 30×128 Werte | Manuelle Prüfung |
| 2b | Layer-Auswahl entspricht Erwartung (kälteste Layer auf CPU) | Log-Output |
| 2c | t/s pp512 + tg128 | Vorher (n_cpu_moe=20, erste 20 Layer) vs. Nachher (n_cpu_moe=20, kälteste 20 Layer) |

## Recherche-Strategien

| Problemtyp | Recherche-Strategie |
|------------|---------------------|
| JSON-Export in C++ | `fprintf` mit manuellem JSON-String (keine Library-Abhängigkeit) |
| Entropy-Berechnung | Wolfram Alpha für Formel, dann C++ Implementierung |
| tensor_buft_overrides Mechanismus | Code in llama-bench.cpp Zeilen 1220-1245 |

## Offene Fragen

- Entropy vs. Gini-Koeffizient vs. Top-K-Concentration? → Default: Entropy (einfach, aussagekräftig)
- Soll die Frequency-Daten akkumuliert oder pro Run gespeichert werden? → Pro Run (ein Benchmark = ein JSON)
