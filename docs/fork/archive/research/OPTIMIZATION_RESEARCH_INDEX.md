# Optimierungs-Recherche Archiv (2026-07-11 bis 2026-07-14)

**Zusammenfassung:** 4 wöchentliche Recherche-Wellen (jeweils 4 parallele Subagents, Web + arXiv) identifizierten 88+ Optimierungs-Ansätze für den Fork. Alle Ergebnisse sind in `docs/fork/ROADMAP.md` eingeflossen (kategorisiert nach Tier 1-4, mit Status ✅/❌/☐ und Eval-Notizen).

## Recherche-Wellen

| Datum | Datei | Fokus | Neue Items |
|-------|-------|-------|------------|
| 2026-07-11 | `archive/research/2026-07-11_OPTIMIZATION_RESEARCH.md` | Quick Wins (MTP, Tensor Split, Vulkan FA), Tier 2 (Mixed Precision, UMA, EAGLE-3) | #1-13 |
| 2026-07-12 | `archive/research/2026-07-12_OPTIMIZATION_RESEARCH.md` | CM1, Pascal L1, UBBoost, LFRU, Conf-KV, MoE Load Balancing | #31-42 |
| 2026-07-13 | `archive/research/2026-07-13_OPTIMIZATION_RESEARCH.md` | Persistent VRAM, REAP Pruning, xKV, BSFA, Pre-Attention Expert | #57-76 |
| 2026-07-14 | `archive/research/2026-07-14_OPTIMIZATION_RESEARCH.md` | MMVQ Fix, Pipeline Cache, NCCL, GEAR, PEARL, CUDA Graph, DSD | #77-88 |

## Status der identifizierten Items

Siehe `docs/fork/ROADMAP.md` für den aktuellen Status aller Items. Die ROADMAP ist dieSingle Source of Truth — diese Archivdateien enthalten nur das Rohmaterial der Recherche.

## Wichtige Erkenntnisse aus den Recherchen

- **Gemma 4 QK-Norm:** Covariance-Term für Expected Attention bringt <0.1% Gain → Mean-only sufficient
- **Pascal (GTX 1070):** Keine Tensor Cores, L1-Cache-Tuning wirkungslos, MMVQ für Q4_K/Q5_K nicht defekt
- **RDNA3 (Mars):** turbo3/4 KV funktioniert, CM2 > CM1, UMA slower than GTT for GPU-Compute
- **MoE-Offloading:** thecodacus Prefetch (2-Slot Sweet-Spot) > Two-Tier Cache > DALI
- **CUDA Graphs:** Nicht implementiert — Decode-Phase profitiert, Prefill nicht
