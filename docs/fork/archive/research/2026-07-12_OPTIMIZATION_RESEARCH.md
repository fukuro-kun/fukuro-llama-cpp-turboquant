# Optimierungs-Ansaetze fuer fukuro-llama-cpp-turboquant

**Recherche:** 2026-07-12, Web + arXiv (4 parallele Subagents)
**Hardware:** Mars (RDNA3/Vulkan/30GB), Styx (Pascal/CUDA/8GB), Hydra (Ampere/CUDA/8GB), Uranus (2x Ada/CUDA/32GB), Venus (GCN/Vulkan)
**Ausschluss:** Treiber-Neuimplementierung, Kernel-Rekompilierung, Hopper-spezifische Features

---

## Bereits im Fork integriert (nicht erneut vorschlagen)

Die folgenden PRs wurden bei dieser Recherche als **bereits im Fork** identifiziert:
- PR #21472 (CUDA Graph Properties Check) — Commit dfe863cbf
- PR #23764 (F16 Attention Mask) — Commit e9482ba53
- PR #22299 (Internal AllReduce Kernel) — Commit cf1de535b
- PR #21611 (LRU Eviction für CUDA Graphs) — Commit 704e83381
- PR #22423 (CPU RMS_NORM+MUL Fusion) — Commit 7aa4f4e91
- PR #18749 (RDNA3 Coopmat Matmul Parameters) — Commit cf119f140
- Warp Shuffle Reduction: 61 Matches in ggml-cuda (bereits genutzt)
- Constant Memory: 26 matches in ggml-cuda (bereits genutzt)

---

## Tier 1: Quick Wins (< 1 Woche)

### 1. K-Quant A-Matrix Transpose (CM1)
- **Was:** Repackt K-quant Gewichte von [row, k_block] zu [k_block, row] fuer sequenzielle Shader-Lesepfade auf CM1 (VK_KHR_cooperative_matrix). Verbessert Cache-Lokalitaet beim Dequantize-MatMul.
- **Systeme:** Mars (RDNA3 mit coopmat)
- **Schwierigkeit:** mittel
- **Existiert:** PR #22970 (open) — unverifiziert
- **Aufwand:** 4-8 Stunden
- **Gain:** +5-8% PP fuer Q4_K, bis +11-15% PP fuer Q6_K

### 2. Pascal L1 Cache Tuning (-Xptxas -dlcm=ca)
- **Was:** Aktiviert L1/Texture Cache fuer globale Loads auf Pascal GP104 (GTX 1070). Standardmaessig cached GP104 nur in L2, diese Flag opt-in fuer unified L1/Texture Cache.
- **Systeme:** Styx (Pascal GTX 1070)
- **Schwierigkeit:** einfach
- **Existiert:** nein (Compiler-Flag, kein PR)
- **Aufwand:** 1-2 Stunden
- **Gain:** unbekannt, potenziell +5-10% bei memory-bound Workloads

### 3. Per-Quant MMVQ/MMQ Batch Threshold fuer AMD MFMA
- **Was:** Quantisierungsspezifische Batch-Schwellen fuer MMVQ vs MMQ auf AMD MFMA-Hardware. K-quants profitieren frueher von MMQ (ab batch=4).
- **Systeme:** Mars (RDNA3), Venus (GCN)
- **Schwierigkeit:** mittel
- **Existiert:** Commit bc81d47 (upstream) — unverifiziert
- **Aufwand:** 2-3 Tage
- **Gain:** bis +76% PP auf MI250X (AMD MFMA), unklar auf RDNA3

---

## Tier 2: Mittelfristig (1-3 Wochen)

### 4. UBBoost: Dynamische Ubatch-Groesse fuer Prefill
- **Was:** Erhoeht ubatch-Groesse nur waehrend Prompt-Processing bei VRAM-konstriktiven Setups. Prefill-Speed bis 2x ohne TG-Beeintraechtigung.
- **Systeme:** Styx (8GB), Hydra (8GB)
- **Schwierigkeit:** mittel
- **Existiert:** Discussion #23262 (RFC, externer Fork)
- **Aufwand:** 1 Woche
- **Gain:** bis 2x PP bei VRAM-konstriktiven Setups

### 5. Auto Parameter Fitting fuer Tensor Parallelism
- **Was:** Automatische Anpassung von Kontextlaenge und GPU-Layern bei TP-Modus. Reduziert manuelle Konfiguration.
- **Systeme:** Uranus (2x RTX 4060 Ti)
- **Schwierigkeit:** mittel
- **Existiert:** PR #22950 (open)
- **Aufwand:** 3-4 Tage
- **Gain:** Vereinfachte TP-Konfiguration, bessere Speicherauslastung

### 6. Row-Packing fuer Dequantize-MatVec
- **Was:** Packt zwei Ausgabe-Rows pro Workgroup fuer Q5_K dequantize-matmul-vector shader. Halbiert Workgroup-Count.
- **Systeme:** Mars (RDNA3), Venus (GCN)
- **Schwierigkeit:** mittel
- **Existiert:** nein (Zinc RDNA4_TUNING.md Referenz)
- **Aufwand:** 4-8 Stunden
- **Gain:** unklar, potenziell +10-20% DMMV

### 7. LFRU Expert Caching fuer MoE Offloading
- **Was:** Frequency-weighted LRU Eviction fuer Expert-Cache auf GPU. Verhindert dass fruehe Layer den Cache monopolisieren.
- **Systeme:** Styx, Hydra
- **Schwierigkeit:** mittel
- **Existiert:** vLLM commit 71ed1fc (Referenz)
- **Aufwand:** 1-2 Wochen
- **Gain:** +5.2% speedup (vLLM Referenz), 15-20% hit rate

### 8. Conf-KV: Confidence-aware KV Cache Eviction
- **Was:** Konvertiert next-token distribution in confidence score fuer per-step cache budget selection. Mixed FP16/INT8 storage.
- **Systeme:** Alle
- **Schwierigkeit:** einfach
- **Existiert:** arXiv:2605.24786
- **Aufwand:** 1-2 Wochen
- **Gain:** KV-Cache Kompression, ergaenzt TurboQuant

### 9. Talon: Confidence-aware Speculative Decoding
- **Was:** Budget-driven adaptive tree expansion fuer speculative decoding. "deep-and-narrow" fuer deterministische Kontexte, "shallow-and-wide" fuer unsichere branches.
- **Systeme:** Alle
- **Schwierigkeit:** einfach
- **Existiert:** arXiv:2601.07353
- **Aufwand:** 2-3 Wochen
- **Gain:** hoehere Acceptance Rate als statische trees

### 10. MoE Load Balancing mit Expert Frequency Tracking
- **Was:** Statistisches Tracking der Expert-Aktivierungsfrequenz. Statische Zuordnung haeufiger Experten zu GPU, seltener zu CPU.
- **Systeme:** Styx, Hydra
- **Schwierigkeit:** mittel
- **Existiert:** nein (Konzept)
- **Aufwand:** 1 Woche
- **Gain:** bessere Load-Balance bei MoE-Offloading

---

## Tier 3: Komplex (3-6 Wochen)

### 11. Paged KV Cache mit Continuous Batching
- **Was:** PagedAttention-basierter KV-Cache und Continuous-Batching-Scheduler nach vLLM-Modell.
- **Systeme:** Alle (besonders Server-Szenarien)
- **Schwierigkeit:** schwer
- **Existiert:** Discussion #21961 (Prototyp)
- **Aufwand:** 2-3 Wochen
- **Gain:** 2.5x aggregate throughput

### 12. Disaggregated Prefill/Decode
- **Was:** Trennt PP und TG auf unterschiedliche Geraete/Maschinen.
- **Systeme:** Uranus (Multi-GPU)
- **Schwierigkeit:** schwer
- **Existiert:** Issue #21266 (RFC)
- **Aufwand:** 2-3 Wochen
- **Gain:** verhindert PP/TG-Interferenz

### 13. GRKV: Global Regression KV Compression
- **Was:** Training-free KV cache compression via ridge-regression-basierte Merge-Schritte.
- **Systeme:** Alle
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2605.31105
- **Aufwand:** 2-3 Wochen
- **Gain:** bessere Performance als existierende Merging-Methoden

### 14. CapKV: Capacity-aware KV Cache Eviction
- **Was:** Eviction als information bottleneck problem mit closed-form mutual information objective.
- **Systeme:** Alle
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2604.25975
- **Aufwand:** 2-3 Wochen
- **Gain:** theoretisch fundierte Eviction

### 15. SliderQuant: Sliding-layer Post-Training Quantization
- **Was:** Beruecksichtigt varying quantization sensitivity von shallow, intermediate und deep layers.
- **Systeme:** Alle
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2603.25284
- **Aufwand:** 2-3 Wochen
- **Gain:** bessere Low-Bit-Quantisierung

### 16. Alloc-MoE: Budget-aware Expert Activation
- **Was:** Optimiert budget allocation auf layer und token level. 1.34x decode speedup bei halbem budget.
- **Systeme:** MoE-Systeme (Mars, Styx, Uranus)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2604.08133
- **Aufwand:** 3-4 Wochen
- **Gain:** 1.34x decode speedup

### 17. HybriMoE: Hybrid CPU-GPU Scheduling
- **Was:** Effiziente CPU-GPU kollaborative inference mit expert offloading und cache management.
- **Systeme:** Styx (MoE-Offloading)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2504.05897
- **Aufwand:** 3-4 Wochen
- **Gain:** 1.33x prefill, 1.70x decode

### 18. CUDA Concurrent Streams fuer QKV Projections
- **Was:** Unabhaengige Ausfuehrung von Q/K/V Projections via CUDA Streams.
- **Systeme:** Styx, Hydra
- **Schwierigkeit:** mittel
- **Existiert:** GGML_CUDA_GRAPH_OPT=1 (Pascal-Tuning noetig)
- **Aufwand:** 3-5 Tage
- **Gain:** Compute-Overlap bei Attention

---

## Tier 4: Langfristig / Forschung (6+ Wochen)

### 19. SpecMD Least-Stale Expert Prefetching
- **Was:** Least-Stale Eviction Policy, reduziert Cache-Misses bis 85x gegenueber LRU.
- **Systeme:** Styx, Hydra
- **Existiert:** arXiv:2508.21706
- **Aufwand:** 2-3 Wochen
- **Gain:** 85x weniger collision misses, 34.7% TTFT reduction

### 20. QUICK Shared Memory Bank Conflict Elimination
- **Was:** Offline-Interleaving von quantisierten Gewichtsmatrizen um bank conflicts zu eliminieren.
- **Systeme:** Hydra (CC 7.5+)
- **Existiert:** arXiv:2402.10076, GitHub: SqueezeBits/QUICK
- **Aufwand:** 3-4 Wochen
- **Gain:** bis 1.94x throughput

### 21. FluxMoE Expert Paging mit Pipeline-Overlap
- **Was:** Decoupled Expert Residency, on-demand Materialisierung mit Pipeline-Overlap.
- **Systeme:** Styx, Hydra
- **Existiert:** arXiv:2604.02715
- **Aufwand:** 3-4 Wochen
- **Gain:** kleineres Working Set, GPU busy

### 22. STAR-KV: Low-rank KV Compression
- **Was:** Soft thresholding fuer adaptive rank control. Bis 75% compression, 6.9x attention speedup.
- **Systeme:** Alle
- **Existiert:** arXiv:2606.08382, GitHub: PriyanshBhatnagar/STAR-KV
- **Aufwand:** 4-6 Wochen
- **Gain:** 75% KV compression, 6.9x attention

### 23. VQKV: Vector Quantization KV Cache
- **Was:** VQ fuer hoch-komprimierte KV-Repraesentationen. 82.8% compression bei 98.6% performance.
- **Systeme:** Alle
- **Existiert:** arXiv:2603.16435
- **Aufwand:** 3-4 Wochen
- **Gain:** 82.8% KV compression

### 24. CompilerKV: Offline Experience Compilation
- **Was:** Kompiliert offline Kompressionserfahrung in wiederverwendbare Entscheidungstabellen.
- **Systeme:** Alle
- **Existiert:** arXiv:2602.08686, GitHub: luckypiggy-orangejuice/CompilerKV
- **Aufwand:** 3-4 Wochen
- **Gain:** robuste Kompression unter engen Memory-Budgets

### 25. SliceMoE: Bit-sliced Expert Caching
- **Was:** Bit-sliced caching unter miss-rate constraints. Redundanz unter Experten minimiert.
- **Systeme:** MoE-Systeme
- **Existiert:** arXiv:2512.12990
- **Aufwand:** 4-5 Wochen
- **Gain:** effiziente MoE-Caching

### 26. DALI: Workload-aware MoE Offloading
- **Was:** Workload-aware offloading framework das underutilization und load imbalance verhindert.
- **Systeme:** Styx
- **Existiert:** arXiv:2602.03495
- **Aufwand:** 4-5 Wochen
- **Gain:** intelligentes MoE-Offloading

### 27. MoBiE: Mixture of Binary Experts
- **Was:** Binarisierung framework fuer MoE-LLMs mit joint SVD decomposition. Ueber 2x inference speedup.
- **Systeme:** MoE-Systeme
- **Existiert:** arXiv:2604.06798, GitHub: MoBiE-Team/MoBiE
- **Aufwand:** 5-6 Wochen
- **Gain:** 2x+ inference speedup

### 28. DASH-Q: Ultra Low-Bit PTQ
- **Was:** Diagonal Hessian approximation und iterative weighted least squares fuer ultra low-bit.
- **Systeme:** Alle
- **Existiert:** arXiv:2604.13806
- **Aufwand:** 4-5 Wochen
- **Gain:** +7.01% zero-shot accuracy im ultra low-bit

### 29. GOOSE: Anisotropic Speculation Trees
- **Was:** Adaptive spine tree mit deep chain und wide branches. Training-free.
- **Systeme:** Alle
- **Existiert:** arXiv:2604.02047
- **Aufwand:** 3-4 Wochen
- **Gain:** hoehere Acceptance Rate

---

## Nicht empfohlen / nicht anwendbar

| Ansatz | Grund |
|--------|-------|
| UMA Buffer Transfer Optimization (PR #22462) | Verwandte UMA-PRs (#22455, #22930, #23770) bereits revertiert, System-RAM langsamer als GTT |
| Host-Visible Memory Buffers (PR #22930) | Bereits revertiert, RCA-Masterplan: System-RAM langsamer als GTT |
| Pipeline Barriers fuer Memcpy (PR #23770) | Bereits revertiert |
| SYCL-spezifische Optimierungen | Keine Intel GPUs im Setup |
| Bindless Descriptor Management | Komplexe Architekturaenderung, geringer ROI |
| CUDA L2 Cache Persistence | CC 8.0+ (Ampere+), nicht auf Pascal |
| DFlash Speculative Decoding | Blackwell-spezifisch |

---

## Priorisierung nach Cost-Benefit

### Sofort umsetzbar (Tier 1 Quick-Wins):
1. **K-Quant Transpose CM1** — 4-8h, +5-15% PP auf Mars
2. **Pascal L1 Cache Tuning** — 1-2h, Compiler-Flag fuer Styx
3. **Per-Quant MMVQ/MMQ Threshold** — 2-3 Tage, AMD MFMA

### Diese / naechste Woche (Tier 2):
4. **UBBoost** — 1 Woche, bis 2x PP auf Styx/Hydra
5. **Row-Packing DMMV** — 4-8h, Mars/Venus
6. **Auto Param Fitting TP** — 3-4 Tage, Uranus
7. **LFRU Expert Caching** — 1-2 Wochen, Styx (siehe auch #14 LFU Recherche)
8. **Conf-KV** — 1-2 Wochen, KV-Cache Kompression (ergaenzt TurboQuant)

### Mittelfristig (Tier 3):
9. **Paged KV Cache** — 2-3 Wochen, 2.5x throughput
10. **GRKV / CapKV** — 2-3 Wochen, KV-Cache Kompression
11. **Alloc-MoE / HybriMoE** — 3-4 Wochen, MoE-Optimierung

### Langfristig (Tier 4):
12. **SpecMD Least-Stale** — 2-3 Wochen, 85x weniger misses
13. **QUICK Bank Conflict** — 3-4 Wochen, 1.94x auf Hydra
14. **STAR-KV / VQKV** — 3-6 Wochen, 75-83% KV compression
