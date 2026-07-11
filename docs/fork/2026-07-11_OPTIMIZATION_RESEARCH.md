# Optimierungs-Ansätze für fukuro-llama-cpp-turboquant

**Recherche:** 2026-07-11, Web + arXiv (4 parallele Recherchen, 50+ Quellen)
**Hardware:** Mars (RDNA3/Vulkan/30GB), Styx (Pascal/CUDA/8GB), Hydra (Ampere/CUDA/8GB), Uranus (2x Ada/CUDA/32GB), Venus (GCN/Vulkan)
**Ausschluss:** Treiber-Neuimplementierung, umfangreiche Kernel-Rekompilierung

---

## Tier 1: Quick Wins (< 1 Woche)

### 1. MTP Logits Copy Optimization
- **Was:** Vermeidet Kopieren von Logits für jeden Token im Batch bei MTP Prompt-Processing. Nur pre-norm wird benötigt. Halbiert MTP-Overhead auf PP.
- **Systeme:** Mars, Styx (beide haben MTP im Fork)
- **Schwierigkeit:** Einfach (Cherry-Pick)
- **Existiert:** PR #23198 (merged Mai 2026)
- **Aufwand:** 2-3 Stunden
- **Gain:** +20% PP mit MTP (MI50/Qwen27B Benchmark)

### 2. Tensor Split Regex Optimization
- **Was:** Macht 29 std::regex Patterns static (file scope) statt per-Token neu kompiliert. Befreit ~40% des Decode-Threads.
- **Systeme:** Uranus (Multi-GPU), prinzipiell alle mit `--tensor-split`
- **Schwierigkeit:** Einfach (Cherry-Pick)
- **Existiert:** PR #24710 (merged)
- **Aufwand:** 1-2 Stunden
- **Gain:** get_split_state drop von ~22% → ~13%

### 3. Pascal CUDA MMVQ Optimization
- **Was:** MMVQ Launch-Config mit 2 warps statt 4 für single-token decode auf Pascal (CC 6.1). FP16 Performance-Detection Fix.
- **Systeme:** Styx (GTX 1070)
- **Schwierigkeit:** Einfach-Mittel (Cherry-Pick)
- **Existiert:** PR #25479
- **Aufwand:** 2-3 Tage
- **Gain:** +3-6% decode auf Pascal (Q4_0: +3.78%, Q4_K_M: +5.81%)

### 4. n-gram / Prompt-Lookup Decoding
- **Was:** Modellfreies Speculative Decoding via n-gram lookup aus generiertem Kontext. Zero extra VRAM, zero extra Download.
- **Systeme:** Alle (besonders Code-Generation mit repetitiven Patterns)
- **Schwierigkeit:** Einfach (bereits verfügbar)
- **Existiert:** In mainline (`--spec-type ngram-mod,ngram-map-k4v`)
- **Aufwand:** 0 Stunden (nur aktivieren/testen)
- **Gain:** Fast nie hurts, hilft bei repetitiven Workloads

### 5. GTT Size Tuning für Mars APU
- **Was:** GTT (Graphics Translation Table) ist standardmäßig auf 50% des System-RAM limitiert (15GB von 30GB). Erhöhung gibt GPU mehr Adressraum.
- **Systeme:** Mars (RDNA3 APU, 30GB shared)
- **Schwierigkeit:** Einfach (Kernel-Parameter, keine Rekompilierung)
- **Existiert:** Nein (Konfiguration)
- **Aufwand:** 1 Stunde
- **Gain:** Ermöglicht Nutzung von >15GB für GPU workloads

### 6. Vulkan MUL_MAT_ID Subgroup Optimization
- **Was:** Subgroup ballot operations für effizientere MoE-Expert-Selection bei Vulkan. Massive Speedup für MoE prompt processing.
- **Systeme:** Mars (RDNA3), Venus (GCN)
- **Schwierigkeit:** Einfach (Cherry-Pick)
- **Existiert:** PR #15524
- **Aufwand:** 1 Tag
- **Gain:** Bis zu **657% Speedup** auf AMD Pro VII, 466% auf RX 6800 XT für MoE PP

### 7. Vulkan FlashAttention Refactor (Subgroup Reductions)
- **Was:** Row splitting (1/4), shared memory staging für K/V loads, Q caching in registers, fused Lf accumulation. Vendor-specific Br selection.
- **Systeme:** Mars (RDNA3), Venus (GCN)
- **Schwierigkeit:** Mittel (Cherry-Pick + Testing)
- **Existiert:** PR #19625 (merged)
- **Aufwand:** 1 Woche
- **Gain:** 10-20% improvement auf scalar FA path
- **Hinweis:** MoltenVK+AMD subgroupShuffleXor bug — Workaround in PR #23218

---

## Tier 2: Mittelfristig (1-3 Wochen)

### 8. Mixed Precision KV Cache (Hot/Cold Tokens)
- **Was:** Recente Tokens in FP16, ältere automatisch quantisiert. Threshold (default 32), group_size (16). Hot/cold types konfigurierbar.
- **Systeme:** Alle mit langem Kontext (besonders Mars 224k, Styx 224k)
- **Schwierigkeit:** Mittel
- **Existiert:** Commit e889fbd (merged in mainline)
- **Aufwand:** 1 Woche
- **Gain:** Reduziert KV-Cache-Größe bei erhaltenér Qualität für recente Tokens

### 9. Vulkan Shared Memory Staging Kernel
- **Was:** Zwei-Phasen-Ansatz für Matrix-Vector: Decode phase (raw uint64 loads → shared memory) + Compute phase (dot product). Shmem-staging.
- **Systeme:** Mars (RDNA3), Venus (GCN)
- **Schwierigkeit:** Mittel
- **Existiert:** PR #20897 (für Intel Arc >2.5x speedup)
- **Aufwand:** 2-3 Tage (RDNA Testing + Anpassung)
- **Gain:** >2.5x auf Intel Arc, potenziell ähnlich auf RDNA

### 10. UMA Zero-Copy für Mars APU
- **Was:** True Zero-Copy für Unified Memory Architectures. Vermeidet Duplizierung von weights in RAM. DMA-BUF für pinned shared memory, cache flushing.
- **Systeme:** Mars (AMD Radeon 760M iGPU), Venus (Vega iGPU)
- **Schwierigkeit:** Mittel
- **Existiert:** PR #22098 (SYCL), PR #22462 (Vulkan UMA optimization)
- **Aufwand:** 1-2 Wochen
- **Gain:** Host-to-Backend transfer 36.888s → 0.327s (**+112x**), UMA read 33→60 GB/s (+82.9%)

### 11. EAGLE-3 Speculative Decoding
- **Was:** Feature-basiertes Speculative Decoding. Multi-Layer Feature Fusion, direkte Token-Prediction. Training-Time Test simuliert Inference während Training.
- **Systeme:** Alle (besonders Mars, Hydra)
- **Schwierigkeit:** Mittel
- **Existiert:** PR #18039 (merged), Code: https://github.com/SafeAILab/EAGLE
- **Aufwand:** 2-3 Wochen (Integration + Draft-Model-Training/Konvertierung)
- **Gain:** Bis zu **6.5x Speedup**, 1.4x besser als EAGLE-2
- **Hinweis:** Benötigt EAGLE-3 Draft-Modell (z.B. RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3)

### 12. Vulkan Cooperative Matrix (Coopmat2) für RDNA3
- **Was:** VK_KHR_cooperative_matrix für Tensor-Core-ähnliche Matrix Multiplies auf RDNA3. WMMA (Wavefront Mixed-precision Multiply Accumulate).
- **Systeme:** Mars (RDNA3 — unterstützt coopmat), Uranus (Ada — via Vulkan)
- **Schwierigkeit:** Mittel-Schwer
- **Existiert:** PR #10597 (basic), PR #19075 (Coopmat1 Refactor für AMD)
- **Aufwand:** 2-3 Wochen
- **Gain:** 2.5x speedup für FlashAttention Q*K^T multiply
- **Hinweis:** Fork hat bereits coopmat2 Feature-Check. turbo3 FA ist deaktiviert (glslc bug) — coopmat könnte das umgehen

### 13. Two-Tier GPU+RAM Expert Cache
- **Was:** Persistenter VRAM expert slot cache (Tier 1) + pinned RAM für warm experts (Tier 2) + unpinned RAM/disk für cold (Tier 3).
- **Systeme:** Styx (8GB VRAM — kritisch!), Mars
- **Schwierigkeit:** Mittel
- **Existiert:** Feature Request #20757 (nicht implementiert)
- **Aufwand:** 1-2 Wochen
- **Gain:** Signifikant für Styx — reduziert H2D-Copies für häufig genutzte Experts

### 14. LFU Caching Policy für MoE Experts
- **Was:** LFU (Least Frequently Used) statt LRU für expert cache. Forschung zeigt LFU ist besser für MoE-Workloads.
- **Systeme:** Styx, Mars
- **Schwierigkeit:** Einfach
- **Existiert:** Nein (aber Forschung belegt)
- **Aufwand:** 2-3 Tage
- **Gain:** 15-20% improvement in expert hit rate

### 15. Efficient VRAM-Constrained xLM (PipeShard)
- **Was:** Benchmark-profile-guided CPU-GPU Hybrid Scheduling. Sub-Layer Model Sharding, Pipelined Copy-Compute, Prioritized Tensor Placement.
- **Systeme:** Styx (8GB), Hydra (8GB), Mars (shared memory)
- **Schwierigkeit:** Mittel
- **Existiert:** https://github.com/deepshnv/pipeshard-mlsys26-ae (llama.cpp fork!)
- **Aufwand:** 2-3 Wochen (Portieren relevanter Teile)
- **Gain:** Deutliche Verbesserung für VRAM-constrained MoE-Inference

### 16. Vulkanised 2026: shmem-staging Matrix-Vector Kernel
- **Was:** Handgeschriebene Shader mit Kernel-Fusion, Graph-Scheduling Optimierungen. Shmem-staging M-V Kernel.
- **Systeme:** Mars (RDNA3), Venus (GCN)
- **Schwierigkeit:** Mittel (Cherry-Pick)
- **Existiert:** Teil von llama.cpp upstream (Vulkanised 2026 Presentation)
- **Aufwand:** 1-2 Wochen
- **Gain:** >2.5x TG Speedup auf Intel Arc (RDNA potenziell ähnlich)

---

## Tier 3: Mittelfristig komplex (3-6 Wochen)

### 17. HOBBIT: Mixed-Precision Expert Offloading
- **Was:** Token-level dynamic expert loading, layer-level adaptive prefetching, sequence-level expert caching. Cache-miss experts in low precision.
- **Systeme:** Styx (MoE-Offloading), Mars (shared memory)
- **Schwierigkeit:** Schwer
- **Existiert:** Paper arXiv:2411.01433, implementiert auf llama.cpp
- **Aufwand:** 4-6 Wochen
- **Gain:** Bis zu **9.93x speedup** vs state-of-the-art für MoE-Offloading

### 18. DALI: Workload-Aware MoE Offloading
- **Was:** Dynamische Expert-Zuweisung CPU/GPU via 0-1 Integer Optimization. Residual-Based Prefetching nutzt Inter-Layer Residuals.
- **Systeme:** Styx (perfekt für CPU-GPU Hybrid)
- **Schwierigkeit:** Mittel
- **Existiert:** Paper arXiv:2602.03495 (kein Code)
- **Aufwand:** 2-3 Wochen
- **Gain:** Erweitert bestehendes MoE-Offloading intelligent

### 19. Expected Attention KV Cache Compression
- **Was:** Training-free Methode schätzt KV-Wichtigkeit durch Vorhersage wie zukünftige Queries attenden. Closed-Form Expected Attention Scores.
- **Systeme:** Alle (besonders Mars 224k Kontext)
- **Schwierigkeit:** Mittel
- **Existiert:** KVPress Library (20+ Techniken), Paper arXiv:2510.00636
- **Aufwand:** 2-3 Wochen
- **Gain:** Ergänzt TurboQuant — zusätzliche KV-Reduktion bei erhaltenér Qualität

### 20. Tensor Parallelism für Uranus (2x RTX 4060 Ti)
- **Was:** `--split-mode tensor` verteilt Tensoren über 2 GPUs statt komplette Layer. Echtes paralleles Rechnen.
- **Systeme:** Uranus (2x RTX 4060 Ti, 32GB total)
- **Schwierigkeit:** Schwer
- **Existiert:** PR #19378 (initial), PR #23225 (KV quantization support)
- **Aufwand:** 3-4 Wochen für stabilen Einsatz
- **Gain:** ~40% boost in token generation
- **Hinweis:** PCIe 3.0 x8 pro GPU limitiert — NVLink nicht verfügbar

### 21. PagedAttention / Paged KV Cache
- **Was:** Seitenbasierte KV-Cache Verwaltung. Reduziert Fragmentierung, Copy-on-Write Sharing zwischen Sequenzen. 247 Sequenzen vs 25 bei gleichem KV-Budget.
- **Systeme:** Uranus (batched serving), Mars (2 Slots)
- **Schwierigkeit:** Schwer
- **Existiert:** PR #22569 (Draft), CUDA-only
- **Aufwand:** 3-4 Wochen
- **Gain:** 2.5x aggregate throughput bei hoher Konkurrenz

### 22. GWQ: Gradient-Aware Weight Quantization
- **Was:** Gradient-basierter PTQ-Ansatz. Top 1% Outliers bleiben FP16, Rest 3-4 bit. Nur 1% Calibration Set nötig.
- **Systeme:** Alle
- **Schwierigkeit:** Mittel
- **Existiert:** Paper arXiv:2411.00850 (kein Code)
- **Aufwand:** 2-3 Wochen
- **Gain:** 1.2x Inference Speedup, alternative zu QAT

### 23. DuoServe-MoE: Dual-Phase Expert Scheduling
- **Was:** Trennt Prefill (dichte Expert-Aktivierung) und Decode (sparse) Phasen. Two-Stream CUDA Pipeline für Prefetch-Overlap. Offline-trainierter Predictor.
- **Systeme:** Styx (MoE-Offloading mit async prefetch)
- **Schwierigkeit:** Mittel-Schwer
- **Existiert:** Paper arXiv:2509.07379 (kein Code)
- **Aufwand:** 3-4 Wochen
- **Gain:** Phase-spezifische Optimierung des Prefetch-Overlaps

### 24. HybriMoE: Hybrid CPU-GPU Scheduling
- **Was:** Dynamic Intra-Layer Scheduling für CPU/GPU Balance. Impact-Driven Inter-Layer Prefetching. Score-based Caching.
- **Systeme:** Styx (MoE-Offloading)
- **Schwierigkeit:** Mittel-Schwer
- **Existiert:** Auf kTransformers Framework implementiert
- **Aufwand:** 3-4 Wochen
- **Gain:** 1.33x Prefill, 1.70x Decode Speedup

---

## Tier 4: Langfristig / Forschung (6+ Wochen)

### 25. llama.cpp 2026 Rewrite Merge
- **Was:** Größte architektonische Überarbeitung: neuer Kernel-Generator, reorganisierter KV-Cache (contiguous per head), vereinheitlichtes Backend-Dispatch.
- **Systeme:** Alle
- **Schwierigkeit:** Schwer (großes Refactoring, viele Merge-Konflikte)
- **Existiert:** In mainline (April 2026 merged)
- **Aufwand:** 4-6 Wochen für vollständigen Merge
- **Gain:** 2.1x Durchsatz auf 70B, 1.4x auf 7B

### 26. Q-Filters / LagKV / MiniCache
- **Was:** Drei neue KV-Compression-Methoden: Q-Filters (Query-Key Geometrie, 32x compression), LagKV (Lag-Relative, +50% vs H2O), MiniCache (depth-wise, 5x compression).
- **Systeme:** Alle mit langem Kontext
- **Schwierigkeit:** Mittel-Schwer
- **Existiert:** Q-Filters: https://github.com/nathangodey/qfilters
- **Aufwand:** 2-3 Wochen pro Methode
- **Gain:** 5-32x KV compression bei erhaltener Qualität

### 27. Pre-Attention Expert Prediction
- **Was:** Expert prediction VOR attention block statt nach previous layer. 93-97% accuracy. 2 linear functions mit ranking-aware loss.
- **Systeme:** Styx, Mars
- **Schwierigkeit:** Mittel-Schwer
- **Existiert:** Forschungspapier (kein Code)
- **Aufwand:** 2-3 Wochen
- **Gain:** 15% absolute accuracy improvement über state-of-the-art prefetching

### 28. Adaptive MTP Speculative Decoding
- **Was:** Runtime-Anpassung von `--spec-draft-n-max` basierend auf Acceptance Rate, Context Length, Memory Pressure. Verhindert Regression.
- **Systeme:** Alle (MTP bereits im Fork aber kein Speedup)
- **Schwierigkeit:** Mittel
- **Existiert:** PR #22931 (Fallback, keine adaptive Anpassung)
- **Aufwand:** 1-2 Wochen
- **Gain:** Verhindert -14% Regression auf Styx, könnte MTP nutzbar machen

### 29. FastKV: Token-Selective Propagation
- **Was:** Early Layers full-context, Late Layers nur kritische Tokens. Entkopplung von Prefill-Compute-Reduktion und KV-Budget.
- **Systeme:** Alle (besonders Long-Context)
- **Schwierigkeit:** Mittel-Hoch
- **Existiert:** https://github.com/dongwonjo/fastkv
- **Aufwand:** 3-4 Wochen
- **Gain:** Bis zu 1.82x Prefill, 2.87x Decode Speedup

### 30. Speculative Expert Prefetching (MoE-SpeQ)
- **Was:** Kleines on-device draft model future experts. Quantized model als natural predictor (zero-training-cost). Adaptive governor mit Amortization Roofline Model.
- **Systeme:** Styx (CUDA), Mars (Vulkan — theoretisch)
- **Schwierigkeit:** Schwer
- **Existiert:** Forschung (kein Code)
- **Aufwand:** 3-4 Wochen
- **Gain:** Bis zu 2.34x speedup über state-of-the-art offloading

---

## Nicht empfohlen / nicht anwendbar

| Ansatz | Grund |
|--------|-------|
| FlashAttention-3 (arXiv:2407.08608) | Hopper-spezifisch (H100), unsere GPUs: Pascal/Ampere/Ada/RDNA3 |
| Programmatic Dependent Launch (PDL) | Hopper+ Feature, nicht auf unserer Hardware |
| GANQ (arXiv:2501.12956) | Erfordert LUT-basierte GEMM Kernels, sehr GPU-spezifisch |
| valkyr (Zig+Vulkan) | Kompletter Stack-Wechsel zu Zig, 8-12 Wochen |
| AnchorTP / Shift Parallelism | Sehr komplex (10-14 Wochen), nur für Uranus |
| Tiled Flash Linear Attention | Für zukünftige Modelle (xLSTM), nicht für aktuelle Transformer |
| NUMA-aware Optimization | Arm Neoverse spezifisch, unsere x86 CPUs haben kein relevantes NUMA |
| Continuous Batching / Chunked Prefill | In llama-server bereits teilweise vorhanden (`--cont-batching`), vLLM-Features nicht portierbar |

---

## Priorisierung nach Cost-Benefit

### Sofort umsetzbar ( diese Woche):
1. **#6 MUL_MAT_ID Subgroup** — 1 Tag, bis 657% MoE PP Speedup auf AMD
2. **#1 MTP Logits Copy** — 2-3 Stunden, +20% PP mit MTP
3. **#2 Tensor Split Regex** — 1-2 Stunden, befreit 40% decode thread
4. **#5 GTT Size Tuning** — 1 Stunde, mehr GPU-Speicher für Mars
5. **#4 n-gram Decoding** — 0 Stunden, nur aktivieren

### Diese / nächste Woche:
6. **#7 Vulkan FA Refactor** — 1 Woche, 10-20% scalar FA improvement
7. **#3 Pascal MMVQ** — 2-3 Tage, +3-6% decode auf Styx
8. **#14 LFU Caching** — 2-3 Tage, 15-20% expert hit rate
9. **#9 Vulkan Shmem-Staging** — 2-3 Tage, >2.5x TG potenziell

### Mittelfristig (2-4 Wochen):
10. **#10 UMA Zero-Copy** — 1-2 Wochen, +112x transfer speed für Mars
11. **#8 Mixed Precision KV** — 1 Woche, hot/cold token management
12. **#13 Two-Tier Expert Cache** — 1-2 Wochen, kritisch für Styx 8GB
13. **#11 EAGLE-3** — 2-3 Wochen, bis 6.5x speedup
14. **#15 PipeShard** — 2-3 Wochen, VRAM-constrained MoE

### Langfristig (4+ Wochen):
15. **#12 Coopmat2 RDNA3** — 2-3 Wochen, 2.5x FA multiply
16. **#17 HOBBIT** — 4-6 Wochen, 9.93x MoE offloading
17. **#20 Tensor Parallelism** — 3-4 Wochen, 40% boost für Uranus
18. **#25 2026 Rewrite Merge** — 4-6 Wochen, system-wide 1.4-2.1x
