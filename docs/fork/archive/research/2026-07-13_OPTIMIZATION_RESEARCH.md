# Optimierungs-Ansätze für fukuro-llama-cpp-turboquant

**Recherche:** 2026-07-13, Web + arXiv (4 parallele Subagents)
**Hardware:** Mars (RDNA3/Vulkan/30GB), Styx (Pascal/CUDA/8GB), Hydra (Ampere/CUDA/8GB), Uranus (2x Ada/CUDA/32GB), Venus (GCN/Vulkan)
**Ausschluss:** Treiber-Neuimplementierung, Kernel-Rekompilierung, Hopper-spezifische Features

---

## Verifizierung: Bereits im Fork

Folgende PRs wurden von Subagents vorgeschlagen, sind aber **bereits im Fork** (via git log verifiziert):

| PR | Beschreibung | Commit |
|----|-------------|--------|
| #22887 | Vulkan MUL_MAT_VEC 4K Iteration F16/32 | 672767aa9 |
| #19376 | Vulkan Coopmat2 FA FP16 Accumulators | 4cae4a13d |
| #23964 | Vulkan WHT Shared Memory Reduction | 82813274b |
| #19645 | CUDA Graphs für MMID BS 1-4 | 7f31e3eff |
| #12183 | Flash Decoding Occupancy Optimization | 80debf98b |
| #16715 | CUDA General GEMV Fusion | eabf7650e |
| #22299 | Internal AllReduce Kernel CUDA | cf1de535b |
| #19769 | NVFP4 Quantization Type | d6835b314 |
| #20518 | Vulkan async and event fixes | 1e528d5f2 |

## Verifizierung: Bereits in ROADMAP

| ROADMAP # | PR/arXiv | Status |
|-----------|----------|--------|
| #18 | DALI (arXiv:2602.03495) | ☐ |
| #19 | Expected Attention (arXiv:2510.00636) | ☐ |
| #21 | PagedAttention (PR #22569) | ☐ |
| #24 | HybriMoE (arXiv:2504.05897) | ☐ |
| #36 | Auto Parameter Fitting TP (PR #22950) | ☐ |
| #45 | CUDA Concurrent Streams (PR #16991) | ❌ |

---

## Tier 1: Quick Wins (< 1 Woche)

### 76. Vulkan UMA Cached Host Memory Preference
- **Was:** Bevorzugt cached host memory (HostVisible|HostCoherent|HostCached) statt write-combining memory auf UMA-Systemen. Verhindert langsame read-backs von WC-Speicher. Setzt memory_property_flags korrekt auf das tatsächlich allokierte Memory.
- **Systeme:** Mars (RDNA3 APU), Venus (GCN)
- **Schwierigkeit:** einfach
- **Existiert:** PR #23762 (open, nicht merged) — **verifiziert via webfetch**
- **Aufwand:** 4-8h
- **Gain:** Auf AMD UMA: Get-BW von 4.6→14.97 GB/s (3x), Set-BW von 3.9→14.52 GB/s (3.7x) bei großen Transfers. Direkter Benefit für Mars APU.

### 67. MMQ Stream-k Disable für Tensor-Split MoE
- **Was:** Deaktiviert MMQ stream-k für tensor-split MoE workloads da stream-k decomposition/fixup overhead schlechter performt als straight xy tiling. Verwendet cp.async für verbleibende tile loads.
- **Systeme:** Uranus (2x RTX 4060 Ti, tensor parallelism)
- **Schwierigkeit:** einfach
- **Existiert:** PR #22170 (merged) — **nicht im Fork (git log verifiziert)**
- **Aufwand:** 2-4h
- **Gain:** 30-50% P2P prompt improvement auf Dual-GPU MoE

### 68. KV Cache Size Limiting mit Demand Paging
- **Was:** --kv-cache-tokens N limitiert KV-Allokation auf N Tokens statt vollem Context. --kv-cache-demand-paged nutzt mmap(MAP_NORESERVE) für lazy physical page allocation. Block-Tracking als Foundation für PagedAttention.
- **Systeme:** Alle (besonders RAM-limitierte: Styx 8GB)
- **Schwierigkeit:** einfach
- **Existiert:** PR #18747 — **nicht im Fork (git log verifiziert)**
- **Aufwand:** 1-2 Wochen
- **Gain:** Reduziert KV-Memory bei langen Contexts, ermöglicht größere Contexts auf VRAM-limitierten GPUs

### 69. Tensor Prefetching (--prefetch-weights)
- **Was:** Überlappt compute der aktuellen Layer mit weight loading für nächste Layer via async copy engine. Nur CUDA, erfordert --no-mmap für pinned memory. Für MoE weniger effektiv bei kleinen ubatch, profitabel bei großen ubatch (1024+).
- **Systeme:** Styx, Hydra, Uranus (alle CUDA)
- **Schwierigkeit:** mittel
- **Existiert:** PR #21067 (draft) — **nicht im Fork**
- **Aufwand:** 1-2 Tage
- **Gain:** Layer-level Prefetch, komplementär zu thecodacus Expert-Prefetch (welcher MoE-spezifisch ist)

### 70. RADV Driver-Specific Shader Optimizations
- **Was:** Nutzt RADV-spezifische Environment-Variables (radv_invariant_geom, radv_no_dynamic_bounds, etc.) für Shader-Compiler-Tuning. Mesa 26.0+ Features.
- **Systeme:** Mars (RADV), Venus (RADV)
- **Schwierigkeit:** einfach
- **Existiert:** nein (Driver-Konfiguration)
- **Aufwand:** 4-8h
- **Gain:** Unbekannt, muss benchmarked werden. RADV-spezifische Optimierungen können 5-15% bringen.

---

## Tier 2: Mittelfristig (1-3 Wochen)

### 71. Persistent VRAM Expert Cache
- **Was:** Persistenter VRAM-Puffer für hot CPU-resident experts mit explizitem expert→slot bookkeeping. Vermeidet synchrone PCIe-Transfers bei cache hits. Invertiertes execution model: MUL_MAT_ID bleibt auf CPU, GPU führt cached rows parallel aus.
- **Systeme:** Styx (8GB VRAM, MoE-Offloading), Uranus
- **Schwierigkeit:** schwer
- **Existiert:** PR #23170 (merged), RFC #24528 (discussion), PR #21614 (closed) — **nicht im Fork**
- **Aufwand:** 1-2 Wochen
- **Gain:** Massiver Speedup möglich bei hohen cache hit rates. Löst das Problem das unser Expert-Cache hatte (Buffer-Recycling) durch persistente Slots.

### 72. MoE Expert Profiling & REAP Pruning
- **Was:** C++ profiler für REAP saliency scores direkt aus GGUF inference via ggml_backend_eval_callback. GGUF pruner für direktes pruning von expert axes.
- **Systeme:** Alle MoE
- **Schwierigkeit:** mittel
- **Existiert:** PR #20454 (open) — **nicht im Fork**
- **Aufwand:** 1-2 Tage
- **Gain:** Ermöglicht intelligente Expert-Placement-Entscheidungen basierend auf echten Aktivierungs-Profilen. Nutzt unsere vorhandene MoE-Frequency-Tracking-Infrastruktur.

### 73. xKV: Cross-Layer KV-Cache Compression
- **Was:** Training-freie Kompression via SVD über Layer-Gruppen, nutzt Alignment dominanter singulärer Vektoren. Selective Reconstruction für wichtige Layer.
- **Systeme:** Alle (besonders Mars/Uranus mit viel Memory)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2503.18893, GitHub: abdelfattah-lab/xKV
- **Aufwand:** 2-3 Wochen
- **Gain:** Bis zu 8× KV-Cache Kompression bei 2-3% Genauigkeitsverlust. Orthogonal zu TurboQuant (Cross-Layer vs Token-Level).

### 74. Block-Sparse Flash Attention (BSFA)
- **Was:** Drop-in Ersatz für FlashAttention, der basierend auf kalibrierten Thresholds irrelevante Value-Blocks überspringt. Training-frei, benötigt nur einmalige Kalibrierung auf 16 Samples.
- **Systeme:** Alle CUDA (Hydra, Uranus, Styx)
- **Schwierigkeit:** einfach
- **Existiert:** arXiv:2512.07011, GitHub: Danielohayon/Block-Sparse-Flash-Attention
- **Aufwand:** 1-2 Wochen
- **Gain:** 1.10-1.24× Speedup bei >99% Genauigkeit. Komplementär zu TurboQuant FA.

### 75. Pre-Attention Expert Prediction
- **Was:** Pre-attention expert prediction mit 2 linear functions und ranking-aware loss vor attention block im selben layer. Unterstützt prefetching im ersten layer.
- **Systeme:** Styx, Hydra, Uranus (alle MoE)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2511.10676
- **Aufwand:** 1-2 Wochen
- **Gain:** 93-97% expert prediction accuracy. Ermöglicht präzisen Prefetch vor MoE-Layer.

### 76. BucketServe: Bucket-Based Dynamic Batching
- **Was:** Bucket-basiertes dynamisches Batching für hohe Throughput und niedrige Latenz unter konkurrierenden Workloads.
- **Systeme:** Alle (multi-request Szenarien)
- **Schwierigkeit:** einfach
- **Existiert:** arXiv:2507.17120
- **Aufwand:** 1-2 Wochen
- **Gain:** Bis zu 3.58× Throughput-Improvement gegenüber UELLM.

### 67. MXFP4 Quantization für gpt-oss
- **Was:** OCP open-standard MXFP4 Format für gpt-oss Modelle mit Unterstützung über alle major Backends (CUDA, Vulkan, Metal, CPU).
- **Systeme:** Alle
- **Schwierigkeit:** einfach
- **Existiert:** Diskussion #15095
- **Aufwand:** 1-2 Wochen
- **Gain:** Ermöglicht native Nutzung von gpt-oss Modellen ohne Konvertierungsaufwand.

### 68. Vulkan Matmul Parameter-Tuning für AMD Coopmat
- **Was:** Optimiert Matmul-Parameter-Kombinationen speziell für AMD GPUs mit KHR_cooperative_matrix Unterstützung.
- **Systeme:** Mars (RDNA3)
- **Schwierigkeit:** mittel
- **Existiert:** PR #18749 (merged) — **Status im Fork unklar (false positive bei git log)**
- **Aufwand:** 1-2 Tage
- **Gain:** +1-3% auf RDNA3. Tuning-Änderung, kein Code-Refactoring.

---

## Tier 3: Komplex (3-6 Wochen)

### 69. FlashMoE: ML-based Cache Replacement
- **Was:** ML-basierte Caching-Strategie die recency und frequency signals adaptiv kombiniert. Offloading inaktiver experts zu SSD für RAM-constrained environments.
- **Systeme:** Styx (VRAM+RAM constrained), Hydra
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:2601.17063
- **Aufwand:** 2-3 Wochen
- **Gain:** Bis zu 51% höhere cache hit rate als LRU/LFU, bis zu 2.6× speedup.

### 70. ST-MoE: Spatio-Temporal Expert Prefetching
- **Was:** Spatio-temporal expert prefetching framework mit lightweight runtime prediction. Exploitiert Korrelation über adjacent MoE layers und consecutive decoding tokens.
- **Systeme:** Styx, Hydra, Uranus (alle MoE)
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:2606.15453
- **Aufwand:** 2-3 Wochen
- **Gain:** 85% expert prediction accuracy, 2.5× durchschnittlicher speedup.

### 71. Efficient CPU-GPU Collaborative MoE
- **Was:** N-index, M-way set-associative Expert-Cache auf GPU mit asynchronem CPU-Compute bei Cache-Misses. Exploitiert consecutive layer (44%) und token (40-60%) reuse patterns.
- **Systeme:** Styx, Hydra (consumer-grade hardware)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2512.16473, GitHub: elsa-lab/MoE-CPU-GPU-Collaborative-Inference
- **Aufwand:** 3-4 Wochen
- **Gain:** Signifikante CPU-GPU Kollaboration, setzt auf vorhandener thecodacus-Infrastruktur auf.

### 72. N4_0 Native 4-bit Float (Blackwell)
- **Was:** Native 4-bit Float Quant die Blackwells FP4 MMA nutzt. +40% PP für dense Modelle. Vereinfachte Variante als Ausgangspunkt.
- **Systeme:** Uranus (Ada — könnte teilweise profitieren)
- **Schwierigkeit:** mittel
- **Existiert:** PR #23572
- **Aufwand:** 2-3 Wochen
- **Gain:** +40% PP für dense Modelle auf Blackwell. Auf Ada (4060 Ti) möglicherweise weniger.

### 73. CascadeInfer: Length-Aware Scheduling
- **Was:** Partitioniert Instanzen in length-specialized groups für reduzierte Request-Length-Heterogenität innerhalb Batches.
- **Systeme:** Alle (multi-instance)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2512.19179
- **Aufwand:** 2-3 Wochen
- **Gain:** Reduziert End-to-End Latenz um bis zu 67% bei 2.89× Throughput-Steigerung.

### 74. Vulkan Descriptor Indexing (Bindless)
- **Was:** Nutzt VK_EXT_descriptor_indexing für bindless Resource-Access, reduziert vkCmdBindDescriptorSets Aufrufe.
- **Systeme:** Mars, Venus
- **Schwierigkeit:** schwer
- **Existiert:** nein (allgemeine Vulkan-Technik)
- **Aufwand:** 2-4 Wochen
- **Gain:** Reduziert Descriptor-Binding-Overhead, besonders bei vielen Tensoren.

### 75. Non-blocking Pipeline Scheduling (xdev)
- **Was:** Event-driven cross-GPU dependencies ermöglichen GPU-Self-Synchronisation während CPU nächsten Microbatch vorbereitet.
- **Systeme:** Uranus (Multi-GPU)
- **Schwierigkeit:** schwer
- **Existiert:** PR #19922
- **Aufwand:** 3-4 Wochen
- **Gain:** Reduziert Pipeline-Bubbles bei Multi-GPU Inference.

### 76. CPU Backend Operator Fusion
- **Was:** Operator Fusion für CPU Backend reduziert memory traffic und kernel launch time durch Kombination benachbarter Operationen.
- **Systeme:** Alle (CPU-only oder Hybrid)
- **Schwierigkeit:** mittel
- **Existiert:** Diskussion #22315
- **Aufwand:** 3-4 Wochen
- **Gain:** Reduziert Memory-Traffic, besonders nützlich für MoE-Offloading (CPU compute path).

---

## Tier 4: Langfristig / Forschung (6+ Wochen)

### 67. Dynamic Expert Quantization (DynaExq)
- **Was:** Adaptiert Expert-Bitbreiten dynamisch basierend auf Aktivierungsdichte statt statischer Offline-Quantisierung.
- **Systeme:** Styx, Hydra, Uranus (MoE-Offloading)
- **Existiert:** arXiv:2511.15015
- **Aufwand:** 4-6 Wochen

### 68. SliceMoE: Bit-Sliced Expert Caching
- **Was:** Cache-Management auf Bit-Slice-Ebene mit Dynamic Bit-Sliced Caching und Predictive Cache Warmup.
- **Systeme:** Styx (8GB, MoE-Offloading)
- **Existiert:** arXiv:2512.12990
- **Aufwand:** 5-7 Wochen

### 69. Mixture of Attentions for Speculative Decoding
- **Was:** Trainiert K neue LM-Heads für parallele Token-Generierung, kombiniert Attention-Mixture mit Speculative Decoding.
- **Systeme:** Alle
- **Existiert:** arXiv:2410.03804
- **Aufwand:** 5-6 Wochen

### 70. MuxWise: Prefill-Decode Multiplexing
- **Was:** Intra-GPU Multiplexing von Prefill und Decode mit bubble-less engine und SLO-aware dispatcher.
- **Systeme:** Alle CUDA
- **Existiert:** arXiv:2504.14489
- **Aufwand:** 4-5 Wochen

### 71. SpecForge Training Framework
- **Was:** Open-Source Framework für EAGLE-3 Draft-Model Training mit target-draft decoupling und hybrid parallelism.
- **Systeme:** Alle (benötigt Training)
- **Existiert:** arXiv:2603.18567, GitHub: sgl-project/SpecForge
- **Aufwand:** 6-8 Wochen

### 72. PFlash: Speculative Prefill
- **Was:** Kleines Draft-Modell bewertet Token-Importanz über Attention; schweres Target prefills nur die wichtigsten Spans.
- **Systeme:** Uranus, Hydra, Styx
- **Existiert:** GitHub: Luce-Org/lucebox-hub/pflash
- **Aufwand:** 6-8 Wochen
- **Gain:** 10.4× TTFT-Speedup bei 128K Kontext

### 73. Radix Tree Prefix Caching
- **Was:** Radix Tree (Trie) Datenstruktur für effizientes partielles Prefix-Matching, inspiriert von SGLangs RadixAttention.
- **Systeme:** Alle (besonders Server)
- **Existiert:** SGLang RadixAttention (extern)
- **Aufwand:** 5-7 Wochen
- **Gain:** 10× Speedup in Produktion (SGLang Referenz)

### 74. SlimCaching: Edge Caching for MoE
- **Was:** Verteiltes Inferenz-System mit Expert-Caching auf Edge-Devices.
- **Systeme:** Alle (distributed setups)
- **Existiert:** arXiv:2507.06567
- **Aufwand:** 5-6 Wochen

### 75. Future-Aware Quantization (FAQ)
- **Was:** Nutzt zukünftige Layer-Aktivierungen zur Quantisierungsführung statt nur aktueller Layer.
- **Systeme:** Alle
- **Existiert:** arXiv:2602.02538
- **Aufwand:** 2-3 Wochen

---

## Top-Empfehlungen nach Hardware

### Styx (GTX 1070, Pascal, 8GB) — MoE-Offloading
1. **#61 Persistent VRAM Expert Cache** (PR #23170) — Löst unser Cache-Problem, 1-2 Wochen
2. **#59 Tensor Prefetching** (PR #21067) — Layer-level Prefetch, 1-2 Tage
3. **#58 KV Cache Size Limiting** (PR #18747) — VRAM-Ersparnis, 1-2 Wochen
4. **#62 MoE Expert Profiling** (PR #20454) — Nutzt unsere Freq-Tracking-Infra, 1-2 Tage

### Mars (RDNA3, Vulkan, 30GB UMA)
1. **#56 Vulkan UMA Cached Host Memory** (PR #23762) — 3x Get-BW auf APU, 4-8h ⭐
2. **#60 RADV Driver Optimizations** — 4-8h, muss benchmarked werden
3. **#68 Vulkan Matmul Parameter-Tuning AMD** (PR #18749) — +1-3%, 1-2 Tage
4. **#64 Block-Sparse Flash Attention** — 1.24× Speedup, 1-2 Wochen

### Uranus (2x RTX 4060 Ti, Ada, 32GB)
1. **#57 MMQ Stream-k Disable** (PR #22170) — 30-50% P2P MoE, 2-4h ⭐
2. **#75 Non-blocking Pipeline Scheduling** (PR #19922) — Pipeline-Bubble-Reduktion
3. **#72 N4_0 Native 4-bit Float** (PR #23572) — +40% PP (Blackwell-bedingt)

### Alle Systeme
1. **#64 BSFA** — Drop-in FA Enhancement, 1-2 Wochen
2. **#63 xKV** — Orthogonale KV-Kompression, 2-3 Wochen
3. **#66 BucketServe** — 3.58× Throughput bei multi-request

---

## Nicht empfohlen / Ausschluss

- **Vulkan Shader Write Compression** — AMD DCC-spezifisch, erfordert tiefes AMD-Shader-Wissen, riskant ohne Hardware-Testing
- **SlimCaching** — Edge/Distributed Setup, nicht relevant für LAN-Infrastruktur
- **Mixture of Attentions** — Erfordert Training neuer LM-Heads, nicht training-frei
