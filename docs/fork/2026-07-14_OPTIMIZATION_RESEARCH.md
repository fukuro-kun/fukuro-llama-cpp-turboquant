# Optimierungs-Ansätze für fukuro-llama-cpp-turboquant

**Recherche:** 2026-07-14, Web + arXiv (4 parallele Subagents)
**Hardware:** Mars (RDNA3/Vulkan/30GB), Styx (Pascal/CUDA/8GB), Hydra (Ampere/CUDA/8GB), Uranus (2x Ada/CUDA/32GB), Venus (GCN/Vulkan)
**Ausschluss:** Treiber-Neuimplementierung, Kernel-Rekompilierung, Hopper-spezifische Features

---

## Tier 1: Quick Wins (< 1 Woche)

### 1. K-Quant MMVQ Path Fix für RDNA3
- **Was:** Issue #21151 zeigt dass Q4_K/Q5_K/Q2_K auf RDNA3 über den MMVQ-Pfad 10-15x langsamer sind als f32-Dequant (2.4% vs 37% Bandbreitennutzung). Fix: Diese Quant-Typen vom MMVQ-Pfad auf non-GCN AMD ausschließen.
- **Systeme:** Mars (RDNA3), Venus (GCN — bereits korrekt)
- **Schwierigkeit:** einfach
- **Existiert:** Issue #21151 (llama.cpp) — **unverifiziert ob upstream gefixt**
- **Aufwand:** 2-4 Stunden
- **Gain:** 10-15x für Q4_K/Q5_K single-token decode auf RDNA3 (falls tatsächlich betroffen)
- **Status:** Fork hat `ggml_vk_should_use_mmvq()` mit AMD-spezifischer Logik (k<2048→false). Aber Q4_K/Q5_K fallen durchs Raster (default→true bei k≥2048). **Zu verifizieren mit Benchmark auf Mars.**

### 2. Vulkan Pipeline Cache Disk Persistence
- **Was:** Speichert Pipeline-Cache-Binaries auf Disk zwischen Programmläufen, um teure Shader-Rekompilierung zu vermeiden. GGML_VK_CACHE_DIR Environment Variable.
- **Systeme:** Alle Vulkan-Systeme (Mars, Venus)
- **Schwierigkeit:** mittel
- **Existiert:** GitHub: Perinban/llama.cpp commit 1b7250c — **unverifiziert**
- **Aufwand:** 1-2 Tage
- **Gain:** Reduziert Startup-Zeit erheblich (Shader-Kompilierung entfällt nach erstem Lauf)

### 3. NCCL Communication Optimization für Multi-GPU
- **Was:** Automatische Nutzung von NCCL für Cross-GPU Reductions in Tensor-Mode statt manueller PCIe-Kopien. Erkennt schnellsten Pfad (NVLink vs PCIe) automatisch.
- **Systeme:** Uranus (2x 4060 Ti)
- **Schwierigkeit:** einfach
- **Existiert:** docs/multi-gpu.md (bereits dokumentiert) — **verifiziert**
- **Aufwand:** 1 Woche
- **Gain:** Bis 2x für Multi-GPU Tensor Parallelism auf Uranus

---

## Tier 2: Mittelfristig (1-3 Wochen)

### 4. GEAR — KV Cache Compression (Quant+LowRank+Sparse)
- **Was:** KV-Cache-Kompression: Quantisiert Mehrheit der Entries auf 4-bit, Low-Rank-Matrix approximiert Quantisierungsfehler, Sparse-Matrix remediert Outlier-Errors. Near-lossless 4-bit KV-Kompression mit 2.38x Throughput.
- **Systeme:** Alle (hardware-agnostisch), besonders Mars (224k Kontext) und Styx (8GB VRAM)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2403.05527 — GitHub: HaoKang-Timmy/GEAR (öffentlich)
- **Aufwand:** 3-6 Wochen (Portierung in ggml KV-Cache-Format)
- **Gain:** Komplementär zu TurboQuant (andere Kompressionsstrategie), 2.38x Throughput

### 5. PEARL — Parallel Speculative Decoding
- **Was:** Löst das Mutual-Waiting-Problem von SD: Pre-Verify verifiziert ersten Draft-Token während Drafting-Phase, Post-Verify generiert weitere Draft-Tokens während Verification-Phase. Bis 4.43x über Auto-Regressive, 1.50x über Vanilla SD.
- **Systeme:** Alle mit SD (Uranus, Hydra, Mars, Styx)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2408.11850 — GitHub: smart-lty/ParallelSpeculativeDecoding (öffentlich, ICLR 2025)
- **Aufwand:** 3-6 Wochen (Integration in bestehende EAGLE-3/MTP-Pipeline)
- **Gain:** 1.50x über Vanilla SD, komplementär zu EAGLE-3

### 6. Fiddler — CPU-GPU Orchestration for MoE
- **Was:** Resource-efficient MoE-Inferenzsystem, das CPU- und GPU-Ressourcen strategisch nutzt. 1.26x Single-Batch, 1.30x Long-Prefill, 11.57x Beam-Search Speedup. ICLR 2025.
- **Systeme:** Styx (MoE-Offload), Mars (UMA), Hydra, Venus
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2402.07033 — GitHub: efeslab/fiddler (öffentlich, ICLR 2025)
- **Aufwand:** 3-6 Wochen (Portierung der Orchestrierungs-Logik)
- **Gain:** 1.26-11.57x je nach Workload

### 7. IQ*_K Quantization mit Importance Matrix
- **Was:** Portiert ik_llama.cpp IQ2_K bis IQ6_K Quantisierung mit layer-wise importance matrix für verbesserte Qualität bei gleicher Bit-Rate. Verwendet Aktivierungs-Statistiken für intelligentere Quantisierung.
- **Systeme:** Alle (besonders bei VRAM-Knappheit)
- **Schwierigkeit:** mittel
- **Existiert:** PR #19726 (CPU Backend portiert)
- **Aufwand:** 2-3 Wochen
- **Gain:** Bessere Qualität bei gleicher Größe als Standard Q-Quants

### 8. CUDA Graph Capture für Decode Phase
- **Was:** Capturiert gesamte Kernel-Sequenz eines Decode-Schritts und replays sie mit einem einzigen Launch statt hunderte einzelner Kernel-Aufrufe. Reduziert CPU-Overhead um bis zu 38%.
- **Systeme:** Uranus, Hydra, Styx (alle CUDA)
- **Schwierigkeit:** mittel
- **Existiert:** PR #9017, NVIDIA Technical Blog
- **Aufwand:** 2-3 Wochen
- **Gain:** 15-38% Launch-Overhead-Reduktion
- **Einschränkung:** Styx mit MoE-Offload erzeugt Split-Buffers → CUDA Graphs deaktiviert. Nur nutzbar bei voller GPU-Offload.

### 9. Wave32/Wave64 Subgroup Size Tuning für RDNA3
- **Was:** RDNA3 unterstützt sowohl Wave32 als auch Wave64. Für memory-bandwidth-bound matmul-vec Operationen ist Wave64 oft optimal. PR #12087 begann Tests hierfür.
- **Systeme:** Mars (RDNA3), Venus (GCN)
- **Schwierigkeit:** mittel
- **Existiert:** PR #12087 (unvollständig)
- **Aufwand:** 2-3 Tage
- **Gain:** Potenziell 5-15% auf matmul-vec Operationen

### 10. Vulkan Push Descriptors (VK_KHR_push_descriptor)
- **Was:** Erlaubt es, Deskriptoren direkt in den Command Buffer zu schreiben statt Deskriptor-Sets zu binden. Reduziert CPU-Overhead bei häufigen Updates pro Layer.
- **Systeme:** Alle Vulkan-Systeme (Mars, Venus)
- **Schwierigkeit:** mittel
- **Existiert:** nein (Vulkan-Standard-Extension)
- **Aufwand:** 3-5 Tage
- **Gain:** Reduziert CPU-Overhead bei Descriptor-Binding

### 11. Dynamic Speculative Decoding (DSD) ohne Draft-Model
- **Was:** Self-speculative decoding mit ngram-map und adaptive skip-streak basierend auf Acceptance Rate. Kein separates Draft-Model erforderlich.
- **Systeme:** Alle (besonders Styx mit limitiertem VRAM)
- **Schwierigkeit:** mittel
- **Existiert:** Commit 72d3b18 (PR #18471)
- **Aufwand:** 2-3 Wochen
- **Gain:** SD ohne Draft-Model-Overhead
- **Hinweis:** Fork hat bereits Adaptive MTP (skip-streak), aber DSD nutzt ngram-map statt MTP.

### 12. Cross-Layer Gate Expert Prediction
- **Was:** Nutzt Cross-Layer-Gating für genauere Expert-Prediction ohne Fine-Tuning. Reduziert Trainingskosten gegenüber anderen gelernten Methoden.
- **Systeme:** Styx, Hydra (MoE-Offloading)
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:2502.12224
- **Aufwand:** 1-2 Wochen
- **Gain:** Höhere Prediction-Accuracy → besseres Prefetching

---

## Tier 3: Komplex (3-6 Wochen)

### 13. SageAttention — INT8 Quantized Attention
- **Was:** Quantisiert QK^T auf INT8 und PV auf FP16/FP8 mit Per-Kernel Smoothing → 2.1x über FlashAttention2. Plug-and-play, nahezu verlustfrei.
- **Systeme:** Hydra (Ampere ✓), Uranus (Ada ✓, FP8 möglich) — **NICHT** Mars (Vulkan), Styx (Pascal CC 6.1), Venus (Vulkan)
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:2410.02367 — GitHub: thu-ml/SageAttention (Triton+CUDA)
- **Aufwand:** 6-10 Wochen (ggml-CUDA-Backend-Integration)
- **Gain:** 2.1x über FlashAttention2

### 14. ScoutAttention — Layer-Ahead CPU Pre-Computation
- **Was:** KV-Cache-Offloading mit GPU-CPU kollaborativer Block-Sparse-Attention. CPU startet Attention-Berechnung eine Layer im Voraus. 2.1x Speedup, Genauigkeit innerhalb 2.4%. DAC 2026.
- **Systeme:** Mars (UMA), Styx (KV-Offload), Hydra, Venus, Uranus
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2603.27138 — keine öffentliche Implementierung
- **Aufwand:** 4-8 Wochen
- **Gain:** 2.1x über bestehende Offloading-Methoden

### 15. HGCA — Hybrid GPU-CPU Attention
- **Was:** Dense Attention auf recent KV in GPU + parallele Sparse Attention auf salient KV in CPU. Attention-Outputs werden via Log-Sum-Exp-Fusion gemerged. Kein Retraining nötig.
- **Systeme:** Mars (UMA), Styx (KV-Offload), Hydra, Venus
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2507.03153 — keine öffentliche Implementierung
- **Aufwand:** 4-8 Wochen
- **Gain:** Längere Sequenzen + größere Batch-Größen

### 16. DMS — Dynamic Memory Sparsification
- **Was:** Lernt Per-Head Eviction-Policy für KV-Cache. 8x Kompression mit nur ~1K Training Steps. +12.0 Punkte AIME 24 für Qwen-R1 32B. NeurIPS 2025.
- **Systeme:** Alle, besonders Mars (224k Kontext, Reasoning-Modelle), Uranus
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:2506.05345 — GitHub: NVIDIA/Model-Optimizer + shisa-ai/FastDMS
- **Aufwand:** 6-10 Wochen
- **Gain:** 8x KV-Kompression, Inference-Time Hyper-Scaling

### 17. ML-SpecQD — Multi-Level SD with Quantized Drafts
- **Was:** Plug-and-Play SD: Nutzt MXFP4 Weight-Only-Quantization des Target-Modells als Draft (direkter Cast, kein Pre-Training nötig!). Multi-Level: MXFP4-Draft wird selbst per SD beschleunigt. Bis 2.72x über BF16.
- **Systeme:** Alle
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2503.13565 — keine offizielle GitHub-Repo
- **Aufwand:** 4-8 Wochen (MXFP4/WOQ-Draft + rekursive SD-Pipeline)
- **Gain:** 2.72x über BF16

### 18. SP-MoE — SD-Aware Expert Prefetching
- **Was:** Nutzt strukturelle Korrespondenz zwischen Draft- und Target-Modell für spekulativen Expert-Prefetch vor Verifikation. 1.07-3.5x TPOT-Speedup.
- **Systeme:** Styx, Mars, Hydra (MoE + SD kombiniert), Uranus
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2510.10302 — Code nicht verlinkt
- **Aufwand:** 4-8 Wochen
- **Gain:** 1.07-3.5x TPOT-Speedup

### 19. ExpertFlow — Predictive Expert Caching + Token Scheduling
- **Was:** Transformer-basiertes Routing-Path-Predictor, das Expert-Nutzung über alle MoE-Layer in einem Forward-Pass schätzt. Bis 10x Throughput, 93.72% GPU-Memory-Reduktion.
- **Systeme:** Alle mit MoE-Offloading
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:2410.17954 — GitHub existiert (basiert nicht auf llama.cpp)
- **Aufwand:** 6-10 Wochen
- **Gain:** 10x Throughput

### 20. SliceMoE — Bit-Sliced Expert Caching
- **Was:** Dynamic Bit-Sliced Caching auf Slice-Level-Granularität mit on-demand Präzision. Bis 1.81x Decode-Latenz-Reduktion. DAC 2026.
- **Systeme:** Styx (8GB VRAM, Cache-kritisch), Mars, Hydra
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:2512.12990 — keine öffentliche Implementierung
- **Aufwand:** 8-12 Wochen
- **Gain:** 1.81x Decode-Latenz-Reduktion

### 21. SpargeAttention — Training-Free Sparse Attention
- **Was:** Two-Stage Online Filter: Stage 1 prädiziert Attention-Map schnell → skip QK^T; Stage 2 ist Online Softmax-aware Filter → skip PV. ICML 2025. Basiert auf SageAttention.
- **Systeme:** Hydra, Uranus (gleiche CUDA-Limits wie SageAttention)
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:2502.18137 — GitHub: thu-ml/SpargeAttn (öffentlich)
- **Aufwand:** 8-12 Wochen
- **Gain:** Training-free sparse attention ohne Genauigkeitsverlust

### 22. SparseInfer — Training-Free Activation Sparsity Prediction
- **Was:** Training-Free Predictor für Aktivierungs-Sparsity in ReLU-basierten LLMs. Vergleicht nur Sign-Bits von Inputs und Weights.
- **Systeme:** Uranus, Hydra, Mars (bei ReLU-konvertierten Modellen)
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:2411.12692 — keine öffentliche Implementierung
- **Aufwand:** 8-12 Wochen
- **Gain:** Speedup für ReLU-basierte Modelle
- **Einschränkung:** Erfordert ReLU-Aktivierung (SiLU→ReLU Konvertierung nötig für Standard-LLMs)

---

## Tier 4: Langfristig / Forschung (3+ Monate)

### 23. Radix Tree Prefix Caching
- **Was:** Erweitert bestehendes Prefix-Caching um token-level Radix Tree Struktur für aggressivere Wiederverwendung von KV-Cache über tausende Requests mit gemeinsamen Prefixes. Inspiriert von SGLangs RadixAttention.
- **Systeme:** Alle Server (besonders mit vielen Concurrent Users)
- **Schwierigkeit:** schwer
- **Existiert:** SGLang Blog, arXiv:2507.07400 (KVFlow)
- **Aufwand:** 3-4 Wochen
- **Gain:** Aggressivere Cache-Wiederverwendung

### 24. Disaggregated Prefill/Decode
- **Was:** Trennt Prompt-Processing und Token-Generation auf unterschiedliche Geräte/Maschinen. Reduziert Interferenz zwischen prefill-heavy und decode-heavy Workloads.
- **Systeme:** Uranus (ideal für 2x 4060 Ti Aufteilung)
- **Schwierigkeit:** schwer
- **Existiert:** Issue #21266 (Diskussion, kein PR)
- **Aufwand:** 6-8 Wochen
- **Gain:** Optimierte Resource-Auslastung bei Multi-GPU

### 25. Vulkan Indirect Dispatch für MoE
- **Was:** vkCmdDispatchIndirect erlaubt es, Dispatch-Parameter von einem GPU-Buffer zu lesen statt von der CPU. Für MoE-Expert-Dispatch könnte dies CPU-GPU-Synchronisation reduzieren.
- **Systeme:** Alle Vulkan-Systeme (Mars, Venus)
- **Schwierigkeit:** schwer
- **Existiert:** nein (Vulkan-Standard-Feature)
- **Aufwand:** 1-2 Wochen
- **Gain:** Reduziert CPU-GPU-Synchronisation bei MoE

### 26. Multi-threaded Command Buffer Recording
- **Was:** Paralleles Aufzeichnen von Command Buffers auf mehreren Threads könnte CPU-Overhead reduzieren.
- **Systeme:** Alle Vulkan-Systeme
- **Schwierigkeit:** schwer
- **Existiert:** PR #9118 (diskutiert, nicht implementiert)
- **Aufwand:** 2-3 Wochen
- **Gain:** Reduziert CPU-Overhead bei Command Buffer Erstellung

---

## Bereits im Fork (verifiziert bei dieser Recherche)

- **Block-load Q3_K/Q6_K (PR #23056):** Bereits im Fork (Commit 473c3f633)
- **FlashAttention Chunking (PR #16829):** Bereits im Fork (Commit 55dd854e8)
- **A-Matrix Transpose (PR #22970):** Evaluiert als #31 ❌ (CM1-only, Mars nutzt CM2)

---

## Top 5 Empfehlungen (nach Aufwand-Nutzen-Ratio)

1. **#1 K-Quant MMVQ Path Fix** — 2-4h, 10-15x für Q4_K auf RDNA3. **Zuerst Benchmark auf Mars ob tatsächlich betroffen.**
2. **#4 GEAR** — 3-6 Wochen, öffentlicher Code, komplementär zu TurboQuant, hardware-agnostisch
3. **#5 PEARL** — 3-6 Wochen, öffentlicher Code (ICLR 2025), integriert in EAGLE-3-Pipeline
4. **#6 Fiddler** — 3-6 Wochen, öffentlicher Code (ICLR 2025), CPU-GPU MoE Orchestration
5. **#2 Vulkan Pipeline Cache Disk Persistence** — 1-2 Tage, reduziert Startup-Zeit erheblich

---

## Hardware-Kompatibilitäts-Matrix (Top Items)

| Ansatz | Mars (RDNA3/Vulkan) | Styx (Pascal/CUDA) | Hydra (Ampere/CUDA) | Uranus (Ada/CUDA) | Venus (GCN/Vulkan) |
|--------|:---:|:---:|:---:|:---:|:---:|
| K-Quant MMVQ Fix | ✅⭐ | ❌ | ❌ | ❌ | ✅ |
| Vulkan Pipeline Cache | ✅⭐ | ❌ | ❌ | ❌ | ✅⭐ |
| GEAR | ✅ | ✅ | ✅ | ✅ | ✅ |
| PEARL | ✅ | ✅ | ✅ | ✅⭐ | ✅ |
| Fiddler | ✅ | ✅⭐ | ✅ | ✅ | ✅ |
| IQ*_K | ✅ | ✅ | ✅ | ✅ | ✅ |
| CUDA Graph | ❌ | ⚠️ (MoE-Offload) | ✅ | ✅⭐ | ❌ |
| SageAttention | ❌ | ❌ | ✅ | ✅⭐ | ❌ |
| ScoutAttention | ✅⭐ | ✅⭐ | ✅ | ✅ | ✅ |

⭐ = besonders geeignet für diese Hardware
