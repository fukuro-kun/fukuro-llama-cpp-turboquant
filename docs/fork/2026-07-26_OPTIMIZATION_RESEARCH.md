# Optimierungs-Ansätze für fukuro-llama-cpp-turboquant

**Recherche:** 2026-07-26, Web + arXiv (4 parallele Subagents)
**Hardware:** Mars (RDNA3/Vulkan/30GB UMA), Styx (Pascal/CUDA/8GB), Hydra (Ampere/CUDA/8GB), Uranus (2x Ada/CUDA/32GB), Venus (GCN/Vulkan)
**Ausschluss:** Treiber-Neuimplementierung, Kernel-Rekompilierung, Hopper/Blackwell-spezifisch

---

## Verifikations-Ergebnisse

### Top 5 Quick-Wins verifiziert:

| PR | Titel | Status | Im Fork? | Fazit |
|----|-------|--------|----------|-------|
| #22887 | MUL_MAT_VEC 4K per Iteration (F16/32) | ✅ MERGED May 2026 | ⏳ Prüfen | +9% tg128 auf Intel BMG, alle Vulkan-Systeme |
| #25479 | Pascal MMVQ nwarps DP4A | 🔓 OPEN Jul 2026 | ❌ Nicht im Fork | +3.07% tg auf Pascal, Cherry-Pick-Kandidat |
| #15524 | MUL_MAT_ID Subgroup Non-Coopmat | ✅ MERGED Aug 2025 | ✅ Bereits im Fork (#6) | +657% auf GCN, bereits integriert |
| #16391 | Host-Memory Prompt Caching (--cram) | ✅ MERGED Oct 2025 | ✅ Bereits im Fork | -cram CLI arg vorhanden |
| #19754 | CUDA Graph Capture Improvement | ✅ MERGED Feb 2026 | ✅ Bereits im Fork | warmup_complete Logic vorhanden |

**Ergebnis:** 3 von 5 Quick-Wins bereits im Fork. 1 Cherry-Pick-Kandidat (#25479). 1 muss geprüft werden (#22887).

---

## Tier 1: Quick Wins (< 1 Woche)

### 1. Pascal MMVQ nwarps Optimierung
- **Was:** Dedizierte MMVQ launch-config für Pascal GPUs (CC 6.1/6.2) mit DP4A. 2 warps statt 4 für single-token decode. Reduziert register pressure und verbessert occupancy.
- **Systeme:** Styx (GTX 1070)
- **Schwierigkeit:** einfach
- **Existiert:** PR #25479 (OPEN, nicht gemerged) — verifiziert
- **Aufwand:** 2-3 Tage (Cherry-Pick)
- **Gain:** +3.07% tg geomean (Q4_0 +4.6%, Q4_K_M +5.4%, Q5_K_M +3.2%, Q8_0 +2.5%)

### 2. MUL_MAT_VEC 4K per Iteration (F16/32)
- **Was:** Erhöht K_PER_ITER von 2 auf 4 für F16/32 im MUL_MAT_VEC Kernel. Verbessert sequenzielle coalesced reads.
- **Systeme:** Mars, Venus, alle Vulkan-Systeme
- **Schwierigkeit:** einfach
- **Existiert:** PR #22887 (MERGED May 2026) — verifiziert. **Im Fork prüfen!**
- **Aufwand:** 1 Tag (Cherry-Pick falls nicht vorhanden)
- **Gain:** +9% tg128 auf Intel BMG (AMD RDNA3 muss verifiziert werden)

### 3. Internal AllReduce Kernel für CUDA
- **Was:** Interner Reduktionskernel für Tensor-Parallelismus als Fallback ohne NCCL. Bis 10% Token-Gen Gewinn.
- **Systeme:** Uranus (2x RTX 4060 Ti)
- **Schwierigkeit:** einfach
- **Existiert:** PR #22299 — unverifiziert
- **Aufwand:** 1-2 Wochen
- **Gain:** ~10% tg auf Uranus (wenn TP genutzt)

### 4. CPU Backend Operator Fusion (RMS_NORM + MUL)
- **Was:** Fused RMS_NORM + MUL Kernel auf CPU-Backend. Single-pass ohne Materialisierung des intermediären Results.
- **Systeme:** Alle (CPU fallback)
- **Schwierigkeit:** einfach
- **Existiert:** PR #22423, Commit 5dc3409 — unverifiziert. **Achtung:** ROADMAP #76 ⏭️ zeigt Regressionen auf Consumer-CPUs.
- **Aufwand:** 1 Woche
- **Gain:** ~5-10% CPU-Performance (aber #76 zeigt Regressionen!)

### 5. ASET Adaptive Skipping
- **Was:** Training-free adaptive activation policy mit router confidence und entropy penalty. Static layer-wise Top-k allocation.
- **Systeme:** Styx, Hydra, Uranus (MoE Decode)
- **Schwierigkeit:** einfach
- **Existiert:** arXiv:findings.acl.2140 — Paper-only
- **Aufwand:** 3-5 Tage
- **Gain:** 10-78% throughput (Paper-Angabe, muss verifiziert werden)
- **Risiko:** Ähnlich wie #44 Alloc-MoE (K-Reduktion) — Quality-Drop wahrscheinlich

---

## Tier 2: Mittelfristig (1-3 Wochen)

### 6. Transfer Queue für AMD RDNA3
- **Was:** Dedizierte Transfer Queue für async transfers auf AMD. Entlastet Compute Queue.
- **Systeme:** Mars (RDNA3)
- **Schwierigkeit:** mittel
- **Existiert:** PR #19976 — unverifiziert
- **Aufwand:** 2-3 Tage
- **Gain:** Gemischt (hilft auf RDNA4, RDNA3 muss getestet werden)

### 7. MUL_MAT_ID Non-Square Tile (128x32)
- **Was:** Nicht-quadratische Tile-Geometrie für MUL_MAT_ID auf AMD mit coopmat. 7-10% prefill speedup bei MoE.
- **Systeme:** Mars (RDNA3 Phoenix)
- **Schwierigkeit:** mittel
- **Existiert:** Discussion #22598 — unverifiziert
- **Aufwand:** 3-4 Tage
- **Gain:** 7-10% pp bei MoE auf Strix Halo (gfx1151)

### 8. RateQuant: Optimal Mixed-Precision KV Cache
- **Was:** Per-quantizer Verzerrungsmodelle via reverse waterfilling für head-spezifische Bit-Allocation. Kein Runtime-Overhead.
- **Systeme:** Alle (CUDA+Vulkan)
- **Schwierigkeit:** einfach
- **Existiert:** arXiv:2605.06675 — Paper-only
- **Aufwand:** 1-2 Wochen
- **Gain:** Bessere Qualität bei gleicher Kompression als TurboQuant? Muss evaluiert werden.

### 9. InnerQ: Hardware-aware KV-Cache Quantization
- **Was:** Inner-dimension grouping mit Dequantization aligned für GPU-Compute-Unit scale factor reuse.
- **Systeme:** Alle (besonders CUDA)
- **Schwierigkeit:** einfach
- **Existiert:** arXiv:2602.23200 — Paper-only
- **Aufwand:** 1-2 Wochen
- **Gain:** Hardware-aligned Dequant, könnte TurboQuant ergänzen

### 10. Backend-agnostic Tensor Parallelism (Meta Device)
- **Was:** Backend-agnostische TP über "meta device" Abstraktion. Automatische Graph-Splits, synchronisiert nur an notwendigen Punkten.
- **Systeme:** Uranus (2x RTX 4060 Ti)
- **Schwierigkeit:** mittel
- **Existiert:** PR #19378 — unverifiziert
- **Aufwand:** 2-3 Wochen
- **Gain:** Besser als layer-mode für Uranus

### 11. Dynamic KV Cache Resize (--kv-dynamic)
- **Was:** Startet KV Cache klein, wächst on-demand. Verhindert GPU OOM bei großem `-c` wenn Nutzung klein.
- **Systeme:** Mars (30GB UMA), Venus
- **Schwierigkeit:** mittel
- **Existiert:** PR #21757 (Draft) — unverifiziert
- **Aufwand:** 2-3 Wochen
- **Gain:** OOM-Vermeidung, kein direkter Speedup

### 12. Speculative Checkpointing für Hybrid Models
- **Was:** Checkpoint/Restore für spekulatives Decoding mit rekurrenten Modulen (DeltaNet/Mamba). Für Qwen3.5/3.6 MoE+SSM.
- **Systeme:** Uranus, Hydra, Styx
- **Schwierigkeit:** mittel
- **Existiert:** PR #19493, #20075, #20925 — unverifiziert
- **Aufwand:** 2-3 Wochen
- **Gain:** Ermöglicht spec-decoding auf Qwen3.5/3.6

### 13. FineMoE Fine-Grained Expert Offloading
- **Was:** Extrahiert fine-grained expert selection patterns aus prompts für adaptive prefetching/caching/offloading.
- **Systeme:** Styx (8GB VRAM, MoE Offloading)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2502.05370 — Paper-only
- **Aufwand:** 1-2 Wochen
- **Gain:** -47% latency, +39% expert hit rate (Paper-Angabe)

### 14. Shared Expert Auxiliary Stream
- **Was:** Shared-expert computation in separatem auxiliary CUDA stream während routed experts im main stream.
- **Systeme:** Hydra, Uranus (MoE mit shared experts)
- **Schwierigkeit:** mittel
- **Existiert:** TensorRT-LLM Referenz — unverifiziert für llama.cpp
- **Aufwand:** 1 Woche
- **Gain:** Overlap computation, reduziert latency

---

## Tier 3: Komplex (3-6 Wochen)

### 15. RDNA3-Specific Flash Attention Block Size Tuning
- **Was:** Autotuned Flash Attention block sizes für RDNA3's 32-lane WMMA fragments und 64KB LDS limit. Split-K decode.
- **Systeme:** Mars (RDNA3)
- **Schwierigkeit:** schwer
- **Existiert:** GitHub: chelokot/flash-attention-rdna3 — unverifiziert
- **Aufwand:** 1-2 Wochen
- **Gain:** Bessere KV cache bandwidth utilization

### 16. VQKV: Vector Quantization KV Cache
- **Was:** VQ-Indizierung statt Skalar-Quantisierung. 82.8% Kompression bei 98.6% Performance.
- **Systeme:** Alle (besonders VRAM-constrained)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2603.16435 — Paper-only. **Achtung:** ROADMAP #50 ⏭️ bereits erfasst.
- **Aufwand:** 2-3 Wochen
- **Gain:** 82.8% Kompression (vs TurboQuant 75-80%)

### 17. NestedKV: Key-Only KV-Cache Kompression
- **Was:** Multi-time-scale cosine anomaly scoring und adaptives Budgeting. Globale, block-level und sliding-window key anchors.
- **Systeme:** Alle (Long-Context)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2605.26678 — Paper-only
- **Aufwand:** 2-3 Wochen
- **Gain:** Training-free Kompression

### 18. HybriMoE: Hybrid CPU-GPU Scheduling
- **Was:** Dynamische intra-layer Scheduling und workload-aware cache replacement für MoE.
- **Systeme:** Styx, Hydra (CPU-GPU Collaborative)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:2504.05897 — Paper-only
- **Aufwand:** 3-4 Wochen
- **Gain:** Bessere Load-Balancing als thecodacus

### 19. SlimInfer: Dynamic Token Pruning
- **Was:** Pruned redundante hidden-state tokens in intermediate Layern mit asynchronem KV-Cache Manager.
- **Systeme:** Alle (Long-Context)
- **Schwierigkeit:** mittel
- **Existiert:** arXiv:AAAI 2026 — Paper-only
- **Aufwand:** 2-3 Wochen
- **Gain:** Token-Reduktion, weniger Compute

### 20. ExpertFlow Adaptive Scheduling
- **Was:** Adaptive expert prefetching mit runtime statistics. Hybrid cross-layer prediction. Model stall time <0.1%.
- **Systeme:** Styx, Hydra, Uranus
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:2510.26730 — Paper-only
- **Aufwand:** 2-3 Wochen
- **Gain:** Stall-Reduktion

---

## Tier 4: Langfristig/Forschung (6+ Wochen)

### 21. Paged KV Cache mit Scheduler
- **Was:** Opt-in paged KV Cache mit Fixed-Size-Blocks und Continuous-Batching-Scheduler. 10x höhere Concurrency.
- **Systeme:** Uranus, Hydra, Styx
- **Schwierigkeit:** schwer
- **Existiert:** PR #22569 (Draft) — unverifiziert
- **Aufwand:** 6-8 Wochen
- **Gain:** 10x Concurrency bei gleichem KV-Budget

### 22. HCSpec: Two-Tier Horizontal Cascade Speculative Decoding
- **Was:** Positionsspezialisierte Draft-Module. Dual-layer dual-path für frühe Schritte, single-layer für spätere.
- **Systeme:** Alle
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:ACL 2026 — Paper-only
- **Aufwand:** 4-6 Wochen
- **Gain:** Höhere SD Acceptance

### 23. DART: Diffusion-Inspired Speculative Decoding
- **Was:** Parallele Logit-Vorhersage für multiple masked Positionen in einem Forward Pass. Eliminiert autoregressive rollouts.
- **Systeme:** Alle (besonders CUDA)
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:2601.19278 — Paper-only
- **Aufwand:** 4-6 Wochen
- **Gain:** Geringe Latenz durch parallele Generierung

### 24. ReSA: Rectified Sparse Attention
- **Was:** Block-sparse attention mit periodischer dense rectification. Refreshed KV-Cache in festen Intervallen.
- **Systeme:** Alle (Long-Context auf Uranus)
- **Schwierigkeit:** schwer
- **Existiert:** arXiv:ACL 2026 — Paper-only
- **Aufwand:** 4-5 Wochen
- **Gain:** Error-bounding für sparse attention

---

## Nicht empfohlen (Hardware-Mismatch oder bereits evaluiert)

| Ansatz | Grund |
|--------|-------|
| PDL (PR #22522) | Hopper-spezifisch (CC >= 90) |
| NVFP4 (PR #23572) | Blackwell-spezifisch |
| DALI (arXiv:2602.03495) | Bereits evaluiert als #18 ❌ NO-GO |
| SpecMD (arXiv:2602.03921) | Bereits in ROADMAP als #46 ⏭️ |
| Fiddler (arXiv:2402.07033) | Bereits in ROADMAP als #82 ⏭️ |
| DuoServe-MoE | Bereits in ROADMAP als #23 ⏭️ |
| Pre-Attention Expert Prediction | Bereits in ROADMAP als #65 ⏭️ |
| CPU Backend Operator Fusion | Bereits in ROADMAP als #76 ⏭️ (Regressionen) |
| UMA Direct Read Threshold (PR #22462) | Bereits evaluiert als #10 ❌ |
| Router Mode (PR #17470) | Bereits im Fork (/models/load, /models/unload) |
| RPC Backend | Bereits im Fork (tools/rpc/) |
| MCP stdio (PR #25736) | Bereits in upstream, fork hat es |
| MUL_MAT_ID Subgroup Non-Coopmat (PR #15524) | Bereits im Fork (#6 ✅) |
| Host-Memory Prompt Caching (PR #16391) | Bereits im Fork (--cram) |
| CUDA Graph Capture Improvement (PR #19754) | Bereits im Fork (warmup_complete) |

---

## Empfohlene nächste Schritte

### Sofort (Tier 1, < 1 Woche):
1. **PR #22887 prüfen:** Ist MUL_MAT_VEC 4K per Iteration bereits im Fork? Wenn nicht → Cherry-Pick (1 Tag, +9% tg auf Vulkan)
2. **PR #25479 Cherry-Pick:** Pascal MMVQ nwarps für Styx (2-3 Tage, +3% tg)

### Kurzfristig (Tier 2, 1-3 Wochen):
3. **RateQuant/InnerQ Eval:** Paper-Recherche ob TurboQuant-Vorteile ergänzt werden können (1-2 Wochen)
4. **PR #22299 Internal AllReduce:** Für Uranus TP-Mode (1-2 Wochen, +10% tg)

### Mittelfristig (Tier 2-3, 2-4 Wochen):
5. **FineMoE Fine-Grained Offloading:** Für Styx MoE-Offloading (1-2 Wochen, -47% latency Paper-Angabe)
6. **Transfer Queue für AMD:** Für Mars RDNA3 (2-3 Tage, muss getestet werden)

### Langfristig (Tier 3-4, 4+ Wochen):
7. **VQKV/NestedKV:** KV-Cache Kompression über TurboQuant hinaus (2-3 Wochen, aber TurboQuant bereits 3-5x)
8. **Paged KV Cache:** Für high-concurrency Szenarien (6-8 Wochen, nur wenn InferenzQuelle multi-user wird)
