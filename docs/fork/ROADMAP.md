# ROADMAP — Fork-Beschleunigung

**Erstellt:** 2026-07-11
**Aktualisiert:** 2026-07-13 (Research-Sweep #3, 2-Slot Prefetch Sweet-Spot, M3 abgeschlossen)
**Quelle:** [Optimierungs-Recherche 2026-07-11](2026-07-11_OPTIMIZATION_RESEARCH.md) + [Recherche 2026-07-12](2026-07-12_OPTIMIZATION_RESEARCH.md) (Web + arXiv, 4 parallele Subagents pro Sweep)
**Hardware:** Mars (RDNA3/Vulkan/30GB), Styx (Pascal/CUDA/8GB), Hydra (Ampere/CUDA/8GB), Uranus (2x Ada/CUDA/32GB), Venus (GCN/Vulkan)
**Ausschluss:** Treiber-Neuimplementierung, Kernel-Rekompilierung, Hopper-spezifische Features

### Zeit-Schätzung (kalibriert 2026-07-12)

Schätzungen sind **Solo-Agent-Aufwand** (inkl. Remote-Builds 5-15 min/Zyklus, Benchmark-Zyklen, Server-Management, Doku, Code-Review), nicht "pure coding time".

**Kalibrierung aus 14h Solo-Session (2026-07-11/12):**
- #3 Pascal MMVQ: geschätzt "2-3 Tage", tatsächlich ~3-4h (mit PR als Vorlage) → **mit PR-Vorlage Faktor 0.5x, ohne PR-Vorlage Faktor 1x**
- #32 L1 Cache: geschätzt "1-2h", tatsächlich ~1h (trivialer CMake-Flag) → **triviale Config-Änderungen sind korrekt geschätzt**
- Forschungs-/Eval-Aufwand pro Item: ~10-15 min (git log/grep) bis ~1h (Subagent + Analyse)
- Doku + Commit pro Item: ~15-30 min
- Build-Zyklus Remote: 5-15 min, Benchmark: 5-10 min + Server-Management

**Faustregel:** Subagent-Schätzungen × 2-3 für Solo-Agent-Realität. Tier 1 "4-8h" → 1-2 Tage. Tier 2 "1 Woche" → 2-3 Wochen. Tier 3 "3-6 Wochen" → 6-12 Wochen.

---

## Meilensteine

| MS | Name | Items | Status |
|----|------|-------|--------|
| **M1** | Quick-Win-Welle | #1-5 (kombiniert) | ✅ abgeschlossen (#2✅, #4✅, #5✅, #1❌) |
| **M2** | Vulkan-Offensive | #6, #7, #9, #10, #12 | ✅ evaluiert (#6✅ bereits integriert, #7✅ bereits integriert, #9❌, #10❌, #12✅ bereits integriert) |
| **M3** | MoE-Offloading v2 | #3, #13, #14, #15 | ✅ abgeschlossen (#3✅, #6✅, #13❌, #14⏭️, #15❌, #37✅, #40⏭️) |
| **M4** | Speculative Decoding v2 | #11, #28 | ✅ abgeschlossen (#11✅ bereits integriert, #28✅ eigene Implementierung) |
| **M5** | Coopmat2 + Multi-GPU | #12, #20, #21 | ⏳ teilweise (#12✅ bereits integriert, #20✅ bereits integriert, #21 offen) |
| **M6** | Forschung | #17, #18, #19, #25, #26, #27, #29, #30, #39, #41-#55, #69-#76 | ☐ offen (Research-Sweep #3: 2026-07-13) |

**Regel:** Solo-Pläne werden nur für den **aktuellen und nächsten Meilenstein** erstellt. Tier 3-4 Pläne entstehen wenn der Meilenstein näher rückt (verhindert veraltete Pläne).

---

## Tier 1: Quick Wins (< 2 Tage Solo-Agent)

*Schätzung inkl. Build/Benchmark/Doku. Subagent-"Stunden" = Solo-Agent-Tage.*

| # | Status | Ansatz | Systeme | Aufwand | Existiert | Solo-Plan | Gain |
|---|--------|--------|---------|---------|-----------|-----------|------|
| 1 | ❌ | MTP Logits Copy Optimization | Mars, Styx | 1-2 Tage | PR #23198 | [M1-Batch](plans/SESSION_PLAN_quickwins-batch1.md) | +20% PP mit MTP — **19 Konflikte in 10 Kern-Dateien, MTP OFF, skipped** |
| 2 | ✅ | Tensor Split Regex Optimization | Uranus | 4-8h | PR #24710 (open) | [M1-Batch](plans/SESSION_PLAN_quickwins-batch1.md) | befreit 40% decode thread — **angewendet, Build grün auf Hydra+Mars** |
| 3 | ✅ | Pascal CUDA MMVQ Optimization | Styx | 1-2 Tage | PR #25479 (draft) | [M3](plans/SESSION_PLAN_pascal-mmvq.md) | +3-6% decode auf Pascal — **manuell portiert, +8.9% Generation auf GTX 1070 (E2B MoE), Code-Review ship-ready. Tatsächlich ~3-4h mit PR-Vorlage** |
| 4 | ✅ | n-gram / Prompt-Lookup Decoding | Alle | 0h | in mainline | [M1-Batch](plans/SESSION_PLAN_quickwins-batch1.md) | hilft bei repetitiven Workloads — **verfügbar (`--spec-type ngram-mod`), kein Speedup auf E2B (39.1 vs 39.2 t/s)** |
| 5 | ✅ | GTT Size Tuning für Mars APU | Mars | 1-2h | Konfiguration | [M1-Batch](plans/SESSION_PLAN_quickwins-batch1.md) | >15GB GPU-Speicher — **bereits konfiguriert (26GB GTT), kein Tuning nötig** |
| 6 | ✅ | Vulkan MUL_MAT_ID Subgroup Optimization | Mars, Venus | 1 Tag | PR #15524 (merged) | [M2](plans/SESSION_PLAN_mulmat-id-subgroup.md) | bis +657% MoE PP auf AMD — **bereits im Fork integriert: MatMulIdType::SUBGROUP, mul_mm_id_funcs.glsl, matmul_id_subgroup_* Shader werden generiert, Pipelines aktiv (subgroup_ballot && subgroup_require_full_support && subgroup_min_size_16)** |
| 7 | ✅ | Vulkan FlashAttention Refactor | Mars, Venus | 1 Woche | PR #19625 (merged) | [M2](plans/SESSION_PLAN_vulkan-fa-refactor.md) | 10-20% scalar FA improvement — **bereits im Fork integriert: Commit 66e999ecc, get_fa_tuning_params_scalar(), row_split, shmem_staging, Q caching in registers, vendor-specific Br selection** |
| 31 | ❌ | K-Quant A-Matrix Transpose (CM1) | Mars | 1-2 Tage | PR #22970 (open) | — | +5-15% PP auf coopmat1 — **Recherche 2026-07-12: PR ist CM1-only, Mars nutzt CM2 (coopmat2). Code-Zeile 5338: "cm1 is used only when cm2 is unavailable". PR bringt keinen Speedup auf CM2-Geräten. Venus (GCN) hat kein coopmat. Nicht lohnenswert für den Fork** |
| 32 | ❌ | Pascal L1 Cache Tuning | Styx | 4-8h | nein (Compiler-Flag) | — | +5-10% bei memory-bound — **Recherche 2026-07-12, -Xptxas -dlcm=ca aktiviert L1/Texture Cache auf Pascal GP104. Benchmark: kein Speedup (pp128: 1453→1445 -0.5%, tg64: 77.25→77.22 -0.04%). GTX 1070 hat nur 48KB L1/Shared per SM, Streaming-Workloads evicten L1 sofort. Flag als GGML_CUDA_L1_CACHE=ON Option verfügbar, aber wirkungslos** |
| 33 | ⏭️ | Per-Quant MMVQ/MMQ Batch Threshold | Mars, Venus | 1-2 Tage | Commit bc81d47 | — | bis +76% PP auf AMD MFMA — **Recherche 2026-07-12: Fork hat bereits ggml_vk_should_use_mmvq() mit vendor+quant-spezifischen Thresholds (AMD: k<2048→false, per-quant Logik). Commit bc81d47 wäre Fein-Tuning, aber Kernlogik bereits vorhanden. +76% wurden auf MI250X gemessen, nicht Phoenix RDNA3. Low-priority, verschoben** |
| 56 | ✅ | Vulkan UMA Cached Host Memory | Mars, Venus | 4-8h | PR #23762 (open) | — | 3x Get-BW auf APU — **Implementiert 2026-07-13: PR #23762 portiert. Bevorzugt HostCached statt write-combining auf UMA. memory_property_flags fix (actual vs requested). Non-coherent flush/invalidate handling. Benchmark Mars (26B Q4_K_M, 2-Slot Prefetch): tg128 16.73→17.90 (+7% auf tg, +11% vs baseline), pp512 184.26→185.14 (+0.5%, Rauschen). Decode profitiert massiv von cached reads, Prefill weniger (write-heavy).** |
| 57 | ❌ | MMQ Stream-k Disable für Tensor-Split MoE | Uranus | 2-4h | PR #22170 (merged) | — | 30-50% P2P PP auf Dual-GPU MoE — **Implementiert+Benchmarked 2026-07-13: GGML_CUDA_DISABLE_MMQ_STREAM_K env var hinzugefügt. Benchmark auf Uranus (2x RTX 4060 Ti): E4B QAT (7.46B): pp512 -64%. 26B IQ4_XS (25.23B) tensor split: pp512 2669→17.48 (**-99.3%**), layer split: pp512 3005→11.59 (**-99.6%**). tg128 unbeeinflusst. Stream-k decomposition ist ESSENZIELL für MMQ-Performance auf Ada GPUs. PR #22170 war für spezifischen Edge Case (stream-k fixup-buffer Race Condition bei src1_ncols != ne11) gedacht, nicht für generelle MoE-Workloads. Env var bleibt im Code (Default: OFF = stream-k ON) für Debugging-Zwecke.** |
| 58 | ⏭️ | KV Cache Size Limiting + Demand Paging | Alle | 1-2 Wochen | PR #18747 (open) | — | Reduziert KV-Memory bei langen Contexts — **Recherche 2026-07-13 (Update 2026-07-13 Tiefen-Recherche): PR #18747 ist noch OPEN (nicht gemerged, 17 Commits, +1397/-32 Zeilen, 15 Dateien). Liefert: (1) --kv-cache-tokens N limitiert KV-Allokation, (2) --kv-cache-demand-paged nutzt mmap(MAP_NORESERVE) für lazy allocation, (3) Block-Tracking-Infrastruktur (llama-kv-block.h, +263 Zeilen) als Foundation für zukünftiges PagedAttention. Was der PR NICHT tut: keine per-token KV memory reduction, kein memory sharing zwischen Sequenzen, kein dynamisches grow/shrink. Komplementär zu TurboQuant (wir haben schon!) — TurboQuant komprimiert KV-Daten (3-4 bit), PR #18747 managed Allokation. Für Styx (8GB, 224k): TurboQuant hat höhere Immediate-Priorität (bereits vorhanden), PR #18747 kann später übernommen werden wenn gemerged. PagedAttention-Paper (arXiv:2309.06180, vLLM) bestätigt Architektur-Ansatz. Verschoben bis PR merged oder bis TurboQuant allein nicht ausreicht.** |
| 59 | ☐ | Tensor Prefetching (--prefetch-weights) | Styx, Hydra, Uranus | 1-2 Tage | PR #21067 (draft) | — | Layer-level Prefetch via async copy — **Recherche 2026-07-13: PR #21067 (draft). Überlappt compute mit weight loading für nächste Layer. Nur CUDA, erfordert --no-mmap. Komplementär zu thecodacus Expert-Prefetch (MoE-spezifisch). Profitabel bei großen ubatch (1024+).** |
| 60 | ❌ | RADV Driver-Specific Shader Optimizations | Mars, Venus | 4-8h | nein (Driver-Config) | — | 5-15% möglicher Speedup — **Recherche 2026-07-13: RADV Environment-Variables (RADV_PERFTEST=geom, RADV_NO_DYNAMIC_BOUNDS) sind graphics-orientiert (geometry shaders, invariant geom). Für reine Compute-Workloads (Vulkan LLM Inference) nicht relevant. Kein messbarer Speedup erwartet.** |

## Tier 2: Mittelfristig (2-6 Wochen Solo-Agent)

*Subagent-"1 Woche" = Solo-Agent 2-3 Wochen (Build-Zyklen, Debugging, Doku).*

| # | Status | Ansatz | Systeme | Aufwand | Existiert | Solo-Plan | Gain |
|---|--------|--------|---------|---------|-----------|-----------|------|
| 8 | ⏭️ | Mixed Precision KV Cache (Hot/Cold) | Alle | 2-3 Wochen | commit e889fbd | später | Reduziert KV-Größe bei erhaltener Qualität — **Eval 2026-07-13: Commit e889fbd (upstream) implementiert kv_layer_mixed mit FP16 hot + Q4_0 cold, threshold=32, group_size=16. Fork hat bereits layer-adaptive Precision (TURBO_LAYER_ADAPTIVE Modes 1-7), aber nicht token-adaptive. Kritischer Blocker: Flash Attention erwartet uniforme Tensor-Typen — hot/cold erfordert Dequantisierung der Cold-Zone bei jedem Attention-Call. Nutzen über turbo3/turbo4 marginal: turbo4 (4.25 bit) hat schon gute Qualität, hot/cold mit FP16 hot zone = 3.75x mehr Speicher für recent tokens bei ähnlichem Gesamtbedarf. Benefit ist Qualität (recent tokens FP16), nicht Speichersparnis. Für produktiv-Setup (Styx 224k, VRAM-limitiert) kontraproduktiv. Postponed bis FA mixed-type support oder Qualität-Priorität steigt.** |
| 9 | ❌ | Vulkan Shared Memory Staging Kernel | Mars, Venus | 1-2 Wochen | PR #20897 (closed) | [M2](plans/SESSION_PLAN_vulkan-shmem-staging.md) | >2.5x TG potenziell — **PR closed ohne Merge (AI-generiert), manuelle Portierung nötig** |
| 10 | ❌ | UMA Zero-Copy für Mars APU | Mars, Venus | 2-4 Wochen | PR #22462 | [M2](plans/SESSION_PLAN_uma-zero-copy.md) | +112x transfer speed — **nicht implementieren: verwandte UMA-PRs (#22455, #22930, #23770) bereits getestet und revertiert, RCA-Masterplan zeigt System-RAM ist langsamer als GTT für GPU-Compute auf Mars, PR #22462 ist open/unstable** |
| 11 | ✅ | EAGLE-3 Speculative Decoding | Alle | 4-6 Wochen | PR #18039 (merged) | [M4](plans/SESSION_PLAN_eagle3.md) | bis 6.5x speedup — **bereits im Fork integriert: Commit 57774253c, LLM_ARCH_EAGLE3 in llama-arch.cpp, eagle3-Modell-Support, --spec-type eagle3 verfügbar** |
| 12 | ✅ | Vulkan Cooperative Matrix (Coopmat2) | Mars, Uranus | 4-6 Wochen | PR #19075 (merged) | [M5](plans/SESSION_PLAN_coopmat2-rdna3.md) | 2.5x FA multiply — **bereits im Fork integriert: coopmat2 Support in ggml-vulkan.cpp, flash_attn_cm2.comp, mul_mm_cm2.comp, dequant_funcs_cm2.glsl, SPV-Shader auf Mars generiert** |
| 13 | ❌ | Two-Tier GPU+RAM Expert Cache | Styx, Mars | 2-4 Wochen | FR #20757 | [M3](plans/SESSION_PLAN_two-tier-expert-cache.md) | kritisch für Styx 8GB — **nicht implementieren: alle PRs closed, thecodacus Memory Pinning deckt Tier 2 bereits ab, geringer ROI für Pascal (compute-bound nicht PCIe-bound)** |
| 14 | ⏭️ | LFU Caching Policy für MoE Experts | Styx, Mars | 1-2 Wochen | nein (Forschung) | [M3](plans/SESSION_PLAN_lfu-caching.md) | 15-20% expert hit rate — **verschoben: Recherche zeigt LFU allein nicht lohnenswert. Besser: LFRU (vLLM PR #37190) + thecodacus Pinning. SpecMD "Least-Stale" (arXiv:2602.03921) 85× besser als LRU. Für Styx (8GB) lohnenswert, für Mars (<5% gain) nicht** |
| 15 | ❌ | PipeShard: VRAM-Constrained MoE | Styx, Hydra, Mars | 4-6 Wochen | arXiv:2604.26334 | [M3](plans/SESSION_PLAN_pipeshard.md) | VRAM-optimierte MoE-Inference — **Eval 2026-07-13: NVIDIA MLSys 2026 Paper, PR #22692 open (+5396 Zeilen, 40 Dateien). Aufwand 4-12 Wochen. Nicht empfohlen: (1) Styx GTX 1070: PCIe 3.0 + keine Tensor Cores = bottleneck-limited, marginaler Benefit. (2) Mars RDNA3 APU: unified memory, kein PCIe-Transfer → PipeShard's pipelined copy-compute irrelevant. (3) Hohe Konfliktgefahr mit TurboQuant (tensor_buft_overrides). (4) PR #22692 nicht gemerged, Cherry-Pick wird zunehmend riskant. Warten auf upstream Merge, dann re-evaluieren.** |
| 16 | ✅ | Vulkanised 2026: shmem-staging M-V Kernel | Mars, Venus | 2-4 Wochen | upstream | später | >2.5x TG auf Intel Arc — **Eval 2026-07-13: Bereits im Fork integriert (Commit 66e999ecc, PR #19625). shmem_staging in ggml-vulkan.cpp (13 Referenzen), flash_attn.comp, flash_attn_cm1.comp. NVIDIA-only Aktivierung. Keine weitere Aktion nötig.** |
| 34 | ✅ | UBBoost: Dynamische Ubatch-Größe | Styx, Hydra | 2-3 Wochen | Discussion #23262 | — | bis 2x PP bei VRAM-konstriktiven Setups — **Implementiert: -ubp/--ubatch-prefill CLI flag, n_ubatch_prefill in cparams. Graph reserviert für max(n_ubatch, n_ubatch_prefill). SWA cache sizing fix für max(n_ubatch, n_ubatch_prefill). Benchmark Styx (Pascal GTX 1070, turbo3/4 KV): E4B ub=256/ubp=512: +20% pp2048, +41% pp8192, +19% tg128 vs ub=512 baseline. 26B ub=256/ubp=512: +18% pp2048, +10% pp8192, +4.5% tg128. Optimal: ub=256, ubp=512. ubp>512 OOM auf 8GB.** |
| 35 | ✅ | Row-Packing für Dequantize-MatVec | Mars, Venus | 1-2 Tage | nein (Zinc Referenz) | — | +10-20% DMMV (erwartet) → **+1% gemessen** — **Implementiert: rm_stdq=2, rm_kq=4 für RDNA3 (Commit acd8dfe3a). Row-packing war bereits teilweise implementiert (NUM_ROWS=2 default, =4 auf GCN). Erhöhung auf RDNA3 brachte nur +1% auf E2B Q4_K (pp128: 493→498, tg64: 40.0→40.4). DMMV ist nicht der Bottleneck für kleine MoE-Modelle.** |
| 36 | ☐ | Auto Parameter Fitting TP | Uranus | 1-2 Wochen | PR #22950 (open) | — | vereinfachte TP-Konfiguration — **Eval 2026-07-13: Nicht im Fork. PR #22950 open (unstable, 2026-06-05). Fork blockiert SPLIT_MODE_TENSOR in fit.cpp:181. Uranus (2x identische RTX 4060 Ti) erfüllt PR-Annahme (homogeneous GPUs). 3-4 Tage Aufwand nach PR-merge oder manueller Portierung.** |
| 37 | ✅ | LFRU Expert Caching | Styx, Mars | 2-4 Wochen | vLLM commit 71ed1fc | — | +5.2% speedup, 15-20% hit rate — **Eval 2026-07-13: thecodacus Prefetch repariert (MUL_MAT_ID Graph-Suche). 2-Slot Sweet-Spot: Styx +28.9% pp512 +2.1% tg8, Mars +8.8% pp512 +3.8% tg128 (reiner Win auf beiden!). 3-Slot: -35% tg auf Mars (Queue-Konflikt RDNA3), 1-Slot: -11.9% pp auf Styx (kein Overlap). Beide Server-Scripts auf GGML_SCHED_PREFETCH_SLOTS=2. Expert-Cache postponed (Buffer-Recycling-Crash).** |
| 38 | ⏭️ | Conf-KV: Confidence-aware KV Eviction | Alle | 2-4 Wochen | arXiv:2605.24786 | — | KV-Cache Kompression ergänzt TurboQuant — **Eval 2026-07-13: Verschoben. Conf-KV nutzt FP16/INT8 mixed precision, konflikt mit TurboQuant turbo3/turbo4 (3.125/4.25 bit). Eviction policy könnte komplementär sein, aber mixed-precision storage braucht signifikante Anpassung an ISWA-Architektur. TurboQuant-native Ansätze priorisieren.** |
| 39 | ☐ | Talon: Adaptive Token Trees | Alle | 4-6 Wochen | arXiv:2601.07353 | — | höhere Spec-Decoding Acceptance — **Recherche 2026-07-12** |
| 40 | ⏭️ | MoE Load Balancing Expert Frequency | Styx, Hydra | 2-3 Wochen | nein (Konzept) | — | bessere Load-Balance bei MoE-Offloading — **Phase 1 implementiert (2026-07-13): Expert Frequency Tracking via C API + LLAMA_MOE_FREQ_TRACK=1. Validiert auf Styx (26B QAT, 30 Layer × 128 Experts). Phase 2 (frequency-guided layer offloading): NEGATIVERGEBNIS. Sowohl pure-entropy als auch swap-Strategie verschlechtern tg128 um -7%. Root Cause: kälteste Layer (höchste Entropy) sind späte Layer (23-27), die auf GPU sein müssen für Generation-Performance. Default "erste N Layer auf CPU" ist optimal weil späte Layer auf GPU bleiben. Per-Expert-Platzierung (statt per-Layer) wäre nötig für echten Benefit, erfordert aber Tensor-Splitting (3D→2D) — tiefer GGML-Eingriff. Frequency-Tracking-Infrastruktur bleibt nützlich für Analyse und zukünftige Per-Expert-Optimierung.** |
| 61 | 🔄 | Persistent VRAM Expert Cache | Styx, Uranus | 1-2 Wochen | PR #24524 (closed) + RFC #24528 | — | Massiver Speedup bei cache hits — **Recherche 2026-07-13 (Update 2026-07-13 Tiefen-Recherche): WICHTIG — PR #23170 ist laut RFC #24528 ein "no-op when made correct" (staging-buffer approach funktioniert nicht korrekt). Die eigentliche Lösung ist PR #24524 (closed wegen Größe, 2222 Zeilen, 11 Dateien): Invertiertes execution model — MUL_MAT_ID bleibt auf CPU, GPU führt cached rows parallel aus (thread 0 dispatches batched matvec), CPU threads compute miss rows concurrently. Zero-config, on by default (--moe-cache 0 disables, --moe-cache N caps per-device budget). Empirisch: 10% der Experten → 80-81% cache hits, Top 30% → 95-96% hits. Safety Rails: decode-only fill, stable shape census, paired gate+up pools, GPU-resident handoff, on-disk hot-set persistence, baseline-sampled bail-out, VRAM surrender bei OOM, multi-GPU sharding. Alle prior attempts (#20757, #21609, #21614, #21620, #23170) bewegten MUL_MAT_ID zur GPU → jeder cache miss auf critical path als synchroner PCIe transfer (~3× decode regression). PR #24524 vermeidet das strukturell. arXiv-Paper: ProMoE (2410.22134, 2.13× prefill/2.84× decode), MoE-Infinity (2401.14361, 3.1-16.7× latency), DuoServe-MoE (2509.07379), Caching Analysis (2511.05814, LFU > LRU). Empfehlung: PR #24524 Design manuell portieren (nicht cherry-picken — zu groß bei Fork-Divergenz), 1-2 Wochen. Styx (PCIe 3.0) profitiert am meisten, Mars (unified memory) weniger.** |
| 62 | ☐ | MoE Expert Profiling & REAP Pruning | Alle MoE | 1-2 Tage | PR #20454 (open) | — | Intelligente Expert-Placement — **Recherche 2026-07-13: PR #20454 (open). C++ profiler für REAP saliency scores aus GGUF inference. Nutzt unsere vorhandene MoE-Frequency-Tracking-Infrastruktur. Ermöglicht datengetriebene Expert-Placement-Entscheidungen.** |
| 63 | ☐ | xKV: Cross-Layer KV-Cache Compression | Alle | 2-3 Wochen | arXiv:2503.18893 | — | 8× KV-Kompression orthogonal zu TurboQuant — **Recherche 2026-07-13: Training-freie SVD-basierte Kompression über Layer-Gruppen. Bis 8× Kompression bei 2-3% Genauigkeitsverlust. Orthogonal zu TurboQuant (Cross-Layer vs Token-Level). GitHub: abdelfattah-lab/xKV.** |
| 64 | ☐ | Block-Sparse Flash Attention (BSFA) | Alle CUDA | 1-2 Wochen | arXiv:2512.07011 | — | 1.10-1.24× Speedup, drop-in FA Enhancement — **Recherche 2026-07-13: Drop-in Ersatz für FlashAttention, überspringt irrelevante Value-Blocks basierend auf kalibrierten Thresholds. Training-frei, 16 Samples Kalibrierung. >99% Genauigkeit. GitHub: Danielohayon/Block-Sparse-Flash-Attention.** |
| 65 | ☐ | Pre-Attention Expert Prediction | Styx, Hydra, Uranus | 1-2 Wochen | arXiv:2511.10676 | — | 93-97% expert prediction accuracy — **Recherche 2026-07-13: Pre-attention prediction mit 2 linear functions vor attention block. Ermöglicht präzisen Prefetch vor MoE-Layer. 93.03% auf DeepSeek V2 Lite, 94.69% auf Qwen3-30B.** |
| 66 | ☐ | BucketServe: Dynamic Batching | Alle | 1-2 Wochen | arXiv:2507.17120 | — | bis 3.58× Throughput — **Recherche 2026-07-13: Bucket-basiertes dynamisches Batching für multi-request Szenarien. Besonders relevant für Server-Workloads (phobos, styx, uranus).** |
| 67 | ☐ | MXFP4 Quantization für gpt-oss | Alle | 1-2 Wochen | Diskussion #15095 | — | Native gpt-oss Modellnutzung — **Recherche 2026-07-13: OCP open-standard MXFP4 Format. Alle major Backends (CUDA, Vulkan, Metal, CPU). Ermöglicht gpt-oss Modelle ohne Konvertierung.** |
| 68 | ✅ | Vulkan Matmul Parameter-Tuning AMD | Mars | 1-2 Tage | PR #18749 (merged) | — | +1-3% auf RDNA3 — **Recherche 2026-07-13: Bereits im Fork (Commits cf119f140, f45eef8cb). Matmul-Parameter-Kombinationen spezifisch für AMD coopmat. Keine weitere Aktion nötig.** |

## Tier 3: Komplex (6-12 Wochen Solo-Agent)

*Subagent-"3-4 Wochen" = Solo-Agent 6-8 Wochen. Erfordert meist User-Planung.*

| # | Status | Ansatz | Systeme | Aufwand | Existiert | Solo-Plan | Gain |
|---|--------|--------|---------|---------|-----------|-----------|------|
| 17 | ☐ | HOBBIT: Mixed-Precision Expert Offloading | Styx, Mars | 8-12 Wochen | arXiv:2411.01433 | später | bis 9.93x MoE offloading |
| 18 | ☐ | DALI: Workload-Aware MoE Offloading | Styx | 4-6 Wochen | arXiv:2602.03495 | später | intelligentes MoE-Offloading |
| 19 | ☐ | Expected Attention KV Cache Compression | Alle | 4-6 Wochen | arXiv:2510.00636 | später | ergänzt TurboQuant |
| 20 | ✅ | Tensor Parallelism für Uranus | Uranus | 6-8 Wochen | PR #19378 (merged) | [M5](plans/SESSION_PLAN_tensor-parallelism.md) | ~40% boost für 2x 4060 Ti — **bereits im Fork integriert: Commit d850df3f5, backend-agnostic tensor parallelism, AllReduce in ggml-cuda/allreduce.cu, Meta-Device in llama.cpp** |
| 21 | ☐ | PagedAttention / Paged KV Cache | Uranus, Mars | 6-8 Wochen | PR #22569 | später | 2.5x aggregate throughput |
| 22 | ☐ | GWQ: Gradient-Aware Weight Quantization | Alle | 4-6 Wochen | arXiv:2411.00850 | später | 1.2x inference speedup |
| 23 | ☐ | DuoServe-MoE: Dual-Phase Expert Scheduling | Styx | 6-8 Wochen | arXiv:2509.07379 | später | phase-spezifisches Prefetch |
| 24 | ☐ | HybriMoE: Hybrid CPU-GPU Scheduling | Styx | 6-8 Wochen | arXiv:2504.05897 | später | 1.33x prefill, 1.70x decode |
| 41 | ☐ | GRKV: Global Regression KV Compression | Alle | 4-6 Wochen | arXiv:2605.31105 | später | training-free KV compression — **Recherche 2026-07-12** |
| 42 | ☐ | CapKV: Capacity-aware KV Eviction | Alle | 4-6 Wochen | arXiv:2604.25975 | später | information-theoretic eviction — **Recherche 2026-07-12** |
| 43 | ☐ | SliderQuant: Sliding-layer PTQ | Alle | 4-6 Wochen | arXiv:2603.25284 | später | bessere Low-Bit-Quantisierung — **Recherche 2026-07-12** |
| 44 | ☐ | Alloc-MoE: Budget-aware Expert Activation | Mars, Styx | 6-8 Wochen | arXiv:2604.08133 | später | 1.34x decode speedup — **Recherche 2026-07-12** |
| 45 | ❌ | CUDA Concurrent Streams QKV | Styx, Hydra | 1-2 Wochen | GGML_CUDA_GRAPH_OPT=1 | später | **Bereits im Fork (PR #16991). Benchmark 2026-07-13: (1) Uranus (RTX 4060 Ti 16GB, E4B voll auf GPU): tg2048 -10.7% mit CUDA Graphs (interleaved Node-Order bricht Graph-Capture), tg512 +1% ohne Graphs (Rauschen). (2) Styx (GTX 1070 8GB, 26B QAT MoE-Offload): kein Effekt — CPU-Offload erzeugt Split-Buffers → CUDA Graphs deaktiviert → Feature aktiviert gar nicht. Nur nutzbar bei single-GPU + voller Offload + CUDA-Graphs-kompatibel, und dann Regression.** |
| 69 | ☐ | FlashMoE: ML-based Cache Replacement | Styx, Hydra | 2-3 Wochen | arXiv:2601.17063 | später | 2.6× speedup, 51% hit rate — **Recherche 2026-07-13** |
| 70 | ☐ | ST-MoE: Spatio-Temporal Prefetching | Styx, Hydra, Uranus | 2-3 Wochen | arXiv:2606.15453 | später | 2.5× speedup, 85% prediction — **Recherche 2026-07-13** |
| 71 | ☐ | Efficient CPU-GPU Collaborative MoE | Styx, Hydra | 3-4 Wochen | arXiv:2512.16473 | später | N-index M-way set-associative cache — **Recherche 2026-07-13** |
| 72 | ☐ | N4_0 Native 4-bit Float | Uranus | 2-3 Wochen | PR #23572 | später | +40% PP (Blackwell-bedingt) — **Recherche 2026-07-13** |
| 73 | ☐ | CascadeInfer: Length-Aware Scheduling | Alle | 2-3 Wochen | arXiv:2512.19179 | später | 67% Latenz-Reduktion — **Recherche 2026-07-13** |
| 74 | ☐ | Vulkan Descriptor Indexing (Bindless) | Mars, Venus | 2-4 Wochen | nein | später | Reduziert Descriptor-Binding-Overhead — **Recherche 2026-07-13** |
| 75 | ☐ | Non-blocking Pipeline Scheduling | Uranus | 3-4 Wochen | PR #19922 | später | Reduziert Pipeline-Bubbles — **Recherche 2026-07-13** |
| 76 | ☐ | CPU Backend Operator Fusion | Alle | 3-4 Wochen | Diskussion #22315 | später | Reduziert Memory-Traffic CPU-Path — **Recherche 2026-07-13** |

## Tier 4: Langfristig / Forschung (3+ Monate Solo-Agent)

*Subagent-"6+ Wochen" = Solo-Agent 3+ Monate. Meist Paper-Implementierungen ohne PR-Vorlage.*

| # | Status | Ansatz | Systeme | Aufwand | Existiert | Solo-Plan | Gain |
|---|--------|--------|---------|---------|-----------|-----------|------|
| 25 | ☐ | llama.cpp 2026 Rewrite Merge | Alle | 3-4 Monate | mainline | später | 2.1x Durchsatz 70B, 1.4x 7B |
| 26 | ☐ | Q-Filters / LagKV / MiniCache | Alle | 1-2 Mo/Methode | GitHub (qfilters) | später | 5-32x KV compression |
| 27 | ☐ | Pre-Attention Expert Prediction | Styx, Mars | 4-6 Wochen | Forschung | später | 15% accuracy improvement |
| 28 | ✅ | Adaptive MTP Speculative Decoding | Alle | 2-4 Wochen | PR #22931 (closed) | [M4](plans/SESSION_PLAN_adaptive-mtp.md) | verhindert MTP-Regression — **Fork hat eigene Implementierung: `LLAMA_MTP_SKIP_STREAK_THRESHOLD` env var (1-32), skip-streak Mechanismus in common/speculative.cpp. PR #22931 wurde closed (Draft), wäre auch nicht kompatibel (upstream draft-mtp vs Fork gemma4-assistant). Funktionsäquivalent vorhanden** |
| 29 | ☐ | FastKV: Token-Selective Propagation | Alle | 2-3 Monate | GitHub (fastkv) | später | 1.82x prefill, 2.87x decode |
| 30 | ☐ | Speculative Expert Prefetching (MoE-SpeQ) | Styx, Mars | 2-3 Monate | Forschung | später | 2.34x offloading speedup |
| 46 | ☐ | SpecMD Least-Stale Expert Prefetching | Styx, Hydra | 4-6 Wochen | arXiv:2508.21706 | später | 85x weniger misses, 34.7% TTFT — **Recherche 2026-07-12** |
| 47 | ☐ | QUICK Bank Conflict Elimination | Hydra | 2-3 Monate | arXiv:2402.10076 | später | bis 1.94x throughput — **Recherche 2026-07-12** |
| 48 | ☐ | FluxMoE Expert Paging | Styx, Hydra | 2-3 Monate | arXiv:2604.02715 | später | kleineres Working Set — **Recherche 2026-07-12** |
| 49 | ☐ | STAR-KV: Low-rank KV Compression | Alle | 3-4 Monate | arXiv:2606.08382 | später | 75% KV compression, 6.9x attention — **Recherche 2026-07-12** |
| 50 | ☐ | VQKV: Vector Quantization KV Cache | Alle | 2-3 Monate | arXiv:2603.16435 | später | 82.8% KV compression — **Recherche 2026-07-12** |
| 51 | ☐ | CompilerKV: Offline Experience Compilation | Alle | 2-3 Monate | arXiv:2602.08686 | später | robuste Kompression — **Recherche 2026-07-12** |
| 52 | ☐ | SliceMoE: Bit-sliced Expert Caching | Mars, Styx | 3-4 Monate | arXiv:2512.12990 | später | effiziente MoE-Caching — **Recherche 2026-07-12** |
| 53 | ☐ | MoBiE: Mixture of Binary Experts | Mars, Styx | 3-4 Monate | arXiv:2604.06798 | später | 2x+ inference speedup — **Recherche 2026-07-12** |
| 54 | ☐ | DASH-Q: Ultra Low-Bit PTQ | Alle | 3-4 Monate | arXiv:2604.13806 | später | +7.01% accuracy ultra low-bit — **Recherche 2026-07-12** |
| 55 | ☐ | GOOSE: Anisotropic Speculation Trees | Alle | 2-3 Monate | arXiv:2604.02047 | später | höhere Acceptance Rate — **Recherche 2026-07-12** |

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

---

## Abhängigkeiten

| Item | Hängt ab von | Grund |
|------|-------------|-------|
| #12 Coopmat2 | #7 Vulkan FA Refactor | Coopmat-FA baut auf Refactor auf |
| #20 Tensor Parallelism | #2 Tensor Split Regex | Regex-Opt ist Voraussetzung für stabiles TP |
| #11 EAGLE-3 | — | Unabhängig, aber Draft-Modell muss existieren |
| #17 HOBBIT | #14 LFU Caching | HOBBIT nutzt erweiterte Caching-Policies |
| #25 2026 Rewrite | Alle M1-M5 | Merge nach Abschluss der geplanten Features |

---

## Priorisierung nach Cost-Benefit

### Sofort umsetzbar (M1 — Quick-Win-Welle):
1. **#6 MUL_MAT_ID Subgroup** — 1 Tag, bis +657% MoE PP auf AMD
2. **#1 MTP Logits Copy** — 2-3h, +20% PP mit MTP
3. **#2 Tensor Split Regex** — 1-2h, befreit 40% decode thread
4. **#5 GTT Size Tuning** — 1h, mehr GPU-Speicher für Mars
5. **#4 n-gram Decoding** — 0h, nur aktivieren

### Diese / nächste Woche (M2 — Vulkan-Offensive):
6. **#7 Vulkan FA Refactor** — 1 Woche, 10-20% scalar FA improvement
7. **#9 Vulkan Shmem-Staging** — 2-3 Tage, >2.5x TG potenziell
8. **#10 UMA Zero-Copy** — 1-2 Wochen, +112x transfer speed für Mars

### Mittelfristig (M3 — MoE-Offloading v2):
9. **#3 Pascal MMVQ** — 2-3 Tage, +3-6% decode auf Styx
10. **#14 LFU Caching** — 2-3 Tage, 15-20% expert hit rate
11. **#13 Two-Tier Expert Cache** — 1-2 Wochen, kritisch für Styx 8GB
12. **#15 PipeShard** — 2-3 Wochen, VRAM-constrained MoE

### Mittelfristig (M4 — Speculative Decoding v2):
13. **#28 Adaptive MTP** — 1-2 Wochen, verhindert MTP-Regression
14. **#11 EAGLE-3** — 2-3 Wochen, bis 6.5x speedup

### Langfristig (M5-M6):
15. **#12 Coopmat2 RDNA3** — 2-3 Wochen, 2.5x FA multiply
16. **#20 Tensor Parallelism** — 3-4 Wochen, 40% boost für Uranus
17. **#17 HOBBIT** — 4-6 Wochen, 9.93x MoE offloading
18. **#25 2026 Rewrite** — 4-6 Wochen, system-wide 1.4-2.1x
