# Wochenrückschau 7.–14. Juli 2026 — Messbare Optimierungsergebnisse

**Erstellt:** 2026-07-14 | **Zeitraum:** 2026-07-07 bis 2026-07-14 | **Quellen:** git-log (161 Commits), CHANGELOG, ROADMAP, SNAPSHOT, Solo-Session-Reports, Trilium-TTT (Tagesnotizen 07.–14.07. vollständig ausgewertet)

---

## Verifikation der Produktiv-Werte (Live-Test 2026-07-14 16:25 UTC)

| System | Metrik | TTT-Doku | Live-Verifikation | Delta |
|--------|--------|----------|-------------------|-------|
| Mars/phobos (LXC 240) | tg t/s (128 tok, QAT, turbo3/4, 256k) | 25.85 | **25.74** | -0.4% (Rauschen) |
| Styx (GTX 1070, MoE-Offload) | tg t/s (128 tok, QAT, turbo3/4, 224k) | 26.02 | **25.70** | -1.2% (Rauschen) |

Beide Produktiv-Server bestätigen die dokumentierten Werte innerhalb der Messungenauigkeit.

---

## Netto-Verbesserungen — produktiv wirksam

| System | Metrik | Vorher (Woche Anfang) | Nachher (Woche Ende) | Delta | Quelle |
|--------|--------|----------------------|----------------------|-------|--------|
| **Styx** 26B A4B | tg t/s | 12.7 (IQ4_NL, kein Pinning) | 25.7 (QAT+Pinning+Prefetch, verifiziert 14.07.) | **+102%** | thecodacus 08.07. + QAT 10.07. |
| **Styx** 26B A4B | tg t/s mit MTP (IQ4_NL) | 21.3 (Config B Baseline) | 31.9 (Kurztest, 11 tok) | **+50%** (nur IQ4_NL) | thecodacus 08.07. — **nicht auf QAT übertragbar** (siehe MTP-Benchmark 14.07.) |
| **Styx** 26B A4B | Kontext | 160k | 224k | **+64k** | QAT 10.07. |
| **Styx** E2B | tg t/s | 65.1 | 70.9 | **+8.9%** | #3 Pascal MMVQ 11.07. |
| **Mars** 26B A4B | tg t/s | (IQ4_NL baseline) | +16.6% via QAT | **+16.6%** | QAT 10.07. |
| **Mars** 26B A4B | pp t/s | (IQ4_NL baseline) | +10% via QAT | **+10%** | QAT 10.07. |
| **Mars** 26B A4B | Kontext | 180k (obsolet) | 256k (Modell-Max) | **+76k** | QAT + 188k-RCA 12.07. |
| **Uranus** E4B+MTP | tg t/s | Crash (head_dim=512) | 103–112 t/s | **∞ (Crash→laufend)** | E4B FA-Fix 12.07. |
| **xtts-api** LAN | Durchsatz | 59.3 chars/s (331s/19.9K) | 184.2 chars/s (108.3s) | **+211% (3.20x)** | P8.6 11.07. |
| **Uranus** InferenzQuelle | System-Prompt Latenz | 7395ms | 429ms | **17x** | Cache-RAM 13.07. |

**Größter einzelner Win:** thecodacus Pinning+Prefetch auf Styx — **tg +102%** über die Woche (12.7 → 25.7 t/s, IQ4_NL-ohne-Pinning → QAT-mit-Pinning+Prefetch), direkt produktivwirksam.

**MTP-Korrektur (14.07.):** Der ursprünglich als "+150%" dokumentierte MTP-Boost verglich Config A (alle Experten CPU, kein Pinning) mit Config B + MTP (mit Pinning) — irreführend. Der korrekte MTP-Boost ist +50% (21.3→31.9 t/s), bezogen auf die gleiche Config B, und gilt nur für IQ4_NL. Mit QAT ist MTP ein Netto-Nachteil (-9% bis -21%), siehe `docs/fork/2026-07-14_MTP_DRAFT_COMPARISON.md`.

---

## Implementierte Optimierungen mit messbarem Gain

| # | Optimierung | System | Gain | Datum |
|---|-------------|--------|------|-------|
| thecodacus | MoE Pinning+Prefetch (`GGML_CUDA_REGISTER_HOST=1`, `GGML_SCHED_PREFETCH_EXPERTS=1`) | Styx (Pascal) | **pp +72–106%, tg +31%** (IQ4_NL Config A→B); MTP +50% nur IQ4_NL, mit QAT -9% bis -21% | 08.07. |
| #3 | Pascal MMVQ (DP4A, PR #25479 portiert) | Styx | **tg +8.9%** (E2B) | 11.07. |
| QAT-Standard | QAT-Modell statt IQ4_NL | Mars | **pp +10%, tg +16.6%, +76k Kontext** | 10.07. |
| QAT-Standard | QAT-Modell statt IQ4_NL | Styx | tg ±0%, **+64k Kontext** | 10.07. |
| E4B FA-Fix | head_dim=512 → TILE-Kernel | Uranus | Crash→103 t/s (turbo4), 112 t/s (f16) | 12.07. |
| #56 | Vulkan UMA Cached Host Memory (PR #23762) | Mars | **tg +7%** (16.73→17.90), +11% vs baseline | 13.07. |
| #37 | Prefetch 2-Slot Sweet-Spot (`GGML_SCHED_PREFETCH_SLOTS=2`) | Styx | **pp +28.9%, +36.2% pp2048, +36.7% pp8192** | 13.07. |
| #37 | Prefetch 2-Slot Sweet-Spot | Mars | **pp +8.8%, tg +3.8%** (reiner Win) | 13.07. |
| #34 | UBBoost (dynamische Ubatch, `-ubp 512`) | Styx | **E4B: +20% pp2048, +41% pp8192, +19% tg128**; 26B: +18% pp2048 | 13.07. |
| #78 | Vulkan Pipeline Cache Disk (`GGML_VK_CACHE_DIR`) | Mars | **-33% Startup** (1.161s→0.781s, kleines Modell) | 14.07. |
| Cache-RAM | `--cache-ram 32768` | Uranus | **17x** System-Prompt-Cache-Hit (7395ms→429ms) | 13.07. |
| #35 | Row-Packing DMMV RDNA3 | Mars | +1% (minimal, unter Erwartung) | 12.07. |
| #85 | Vulkan Push Descriptors (`VK_KHR_push_descriptor`) | Mars | ±0% (kein RADV-Benefit) | 14.07. |

---

## Evaluiert und verworfen — mit Messung

| # | Ansatz | Messung | Begründung | Datum |
|---|--------|---------|------------|-------|
| #57 | MMQ Stream-k Disable | **pp -64% bis -99.6%** (26B: 2669→17.5 t/s) | Stream-k essenziell für MMQ auf Ada | 13.07. |
| #45 | CUDA Concurrent Streams QKV | tg **-10.7%** (Uranus), ±0% (Styx) | Bricht CUDA-Graph-Capture; bei MoE-Offload inaktiv | 12.07. |
| #32 | Pascal L1 Cache Tuning | pp -0.5%, tg -0.04% | GTX 1070 nur 48KB L1/SM, sofort evicted | 11.07. |
| #77 | K-Quant MMVQ Path Fix | pp ±0.5%, tg MMVQ +3% | Issue #21151 auf Phoenix/RADV nicht reproduzierbar | 14.07. |
| #61 | Persistent VRAM Expert Cache | pp -1%, tg -8.5% (Bail-Out korrekt) | Pascal ohne FA: CPU schneller als Cache. Erst Ampere+ relevant | 14.07. |
| #84 | Wave32/Wave64 Tuning | (nicht gemessen) | Wave64 bereits optimal (20-22% schneller als ROCm Wave32) | 14.07. |
| MTP Q4_0 | MTP Draft auf 26B | Mars -2.4%, Styx -14% | 48-57% Acceptance reicht nicht, Draft-Overhead überwiegt | 10.07. |
| #40 Phase 2 | Frequency-guided MoE Offloading | tg **-7%** | Kälteste Layer = späte Layer, die auf GPU müssen | 13.07. |

---

## Forschungsertrag vs. Aufwand

**ROADMAP-Bewegung in der Woche:** 22✅ → 25✅ (+3 completed), 14❌ → 16❌ (+2 rejected), 7⏭️ → 10⏭️ (+3 postponed), 53☐ → 46☐ (-7 evaluiert).

**4 Research-Sweeps** (je 4 parallele Subagents): ~72 neue Ansätze identifiziert, 11 in ROADMAP aufgenommen als #56–#87.

**Pattern — was funktioniert hat:**
1. **PR-Portierungen** mit konkretem Code (thecodacus, #3 Pascal MMVQ, #56 UMA Cached)
2. **Modell-Wechsel** (QAT statt IQ4_NL — 0 Code, nur Config)
3. **Bug-Fixes** (E4B FA Crash, 188k-Klippe RCA, Prefetch Graph-Suche)

**Pattern — was nicht funktioniert hat:**
- Vulkan-Micro-Optimierungen auf RADV (#85 Push Descriptors ±0%, #77 MMVQ ±0%, #84 Wave64 bereits optimal, #35 Row-Packing +1%). RADV ist bereits gut optimiert — auf Vulkan/RDNA3 gibt es kaum noch niedrighängende Früchte.
- Research-Sweeps lieferten vor allem **Negativ-Ergebnisse** — wertvoll zur Vermeidung von Zeitverschwendung, aber nicht direkt als Speedup messbar.

---

## Kritische Einschätzung

**Die Woche war sehr produktiv für Styx und Mars** (tg +31%/+16.6%, Kontext +64k/+76k), brachte aber auf Vulkan/RADV-Seite nur Negativ-Erkenntnisse. Der Upstream-Rebase (Handoff Option A) würde vermutlich mehr bringen als weitere Vulkan-Micro-Optimierungen — viele Tier-2-Items (#86 DSD, #83 IQ*_K, #67 MXFP4) sind upstream bereits implementiert.

**Aufwand:** 161 Commits, ~4 Research-Sweeps, ~20 evaluierte ROADMAP-Items.
**Ertrag:** 5 Optimierungen mit messbarem produktiven Gain (thecodacus, #3, QAT, #56, #37 Prefetch 2-Slot), 1 Crash-Fix (E4B), 1 Infrastruktur-Win (xtts-api 3.2x).

---

## Begleitende Notizen

- **188k-"Klippe" geklärt (12.07.):** Kein Vulkan-Bug, sondern OOM-Artefakt bei konkurrierenden llama-servern (2×16 GB GTT > 26 GB Limit). Solo-Betrieb 224k/256k: 28-32 t/s. 180k-Grenze obsolet.
- **Mars LXC-Migration (12.07.):** Bare-metal → LXC 240 (phobos), 256k Kontext (Modell-Maximum), 2×128k Slots. Performance identisch zu bare-metal.
- **Styx Btrfs-Migration (09.07.):** Pinning-Feature wieder nutzbar (vorher durch Kernel-Bug lahmgelegt), 24.6 t/s bestätigt.
- **Styx Crash-Loop (14.07. 16:18–16:25):** Zwei festgefahrene `llama-cli`-Test-Prozesse (seit 06:15/06:28, `-ngl 0` auf 26B-Pruned) blockierten 2880 MiB VRAM → Produktiv-Service OOM-Crash-Loop (2765 Restarts). Gefixt durch Kill der Test-Prozesse + Service-Neustart. **Lehre:** Test-Prozesse mit `-ngl 0` auf 26B-Modellen können festfahren und müssen explizit beendet werden; `timeout`-Wrapper empfohlen.
- **MTP-Draft-Vergleich (14.07.):** Benchmark mit QAT + Q4_K_M und Q4_0 Drafts auf Styx. Ergebnis: MTP ist mit QAT ein Netto-Nachteil (Q4_0: -9%, Q4_K_M: -21%). Q4_0 ist der bessere Draft (höhere Akzeptanz, geringerer Overhead). Der 08.07. Wert von 31.9 t/s gilt nur für IQ4_NL Config B und ist nicht auf QAT übertragbar. Siehe `docs/fork/2026-07-14_MTP_DRAFT_COMPARISON.md`.
