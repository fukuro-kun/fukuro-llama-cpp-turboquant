# CHANGELOG — fukuro-llama-cpp-turboquant

Chronologische Auflistung aller bedeutsamen Änderungen. Solo-Session-Agenten tragen hier ihren Fortschritt ein.

Format: `YYYY-MM-DD — <type>: <Was> — <Warum>`

---

## 2026-07-11

### Solo-Session (05:00–15:30)

- **feat: #3 Pascal CUDA MMVQ (PR #25479) — manuell portiert** — `MMVQ_PARAMETERS_PASCAL_DP4A` zum Enum hinzugefügt, Pascal CC 6.1/6.2 Detection, 2 Warps statt 4 für single-token decode (bandwidth-bound auf kleinen SMs). +14 -3 Zeilen in `mmvq.cu`. Code-Review: ship-ready, keine Issues. Build grün auf Hydra (CUDA) und Styx (CUDA CC 6.1). Commit `5c1884929`. **Benchmark auf Styx (GTX 1070, E2B MoE): Generation +8.9% (65.1 → 70.9 t/s), Prompt +0.7% (908.9 → 915.3 t/s).** Besser als die im PR versprochenen +3-6%.
- **❌ #13 Two-Tier Expert Cache (FR #20757) — nicht implementieren** — Alle 4 PRs (#21609, #21614, #23170, #24524) closed ohne Merge. thecodacus Memory Pinning im Fork deckt Tier 2 (pinned RAM) bereits ab. Geringer ROI für Pascal (compute-bound nicht PCIe-bound). RFC #24528 offen aber nicht umgesetzt.
- **❌ #10 UMA Zero-Copy (PR #22462) — nicht implementieren** — Verwandte UMA-PRs (#22455, #22930, #23770) wurden bereits getestet und revertiert. RCA-Masterplan zeigt: System-RAM ist langsamer als GTT für GPU-Compute auf Mars. PR #22462 ist open/unstable, verfolgt einen ähnlichen Ansatz.
- **✅ #6 MUL_MAT_ID Subgroup (PR #15524) — bereits integriert** — PR ist upstream gemerged (Aug 2025). Fork hat bereits `MatMulIdType::SUBGROUP`, `mul_mm_id_funcs.glsl` mit `subgroupBallot`, `matmul_id_subgroup_*` Shader werden generiert, Pipelines aktiv (`subgroup_ballot && subgroup_require_full_support && subgroup_min_size_16`). Vorheriger Revert war auf fehlerhaften Merge zurückzuführen, aktueller Stand ist sauber.
- **feat: #2 Tensor Split Regex static (PR #24710)** — 29 std::regex Patterns von `const` auf `static const` in `src/llama-model.cpp`. Verhindert pro-Token Regex-Recompilation im Tensor-Split-Modus. Build grün auf Hydra (CUDA) und Mars (Vulkan). Commit `bc1eb0207`
- **❌ #1 MTP Logits Copy (PR #23198) — skipped** — 19 Merge-Konflikte in 10 Kern-Dateien (speculative.cpp, llama-context.cpp, qwen35.cpp, etc.). Fork hat `nextn`-Support, PR fügt `pre_norm`-Support hinzu — strukturelle Konflikte. MTP ist OFF in Produktion. Zu hoher Aufwand für geringen Nutzen.
- **❌ #6 MUL_MAT_ID Subgroup (PR #15524) — revertiert** — 23 Merge-Konflikte in 3 Vulkan-Dateien (18 in ggml-vulkan.cpp). Subagent löste Konflikte, aber Shader-Generierung wurde beschädigt (undefined references für `matmul_id_*` Shader). Erfordert manuelle Portierung mit tiefem Vulkan-Shader-Verständnis. Für spätere Session.
- **✅ #5 GTT Size Tuning — bereits konfiguriert** — Mars hat bereits `amdgpu.gttsize=26624` (26GB) in `/proc/cmdline`. GTT ist 26GB, VRAM ist 1GB (APU). Kein weiteres Tuning nötig.
- **⏳ #4 n-gram Decoding — verfügbar, kein Speedup auf E2B** — `--spec-type ngram-mod` auf Mars verfügbar und aktiviert (Log bestätigt `common_speculative_impl_ngram_mod`). Benchmark mit E2B-Modell (2.9GB) und repetitivem Counting-Prompt (256 tokens): **Baseline 39.2 t/s vs n-gram-mod 39.1 t/s** — kein messbarer Speedup. Erwartet für kleine Modelle mit nicht-perfekt-repetitiven Patterns. n-gram ist verfügbar für User die es nutzen wollen, aber kein Default-Speedup.
- **⏭️ #7 Vulkan FA Refactor (PR #19625) — verschoben** — Abhängig von #6 (MUL_MAT_ID), das revertiert wurde. Shader-Basis muss zuerst stabilisiert werden.
- **❌ #9 Vulkan Shmem-Staging (PR #20897) — PR closed** — PR wurde geschlossen ohne Merge (AI-generiert). Manuelle Portierung nötig, Risiko unklar.
- **docs: ROADMAP.md aktualisiert** — Status für #1, #2, #4, #5, #6, #7, #9 aktualisiert. M1 teilweise, M2 blockiert.

### Vorherige Änderungen

- **docs: ROADMAP-Workflow etabliert** — `docs/fork/ROADMAP.md` mit 30 Optimierungs-Ansätzen (M1-M6), Solo-Pläne für M1+M2 in `docs/fork/plans/`. Workflow in AGENTS.md dokumentiert.
- **docs: Commit-Format formalisiert** — `<type>: <Was> — <Warum>` (feat, docs, fix, security, refactor, bench)
- **docs: Optimierungs-Recherche (Web + arXiv, 30 Ansätze)** — 4 parallele Subagents, 50+ Quellen, 30 Ansätze kategorisiert nach Tier. Siehe `docs/fork/2026-07-11_OPTIMIZATION_RESEARCH.md`
- **fix: AGENTS.md Kontext-Tabelle korrigiert** — Styx "229k (lädt)" → "224k (lädt)" (229376 = 224×1024, nicht 229k)
- **feat: fork-speed-research Skill** — Projekt-spezifischer Skill in `.devin/skills/fork-speed-research/` für monatliche Recherche mit 4 parallelen Subagents

## 2026-07-10

- **feat: QAT Produktiv-Standard** — Services auf Mars und Styx von IQ4_NL auf QAT-UD-Q4_K_XL umgestellt. Kontextfenster von 180k/160k auf 224k erweitert. Mars 25.85 t/s, Styx 26.02 t/s. Commit `517ec94d1`
- **feat: Adapter für neue GGUF-Metadata-Keys** — QAT + MTP Q4_0 Draft Modelle laden korrekt. Commit `f9a3dfc62`
- **docs: Kontextfenster korrigiert** — Modell-Maximum ist 256K (262144), nicht 320k. Commit `0b62e3e1f`

## 2026-07-09

- **docs: Vulkan KV-Cache Benchmark** — turbo3/4 ist optimale Vulkan-Konfig (K=turbo3, V=turbo4). +31% schneller als turbo4/4 bei pp@96k-128k. Siehe `docs/fork/2026-07-09_VULKAN_KV_CACHE_BENCHMARK.md`

## 2026-07-08

- **feat: thecodacus MoE-Optimierungen** — Memory Pinning + Async Expert Prefetch. +72-106% pp, +30% tg auf GTX 1070. Siehe `docs/fork/2026-07-08_SOLO_SESSION_REPORT.md`
- **docs: FINALE EMPFEHLUNG aktualisiert** — QAT als Standard, Kontextfenster-Tests, MTP Q4_0 Ergebnisse
