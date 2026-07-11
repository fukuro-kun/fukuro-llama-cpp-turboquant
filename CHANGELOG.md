# CHANGELOG — fukuro-llama-cpp-turboquant

Chronologische Auflistung aller bedeutsamen Änderungen. Solo-Session-Agenten tragen hier ihren Fortschritt ein.

Format: `YYYY-MM-DD — <type>: <Was> — <Warum>`

---

## 2026-07-11

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
