# SESSION_PLAN — UBBoost + MoE Load Balancing (2026-07-13)

**Session-Ziel:** #34 UBBoost (dynamische ubatch-Größe) und #40 MoE Load Balancing (Expert-Frequency-Tracking) implementieren, benchmarken, dokumentieren. Nahtlos weiter mit nächsten ROADMAP-Items.

## Entscheidungen (Briefing 2026-07-13 03:25)

| Frage | Entscheidung | Begründung |
|-------|-------------|------------|
| Items | #34 + #40, dann nahtlos weiter | Styx als Test-System, ergänzen sich |
| GPU-Freigabe Styx | Service stoppen → benchmarken → wiederherstellen | Produktiv-Service darf temporär gestoppt werden |
| GPU-Freigabe Mars | Gleiches Prinzip | Für #8 falls drangekommen |
| Uranus | xtts-api unantastbar, llama.cpp frei+restore | Andere Session testet mit xtts-api |
| Codeberg-Push | Pro Phase bei positivem Ergebnis, negativ lokal | Mittelweg |
| Styx Fork-Stand | Pull auf e09da1df4 + Rebuild | Aktuell halten |
| Subagent-Limit | Max 2 (verschräfte Rate-Limits) | Session-spezifisch |
| Arbeitsmodus | Autonom, keine Zwischenberichte, erst wieder melden wenn nichts Sinnvolles mehr zu tun | 12h Solo-Session |

## Schritte

### Phase 0: Vorbereitung
- [x] LOKAL.md gelesen
- [x] Styx Service gestoppt
- [x] Styx Fork gepullt auf e09da1df4
- [x] Styx Build gestartet (Hintergrund)
- [x] Code-Analyse #34 (n_ubatch Architektur verstanden)
- [ ] SESSION_PLAN geschrieben ← du bist hier
- [ ] hydra Build (parallel, CPU-only Korrektheitstest)

### Phase 1: #34 UBBoost Implementierung
- [ ] `n_ubatch_prefill` zu `llama_cparams` hinzufügen
- [ ] `llama_context::sched_reserve`: Graph für max(n_ubatch, n_ubatch_prefill) reservieren
- [ ] `llama_context::decode`: n_ubatch_prefill für Prefill verwenden
- [ ] CLI flag `--ub-prefill` in common/arg.cpp
- [ ] Public API in include/llama.h
- [ ] Build auf hydra
- [ ] CPU-only Korrektheitstest auf hydra

### Phase 2: #34 Benchmark auf Styx
- [ ] Styx Build fertig → Binary verifizieren
- [ ] Baseline: llama-bench mit n_ubatch=512 (Default)
- [ ] UBBoost: llama-bench mit --ub-prefill 1024/2048/4096
- [ ] Modell: 26B QAT, pp512/pp2048/pp8192, tg128/tg512
- [ ] OOM-Verhalten dokumentieren
- [ ] Ergebnis: ✅ (Speedup) / ❌ (kein Speedup/OOM) / ⏭️ (verschoben)

### Phase 3: #40 MoE Load Balancing
- [ ] Code-Analyse: --n-cpu-moe Verwendung, Expert-Tracking
- [ ] Expert-Frequency-Logging implementieren (welche Experts wie oft)
- [ ] Benchmark auf Styx: Frequency-Daten sammeln mit 26B QAT
- [ ] Analyse: statisch vs datengesteuert
- [ ] Ergebnis: ✅/❌/⏭️

### Phase 4: Nahtlos weiter
- [ ] Nächstes sinnvolles Item: #36 Auto-TP (Uranus) oder #8 Mixed Precision KV (Mars)
- [ ] Bei #36: PR #22950 portieren, Uranus llama.cpp-Server testen
- [ ] Bei #8: Commit e889fbd analysieren, TurboQuant-Kompatibilität prüfen

### Phase 5: Session-Abschluss
- [ ] Styx Service wiederherstellen (start + health check)
- [ ] ROADMAP #34, #40 Status aktualisieren
- [ ] CHANGELOG-Einträge
- [ ] TTT-Eintrag
- [ ] Retrospektive
- [ ] Push bei positiven Ergebnissen

## Verifikations-Strategie

| Schritt | Metrik | Vorher/Nachher |
|---------|--------|----------------|
| #34 UBBoost | pp t/s auf Styx | baseline n_ubatch=512 vs --ub-prefill 1024/2048/4096 |
| #34 OOM | VRAM-Verbrauch | nvidia-smi vor/während Benchmark |
| #40 MoE | Expert-Frequency-Verteilung | statisch vs gemessen |
| Korrektheit | Token-Output identisch | gleicher Prompt, gleiche Tokens |

## Defaults

- n_ubatch_prefill default = n_ubatch (kein Verhalten ohne Flag)
- Bei OOM: automatisch auf n_ubatch zurückfallen
- Bei unklaren Architektur-Entscheidungen: pragmatisch + reversibel

## Recherche-Strategien

| Problemtyp | Recherche-Strategie |
|------------|---------------------|
| n_ubatch Verwendung | grep + read (bereits gemacht) |
| RFC #23262 Details | web_search / brave-search |
| MoE Expert-Tracking | grep --n-cpu-moe + code lesen |
| TurboQuant-Interaktion (#8) | grep turbo + ISWA-Architektur |

## Recherche-Fallbacks

| Block | Mögliches Problem | Recherche-Fallback |
|-------|-------------------|-------------------|
| #34 Build | CMake-Fehler | CMakeLists.txt prüfen, incremental |
| #34 OOM auf Styx | VRAM zu klein | n_ubatch_prefill reduzieren, OOM-Grenze dokumentieren |
| #40 MoE | Expert-Tracking zu langsam | Logging nur im Debug-Modus |
| Styx Build | Build-Fehler | git diff, inkrementell zurückbauen |
