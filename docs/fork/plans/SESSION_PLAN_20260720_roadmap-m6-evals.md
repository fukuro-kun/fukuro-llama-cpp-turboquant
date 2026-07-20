# SESSION_PLAN_20260720 — ROADMAP M6 Tiefen-Evals ✅ ABGESCHLOSSEN

**Status:** ✅ Komplett (2026-07-21). Alle 4 Phasen durchgeführt. 11 Tier-3 Tiefen-Evals + 1 Tier-2 Eval + Rebase-Audit Fix.

**Session-Ziel:** Die 25 offenen ROADMAP-Items (M6 Forschung) systematisch tiefen-evaluieren wie #21 PagedAttention — pro Item entscheiden: ❌ verworfen (mit Begründung) oder "Portierung machbar, X Wochen" mit Solo-Plan. Plus Rebase-Audit nach verlorenen Features.

**Start:** 2026-07-20 05:10 Uhr
**User-Status:** fukuro schläft (05:10 Uhr), Solo-Session aktiviert

## Entscheidungen (aus vorheriger Session)

| Frage | Entscheidung | Begründung |
|-------|--------------|------------|
| Wellen-Reihenfolge | A→B→D→C (am Ende) | User-Auswahl |
| Nach Welle 1: was tun? | "autonom die roadmap abarbeiten" | User-Anweisung |
| Styx Update | Am Session-Ende | Styx noch in Nutzung |
| Uranus Service-Restart | NICHT restarten | vorleser-Training auf Port 18082 |

## Schritte (Todo-Liste)

### Phase 1: Rebase-Audit (höchster Wert — Zufallsfund skip_streak)
- [ ] 1.1 Systematische Diff-Suche: `88bd4f052^..master` nach Fork-Features die fehlen
- [ ] 1.2 Gefundene Lücken dokumentieren + bei Treffern re-applien
- [ ] 1.3 Build + Test + Commit + Push

### Phase 2: #36 Auto Parameter Fitting TP (einziger Tier-2)
- [ ] 2.1 PR #22950 Status + Merge-Readiness prüfen
- [ ] 2.2 Fork-Kompatibilität (fit.cpp:181 SPLIT_MODE_TENSOR Blockade)
- [ ] 2.3 Entscheidung: ❌ verworfen ODER Solo-Plan für Portierung

### Phase 3: Tier-3 Tiefen-Evals (nach Priorisierung)
- [ ] 3.1 #73 CascadeInfer — arXiv:2512.19179, 67% Latenz-Reduktion
- [ ] 3.2 #72 N4_0 Native 4-bit Float — PR #23572, +40% PP
- [ ] 3.3 #71 Efficient CPU-GPU Collaborative MoE — arXiv:2512.16473
- [ ] 3.4 #76 CPU Backend Operator Fusion — Diskussion #22315
- [ ] 3.5 #74 Vulkan Descriptor Indexing (Bindless)
- [ ] 3.6 #75 Non-blocking Pipeline Scheduling — PR #19922
- [ ] 3.7 #18 DALI — arXiv:2602.03495
- [ ] 3.8 #22 GWQ — arXiv:2411.00850
- [ ] 3.9 #23 DuoServe-MoE — arXiv:2509.07379
- [ ] 3.10 #43 SliderQuant — arXiv:2603.25284
- [ ] 3.11 #44 Alloc-MoE — arXiv:2604.08133

### Phase 4: Styx Update (am Session-Ende)
- [ ] 4.1 Styx auf a1b2500cd aktualisieren (git pull + build + service restart)

## Verifikations-Strategie

| Schritt | Verifikation |
|---------|-------------|
| Rebase-Audit | `git diff 88bd4f052^..master -- <feature>` zeigt Feature fehlt → grep im master findet keine Referenz → re-applien → Build grün |
| Tiefen-Eval pro Item | PR/Paper-Analyse + Fork-Codebase-Check + Hardware-Kompatibilität → Entscheidung ❌/☐ mit Begründung |
| Build | `cmake --build build --target llama-server` Exit 0 |
| Push | Code-Review vor Push (Grundprinzip 5) |

## Recherche-Strategien

| Problemtyp | Recherche-Strategie |
|------------|---------------------|
| PR-Status | `webfetch github.com/ggml-org/llama.cpp/pull/<id>` |
| Paper-Verfügbarkeit | arxiv-mcp (Skill `arxiv-mcp`) |
| Fork-Kompatibilität | `grep` + `read` im Codebase |
| Referenzcode | web_search + GitHub-Repo-Suche |

## Recherche-Fallbacks

| Block | Mögliches Problem | Recherche-Fallback |
|-------|-------------------|-------------------|
| Rebase-Audit | Diff zu groß | Nach Feature-Namen greppen, nicht ganzen Diff lesen |
| PR #22950 | PR geschlossen/merged seit letzter Eval | webfetch PR-Page, Status verifizieren |
| Paper-only Items | Kein Code verfügbar | arXiv-Paper lesen, Implementierungsaufwand schätzen |
| Tier-3 Items | Alle 6+ Wochen → in 48h nicht implementierbar | Fokus auf Eval, nicht Implementierung |

## Defaults

1. **Bei unklarem PR-Status:** Neu webfetchen, nicht alte Eval vertrauen
2. **Bei Paper-only:** ❌ wenn kein Referenzcode + >4 Wochen Aufwand, sonst ☐ mit Solo-Plan
3. **Bei Fork-Inkompatibilität:** ❌ mit Begründung (wie #21)
4. **Bei hohem Aufwand (>4 Wochen):** ☐ "später" mit Re-Eval-Hinweis, nicht ❌

## Offene Fragen

Keine — User-Anweisung ist klar: "autonom die roadmap abarbeiten"
