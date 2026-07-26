# Momentaufnahme — fukuro-llama-cpp-turboquant

**Datum:** 2026-07-26 | **Branch:** `master` | **Letzter Commit:** `04ed54f16` (Code-Cleanup -358 Zeilen)

---

## Was zuletzt passiert ist

### 2026-07-26: MoE-Optimierung Tier-3 erschöpft + Code-Cleanup + Research-Sweep #5

Solo-Session mit Auto-Fortsetzung (4×). Drei Tier-3 ROADMAP-Items implementiert, benchmarked und als ❌ NO-GO evaluiert, dann toter Code entfernt und monatliche Recherche durchgeführt:

1. **#71 Set-Associative MoE Cache** — -25% bis -50%. Commit `e1d9ae2bf`.
2. **#18 DALI Workload-Aware Cache** — ±0%. Commit `d2d3c8ce6`.
3. **#44 Alloc-MoE Phase 0** — K=6 +13.5% PPL. Commit `3504a2c00`.
4. **Code-Cleanup** — -358 Zeilen toter Code aus 3 ❌ Experimenten entfernt. Nur LRU + Heuristic bleiben. Commit `04ed54f16`.
5. **Research-Sweep #5** — 4 parallele Subagents, 51 Items, 5/5 Quick-Wins bereits im Fork. 9 neue ROADMAP-Items (#89-97). Commits `311790c79`, `a13f4087c`.

**Code-Review** (review-swe Subagent): 5 P1 + 3 P2 Issues gefunden, alle gefixt. Commits `d9346c051`, `f942ef2b9`, `8d3ac34bb`.

### Echte Speedups (letzte 2 Wochen)

| Datum | Was | Speedup | Commit |
|-------|-----|---------|--------|
| 2026-07-20 | MoE-Default 10→6 (Uranus) | +26% tg, +85% pp | `ffd4845a0` |
| 2026-07-15 | 2-Slot Prefetch (Styx) | +28.9% pp, +2.1% tg | `9651e5fba` |
| 2026-07-15 | 2-Slot Prefetch (Mars) | +8.8% pp, +3.8% tg | `5c60d0b7b` |
| 2026-07-10 | QAT Produktiv-Standard | +10% pp, +16.6% tg (Mars) | `517ec94d1` |
| 2026-07-09 | turbo3/turbo4 mixed K/V | +31% pp@96k-128k | (Benchmark doc) |

**Hinweis:** Diese Session (2026-07-26) brachte **keinen** neuen Speedup — alle 3 MoE-Experimente waren ❌ NO-GO. Die letzten echten Speedups waren vom 2026-07-20 (MoE-Default Tuning) und 2026-07-15 (2-Slot Prefetch).

### 2026-07-25: /slots progress + n_tokens_total

`/slots` JSON-Endpoint erweitert um `progress` (0.0–1.0) und `n_tokens_total` pro Slot. Ermöglicht janus-Router zwischen "Backend arbeitet" und "Backend hängt" zu unterscheiden. Commit `ef8459dcf`.

## Wo wir stehen — das große Bild

### ROADMAP Meilensteine

| Meilenstein | Status | Items |
|-------------|--------|-------|
| M1-M5 | ✅ Abgeschlossen | TurboQuant, Vulkan, MTP, UBBoost, etc. |
| M6 (Forschung) | 🟡 In Arbeit | MoE-Tier-3 erschöpft (4×❌), Research-Sweep #5 komplett, Fork saturiert |

### Tier-3 MoE-Optimierung: ERSCHÖPFT (4× ❌)

| Item | Policy | Ergebnis | Commit |
|------|--------|----------|--------|
| #69 | Heuristic (freq+recency) | -3.1% → ❌ | (vor Session) |
| #71 | Set-Associative (N×M) | -25% bis -50% → ❌ | `e1d9ae2bf` |
| #18 | Workload-Aware (windowed) | ±0% → ❌ | `d2d3c8ce6` |
| #44 | Alloc-MoE (K-Reduktion) | K=6 +13.5% PPL → ❌ | `3504a2c00` |

**Fazit:** LRU Cache + K=8 ist optimal für Gemma-4-26B-A4B. PCIe-Transfer und Expert-Diversität sind die Bottlenecks. Code bereinigt (-358 Zeilen), nur LRU + Heuristic bleiben.

### Erkenntnisse für künftige Solo-Sessions

- **Mars hat coopmat2**, nicht coopmat1. Coopmat1-Optimierungen greifen nicht. Subagent-Prompts müssen coopmat2-Status explizit erwähnen.
- **Fine-grained MoE (128 Experten, K=8) hat zu wenig Spielraum für K-Reduktion.** Keine weiteren K-Reduktions-Experimente an Gemma-4-A4B.
- **PCIe-Transfer dominiert über Eviction-Policy.** Bei 24:1 Oversubscription ist Hit-Rate begrenzt durch Cache-Größe, nicht durch Eviction-Strategie.
- **Research-Sweeps sollten Items vor ROADMAP-Eintrag verifizieren.** 5/5 Quick-Wins waren bereits im Fork.
- **Fork ist saturiert.** Keine neuen umsetzbaren Quick-Wins. Neue Optimierungen erfordern Paper-Implementierungen (Tier 2-3) oder Upstream-Sync.
- **Code-Review findet echte Bugs.** P1#3 (pool-init locking) war Deadlock-Risk. Code-Review ist wertvoll, nicht optional.
- **Bei erschöpften Auto-Fortsetzungen: Code-Cleanup ist immer möglich.** Toter Code aus ❌ NO-GO Experimenten proaktiv aufräumen.

### Produktiv-Status

| System | Modell | Ctx | Service | Status |
|--------|--------|-----|---------|--------|
| Mars (LXC phobos) | 26B-A4B QAT | 256k | llama-server.service | ✅ |
| Styx | 26B-A4B QAT | 196k | llama-server-styx.service | ✅ |
| Venus | 26B-A4B QAT | 256k | llama-server-venus.service | ✅ (Suspend 08-13h) |
| Uranus | 26B-A4B QAT | — | llama-server (user) | ✅ |

## Aktuell in Arbeit (uncommitted)

Keine uncommitteten Änderungen. Alle 12 Commits der Session gepusht.

## Offene Aufgaben

### [D]evin (Auto-Fortsetzung)
- Keine weiteren Auto-Fortsetzungs-Arbeiten ohne User-Entscheidung
- Tier-2 Paper-Implementierungen (#92-97) brauchen User-Entscheidung (1-3 Wochen Aufwand)

### [F]ukuro
- **Upstream-Sync** — Fork ist ~683 Builds hinter upstream (b10133). Großer Rebase/Audit.
- **Tier-2 Re-Eval** — #38 Conf-KV, #63 xKV (KV-Compression jenseits TurboQuant)
- **Tier-2 Paper-Impl.** — #92 RateQuant, #93 InnerQ, #94 FineMoE (1-3 Wochen)
- **#88 MTP+TP Test** — wenn Uranus frei ist

## Nächste Schritte

1. **User-Entscheidung:** Upstream-Sync, Tier-2 Paper-Implementierung, oder anderes Thema
2. **Upstream-Sync** (falls gewünscht): ~683 Builds Audit, 2-4 Tage Aufwand
3. **Tier-2 Re-Eval** (falls gewünscht): #38 Conf-KV oder #63 xKV als KV-Compression-Fokus
