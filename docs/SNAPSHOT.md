# Momentaufnahme — fukuro-llama-cpp-turboquant

**Datum:** 2026-07-26 | **Branch:** `master` | **Letzter Commit:** `3504a2c00` (#44 Alloc-MoE Phase 0 ❌ NO-GO)

---

## Was zuletzt passiert ist

### 2026-07-26: MoE-Optimierung Tier-3 erschöpft (3 ❌ in einer Session)

Solo-Session mit Auto-Fortsetzung (2×). Drei Tier-3 ROADMAP-Items implementiert, benchmarked und als ❌ NO-GO evaluiert:

1. **#71 Set-Associative MoE Cache** (`POLICY_SET_ASSOC_LRU`) — N×M Slots mit LRU pro Set. Benchmark: -25% bis -50% bei 512MB, Rauschen bei 1024MB. Root Cause: 128 Experten × 30 Layer = 3840 Entries → 160-320 Slots = 24:1 Oversubscription. Fully-associative LRU hat keine Conflict-Misses. Commit `e1d9ae2bf`.

2. **#18 DALI Workload-Aware Cache** (`POLICY_WORKLOAD`) — Sliding-window workload accumulation, periodic reset. Benchmark: +81% im ersten Run (Artefakt!), +2.5% bei Verification, -1.1% bis +0.4% bei Final. Root Cause: PCIe-Transfer dominiert, nicht Eviction-Policy. Commit `d2d3c8ce6`.

3. **#44 Alloc-MoE Phase 0** (`LLAMA_MOE_K_OVERRIDE`) — K-Reduktion Quality-Benchmark. Wichtige Korrektur: Gemma-4-26B-A4B nutzt K=8 (nicht K=2). Perplexity: K=6 +13.5%, K=4 +22.9%, K=2 +66.5%. Speed: K=6 +24%, K=4 +38%. Go/No-Go >5% PPL-Drop → ❌. Commit `3504a2c00`.

**Code-Review** (review-swe Subagent): 5 P1 Issues gefunden, alle gefixt (queued-slot detection, off-by-one, pool-init locking, invalidate clear, Kommentar). Commits `d9346c051`, `f942ef2b9`, `8d3ac34bb`.

### 2026-07-25: /slots progress + n_tokens_total

`/slots` JSON-Endpoint erweitert um `progress` (0.0–1.0) und `n_tokens_total` pro Slot. Ermöglicht janus-Router zwischen "Backend arbeitet" und "Backend hängt" zu unterscheiden. Commit `ef8459dcf`.

## Wo wir stehen — das große Bild

### ROADMAP Meilensteine

| Meilenstein | Status | Items |
|-------------|--------|-------|
| M1-M5 | ✅ Abgeschlossen | TurboQuant, Vulkan, MTP, UBBoost, etc. |
| M6 (Forschung) | 🟡 In Arbeit | Tier-3 MoE-Items erschöpft (3×❌), Tier-2 re-evaluieren |

### Tier-3 MoE-Optimierung: ERSCHÖPFT

| Item | Policy | Ergebnis | Commit |
|------|--------|----------|--------|
| #69 | Heuristic (freq+recency) | -3.1% (kurz), +1.8% (lang) → ❌ | (vor dieser Session) |
| #71 | Set-Associative (N×M) | -25% bis -50% → ❌ | `e1d9ae2bf` |
| #18 | Workload-Aware (windowed) | +2.5% (kurz), -1.1% (lang) → ❌ | `d2d3c8ce6` |
| #44 | Alloc-MoE (K-Reduktion) | K=6 +13.5% PPL → ❌ | `3504a2c00` |

**Fazit:** LRU Cache + K=8 ist nahezu optimal für Gemma-4-26B-A4B. PCIe-Transfer und Expert-Diversität sind die Bottlenecks, nicht Eviction-Policy oder K-Anzahl.

### Tier-2: ⏭️ Re-Evaluierung nötig

Verbleibende ⏭️ Items die reif für Re-Evaluierung sein könnten:
- **#14 LFU Caching** — "SpecMD 85× besser als LRU" — aber MoE-Cache-Thema erschöpft
- **#38 Conf-KV** — KV-Eviction, komplementär zu TurboQuant
- **#40 Phase 2** — Per-Expert-Platzierung (erfordert Tensor-Splitting)
- **#63 xKV** — Cross-Layer KV-Compression (8× Kompression)

### Produktiv-Status

| System | Modell | Ctx | Service | Status |
|--------|--------|-----|---------|--------|
| Mars (LXC phobos) | 26B-A4B QAT | 256k | llama-server.service | ✅ |
| Styx | 26B-A4B QAT | 196k | llama-server-styx.service | ✅ |
| Venus | 26B-A4B QAT | 256k | llama-server-venus.service | ✅ (Suspend 08-13h) |
| Uranus | 26B-A4B QAT | — | llama-server (user) | ✅ |

## Aktuell in Arbeit (uncommitted)

Keine uncommitteten Änderungen. Alle 8 Commits der Session gepusht.

## Offene Aufgaben

### [D]evin (Auto-Fortsetzung)
- Keine weiteren Tier-3 Items ohne User-Entscheidung
- Tier-2 Re-Evaluierung möglich (braucht User-Input welche Items priorisiert)

### [F]ukuro
- Entscheidung: Tier-2 Re-Evaluierung oder anderes Thema (z.B. neues Modell, Vulkan-Optimierung)
- Hydra GPU-Ausnahme war auf diese Session begrenzt — bei nächsten GPU-Benchmarks wieder Pascal-Host verwenden

## Nächste Schritte

1. **User-Entscheidung:** Welche Richtung nach Tier-3 Erschöpfung? (Tier-2 Re-Eval, neues Thema, oder Pause)
2. **Tier-2 Re-Eval** (falls gewünscht): #38 Conf-KV oder #63 xKV als KV-Compression-Fokus (komplementär zu TurboQuant)
3. **MoE-Cache-Code aufräumen:** 3 neue Policies (Default OFF) — evtl. dead code entfernen wenn langfristig ungenutzt
