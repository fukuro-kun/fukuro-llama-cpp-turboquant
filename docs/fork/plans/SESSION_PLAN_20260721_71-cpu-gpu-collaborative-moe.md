# SESSION_PLAN: #71 Efficient CPU-GPU Collaborative MoE

**Erstellt:** 2026-07-21
**Status:** Plan fertig, Implementierung ausstehend
**ROADMAP-Item:** #71 Efficient CPU-GPU Collaborative MoE (Tier 3, 2 Wochen)
**Vorgänger:** #69 FlashMoE Heuristic Benchmark (✅ 2026-07-21: ❌ NO-GO, Voraussetzung erfüllt)

## Session-Ziel

Implementiere einen **N-index M-way set-associative Cache** für den bestehenden MoE-Expert-Cache in `ggml/src/ggml-cuda/moe-cache.cu`. Das ersetzt die aktuelle fully-associative + HashMap-Struktur durch eine set-associative Organisation (N×M Slots, LRU pro Set), die laut Paper (arXiv:2512.16473) 4.4× Speedup auf RTX 4090 + 24 Cores bringt — auf Styx/Hydra (8GB, PCIe-Transfer-Bottleneck) realistisch 5-15%.

## Warum jetzt?

- **#69 Heuristic-Benchmark abgeschlossen (2026-07-21):** Heuristic -3.1% vs LRU → ❌ NO-GO. Die Voraussetzung für #71 (Konfundierung vermeiden) ist erfüllt.
- **Fork hat bereits 1883-Zeilen MoE-Cache:** `moe-cache.cu` mit fully-associative + HashMap, LRU + heuristic policy, async fetch, CPU-miss-compute. #71 ist ein **Upgrade** der Cache-Organisation, kein Neubau.
- **MIT-Referenzcode verfügbar:** `github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference` (PyTorch, aber Algorithmus portierbar).
- **Styx + Hydra profitieren:** 8GB VRAM, PCIe-Transfer-Bottleneck — genau das Szenario für set-associative Caching.

## Architektur: Set-Associative Cache

```
Aktuell (fully-associative + HashMap):
┌─────────────────────────────────────────┐
│  HashMap: (blk, eid) → slot_idx         │
│  Slots:   [slot_0, slot_1, ..., slot_N]  │  LRU global
│  Lookup:  HashMap + linear scan         │
└─────────────────────────────────────────┘

Ziel (N-index M-way set-associative):
┌─────────────────────────────────────────┐
│  Hash: (blk, eid) % N → set_idx         │
│  Sets:   [set_0, set_1, ..., set_N-1]   │
│  Slots:  M slots pro Set, LRU pro Set   │
│  Lookup: 1 Hash + M-Vergleiche          │
└─────────────────────────────────────────┘
```

**Vorteile:**
- O(1) Lookup mit vorhersagbarer Latenz (M Vergleiche, nicht N)
- Bessere Cache-Locality (Sets sind contiguous im Memory)
- Kein HashMap-Overhead (Hash direkt auf Set-Index)
- LRU pro Set ist lokaler → weniger Cache-Thrashing bei wechselnden Experten

**Nachteile:**
- M-1 Slots pro Set sind "verschenkt" wenn ein Set heiß ist und andere kalt
- Konflikts-Misses wenn Hash-Funktion schlecht ist (mit #40-Frequency-Daten als Hash-Input mitigierbar)

## Phasen

### Phase 1: Design + Hash-Funktion (2-3 Tage)

1. **Cache-Organisation festlegen:** N (Anzahl Sets) und M (Wege pro Set)
   - Paper: N=64, M=4 (256 Slots total)
   - Fork: Budget-basiert — `budget_mb / slot_size / M = N`
   - Bei 512MB Budget + 2MB Slot = 256 Slots → N=64, M=4
2. **Hash-Funktion:** `(blk * n_expert + eid) % N` als Basis
   - Optional: #40-Frequency-Daten als zusätzlicher Hash-Input (heiße Experten in unterschiedliche Sets streuen)
3. **Datenstruktur:** `moe_cache_set { slot slots[M]; uint32_t lru_head; }` ersetzt `moe_cache_slot`-Array + HashMap
4. **Lookup:** Hash → Set → linear scan M Slots → hit/miss
5. **Eviction:** LRU innerhalb des Sets (M-1 Verschiebungen, nicht N)

### Phase 2: Implementierung (4-5 Tage)

1. **`moe-cache.cu` refactor:** Neues `moe_cache_set`-Struct, alte HashMap entfernen
2. **Lookup-Funktion:** `moe_cache_lookup(blk, eid) → slot_idx | MISS`
3. **Insert-Funktion:** `moe_cache_insert(blk, eid, weight)` mit LRU-Eviction im Set
4. **Async-Worker:** Bestehende Insert-Worker wiederverwenden, nur Lookup-Pfad ändert
5. **Policy-Variante:** `POLICY_SET_ASSOC_LRU` als neue Option (LRU/Heuristic bleiben für A/B-Tests)

### Phase 3: Benchmark + Validierung (3-4 Tage)

1. **Benchmark auf Styx:** tg128 + tg512, Set-Associative vs LRU vs Heuristic
   - Budget: 256MB, 512MB, 1024MB (falls VRAM erlaubt)
   - N/M-Variation: 64/4, 128/2, 32/8
2. **Go/No-Go-Gate:** >5% Speedup vs LRU → Go für Produktiv
3. **Regression-Test:** Cache-Stats vergleichen (Hit-Rate, Eviction-Rate, Latency)
4. **Hydra-Test:** Auf Hydra (RTX 3070 Mobile, 8GB) verifizieren — GPU ist gesperrt für Inference, nur CPU+Cache-Test mit `-ngl 0` oder kurzer GPU-Burst

### Phase 4: Doku + Commit (1-2 Tage)

1. **ROADMAP #71 → ✅** (oder ❌ mit Begründung falls <5%)
2. **CHANGELOG-Eintrag**
3. **Design-Doc Update:** `docs/fork/2026-07-15_MOE_CACHE_PREDICTION_DESIGN.md` um Set-Associative-Sektion erweitern
4. **Trilium-Update:** Projekt-Note `eiba6WJDfTiq` + Benchmark-Note `WjqL5Ky9Z3Hf`
5. **TTT-Eintrag**

## Risiko-Assessment

| Risiko | Impact | Mitigation |
|--------|--------|------------|
| Hash-Kollisionen bei 128 Experten | Mittel | Hash-Funktion mit #40-Frequency-Daten optimieren |
| Set-Associative schlechter als Fully-Associative bei kleinem Cache | Niedrig | A/B-Test mit verschiedenen N/M-Konfigurationen |
| Vulkan-Hosts (Mars/Venus) profitieren nicht | Niedrig | CUDA-only, Vulkan nicht berührt |
| Paper's 4.4× nicht reproduzierbar auf 8GB | Mittel | Realistische Erwartung: 5-15%, Go/No-Go bei 5% |
| #40-Frequency-Tracking als Hash-Input kollidiert mit bestehendem Use | Niedrig | Read-only Zugriff, keine Änderung an #40 |

## Verifikation

- [ ] `moe-cache.cu` kompiliert mit `POLICY_SET_ASSOC_LRU`
- [ ] Benchmark: tg128 Set-Associative vs LRU auf Styx
- [ ] Hit-Rate ≥ LRU-Hit-Rate (39.7% bei 512MB Budget)
- [ ] Cache-Stats: Eviction-Rate, Lookup-Latenz dokumentiert
- [ ] Keine Regression bei PP (Cache ist decode-only)
- [ ] Hydra-Build erfolgreich (CPU-Only-Test)

## Referenzen

- **Paper:** arXiv:2512.16473 (Efficient CPU-GPU Collaborative MoE)
- **Referenzcode:** `github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference` (MIT, PyTorch)
- **Fork MoE-Cache:** `ggml/src/ggml-cuda/moe-cache.cu` (1883 Zeilen, fully-associative + HashMap)
- **#69 Benchmark:** `docs/fork/2026-07-15_MOE_CACHE_PREDICTION_DESIGN.md` (Heuristic -3.1% vs LRU)
- **#40 Frequency Tracking:** `src/llama-context.cpp:1404,2585`, `src/llama-graph.cpp:1592`
