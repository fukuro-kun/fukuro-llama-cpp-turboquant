# SESSION_PLAN: LFRU Expert Caching (#37)

**Erstellt:** 2026-07-13 (Solo-Session)
**Status:** In Arbeit
**ROADMAP-Item:** #37 LFRU Expert Caching (Tier 2, 2-4 Wochen)

## Session-Ziel

Implementiere einen persistenten LFRU (Least Frequently Recently Used) GPU-Cache für MoE-Experten, der die selektive Copy-on-Demand-Logik im Scheduler erweitert. Hot experts bleiben auf GPU, cold experts werden on-demand hochgeladen. Basis: #40 Expert Frequency Tracking + thecodacus async prefetch.

## Entscheidungen

| Frage | Entscheidung | Begründung |
|-------|-------------|------------|
| Cache-Granularität? | Per-Layer, per-Expert | 3D Tensor wird in 2D Slices zerlegt (bereits durch selective copy path) |
| Eviction Policy? | LFRU: score = freq / (clock - last_access + 1) | vLLM validiert: +5.2% vs LRU, verhindert deep-layer starvation |
| Cache-Size Konfiguration? | Env var `GGML_EXPERT_CACHE_SIZE` (default: 0 = disabled) | Konsistent mit `GGML_SCHED_PREFETCH_EXPERTS` |
| Frequency-Quelle? | Runtime counter (incrementiert bei jedem Cache-Zugriff) | Simpler als #40 Integration, keine Scheduler→Context-Abhängigkeit |
| Async Upload bei Miss? | Ja, via `prefetch_backend` (thecodacus) | Überlappt Upload mit Compute |
| CUDA Graphs Kompatibilität? | Nein (wie vLLM `--enforce-eager`) | Dynamische Cache-Slots inkompatibel mit Graph-Capture |

## Schritte

- [x] #40 Expert Frequency Tracking (implementiert + reviewed)
- [x] #8 Mixed Precision KV Cache (evaluiert, postponed)
- [x] #37 Analyse (thecodacus prefetch + vLLM LFRU + selective copy path)
- [ ] 1. LFRU Cache-Struct in ggml-backend.cpp
- [ ] 2. Eviction Logic (score = freq / age)
- [ ] 3. Integration mit selective copy path (lines 1720-1804)
- [ ] 4. Expert-ID Remapping (cache slot → kernel IDs)
- [ ] 5. Build + Smoke-Test auf Hydra (CPU-only)
- [ ] 6. Benchmark auf Styx (Vorher/Nachher)
- [ ] 7. Code-Review + Doku

## Verifikations-Strategie

| Schritt | Metrik | Vergleich |
|---------|--------|-----------|
| 1-4 | Build grün | `cmake --build build` |
| 5 | Smoke-Test: Modell lädt, keine Crashes | `llama-bench -p 64 -n 8` |
| 6 | tg128 t/s auf Styx | Vorher (no cache) vs. Nachher (cache=8) |
| 6 | Cache hit rate | Log-Output: hits/total per layer |

## Architektur

### Current Selective Copy Path (ggml-backend.cpp:1720-1804)

```
For each MUL_MAT_ID node:
  1. Read routing IDs from ids_tensor
  2. Build used_ids bitset (which experts are needed)
  3. Group consecutive used experts
  4. For each group: ggml_backend_tensor_set_async to GPU buffer
  5. Remap IDs: original expert ID → compacted buffer index
  6. After compute: GPU buffer freed (no persistence)
```

### LFRU-Enhanced Path

```
For each MUL_MAT_ID node:
  1. Read routing IDs from ids_tensor
  2. For each needed expert (layer_id, expert_id):
     a. Check if (layer_id, expert_id) is in cache
     b. HIT: use cached slot, update freq++, last_access=clock
     c. MISS: evict lowest-score expert if cache full,
        upload new expert to freed slot, add to cache
  3. Remap IDs: original expert ID → cache slot index
  4. After compute: KEEP GPU buffer (persistent cache)
```

### Cache Struct

```cpp
struct expert_cache_entry {
    int layer_id;
    int expert_id;
    uint64_t frequency;
    uint64_t last_access;
    int slot_index;           // index into cache buffer
    bool valid;
};

struct expert_cache {
    std::map<std::pair<int,int>, expert_cache_entry> entries;  // (layer,expert) → entry
    std::vector<int> free_slots;    // available slot indices
    uint64_t clock;                 // global step counter
    int n_slots_per_layer;          // cache capacity per layer
    size_t slot_size;               // bytes per expert slice
    ggml_backend_buffer_t buffer;   // persistent GPU buffer
};
```

## Recherche-Strategien

| Problemtyp | Recherche-Strategie |
|------------|---------------------|
| Selective copy path Details | Read ggml-backend.cpp:1720-1804 carefully |
| Expert slice extraction | Check how `ggml_backend_tensor_set` handles 3D tensor slices |
| ID remapping mechanism | Read how `ids_tensor` is modified for compacted buffer |
| CUDA async upload | thecodacus patch: `ggml_backend_tensor_set_async` |

## Recherche-Fallbacks

| Block | Mögliches Problem | Recherche-Fallback |
|-------|-------------------|-------------------|
| Cache buffer alloc | VRAM OOM | Reduce n_slots, log cache size |
| ID remapping | Kernel expects original IDs | Check mmid.cu for ID handling |
| Async upload race | Slot overwritten during compute | Event synchronization wie thecodacus |
| Build errors | C++ struct in C file | Move to ggml-backend.cpp (C++ file) |

## Offene Fragen

- Wie funktioniert die ID-Remapping im Detail? (muss Code lesen)
- Kann der Cache-Buffer pro-Layer oder global sein? (pro-Layer ist simpler)
- Wie wird mit gate/up/down (3 Tensoren pro Expert) umgegangen? (3 Slices pro Expert)
