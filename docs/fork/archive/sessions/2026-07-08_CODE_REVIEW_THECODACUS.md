# Code-Review: thecodacus MoE-Optimierungen

**Datum:** 2026-07-08
**Reviewer:** KI-Agent (Solo-Session)
**Commits:** 20f5994, 1163cb3, 5f83fbb (thecodacus/llama.cpp, Branch fable5/prefetch-experts)
**Fork-Commit:** a4215b3d6

## Überblick

3 Patches, +232 Zeilen in 4 Dateien. Code ist sauber, gut strukturiert und produktionsreif.

## Patch 1: Memory Pinning (llama-mmap.cpp/h, llama-model-loader.cpp)

**Bewertung: ✅ Sauber, keine Issues**

- `register_host` Methode mit Function-Pointer für reg/unreg — saubere Abstraktion
- Page-aligned Registration (expandiert zu Page-Boundaries) — korrekt
- Destructor unpinned vor Unmap — korrekte Reihenfolge
- `_POSIX_MAPPED_FILES` Guard für Portabilität
- `GGML_UNUSED` für non-POSIX Plattformen
- Backend-Lookup via `ggml_backend_reg_get_proc_address` — saubere Abstraktion

## Patch 2+3: Async Expert Prefetch (ggml-backend.cpp)

**Bewertung: ✅ Sauber, keine kritischen Issues**

### Architektur
- `GGML_SCHED_MAX_PREFETCH_SLOTS 8` — sinnvolles Limit
- Default 3 Slots (deckt gate/up/down einer MoE-Layer)
- Zweite Backend-Instanz auf gleichem Device für async Uploads
- Event-basierte Synchronisation: `prefetch_ready` / `prefetch_free`

### Korrektheit
- **UAF Fix (Patch 3):** `prefetch_saved_buffer` und `prefetch_saved_data` restaurieren den Tensor's original buffer/data nach Kernel-Launch — kritischer Fix, korrekt implementiert
- Capability-Check: `props.caps.async && props.caps.events` — nur auf unterstützenden Backends aktiv
- Slot-Sizing: `ggml_backend_sched_prefetch_max_size` sized Slots einmal für den größten Expert-Tensor
- Graceful Degradation: Bei Allokationsfehler für Slot >= 2, Weiterarbeit mit weniger Slots
- `ggml_backend_sched_prefetch_disable` räumt sauber auf (synchronize + free)
- Free-Funktion iteriert `GGML_MAX_PREFETCH_SLOTS` (nicht `prefetch_n_slots`) — korrekt, da Slot-Count reduziert worden sein könnte

### Heuristik
- Prefetch nur wenn `ids->ne[0]*ids->ne[1] >= 2*n_expert` — "virtually every expert is used" (großer Batch)
- Bei kleinen Batches (Decode) wird der reguläre Copy-Pfad verwendet — korrekt, da bei wenigen aktiven Experten das Warten auf Routing-IDs effizienter ist

### Thread-Safety
- Separate Backend-Instanz → kein Shared-State-Problem
- Events koordinieren Upload- und Compute-Stream

### Env-Var Parsing
- `atoi` für `GGML_SCHED_PREFETCH_EXPERTS` — `atoi("invalid")` returns 0 → deaktiviert Prefetch. Safe.
- `<= 1` → Default 3 Slots. `> 1` → verwendet Wert direkt. Clean.

### Ressourcen-Management
- Alle Ressourcen (backend, buffers, events) werden in `ggml_backend_sched_free` korrekt freigegeben
- `ggml_backend_sched_synchronize` synct auch `prefetch_backend` — korrekt

## Fazit

**Code ist produktionsreif.** Keine Änderungen nötig. Die UAF-Fix in Patch 3 ist kritisch und korrekt implementiert. Die Capability-Checks stellen sicher, dass der Code nur auf unterstützenden Backends aktiv wird. Die Graceful Degradation bei Allokationsfehlern ist robust.

**Empfehlung:** Merge nach master (bereits erfolgt, Commit a4215b3d6).
