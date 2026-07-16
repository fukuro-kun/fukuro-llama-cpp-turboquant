# Turbo4 llama-server: Graph-Scheduler/Backend-Assignment Problem
 
**Datum:** 2026-06-20
**Status:** � Root Cause bestätigt — Hypothese A verifiziert (-np 1 funktioniert)
**Priorität:** Hoch (Blockiert Produktiv-Einsatz von turbo4 im Server-Modus)
 
---
 
## Zusammenfassung
 
**KORREKTUR:** Der Fehler ist kein SEGV und kein OOM. Der **Graph-Scheduler** kann einen turbo4-FlashAttention-Tensor nicht auf Vulkan platzieren und stuft ihn auf CPU zuruck. Dadurch wird Flash Attention deaktiviert, was aber fur quantized V-Cache zwingend erforderlich ist.
 
---
 
## Fehlerlog (Kritisch)
 
```
sched_reserve: layer 0 is assigned to device Vulkan0 but the Flash Attention tensor
              is assigned to device CPU (usually due to missing support)
sched_reserve: Flash Attention was auto, set to disabled
llama_init_from_model: failed to initialize the context:
              quantized V cache was requested, but this requires Flash Attention
```
 
---
 
## Analyse
 
### Was passiert
 
1. **Turbo4-FA-Tensor fallt auf CPU zuruck:** Der Graph-Scheduler kann nicht alle turbo4-Tensors auf Vulkan platzieren.
2. **FA wird deaktiviert:** Wenn ein FA-Tensor auf CPU liegt, wird Flash Attention automatisch deaktiviert.
3. **Abbruch:** Quantized V-Cache (turbo3/turbo4) erfordert zwingend Flash Attention. Ohne FA bricht die Initialisierung ab.
 
### Warum CLI funktioniert, Server nicht
 
| Aspekt | llama-cli | llama-server |
|--------|-----------|--------------|
| Graph | Einfach (Single Forward) | Komplex (Multi-Slot, Batch, Prompt-Cache) |
| FA-Tensor | Bleibt auf Vulkan | Fallt auf CPU zuruck |
| Ergebnis | ✅ Funktioniert | ❌ Abbruch |
 
---
 
## Was funktioniert vs. Was nicht
 
| Modus | K-Cache | V-Cache | Kontext | Ergebnis |
|-------|---------|---------|---------|----------|
| `llama-cli` | turbo4 | turbo4 | 2048 | ✅ KOHARENT |
| `llama-server` | turbo3 | turbo3 | 170000 | ✅ LAUFT |
| `llama-server` | turbo4 | turbo3 | 2048 | ❌ **Abbruch** (FA auf CPU) |
| `llama-server` | turbo4 | turbo4 | 2048 | ❌ **Abbruch** (FA auf CPU) |
 
---
 
## Root Cause Hypothesen
 
1. **Backend-Assignment:** `ggml-backend.cpp` oder `llama-graph.cpp` weist turbo4 FA-Tensor nicht korrekt Vulkan zu
2. **Alignment/Stride:** turbo4 Block-Layout (68 Bytes, nicht-Power-of-2) passt nicht zur Backend-Heuristik
3. **Missing Pipeline:** Ein Zwischentensor im Server-Graph braucht einen Shader, der fur turbo4 nicht registriert ist
 
---
 
## Untersuchungs-Plan
 
- [x] `ggml-backend.cpp`: Wie werden Tensors Backend zugewiesen? — Siehe `ggml_backend_sched_backend_id_from_cur` (Zeilen 913-968). Input-Tensoren (GGML_TENSOR_FLAG_INPUT) werden immer auf CPU gelegt (Zeile 937-940).
- [x] `llama-graph.cpp`: Server-Graph vs. CLI-Graph Unterschiede — Server Default: `n_parallel = 4` (Multi-Slot), CLI: `n_seq_max = 1`
- [ ] Prufen, ob turbo4 FA-Tensor ein spezielles Alignment braucht
- [ ] Test: `llama-server` mit `--batch-size 1` (vereinfachter Graph)
- [ ] Test: `llama-server -np 1` mit turbo4 (Hypothese A bestätigen)

## Root-Cause-Analyse (2026-07-15 Tiefen-Recherche)

### Vulkan FA-Support: turbo4 ist erlaubt

`ggml-vulkan.cpp` Zeilen 17124-17127: `fa_kv_ok()` gibt `true` für `GGML_TYPE_TURBO4_0` zurück. Der Vulkan-Backend **sagt** er unterstützt turbo4 FA. Das Problem liegt nicht im Vulkan-Support-Check.

### auto_fa Check verwendet synthetischen Graph

`llama-context.cpp` Zeile 517:
```cpp
auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
```

Der Check verwendet `n_tokens = 1` (minimaler Graph). Der echte Server-Betrieb verwendet `n_tokens` von 512-8192+ (Prefill) bzw. `n_seqs` (Decode). Der synthetische Graph repräsentiert möglicherweise nicht die echte Backend-Zuweisung.

### Drei Hypothesen

**Hypothese A (wahrscheinlich):** `n_seqs > 1` (Server Default: 4) verursacht einen anderen Graph mit Input-Tensoren die auf CPU gelegt werden. Die Backend-Expansion (Pass 2) kann den FA-Tensor nicht auf Vulkan ziehen weil er von einem CPU-Input abhängt.

**Hypothese B:** `n_tokens = 1` im auto_fa Check vs. `n_tokens > 1` im echten Betrieb. Andere Tensor-Shapes könnten andere Scheduler-Entscheidungen provozieren.

**Hypothese C:** Scheduler-Reset zwischen auto_fa Check und echter Graph-Reservierung ändert die Backend-Zuweisung.

### Test-Plan (Priorität)

1. **`llama-server -np 1` mit turbo4** — wenn das funktioniert → Hypothese A bestätigt
2. **`llama-server -np 1 -c 2048` mit turbo4** — Minimal-Konfiguration
3. **Vergleich: `llama-cli` mit `-np 4`** — wenn das crasht → Hypothese A bestätigt (CLI mit Multi-Slot reproduziert Server-Bug)

### Fix-Vorschläge

**Fix 1 (empfohlen, minimal):** auto_fa Check mit realistischerem `n_tokens`:
```cpp
const uint32_t n_tokens_check = std::max(cparams.n_ubatch, 32u);
auto * gf = graph_reserve(n_tokens_check, n_seqs, n_outputs, mctx.get(), true);
```

**Fix 2 (komplex):** Backend-Assignment für FA-Tensoren erzwingen — quantized KV muss auf demselben Device wie Layer-Gewichte liegen.

**Fix 3 (generisch):** Scheduler-Logik anpassen — Input-Tensoren die Teil einer FA-Op mit quantized KV sind, nicht auf CPU legen.

## Test-Ergebnis (2026-07-15)

**Test:** `llama-server -np 1 -ctk turbo4 -ctv turbo4 -fa on -c 4096` auf Styx.
**Ergebnis: ✅ Server startet fehlerfrei, kein SEGV.** Hypothese A bestätigt: Problem liegt bei `n_seqs > 1` (Server Default: 4).
**Workaround:** `-np 1` für turbo4, oder turbo3/turbo4 mixed für Multi-Slot.