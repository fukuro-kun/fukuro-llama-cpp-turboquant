# Turbo4 llama-server: Graph-Scheduler/Backend-Assignment Problem
 
**Datum:** 2026-06-20
**Status:** 🔴 OFFEN (Root Cause identifiziert)
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
 
- [ ] `ggml-backend.cpp`: Wie werden Tensors Backend zugewiesen?
- [ ] `llama-graph.cpp`: Server-Graph vs. CLI-Graph Unterschiede
- [ ] Prufen, ob turbo4 FA-Tensor ein spezielles Alignment braucht
- [ ] Test: `llama-server` mit `--batch-size 1` (vereinfachter Graph)