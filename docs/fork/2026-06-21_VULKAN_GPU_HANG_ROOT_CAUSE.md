# Vulkan GPU-Hang Root Cause: amdgpu.lockup_timeout + nodes_per_submit

**Datum:** 2026-06-21
**Issue:** ggml-org/llama.cpp#21724
**Status:** Root Cause identifiziert

---

## Root Cause

Der GPU-Hang auf Mars (AMD 760M, RADV PHOENIX) wird durch eine Kombination aus zwei Faktoren verursacht:

1. **`amdgpu.lockup_timeout` = 2000ms (Default)**
   Der Linux amdgpu-Kernel-Treiber hat einen Watchdog, der GPU-Jobs nach 2000ms abbricht, wenn sie nicht fertig werden.

2. **`nodes_per_submit = 100` in ggml-vulkan.cpp**
   Der Vulkan-Backend batched bis zu 100 Graph-Nodes pro `vkQueueSubmit`. Auf einer langsamen APU wie der 760M dauert ein solcher Batch bei großen Workloads (pp≥16384, ctx≥186k) länger als 2 Sekunden.

**Folge:** Der Kernel-Treiber denkt die GPU ist abgestürzt → Ring Reset → `vk::DeviceLostError` → GPU-Hang → Prozess in D-State → System unresponsive.

## Symptome erklärt

| Symptom | Erklärung |
|---------|-----------|
| pp512-8192 funktioniert | Batches <2s |
| pp16384+ hängt | Batches >2s → GPU reset |
| TG bei 188k hängt | Großer Kontext → lange Batches → GPU reset |
| KV-Cache-Allokation bei 186k hängt | Große Buffer-Initialisierung → lange Batches |
| Nach Reboot funktioniert alles kurz | Pipeline-Cache leer → Shader neu kompiliert → andere Batch-Größen |
| Pipeline-Cache-Korruption | GPU reset während Shader-Kompilierung → korrupter Cache |

## Fixes

### Fix 1: amdgpu.lockup_timeout erhöhen (Workaround, sofort wirksam)
```bash
sudo sh -c 'echo 60000 > /sys/module/amdgpu/parameters/lockup_timeout'
```
Setzt den Timeout von 2s auf 60s. Nicht persistent (nach Reboot weg).

### Fix 2: nodes_per_submit reduzieren (Code-Fix)
In `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:
```cpp
// Vorher:
int nodes_per_submit = 100;

// Nachher:
int nodes_per_submit = 1;  // oder kleiner Wert für APUs
```

### Fix 3: Auto-Detect für APU (optimale Lösung)
```cpp
int nodes_per_submit = device->uma ? 1 : 100;
```

## Validierung

1. `lockup_timeout=60000` setzen
2. pp16384 testen (vorher HANG, sollte jetzt funktionieren)
3. TG bei 188k testen (vorher 0.09 t/s, sollte jetzt funktionieren)
4. KV-Cache-Allokation bei 186k testen (vorher HANG, sollte jetzt funktionieren)
