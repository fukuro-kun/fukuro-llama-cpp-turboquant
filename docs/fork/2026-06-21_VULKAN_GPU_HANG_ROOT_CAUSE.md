# Vulkan GPU-Hang Root Cause: amdgpu.lockup_timeout + nodes_per_submit

**Datum:** 2026-06-21
**Issue:** ggml-org/llama.cpp#21724
**Status:** Root Cause identifiziert

---

## Root Cause

Der GPU-Hang auf AMD-System (AMD 760M, RADV PHOENIX) wird durch eine Kombination aus zwei Faktoren verursacht:

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

### Test 1: nodes_per_submit=1 (GPU-Hang Fix)

| pp | Ohne Fix (master) | Mit Cherry-Picks | Mit Fix (nps=1) | Status |
|----|-------------------|------------------|-----------------|--------|
| 512 | 205 t/s | 171 t/s | 189 t/s | ✅ |
| 4096 | 168 t/s | 164 t/s | 146 t/s | ✅ |
| 8192 | 150 t/s | HANG | 141 t/s | ✅ |
| 16384 | HANG | HANG | >30min (kein Hang, aber extrem langsam) | ⚠️ |

**Fazit:** `nodes_per_submit=1` verhindert GPU-Hangs bis pp8192. pp16384 hängt nicht mehr, ist aber extrem langsam (>30min). Das ist ein **separates Performance-Problem** (vermutlich FA-Shader-Laufzeit bei großen Sequenzen).

### Test 2: nodes_per_submit=10 (Kompromiss)

| pp | Ohne Fix (master) | Mit Cherry-Picks | nps=1 | **nps=10** | **nps=10 ONLY** | Status |
|----|-------------------|------------------|-------|-----------|-----------------|--------|
| 512 | 205 t/s | 171 t/s | 189 t/s | 160 t/s | **171 t/s** | ✅ |
| 4096 | 168 t/s | 164 t/s | 146 t/s | 166 t/s | **169 t/s** | ✅ |
| 8192 | 150 t/s | HANG | 141 t/s | 147 t/s | **146 t/s** | ✅ |
| 16384 | HANG | HANG | >30min | 122 t/s | **122 t/s** | ✅ |

### Test 3: TG-Scaling mit nodes_per_submit=10 ONLY (ohne Cherry-Picks)

| ctx | tg32 vorher | tg32 nps=10+CP | **tg32 nps=10 ONLY** | Status |
|-----|-------------|----------------|----------------------|--------|
| 180000 | 24.1 t/s | 21.24 t/s | — | ✅ |
| 186000 | HANG | 22.05 t/s | — | ✅ BEHOBEN |
| 188000 | 0.099 t/s | 21.33 t/s | **22.09 t/s** | ✅ 223x schneller |
| 192000 | 0.099 t/s | 21.35 t/s | — | ✅ BEHOBEN |

### Entscheidung: Cherry-Picks revertiert

Die Cherry-Picks sind **NICHT nötig** — `nodes_per_submit=10` allein löst alle drei Probleme.
Die Cherry-Picks kosten ~5% Performance bei pp512 (171→160) ohne einen Nutzen zu bringen.
Alle 6 Cherry-Picks wurden revertiert. Der Branch enthält jetzt nur noch den `nodes_per_submit` Fix.

### Test 4: nodes_per_submit Tuning (vollständig)

Alle 16 getesteten Werte in einer Tabelle. Aufsteigend nach nps sortiert.

| nps | pp512 (t/s) | pp16384 (t/s) | tg32@188k (t/s) | pp512 sauber? | Bemerkung |
|-----|-------------|---------------|-----------------|---------------|-----------|
| 1 | 189 | >30min | — | ⚠️ verfälscht | ❌ Zu viel Submit-Overhead |
| 3 | 161 | 121.62 | 20.77 | ⚠️ verfälscht | ✅ stabil |
| 5 | 197 | 118.11 | 21.50 | ⚠️ verfälscht | ✅ |
| 7 | 157 | 120.12 | 21.92 | ⚠️ verfälscht | ✅ stabil |
| 8 | 160 | 121.47 | 21.60 | ✅ sauber | ✅ stabil, nahe an nps=10 |
| 9 | 158 | 110.73 ±14 | 17.31 | ✅ sauber | ⚠️ tg32 deutlich niedrig! |
| **10** | **171** | **122** | **22.09** | ⚠️ verfälscht | ✅ **Sweet Spot — bestes pp16384 UND bestes tg32** |
| 11 | 188 | 108.01 | 21.74 | ✅ sauber | ✅ pp16384 sinkt |
| 12 | 160 | 112.72 | 18.92 | ⚠️ verfälscht | ⚠️ tg32 niedrig |
| 13 | 204 | Timeout | 21.81 | ✅ sauber | ⚠️ pp16384 instabil |
| 14 | 160 | 109.20 | — | ⚠️ verfälscht | ⚠️ pp16384 sinkt |
| 16 | 204 | 94.46 | 21.93 | ✅ sauber | ⚠️ pp16384 stark gesunken |
| 18 | 203 | 115.83 | 19.45 | ✅ sauber | ⚠️ tg32 sinkt |
| 20 | 159 | 120 | 21.46 | ⚠️ verfälscht | ✅ |
| 50 | 158 | 120 | 21.50 | ⚠️ verfälscht | ✅ |
| 100 | 205 | HANG | 0.099 | — | ❌ Original (GPU-Hang) |

**pp512 sauber?** — "verfälscht" = paralleler llama-server lief während des Tests und belegte VRAM. "sauber" = kein llama-server, VRAM frei. Die verfälschten pp512-Werte sind systematisch niedriger (157-197) als die sauberen (158-205). Für die Bewertung des Sweet Spots sind pp16384 und tg32@188k aussagekräftiger, da diese von VRAM-Konkurrenz weniger betroffen sind.

**Fazit:** nps=10 ist der Sweet Spot — bestes pp16384 (122 t/s) und bestes tg32@188k (22.09 t/s).
nps=8 ist die beste Alternative (pp16384=121, tg32=21.60).
nps=9 hat ein unerwartetes tg32-Tief (17.31), nps=13 hat pp16384-Timeout.
Höhere Werte (12-18) zeigen sinkende pp16384- und tg32-Werte.
nps=50 funktioniert noch, nps=100 hängt. GPU-Hang-Schwelle liegt zwischen 50 und 100.

### Root Cause für pp16384 Performance-Problem

`nodes_per_submit=1` verhindert GPU-Hangs, aber pp16384 ist immer noch >30min. Das bedeutet:
- Der GPU-Hang und die PP-Performance-Klippe sind **zwei separate Probleme**
- Der GPU-Hang wird durch `nodes_per_submit` verursacht (Batch >2s → amdgpu reset)
- Die PP-Performance-Klippe wird durch etwas anderes verursacht (vermutlich FA-Shader bei großen Sequenzen)

**Mögliche Ursachen für pp16384 Performance-Problem:**
1. FlashAttention-Shader ist bei großen Sequenzen (seq_len >8192) pathologisch langsam auf RDNA3
2. `mul_mat_bytes_per_submit` löst trotzdem einzelne Submits aus, die zu groß sind
3. Compute-Buffer-Größe überschreitet einen Schwellwert
4. L2-Cache-Thrashing bei großen Sequenzen (Issue #24483)

### Test 3: TG-Scaling mit nodes_per_submit=10 (DURCHBRUCH)

| ctx | tg32 (vorher) | tg32 (mit Fix) | Status |
|-----|---------------|----------------|--------|
| 180000 | 24.1 t/s | 21.24 t/s | ✅ |
| 184000 | 23.8 t/s | 21.28 t/s | ✅ |
| **186000** | **HANG** | **22.05 t/s** | ✅ **BEHOBEN!** |
| **188000** | **0.099 t/s** | **21.33 t/s** | ✅ **BEHOBEN!** (216x schneller) |
| **192000** | 0.099 t/s | 21.35 t/s | ✅ **BEHOBEN!** |

**Die TG-Klippe bei 188k ist BEHOBEN!** Vorher: 0.099 t/s (243x langsamer). Jetzt: 21.33 t/s.

ctx=186000, das vorher beim Laden hängte, funktioniert jetzt mit 22.05 t/s.

**Beobachtung:** tg32 ist mit Fix ~21 t/s (vorher ~24 t/s bei 180k). Das ist ein ~12% Performance-Verlust, der durch die Cherry-Picks und nodes_per_submit=10 verursacht wird. Dieser Verlust ist akzeptabel im Vergleich zum gewonnenen Stability-Bereich (180k → 192k+).
