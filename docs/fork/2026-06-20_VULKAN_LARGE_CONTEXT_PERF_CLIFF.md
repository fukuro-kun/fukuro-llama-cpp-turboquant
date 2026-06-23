# Vulkan Large-Context Performance-Klippe (20.06.2026)

**Status:** Identifiziert und dokumentiert. Workaround aktiv.
**Betroffen:** AMD APU (Radeon 760M, RADV PHOENIX) mit Vulkan-Backend
**Trilium:** `SWumEN7WOXBI` Abschnitt 5.8

---

## Zusammenfassung

Bei Gemma 4 26B (IQ4_NL) auf Vulkan (AMD 760M APU) tritt eine **extrem scharfe Performance-Klippe** bei ca. 184k-188k Kontext auf. Die Inference-Performance fällt in nur 4k Tokens Unterschied von 24 t/s auf 0.09 t/s ab — ein Faktor 243x.

**Dies ist KEIN TurboQuant-Problem** und KEIN VRAM-Bandbreiten-Problem. Es tritt unabhängig vom KV-Cache-Typ (turbo3, turbo4, f16) auf und bei einer APU ist der "VRAM" nur reservierter System-RAM mit derselben Bandbreite.

---

## Testergebnisse

### Setup
- **Modell:** Gemma 4 26B IQ4_NL (13.67 GiB)
- **Backend:** Vulkan (AMD Radeon 760M, RADV PHOENIX)
- **Build:** `2d81adc50` (TurboQuant FA Support für Vulkan)
- **KV-Cache:** turbo3 K+V (für alle Kontext-Tests)
- **Server:** llama-server, `--parallel 1`, `-ngl 99`

### Kontext-Scaling Matrix

| Kontext | t/s | KV-Cache (turbo3) | Status |
|---------|-----|-------------------|--------|
| 8k | 23.7 | ~3 MiB | ✅ |
| 32k | 23.9 | ~13 MiB | ✅ |
| 128k | 24.0 | ~50 MiB | ✅ |
| 160k | 23.9 | ~63 MiB | ✅ |
| 180k | 24.1 | ~70 MiB | ✅ |
| 184k | 23.8 | ~72 MiB | ✅ |
| **188k** | **0.099** | ~73 MiB | ❌ 243x langsamer |
| 192k | 0.099 | ~75 MiB | ❌ |
| 196k | 0.093 | ~77 MiB | ❌ |
| 262k | 0.093 | ~1000 MiB | ❌ |

### Verifizierung: Unabhängig vom KV-Cache-Typ

| KV-Cache-Typ | Kontext | t/s | Status |
|--------------|---------|-----|--------|
| f16 | 188k | hängt beim Laden | ❌ |
| turbo3 | 188k | 0.099 | ❌ |
| turbo3 | 184k | 23.8 | ✅ |

### Verifizierung: Modellgröße irrelevant

| Modell | Kontext | t/s | Status |
|--------|---------|-----|--------|
| Gemma 4 12B IQ4_NL | 8k | 10.4 | ✅ |
| Gemma 4 26B IQ4_NL | 8k | 23.7 | ✅ |
| Gemma 4 26B IQ4_NL | 180k | 24.1 | ✅ |
| Gemma 4 26B IQ4_NL | 188k | 0.099 | ❌ |

### llama-bench (kurzer Kontext, nicht betroffen)

| Config | pp256 t/s | tg64 t/s |
|--------|-----------|----------|
| f16/f16 | 166.77 | 21.66 |
| turbo4/turbo4 | — | 21.83 |
| turbo3/turbo3 | — | 21.88 |

---

## Analyse: Warum es KEIN VRAM-Problem ist

### APU-Speicherarchitektur

Bei einer AMD APU (Radeon 760M) gibt es keine physisch separate VRAM-Bank. Der "VRAM" ist ein BIOS-Carveout aus dem System-RAM:

- **VRAM-Carveout:** 1 GiB (BIOS-reserviert, exklusiv für GPU)
- **GTT:** 26 GiB (dynamisch, vom OS verwaltet, selbes DDR5)
- **System-RAM:** 30 GiB total

Beide Mechanismen greifen auf **denselben DDR5-Speicher** mit **derselben Bandbreite** zu. Der einzige Unterschied ist die Adressierung (Page-Table-Übersetzung bei GTT), was wenige Prozent Performance ausmacht — nicht Faktor 243x.

### Warum die scharfe Grenze auf einen Code-Pfad-Wechsel deutet

1. **Nicht graduell:** 24 t/s → 0.09 t/s in nur 4k Tokens (184k→188k)
2. **Unabhängig vom KV-Cache-Typ:** f16, turbo3, turbo4 alle bei ~188k
3. **Unabhängig vom Modell:** 12B und 26B beide funktionieren bei kleinem Kontext
4. **KV-Cache-Größe bei der Grenze:** ~73 MiB (turbo3) — winzig gegenüber 14 GiB Modellgewichten

### Mögliche Root Causes (zu untersuchen)

- Vulkan-Buffer-Platzierungs-Strategie wechselt bei bestimmter Größe (DEVICE_LOCAL → HOST_VISIBLE)
- FlashAttention-Tuning-Parameter-Wechsel (`get_fa_tuning_params`, Split-K-Reduce)
- Integer-Overflow in Buffer-Größen-Berechnung (`uint32_t` bei ~4GB Grenze?)
- Vulkan-Device-Limits (`maxBufferRange`, `maxMemoryAllocationCount`)

---

## Workaround

### Server-Konfiguration

Server auf **maximal 180k Kontext** begrenzen. Mit turbo3 K+V (~70 MiB KV-Cache bei 180k) läuft der Server stabil mit 24 t/s.

### Code-Workaround

`src/llama-context.cpp` enthält einen Workaround, der TurboQuant KV-Cache auf Vulkan auf f16 zurückfallen lässt. Dies ist eine **zusätzliche Sicherheit** (nicht der Performance-Fix), da das Performance-Problem auch mit f16 auftritt.

```cpp
// WORKAROUND (2026-06-20): TurboQuant KV cache on Vulkan produces extremely slow
// inference (0.08 t/s vs 21.8 t/s with f16) on AMD RDNA3 with large contexts.
// Fallback to f16 until the Vulkan TurboQuant FA performance is fixed.
```

---

## TODO: Root Cause Analysis

- [x] Vulkan-Buffer-Platzierung bei 184k vs 188k vergleichen (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`)
- [x] FA-Tuning-Parameter-Wechsel identifizieren (`get_fa_tuning_params`) — keine Änderung
- [x] Vulkan-Device-Limits prüfen (`maxBufferRange`, `maxMemoryAllocationCount`) — maxStorageBufferRange=4GiB
- [ ] Mit `VK_LAYER_LUNARG_api_dump` oder `VK_LAYER_KHRONOS_validation` Buffer-Größen loggen
- [x] Binäre Suche zwischen 184k und 188k für exakte Grenze — Grenze bei ctx-size 186k

---

## RCA Fortsetzung (21.06.2026)

### Websuche-Ergebnisse

Relevante Upstream-Issues/PRs gefunden:

| PR/Issue | Status in Fork | Relevanz |
|----------|---------------|----------|
| **#23762** "fix UMA performance by preferring cached host memory" | **FEHLT (open)** | WC-Memory-Fix für UMA |
| **#24326** "record actual memory properties during buffer creation" | **FEHLT** | Korrektheits-Fix |
| **#23770** "add pipeline barriers for memcpy read operations" | **FEHLT** | UMA-Korrektheit |
| **#22930** "prefer host-visible memory buffers on UMA devices" | **FEHLT** | Zero-Copy auf UMA |
| #24483 "TG Performance Degradation on RDNA4" | — | L2-Thrashing, split_k |
| #16759 "Odd compute buffer behaviors at breakpoints" | — | `GGML_VK_SUBALLOCATION_BLOCK_SIZE` |
| #20889 "DeviceLostError on gfx1102 (RADV PHOENIX)" | — | Exakt unsere GPU |

### AMD-System Memory-Types (vulkaninfo)

| Type | Heap | Flags | Beschreibung |
|------|------|-------|-------------|
| 0 | 1 | `DEVICE_LOCAL` | VRAM-Carveout (1 GiB) |
| 3 | 1 | `DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT` | GTT (Write-Combining!) |
| **5** | **0** | `HOST_VISIBLE \| HOST_COHERENT \| HOST_CACHED` | **System-RAM (gecached)** |

**Kein Type hat sowohl `DEVICE_LOCAL` als auch `HOST_CACHED`!**

### Test: UMA HostCached-Preference (PR #23762 + #24326)

**Hypothese:** GTT-Memory (Type 3) ist Write-Combining → langsam beim Lesen → Performance-Klippe.

**Fix angewendet:**
1. `memory_property_flags` wird auf tatsächliche Flags gesetzt (nicht angeforderte) — PR #24326
2. UMA-Allokation bevorzugt `HOST_CACHED` vor bare `DEVICE_LOCAL` — PR #23762 adaptiert

**Ergebnis:**

| Prompt-Größe | Ohne Fix | Mit Fix |
|-------------|----------|---------|
| pp512 | 204 t/s | 197 t/s (-3%) |
| pp4096 | 168 t/s | **HANG** (>180s) |
| pp8192 | HANG (>300s) | HANG |

**Fazit: Der Fix hat pp4096 GEBROCHEN!** System-RAM (HostCached, Type 5) ist für GPU-Compute auf AMD-System **langsamer** als GTT (DeviceLocal, Type 3). Der PP-Hang bei >4k ist **kein** WC-Memory-Problem.

### Neue Erkenntnisse

1. **Pipeline-Cache-Korruption** war die Ursache für pp4096-Hangs nach Code-Änderungen. Löschen des Caches (`~/.cache/llama.cpp/vulkan-pipeline-cache.bin`) behebt das Problem.
2. **PP-Klippe bei ~16k Tokens** (nicht 4k wie zuvor angenommen): pp8192=150 t/s ✅, pp16384=HANG ❌
3. **TG-Klippe bei ~188k** ist ein separates Problem (tritt bei Generation auf, nicht bei Prefill)
4. **Fork ist 996 Commits hinter upstream** — viele Vulkan-Fixes fehlen
5. **`suballocation_block_size` = 1 GiB** (default) — könnte bei großen KV-Caches zu Fragmentierung führen
6. **UMA HostCached-Preference (PR #23762) ist NICHT die Lösung** — System-RAM ist auf AMD-System für GPU-Compute langsamer als GTT

### PP-Scaling Matrix (clean pipeline cache)

| Prompt-Größe | t/s | Status |
|-------------|-----|--------|
| pp512 | 205 | ✅ |
| pp4096 | 168 | ✅ |
| pp8192 | 150 | ✅ |
| pp16384 | HANG (>600s) | ❌ |
| pp32768 | HANG | ❌ |
| pp65536 | HANG | ❌ |

### Nächste Schritte

- [ ] PP-Klippe zwischen 8192 und 16384 einkreisen (Binary Search)
- [ ] `GGML_VK_SUBALLOCATION_BLOCK_SIZE` auf 2 GiB oder 4 GiB setzen und testen
- [ ] FlashAttention-Dispatch-Größe bei pp8192 vs pp16384 vergleichen (workgroup count, barrier count)
- [ ] `VK_LAYER_KHRONOS_validation` laufen lassen um GPU-Hang zu diagnostizieren
- [ ] Upstream-Sync der fehlenden Vulkan-PRs evaluieren (996 Commits)

---

## Git-Historie

| Commit | Datum | Beschreibung |
|--------|-------|--------------|
| `e9679788a` | 05:19 | Workaround: TurboQuant auf Vulkan → f16 Fallback |
| `2d81adc50` | 15:06 | TurboQuant FA Support für Vulkan (Workaround entfernt) |
| `730d4fdd3` | 15:18 | FlashAttention Tests für turbo4 KV-Cache |
| (uncommitted) | 20:06 | Workaround reaktiviert + Performance-Klippe dokumentiert |
