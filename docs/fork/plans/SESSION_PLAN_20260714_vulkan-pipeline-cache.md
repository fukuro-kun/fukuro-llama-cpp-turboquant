# SESSION_PLAN: #78 Vulkan Pipeline Cache Disk Persistence

**Erstellt:** 2026-07-14 (Solo-Session)
**Status:** ✅ Implementiert und verifiziert
**ROADMAP-Item:** #78 (Tier 1, 1-2 Tage)
**Quelle:** Perinban/llama.cpp commit 1b7250c

## Session-Ziel

Implementiere Vulkan Pipeline Cache Disk Persistence: Speichere Pipeline-Cache-Binaries auf Disk zwischen Programmläufen, um teure Shader-Rekompilierung zu vermeiden. `GGML_VK_CACHE_DIR` Environment Variable steuert das Cache-Verzeichnis.

## Architektur

### Kern-Änderungen in `ggml-vulkan.cpp`

1. **`vk_device_struct` erweitern:** `vk::PipelineCache pipeline_cache` + `std::string pipeline_cache_path`
2. **Cache Loading (Device-Init):** Lese Cache-Datei, validiere `pipelineCacheUUID`, erstelle `VkPipelineCache` mit initial data
3. **Cache Saving (Device-Destruction):** `vkGetPipelineCacheData`, schreibe atomar auf Disk
4. **Pipeline Creation:** `createComputePipeline(device->pipeline_cache, ...)` statt `VK_NULL_HANDLE`

### UUID-Validierung

Vulkan Pipeline Caches sind driver-spezifisch. Der `pipelineCacheUUID` aus `VkPhysicalDeviceProperties` muss mit dem Header der Cache-Datei übereinstimmen. Bei Mismatch → Cache verwerfen, neu kompilieren.

### Atomares Schreiben

Cache wird in Temp-Datei geschrieben, dann `rename()` — verhindert korrupte Cache-Dateien bei Abbruch.

## Schritte

| # | Status | Aufgabe | Verifikation |
|---|--------|---------|--------------|
| 1 | ☐ | `vk_device_struct` Felder hinzufügen | Build grün |
| 2 | ☐ | Cache Loading in `ggml_vk_get_device` | Log-Output bei Cache-Hit/Miss |
| 3 | ☐ | `createComputePipeline` mit Cache | Pipeline-Erstellung funktioniert |
| 4 | ☐ | Cache Saving in Device-Destructor | Cache-Datei existiert nach Beenden |
| 5 | ☐ | UUID-Validierung | Stale Cache wird verworfen |
| 6 | ☐ | Build auf Hydra | Build grün |
| 7 | ☐ | Test auf Mars: 2x llama-bench, 2. Lauf schneller | Startup-Zeit reduziert |
| 8 | ☐ | Code-Review via review-swe | Keine P0/P1 Issues |
| 9 | ☐ | ROADMAP/CHANGELOG/TTT aktualisieren + commit + push | Dokumentation aktuell |

## Verifikations-Strategie

| Schritt | Metrik | Vorher | Nachher (erwartet) |
|---------|--------|--------|-------------------|
| Cache-Datei | Existiert nach Beenden | nein | ja (~1-10MB) |
| Startup-Zeit 2. Lauf | Zeit bis erste Inference | Xs | <Xs (Shader-Kompilierung entfällt) |
| UUID-Mismatch | Stale Cache erkannt | — | Cache verworfen, neu kompiliert |
| Pipeline-Erstellung | Alle Pipelines kompilieren | ja | ja (mit Cache-Beschleunigung) |

## Defaults

- `GGML_VK_CACHE_DIR` unset → kein Disk-Cache (in-memory only, wie bisher)
- Cache-Datei pro Device: `vk_pipeline_cache_{idx}.bin`
- Atomares Schreiben: Temp-Datei + `rename()`
- Bei korrupter Cache-Datei: verwerfen, neu kompilieren (kein Crash)
- UUID-Mismatch: verwerfen, neu kompiliern
