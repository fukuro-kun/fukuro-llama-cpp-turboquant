# ggml/ — AGENTS.md (Child-DOX)

**Zweck:** GGML-Bibliothek — Tensor-Operationen, Backend-Abstraktion, Quantisierungsformate, Speicherverwaltung und GGUF-Dateiformat. Enthaelt die TurboQuant KV-Kompressions-Implementierung.

**Eigentuemer:** fukuro + KI-Agent

**Geltungsbereich:** Alle Pfade unter `ggml/`.

---

## Lokale Vertraege

### TurboQuant (Fork-spezifisch)

- Implementierung in `src/ggml-turbo-quant.c`
- Aenderungen nur in Absprache mit `src/llama-quant.cpp` und `src/llama.cpp` (KV-Cache-Integration)
- Neue Quantisierungstypen muessen in `ggml-quants.c`, `ggml-common.h` und `ggml.h` registriert werden

### Backend-Regeln

- Jedes Backend (CPU, CUDA, Vulkan, Metal, etc.) implementiert das Interface aus `ggml-backend.cpp`
- Neue Ops in `ggml.c` erfordern Fallback in `ggml-cpu/` und ggfs. Implementation in CUDA/Metal/Vulkan
- `ggml-backend-meta.cpp` und `ggml-backend-reg.cpp` enthalten Registrierungslogik — bei neuen Backends anpassen

### Sprachtrennung

- `src/ggml.c`, `src/ggml-quants.c`, `src/ggml-turbo-quant.c`: C
- `src/ggml-backend.cpp`, `src/gguf.cpp`, `src/ggml-opt.cpp`: C++
- `include/*.h` / `include/*.hpp`: Oeffentliche APIs — ABI-Stabilitat beachten

---

## Arbeitsanleitung

### Schluesseldateien

| Datei | Inhalt |
|-------|--------|
| `src/ggml.c` | Kern-Tensor-Operationen, Graphen-Ausfuehrung, Speicher-Allokation |
| `src/ggml-quants.c` | Quantisierungsformate (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q8_K, IQ-Formate) |
| `src/ggml-turbo-quant.c` | **TurboQuant KV/Weights-Kompression** (Fork-spezifisch) |
| `src/ggml-cuda/moe-cache.cu` | **MoE Expert Cache** (Fork-spezifisch) — async CPU→GPU fetch, 4 Eviction-Policies: LRU (Default), Heuristic, Set-Associative, Workload-Aware. Env: `GGML_CUDA_MOE_CACHE=1`, `GGML_CUDA_MOE_CACHE_POLICY=lru\|heuristic\|set-assoc-lru\|workload`. Siehe `docs/fork/ROADMAP.md` #18/#69/#71 für Benchmark-Ergebnisse. |
| `src/ggml-backend.cpp` | Backend-Abstraktion, Scheduling, Buffer-Management |
| `src/ggml-backend-meta.cpp` | Backend-Metadaten und Aufzaehlung |
| `src/ggml-backend-reg.cpp` | Backend-Registrierung (Dynamisches Laden) |
| `src/gguf.cpp` | GGUF-Dateiformat: Lesen, Schreiben, Metadaten |
| `src/ggml-alloc.c` | Graph-Allocator, temporäre Buffer |
| `src/ggml-opt.cpp` | Optimierer fuer Training/Finetuning |
| `include/ggml.h` | Haupt-C-API |
| `include/ggml-backend.h` | Backend-API |
| `include/gguf.h` | GGUF-C-API |

### Backends (Unterverzeichnisse in `src/`)

| Backend | Verzeichnis | Beschreibung |
|---------|-------------|--------------|
| CPU | `ggml-cpu/` | Referenz-Implementation, AVX/AVX2/NEON/AMX |
| CUDA | `ggml-cuda/` | NVIDIA CUDA (GGML_CUDA) |
| Vulkan | `ggml-vulkan/` | Vulkan Compute (GGML_VULKAN) |
| Metal | `ggml-metal/` | Apple Metal (GGML_METAL) |
| HIP | `ggml-hip/` | AMD ROCm (GGML_HIP) |
| SYCL | `ggml-sycl/` | Intel SYCL/oneAPI (GGML_SYCL) |
| BLAS | `ggml-blas/` | OpenBLAS/Accelerate/MKL (GGML_BLAS) |
| OpenCL | `ggml-opencl/` | OpenCL (GGML_OPENCL) |
| CANN | `ggml-cann/` | Huawei CANN (GGML_CANN) |
| RPC | `ggml-rpc/` | Netzwerk-RPC fuer verteilte Inferenz |
| WebGPU | `ggml-webgpu/` | WebGPU (GGML_WEBGPU) |
| MUSA | `ggml-musa/` | Moore Threads MUSA |
| OpenVINO | `ggml-openvino/` | Intel OpenVINO |
| Hexagon | `ggml-hexagon/` | Qualcomm Hexagon |
| ZenDNN | `ggml-zendnn/` | AMD ZenDNN |
| zDNN | `ggml-zdnn/` | IBM zDNN |
| VirtGPU | `ggml-virtgpu/` | Virtuelles GPU-Backend |

### Workflow bei Aenderungen

1. **Tensor-Ops:** `ggml.c` aendern → `ggml-impl.h` pruefen → CPU-Fallback in `ggml-cpu/` ergaenzen
2. **Quantisierung:** `ggml-quants.c` aendern → `ggml-common.h` (Shuffles/Masks) und `ggml.h` (Enum) synchronisieren
3. **TurboQuant:** `ggml-turbo-quant.c` aendern → `src/llama-quant.cpp` und KV-Cache-Code pruefen
4. **Backends:** Backend-spezifische Op-Implementation anpassen, `ggml-backend.cpp` nur bei Interface-Aenderungen
5. **GGUF:** `gguf.cpp` aendern → `gguf-py/` bei Format-Aenderungen mitaktualisieren

---

## Verifikation

- [ ] Build erfolgreich mit mindestens einem Backend (`cmake --build build -j$(nproc)`)
- [ ] `./build/bin/test-backend-buffer` laeuft durch (Backend-Grundfunktion)
- [ ] `./build/bin/test-quantize` laeuft durch (Quantisierungs-Rundtrip)
- [ ] Bei TurboQuant-Aenderungen: `./build/bin/test-tq` oder Inferenz mit TurboQuant-Modell erfolgreich
- [ ] Keine neuen Warnungen in ggml-Quellen

---

## Child-DOX-Index

| Pfad | Zweck | Status |
|------|-------|--------|
| `src/ggml-cpu/` | CPU-Referenz-Backend, SIMD-Optimierungen | [~] Kein eigenes DOX (zu klein) |
| `src/ggml-cuda/` | NVIDIA CUDA-Backend | [~] Kein eigenes DOX |
| `src/ggml-vulkan/` | Vulkan Compute-Backend | [~] Kein eigenes DOX |
| `src/ggml-metal/` | Apple Metal-Backend | [~] Kein eigenes DOX |
| `include/` | Oeffentliche C/C++-Header | [~] Kein eigenes DOX |

*Hinweis: Backend-Unterverzeichnisse sind derzeit ohne eigenes Child-DOX; bei komplexen Aenderungen einzelne Backend-AGENTS.md erstellen.*
