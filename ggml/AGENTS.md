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
- Typnummern sind GGUF-/RPC-/KV-State-Vertraege: `Q2_0=42`, `TURBO2_0=47`; TURBO3/4 und TQ3_1S/TQ4_1S behalten 43/44/45/46. C-/Python-Enum und Backend-Konstanten synchron halten; nach Aenderungen Bibliotheken/RPC-Teilnehmer gemeinsam neu bauen. Alte TURBO2-KV-Sessions sind nicht kompatibel; Typ 42 nicht heuristisch als TURBO2 interpretieren.

### Backend-Regeln

- Jedes Backend (CPU, CUDA, Vulkan, Metal, etc.) implementiert das Interface aus `ggml-backend.cpp`
- Neue Ops in `ggml.c` erfordern Fallback in `ggml-cpu/` und ggfs. Implementation in CUDA/Metal/Vulkan
- `ggml-backend-meta.cpp` und `ggml-backend-reg.cpp` enthalten Registrierungslogik — bei neuen Backends anpassen

### CUDA-Expert-Prefetch (Fork-spezifisch)

- Prefetch ist **opt-in**: ohne gesetzte `GGML_SCHED_PREFETCH_EXPERTS` bleibt er aus. `GGML_SCHED_PREFETCH_EXPERTS=1` aktiviert den Pfad mit Default 3 Slots; `>=2` setzt die Slot-Zahl direkt; `GGML_SCHED_PREFETCH_SLOTS=N` ueberschreibt die Slot-Zahl unabhaengig. Er prefetcht **volle Expert-Tensoren** ueber einen zweiten Backend-/Event-Pfad — das ist nicht aequivalent zu Strata-artigem Double-Buffer-Streaming.
- `GGML_SCHED_PREFETCH_ALL=1` (nur wirksam zusammen mit `PREFETCH_EXPERTS`) staget zusaetzlich eligible Expert-Tensoren **spaeterer Splits** voraus (Cross-Split-Lookahead): bei ~1 eligiblem Input pro Split reicht Lookahead-Tiefe 1 nicht, um ~440-MiB-Kopien hinter Compute zu verstecken — die Slots halten den Copy-Stream dauerhaft beschaeftigt. Eligible Inputs ohne freien Slot bleiben im selektiven Per-Expert-Pfad (gemessen bei ub8192 besser als slotlose Volluploads). Pro Device ein eigener Slot-Ring; Slot-Groesse = groesster eligible Expert-Tensor; ~3,5 GB/Device bei 8 Slots.
- Prefetch-State ist **pro Split-Backend** (Slot-Ring, Ready-/Free-Events, eigene Copy-Instanz pro Device) — ein CUDA1-Input landet nie in einem CUDA0-Slot.
- `GGML_SCHED_PREFETCH_ALL=1` (opt-in): staged **jeden** eligiblen Expert-Input eines Splits (Default: nur den ersten). Slot-Sicherheitsregel: ein Slot darf nur wiederbefuellt werden, wenn sein letzter Nutzer ein frueherer Split war (`holder < split_id`) — sonst kreisfaelliges Event-Warten. Eligible Inputs ohne freien Slot nehmen den generischen Async-Vollupload statt des selektiven Pfads — dadurch entfaellt der blockierende IDs-Readback (`ggml_backend_synchronize` pro Layer) bei grossen Batches komplett. Kleine Batches bleiben beim selektiven Pfad (`n_tokens*n_used < 2*n_expert`).
- Die Standalone-Prefetch-Slots tragen seit dem P1b-Fix ein `+512 B` Padding mit genulltem Tail — MMQ liest wenige Bytes ueber das Tensorende hinaus; das Padding ist fuer MMQ erforderlich, nicht optional.
- Siehe Plan `docs/fork/plans/SESSION_PLAN_20261006_strata-optllama-uranus-eval.md` §10.3.

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
| `src/ggml-cuda/moe-cache.cu` | **MoE Expert Cache** (Fork-spezifisch) — async CPU→GPU fetch, 2 Eviction-Policies: `lru` (Default) und `heuristic` (Aliases `0`/`1`). Env: `GGML_CUDA_MOE_CACHE` ist per Default aktiv (nur `<= 0` deaktiviert), `GGML_CUDA_MOE_CACHE_POLICY=lru\|heuristic`. Die Policies `set-assoc`/`workload` wurden im Juli-Cleanup entfernt. Siehe `docs/fork/ROADMAP.md` #18/#69/#71 für Benchmark-Ergebnisse. |
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
- [ ] `./build/bin/test-backend-ops` laeuft durch (Backend-Grundfunktion; auf dem Dev-Host nur `-b CPU`)
- [ ] `./build/bin/test-quantize-fns` laeuft durch (Quantisierungs-Rundtrip)
- [ ] Bei TurboQuant-Aenderungen: `./build/bin/test-turbo-quant` oder Inferenz mit TurboQuant-Modell erfolgreich
- [ ] Bei KV-State-Aenderungen: `test-save-load-state` / `test-state-restore-fragmented` / `test-recurrent-state-rollback` (LABEL `model`, benoetigen Modell-Fixture)
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
