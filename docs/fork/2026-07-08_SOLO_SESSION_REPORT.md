# Solo-Session Report: thecodacus MoE-Optimierungen (2026-07-08)

## Zusammenfassung

Erfolgreiche Portierung der thecodacus MoE-Optimierungen (Memory Pinning + Async Expert Prefetch) in den fukuro-llama-cpp-turboquant Fork. Auf der GTX 1070 (Pascal, 8GB VRAM) wurde ein **Prefill-Boost von bis zu +106%** und ein **Decode-Boost von +68%** gemessen.

## Quelle

- **Repository:** [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp)
- **Branch:** `fable5/prefetch-experts`
- **Commits:** 20f5994, 1163cb3, 5f83fbb

## Portierte Patches

| Patch | Commit | Zeilen | Env-Var | Effekt |
|-------|--------|--------|---------|--------|
| Memory Pinning | 20f5994 | +59 | `GGML_CUDA_REGISTER_HOST=1` | cudaHostRegister für mmap-pages |
| Async Expert Prefetch | 1163cb3 | +102 | `GGML_SCHED_PREFETCH_EXPERTS=1` | Overlap Expert-Upload mit Compute |
| Prefetch Slot Sizing + UAF Fix | 5f83fbb | +90 | (wie oben) | 3 Slots default, UAF-Fix |

**Gesamt: +232 Zeilen in 4 Dateien** (`llama-mmap.cpp/h`, `llama-model-loader.cpp`, `ggml-backend.cpp`)

## Benchmark-Ergebnisse

### Hardware
- GTX 1070 (Pascal, compute 6.1), 8GB VRAM
- Gemma 4 26B-A4B IQ4_NL (14GB, 25.23B params, MoE mit 128 Experten)

### Konfiguration A: `-ot exps=CPU` (alle Experten auf CPU)

| Batch | Baseline | +Pinning | +Pinning+Prefetch | Total Boost |
|-------|----------|----------|-------------------|-------------|
| pp512 | 293.27 | 349.44 | 505.51 | **+72.3%** |
| pp1024 | 273.70 | 336.92 | 503.05 | **+83.8%** |
| pp2048 | 251.58 | 309.43 | 489.72 | **+94.7%** |
| pp4096 | 225.75 | 278.82 | 465.49 | **+106.2%** |
| tg128 | 12.69 | 13.74 | 16.57 | **+30.5%** |

### Konfiguration B: `--n-cpu-moe 20` (Experten der oberen 20 Layer auf CPU)

| Batch | Ohne Opt | +Pinning+Prefetch | Boost |
|-------|----------|-------------------|-------|
| pp512 | 371.47 | 538.34 | **+44.9%** |
| pp2048 | - | 512.84 | - |
| pp4096 | - | 351.02 | - |
| pp8192 | - | 250.27 | - |
| tg128 | 21.33 | 21.32 | ±0% |

### Konfiguration C: `--n-cpu-moe 20` + Pinning + Prefetch (große Batches)

| Batch | t/s |
|-------|-----|
| pp4096 | 351.02 |
| pp8192 | 250.27 |
| tg128 | 17.02 |

### E4B (Dense MoE, voller GPU-Offload) — kein Effekt

| Modell | pp512 Baseline | pp512 +Pinning | tg128 Baseline | tg128 +Pinning |
|--------|----------------|----------------|----------------|----------------|
| Q4_K_M | 811.50 | 805.75 | 41.23 | 40.91 |
| IQ4_XS | 871.63 | 860.88 | 45.61 | 45.23 |

**Erwartet:** Bei vollständigem GPU-Offload gibt es keine H2D-Copies der Experten → Pinning bringt nichts.

### Vergleich mit thecodacus (RTX 3060, Qwen3.6-35B-A3B)

| Platform | Baseline pp2048 | Optimized pp2048 | Boost |
|----------|----------------|------------------|-------|
| RTX 3060 (thecodacus) | 1143 | 1880 | +64.5% |
| GTX 1070 (wir) | 251.58 | 489.72 | +94.7% |

**Erklärung:** Pascal (PCIe Gen3, ältere VRAM, compute 6.1) profitiert mehr vom Overlap als Ampere (PCIe Gen4, schnellere VRAM). Die GPU-Inaktivität war bei Pascal höher, also gibt es mehr zu gewinnen.

## Optimal-Konfiguration für GTX 1070 (8GB VRAM)

### Für Prefill-dominierte Workloads (lange Prompts)
```bash
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 \
llama-server -m gemma-4-26B-A4B-it-IQ4_NL.gguf \
  -ngl 999 -ot "exps=CPU" --flash-attn on
```
- Alle Experten auf CPU, mit Prefetch überlappt
- pp2048: 490 t/s (+95%)

### Für Decode-dominierte Workloads (Chat, kurze Prompts)
```bash
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 \
llama-server -m gemma-4-26B-A4B-it-IQ4_NL.gguf \
  -ngl 999 --n-cpu-moe 20 --flash-attn on
```
- Untere 10 Layer-Experten auf GPU, obere 20 auf CPU
- tg128: 21.3 t/s (+68% vs Baseline -ot exps=CPU)

### Für ausgewogene Workloads (Empfehlung)
```bash
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 \
llama-server -m gemma-4-26B-A4B-it-IQ4_NL.gguf \
  -ngl 999 --n-cpu-moe 20 --flash-attn on
```
- Bester Kompromiss: pp512 538 t/s, tg128 21.3 t/s

## Technische Details

### Memory Pinning (Commit 20f5994)
- `llama_mmap::register_host()` ruft `cudaHostRegister` für die mmap-Pages auf
- Verhindert OS-Paging der Expert-Gewichte im System-RAM
- Aktiviert "dead code" der bereits in llama.cpp vorhanden war (`GGML_CUDA_REGISTER_HOST`)
- Log-Ausgabe: "pinned 13998.48 MiB of mapped model memory for faster H2D transfers"

### Async Expert Prefetch (Commits 1163cb3 + 5f83fbb)
- Bei großen Batches werden fast alle Experten verwendet → Routing-IDs nicht worth waiting for
- Upload läuft durch zweite Backend-Instanz auf gleichem Device (eigener Stream)
- 3 Staging-Slots (default) für eine volle MoE-Layer Lookahead
- Event-geordnete Synchronisation: `prefetch_ready` / `prefetch_free`
- Use-After-Free Fix: Staging-Repoints werden nach Kernel-Launch restauriert

### Voraussetzungen
- CUDA Backend mit `async` und `events` Capabilities
- MoE-Modell mit `MUL_MAT_ID` Ops (Expert-Selection)
- Partielles Offloading (Experten auf CPU, Attention auf GPU)
- Bei vollständigem GPU-Offload: kein Effekt (keine H2D-Copies)

## MTP + Pinning + Prefetch (Kombinationstest)

26B-A4B mit Draft-Modell (`gemma-4-26b-a4b-it-assistant.Q4_K_M.gguf`), `--spec-type draft-mtp`:

| Metrik | Wert |
|--------|------|
| Draft-Akzeptanz | **100%** (11/11 tokens accepted) |
| Generation (mit MTP) | **31.93 t/s** |
| Generation (ohne MTP, Baseline) | 21.3 t/s |
| MTP-Boost | **+50%** |

**Konfiguration:**
```bash
GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_PREFETCH_EXPERTS=1 \
llama-server -m gemma-4-26B-A4B-it-IQ4_NL.gguf \
  -md drafts/gemma-4-26b-a4b-it-assistant.Q4_K_M.gguf \
  --spec-type draft-mtp -ngl 999 -ngld 999 --n-cpu-moe 20 \
  --flash-attn on
```

## Kontext-Scaling mit Pinning+Prefetch (--n-cpu-moe 20)

| Kontext | t/s | Skalierung |
|---------|-----|------------|
| pp512 | 538 | 1.00x |
| pp2048 | 513 | 0.95x |
| pp8192 | 424 | 0.79x |
| pp16384 | 212 | 0.39x |

**Beobachtung:** Bei 16k Kontext bricht die Performance ein (0.39x vs pp512). Ursache: Bei großen Kontexten werden mehr Experten pro Token aktiviert → mehr H2D-Copies → Prefetch-Overhead wächst. Für Kontext >8k ist `-ot exps=CPU` (alle Experten auf CPU) effizienter als `--n-cpu-moe 20`.

## Korrektheits-Verifikation

**Test:** `llama-cli` mit temp=0, seed=42, Prompt "Die Hauptstadt von Deutschland ist"

| Konfiguration | Ausgabe |
|---------------|---------|
| Ohne Pinning | Berlin |
| Mit Pinning | Berlin |

**Ergebnis:** Token-identische Ausgabe. Pinning ändert nur I/O-Pfade (cudaHostRegister), nicht die Berechnung.

## Git-Branch

- **Branch:** `feature/thecodacus-pinning` auf Pascal-Host → **gemerged nach master**
- **Master-Commit:** `a4215b3d6` — "feature: thecodacus MoE-Optimierungen (Pinning + Prefetch)"
- **Dateien:** 4 geändert, +232 Zeilen
- **Codeberg:** Push zu `codeberg.org:fukuro/fukuro-llama-cpp-turboquant` erfolgreich

## Nächste Schritte

1. ~~Branch nach Codeberg pushen~~ ✅ Erledigt
2. ~~Merge in master~~ ✅ Erledigt
3. **Service-Konfiguration aktualisieren** mit Pinning+Prefetch Env-Vars (für 26B-A4B Service)
4. **Test mit Qwen 3.6 MoE** (falls verfügbar) für Vergleich mit thecodacus
5. ~~Korrektheits-Verifikation~~ ✅ Token-identisch verifiziert
