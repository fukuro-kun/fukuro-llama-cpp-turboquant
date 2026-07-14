# SESSION_PLAN: #61 Persistent VRAM Expert Cache (PR #24524 Design)

**Erstellt:** 2026-07-14
**Status:** Plan fertig, Implementierung ausstehend
**ROADMAP-Item:** #61 Persistent VRAM Expert Cache (Tier 2, 1-2 Wochen)
**Vorgänger:** #37 LFRU Expert Caching (postponed — falscher Architektur-Ansatz), #40 MoE Frequency Tracking (Phase 1 ✅)

## Session-Ziel

Implementiere einen persistenten VRAM-Cache für MoE-Experten nach dem **PR #24524 Design** (invertiertes Execution-Model). Im Gegensatz zu allen prior attempts (inkl. unserem #37) wird MUL_MAT_ID **nicht** auf die GPU verschoben. Stattdessen bleibt die Operation auf der CPU, aber Thread 0 im CPU-Kernel dispatcht die Cache-Hit-Rows als batched matvec an die GPU, während die übrigen CPU-Threads die Miss-Rows parallel berechnen. Misses kosten nichts extra — Worst-Case = Vanilla-CPU-Pfad.

## Warum nicht #37 fixen?

#37 (LFRU Expert Caching) hatte den gleichen Architektur-Fehler wie alle closed PRs (#21609, #21614, #21620, #23170): MUL_MAT_ID wird auf GPU verschoben, Expert-Weights werden bei Cache-Miss synchron über PCIe kopiert. Das führt zu ~3× Decode-Regression bei Cache-Misses (gemessen von batot1 auf #20757). Selbst bei 97-99% Hit-Rate war der Metal-Slot-Pool 2× langsamer wegen per-layer Sync-Points.

**PR #24524 vermeidet das strukturell:** MUL_MAT_ID bleibt auf CPU. GPU berechnet nur Hit-Rows asynchron. Kein synchroner PCIe-Transfer auf dem Critical-Path. Ergebnisse: +7-25% auf großen Modellen, 16/16 positiv oder parity.

## Architektur: PR #24524 Inverted Execution Model

```
                    ┌─────────────────────────────────────────┐
                    │         CPU mul_mat_id Kernel           │
                    │                                         │
                    │  Thread 0:                              │
                    │  1. Lese routing IDs                    │
                    │  2. Partitioniere in hits/misses        │
                    │  3. Dispatch hit-rows → GPU (batched)   │
                    │  4. Warte auf GPU-Completion             │
                    │  5. Sammle Ergebnisse in dst            │
                    │                                         │
                    │  Threads 1..N:                          │
                    │  - Berechne miss-rows (wie bisher)      │
                    │  - Ergebnisse in dst                    │
                    └─────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  GPU Cache Pool   │
                    │  (spare VRAM)     │
                    │  - Hot experts    │
                    │  - Paired gate+up │
                    │  - Fused SwiGLU   │
                    └───────────────────┘
```

### Komponenten (3-Commit-Struktur wie PR #24524)

**Commit 1: ggml — MoE Expert Cache API + CPU/Scheduler Integration**
- `ggml_moe_cache` function table (backend-agnostic, zero-initialized)
- CPU `mul_mat_id`: Thread 0 plant hits/misses, dispatcht hit-rows als batched GPU-launch
- CPU `swiglu`: überspringt dst-rows die der Cache fused auf GPU berechnet hat
- Scheduler: `redirect_offer` / `redirect_finalize` / `invalidate` hooks
  - `redirect_offer`: Bietet GPU-side copy tensor eines CPU MUL_MAT_ID dst dem Cache an
  - `redirect_finalize`: Cache populiert es direkt, überspringt host round-trip
  - `invalidate`: Host weight-buffer teardown benachrichtigt Cache (async fills lesen nie freed memory)

**Commit 2: cuda — MoE Expert Cache Implementierung**
- Cache hot CPU-resident MoE expert weights in spare VRAM
- Adaptive fill: decode-only (prompt routing thrasht den Cache)
- Stable shape census bevor VRAM ausgegeben wird
- Paired gate+up pools mit fused gate+up+SwiGLU dispatch
- GPU-resident handoff von down-projection results
- VRAM surrender bei allocator OOM
- Baseline-sampled bail-out (sustained slower-than-CPU → trim + disable)

**Commit 3: CLI + Integration**
- `--moe-cache 0` (disable) / `--moe-cache N` (cap budget N MiB)
- Default: on (zero-config)
- Integration mit #40 Frequency Tracking für hot-set decisions
- Non-CUDA builds: zero-initialized no-op

### Safety Rails (aus PR #24524 übernommen)

| Rail | Zweck |
|------|-------|
| Decode-only fill | Prompt routing thrasht den Cache → nur bei decode füllen |
| Stable shape census | Keine VRAM ausgeben bevor Tensor-Shapes stabil sind |
| Paired gate+up pools | gate+up müssen zusammen im Cache sein (fused SwiGLU) |
| GPU-resident handoff | Down-projection results direkt an GPU consumer splits |
| On-disk hot-set persistence | Hot-set über Runs hinweg persistieren |
| Baseline-sampled bail-out | sustained slower-than-CPU → trim + disable for run |
| VRAM surrender | OOM → Cache gibt VRAM zurück |
| Multi-GPU sharding | Cache pro-GPU, nicht global |

## Entscheidungen

### Architektur-Entscheidungen (aus vorheriger Session)

| Frage | Entscheidung | Begründung |
|-------|-------------|------------|
| Architektur? | PR #24524 inverted model (MUL_MAT_ID auf CPU) | Alle prior attempts mit MUL_MAT_ID auf GPU failed (~3× regression bei misses) |
| Cherry-pick oder manuell portieren? | Manuell portieren | PR ist closed, 2222 Zeilen, 11 Dateien, Fork-Divergenz zu groß |
| Cache-Size Konfiguration? | `--moe-cache N` (N MiB, 0=disable, default=auto) | Konsistent mit PR #24524 zero-config Ansatz |
| Frequency-Quelle? | #40 Expert Frequency Tracking + runtime counter | #40 Infrastruktur bereits vorhanden, kann hot-set seeden |
| Vulkan-Support? | Nein, erst CUDA-only | PR #24524 ist CUDA-only. Vulkan benötigt separaten Backend-Hook (später) |
| thecodacus Prefetch Koexistenz? | Ja, komplementär | Prefetch = cold transfers bei large ubatch (prompt), Cache = hot re-transfers bei ubatch 1 (decode) |
| CUDA Graphs Kompatibilität? | Nein (wie vLLM enforce-eager) | Dynamische Cache-Slots inkompatibel mit Graph-Capture |

### Solo-Session Interview-Protokoll (2026-07-14, Phase 0)

| Frage | Entscheidung | Begründung |
|-------|-------------|------------|
| Session-Scope? | So weit wie möglich (alle 4 Phasen) | Solo-Session Autonomie, stoppe nur bei Blockern |
| PR #24524 Code fetchen? | Ja, 1 Subagent für PR-Commits als Referenz | Echter Code als Referenz für thread-sync + CUDA cache pool Logic |
| Styx-Server stoppen? | Jederzeit | Solo-Session autorisiert, Server danach wiederherstellen |
| LFRU Expert Cache entfernen? | Ja, in Phase 3 | Toter Code (0% hit-rate, Buffer-Recycling-Crash). **WICHTIG:** Nur `expert_valid`/`tensor_copied`/`GGML_EXPERT_CACHE` entfernen. **thecodacus Prefetch (GGML_SCHED_PREFETCH_SLOTS=2, +28.9% pp) bleibt unangetastet!** |

## Schritte

### Phase 1: API + CPU Integration (Commit 1, 3-4 Tage)

- [ ] 1. `ggml_moe_cache` function table definieren (`ggml/include/ggml.h` + `ggml/src/ggml.c`)
  - Struct: `ggml_moe_cache { init, offer, finalize, invalidate, query, dispatch }`
  - Zero-initialized global, populated by GPU backend at registry time
- [ ] 2. CPU `mul_mat_id` Kernel modifizieren (`ggml/src/ggml-cpu/`)
  - Thread 0: partition hits/misses gegen cache
  - Dispatch hit-rows als batched GPU launch
  - Andere Threads: miss-rows wie bisher
  - Ergebnisse in dst sammeln
- [ ] 3. CPU `swiglu` modifizieren
  - `glu_hits` mask: überspringe rows die Cache fused berechnet hat
- [ ] 4. Scheduler hooks (`ggml/src/ggml-backend.cpp`)
  - `redirect_offer`: GPU-side copy tensor eines CPU MUL_MAT_ID dst → Cache
  - `redirect_finalize`: Cache populiert direkt, kein host round-trip
  - `invalidate`: weight-buffer teardown → Cache invalidiert async fills
- [ ] 5. Build-Check auf Hydra (CPU-only, `-DGGML_CUDA=OFF`)

### Phase 2: CUDA Implementation (Commit 2, 4-5 Tage)

- [ ] 6. `ggml_moe_cache` CUDA registration (`ggml/src/ggml-cuda/ggml-cuda.cu`)
  - Cache pool allocation in spare VRAM
  - Expert weight storage (paired gate+up)
  - Fused gate+up+SwiGLU dispatch kernel
- [ ] 7. Adaptive fill logic
  - Decode-only fill (detect via n_tokens == 1)
  - Stable shape census (N forward passes mit gleichen shapes bevor VRAM)
  - Frequency-guided hot-set selection (nutzt #40 Daten wenn verfügbar)
- [ ] 8. Safety rails
  - VRAM surrender bei OOM
  - Baseline-sampled bail-out
  - On-disk hot-set persistence (optional, später)
- [ ] 9. Build-Check auf Hydra (CUDA, `-DGGML_CUDA=ON`)

### Phase 3: CLI + Integration (Commit 3, 2-3 Tage)

- [ ] 10. `--moe-cache` CLI option (`common/arg.cpp`)
  - `--moe-cache 0` = disable
  - `--moe-cache N` = cap budget N MiB
  - Default: auto (zero-config)
- [ ] 11. #40 Frequency Tracking integration
  - Hot-set seeding aus frequency data
  - Cache fill priority basierend auf frequency
- [ ] 12. #37 Code entfernen (nach erfolgreichem Test)
  - `expert_valid` map, `tensor_copied` set aus `ggml-backend.cpp`
  - `GGML_EXPERT_CACHE` env var entfernen
- [ ] 13. Smoke-Test auf Hydra (CPU-only, kein Crash)

### Phase 4: Test + Benchmark auf Styx (2-3 Tage)

- [ ] 14. Produktiv-Server auf Styx stoppen
- [ ] 15. Build auf Styx (CUDA, Pascal)
- [ ] 16. Benchmark Vorher (baseline, no cache)
  - `llama-bench -p 512 -n 128 -ngl 99 -ncmoe 20`
  - `llama-bench -p 512 -n 8 -ngl 99 -ncmoe 20`
- [ ] 17. Benchmark Nachher (cache on, zero-config)
  - Gleiche Befehle + `--moe-cache 0` (off) vs default (on)
- [ ] 18. Cache hit-rate logging
  - hits/total per layer
  - VRAM usage
  - Bail-out events
- [ ] 19. Code-Review via `code-review` Skill
- [ ] 20. Produktiv-Server auf Styx neu starten (mit cache default)

## Verifikations-Strategie

| Schritt | Metrik | Vergleich |
|---------|--------|-----------|
| 1-5 | Build grün (CPU-only) | `cmake --build build` auf Hydra |
| 6-9 | Build grün (CUDA) | `cmake --build build` auf Hydra mit CUDA |
| 10-13 | Smoke-Test: Modell lädt, keine Crashes | `llama-bench -p 64 -n 8` auf Hydra |
| 14-18 | tg128 t/s auf Styx | Vorher (no cache) vs. Nachher (cache on) |
| 18 | Cache hit rate | Log-Output: hits/total per layer |
| 18 | VRAM usage | Soll: < 50% of spare VRAM |
| 18 | Bail-out frequency | Soll: 0 nach warm-up |
| 19 | Code-Review | `code-review` Skill, ship-ready |

## Architektur: Code-Integration-Points

### Dateien die modifiziert werden

| Datei | Änderung | Phase |
|-------|----------|-------|
| `ggml/include/ggml.h` | `ggml_moe_cache` struct + API deklaration | 1 |
| `ggml/src/ggml.c` | `ggml_moe_cache` global (zero-initialized) | 1 |
| `ggml/src/ggml-cpu/quants.c` (oder ähnlich) | CPU `mul_mat_id` thread 0 dispatch | 1 |
| `ggml/src/ggml-cpu/llama-cpu.c` | CPU `swiglu` glu_hits mask | 1 |
| `ggml/src/ggml-backend.cpp` | redirect_offer/finalize/invalidate hooks | 1 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | CUDA cache registration + kernels | 2 |
| `ggml/src/ggml-cuda/moe-cache.cu` (neu) | CUDA MoE cache implementation | 2 |
| `common/arg.cpp` | `--moe-cache` CLI option | 3 |
| `common/common.h` | `moe_cache` in params struct | 3 |
| `src/llama-context.cpp` | #40 frequency → cache hot-set seeding | 3 |
| `ggml/src/ggml-backend.cpp` | #37 code entfernen (expert_valid, tensor_copied) | 3 |

### Was NICHT modifiziert wird

- `src/llama-graph.cpp` — Graph-Building bleibt unverändert (MUL_MAT_ID bleibt wo es ist)
- `src/llama-model-loader.cpp` — tensor_buft_overrides bleiben unverändert
- `ggml/src/ggml-vulkan/` — Vulkan-Support später (separater Session-Plan)

## Recherche-Strategien

| Problemtyp | Recherche-Strategie |
|------------|---------------------|
| CPU mul_mat_id thread-0 dispatch | PR #24524 Commit 1 (618cdb4) lesen |
| CUDA cache pool management | PR #24524 Commit 2 (708c18c) lesen |
| Scheduler redirect hooks | PR #24524 Commit 1, sched integration Teil |
| Fused gate+up+SwiGLU | PR #24524 Commit 2, paired pool logic |
| #40 frequency → hot-set | Unsere #40 Implementierung in llama-context.cpp |

## Recherche-Fallbacks

| Block | Mögliches Problem | Recherche-Fallback |
|-------|-------------------|-------------------|
| CPU thread-0 dispatch | Race condition mit anderen Threads | PR #24524 thread sync mechanism |
| CUDA cache OOM | VRAM zu klein für hot-set | VRAM surrender logic, auto-shrink |
| Fused SwiGLU | Falsche row-skip mask | test-backend-ops MUL_MAT_ID 789/789 |
| Scheduler redirect | Async fill nach buffer free | invalidate hook timing |
| Build errors | C++ struct in C file | ggml.c → function table als extern C |

## Offene Fragen

- Wie funktioniert die thread-0 dispatch-Synchronisation im Detail? (muss PR Commit 1 lesen)
- Kann der Cache pro-Layer oder global sein? (PR: pro-device, multi-GPU sharding)
- Wie wird mit gate_up (merged) vs separate gate/up umgegangen? (PR: paired pools)
- On-disk persistence: welches Format? (PR: hot-set file, später implementierbar)
- Vulkan-Support: wann? (Separater Plan, nach CUDA-Erfolg)

## Risiko-Bewertung

| Risiko | Wahrscheinlichkeit | Impact | Mitigation |
|--------|-------------------|--------|------------|
| Cache thrashing bei wechselnden Experten | Mittel | Mittel | Decode-only fill, frequency-guided |
| VRAM OOM auf Styx (8GB) | Hoch | Niedrig | VRAM surrender, auto-shrink, --moe-cache N cap |
| CPU thread sync bugs | Mittel | Hoch | PR #24524 sync mechanism übernehmen, test-backend-ops |
| Build-Komplexität (11 Dateien) | Niedrig | Mittel | 3-Commit-Struktur, inkrementell testbar |
| Fork-Divergenz bei upstream sync | Hoch | Niedrig | Isolierte Module, klare Feature-Grenzen |

## Erwartete Ergebnisse

### Styx (GTX 1070, PCIe 3.0, 8GB VRAM, MoE-Offload)
- **Best Case:** +25-50% tg128 (wie PR #24524 auf ähnlich constrained setup)
- **Realistic:** +10-20% tg128 (Pascal hat keine Tensor Cores, GPU matvec slower)
- **Worst Case:** Parity (cache dormant, bail-out aktiviert)

### Mars (RDNA3 APU, unified memory)
- **Erwartung:** Wenig Benefit (unified memory = kein PCIe-Transfer-Overhead)
- **Cache könnte sogar schaden** wenn GPU-Cache-Lookup langsamer als RAM-access
- **Empfehlung:** Auf Mars deaktivieren (`--moe-cache 0`)

### Uranus (2x RTX 4060 Ti, 32GB VRAM)
- **Best Case:** +7-25% tg128 (wie PR #24524 auf 4x RTX 3090)
- **Realistic:** +5-15% (Ada hat Tensor Cores, aber weniger spare VRAM als 3090)
