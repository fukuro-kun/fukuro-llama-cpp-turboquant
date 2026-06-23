# Branch-Uebersicht

Dieses Repository enthaelt mehrere Branches mit unterschiedlichen Zwecken.

> **Fork-Lineage und Feature-Vergleich:** Siehe [FORKS.md](FORKS.md) fuer die vollstaendige Abstammungskette und den detaillierten Unterschied zwischen den Fork-Ebenen.

---

## `master` — Hauptentwicklung

**Enthaelt:** Alles — Vulkan-Turbo3 (fuer AMD), AtomicBot Upstream, unsere Fixes

| Komponente | Status |
|-----------|--------|
| Vulkan-Turbo3 (AMD) | 532 Commits |
| AtomicBot upstream | Integriert (2026-06-13) |
| Gemma4Assistant | Integriert |
| MTP Tensor Fix | Integriert |

**Fuer:** Lokale Entwicklung auf allen GPUs (NVIDIA CUDA, AMD ROCm/Vulkan, Intel)

---

## `feature/turboquant-kv-cache-sync` — Sauberer Sync

**Zweck:** Minimalste Divergenz zu AtomicBot-ai fuer Zusammenarbeit

**Enthaelt:**
- AtomicBot-ai `feature/turboquant-kv-cache` (aktueller Stand)
- Unsere 2 Cherry-Picks:
  1. `287ad1b90` — Gemma4UnifiedAssistantForCausalLM Unterstuetzung
  2. `16fe4bc2e` — MTP tensor names alignment + Draft-Kompatibilitaet

**Divergenz zu upstream:** 2 Commits ahead, 0 behind

**Verwendung:**
- PRs an AtomicBot-ai erstellen
- Referenz fuer saubere Zusammenarbeit
- **Nicht** fuer direkte Entwicklung

**Warum existiert dieser Branch?**
- `master` hat 532 zusaetzliche Vulkan-Turbo3 Commits
- `feature/turboquant-kv-cache` (alt) hat veraltete Upstream-Merges
- Dieser Branch zeigt: "Wir haben nur 2 sinnvolle Aenderungen mehr als du"

---

## `feature/turboquant-kv-cache` — Veraltet

**Status:** Obsolet. Wurde durch `feature/turboquant-kv-cache-sync` ersetzt.

**Enthaelt:** Alte Upstream-Merges von AtomicBot-ai (vor dem 2026-06-13 Sync)

---

## `feature/llama-bench-mtp` — Benchmark-Erweiterungen

**Enthaelt:** llama-bench mit MTP-Akzeptanzrate-Messung

**Quelle:** Sujit Vasanth / AtomicBot-ai

**Relevanz:** Cherry-Pick-Quelle fuer `convert_hf_to_gguf.py` (Gemma4Assistant)

---

## `feature/vulkan-mulmat-turbo3` / `-clean` — Vulkan Experimente

**Enthaelt:** Vulkan-Turbo3 Optimierungen

**Status:** Experimentell, teilweise in `master` integriert

---

## `feature/uma-igpu-attention` — AMD iGPU

**Enthaelt:** UMA (Unified Memory Architecture) Attention fuer AMD iGPUs

**Relevant fuer:** AMD iGPUs und APUs (Unified Memory Architecture)

---

## `feature/convert-drafts-to-dot` — Draft-Naming

**Enthaelt:** Aenderung der Draft-Tensor-Namen auf `.` (dot) Notation

**Status:** Optional, nicht in `master`

---

## `feature/diffusion-gemma-v2` — DiffusionGemma Entropy-Bound Decoder

**Zweck:** DiffusionGemma-Output-Qualitaet an Unsloth-Referenz (PR #24423) annaehern

**Enthaelt:**
- Chat-Template Integration (Phase 1.1)
- Self-Conditioning (SC) Tensoren laden (Phase 1.2)
- Default-Params an Referenz anpassen (Phase 1.3)
- KV-Cache Path (PREFILL → DECODE) (Phase 2.2)
- Multi-Block Processing (Phase 2.3)
- llama-server Integration (Phase 4.1)

**Abgeleitet von:** `master`

**Arbeitsbranch:** Aktive Entwicklung, spaeter per Merge in `master`

**Status:** In Entwicklung

---

## `feature/cuda-fast-wht` — CUDA Fast Walsh-Hadamard Transform

**Zweck:** CUDA-Optimierung fuer die Walsh-Hadamard-Transform (WHT), Kernoperation des TurboQuant KV-Cache-Systems.

**Enthaelt:**
- `a817a22bc` — Enum `GGML_HINT_SRC0_IS_HADAMARD` + CPU-WHT (bereits in `master`)
- `c1f1e28d2` — CUDA-WHT Kernel (`fwht.cu`, `fwht.cuh`) auf direkte CUDA-Syntax umgeschrieben
- `192d8ae8b` — PDL-Sync und Fallback-Verbesserungen

**Abgeleitet von:** `master` (nach `a817a22bc`)

**Status:** Abgeschlossen, Build erfolgreich, Benchmark bestanden

**Performance:** +11% pp512 (TinyLlama 1B, TurboQuant KV-Cache)

**Merge-Status:** Bereit fuer Merge in `master`

---

## Build mit CUDA (Wichtig!)

**Problem:** glibc >= 2.43 (z.B. Ubuntu 26.04) fuegt `noexcept` zu `rsqrt` hinzu. CUDA 13.1 kann das nicht verarbeiten.

**Loesung:** `scripts/build-cuda-glibc-patch.sh`
- Patched `mathcalls.h` temporaer fuer den Build
- Stellt die System-Datei nach dem Build wieder her
- Verwendet `gcc-13` als Host-Compiler

```bash
./scripts/build-cuda-glibc-patch.sh
```

**Alternativen (falls Patch nicht gewuenscht):**
1. Docker mit aelterer glibc verwenden
2. Lokale Header-Kopie fuer NVCC bereitstellen
3. Auf CUDA 13.2+ warten (offizieller Fix erwartet)

---

## Empfohlener Workflow

```bash
# Entwicklung (Primary Remote = Codeberg):
git checkout master

# Sync mit AtomicBot pruefen:
git fetch upstream
git log --oneline upstream/feature/turboquant-kv-cache..feature/turboquant-kv-cache-sync

# PR erstellen:
git push origin feature/turboquant-kv-cache-sync
# → Auf GitHub: Create Pull Request gegen AtomicBot-ai/atomic-llama-cpp-turboquant
```

---

## Cherry-Pick-Branch-Strategie

Um upstream-Verbesserungen **isoliert** und **nachvollziehbar** zu integrieren, werden Cherry-Picks auf dedizierten Feature-Branches durchgefuehrt und spaeter einzeln in `master` gemergt.

| Feature-Gruppe | Branch | Von | Upstream-Commits | Abhaengigkeiten | Merge-Ziel |
|----------------|--------|-----|-----------------|----------------|------------|
| **Vulkan-WHT** | `feature/diffusion-gemma-v2` | `master` | `48e7078ee`, `e82beaa60` | `GGML_HINT_SRC0_IS_HADAMARD` Enum | `master` |
| **coopmat2 Feature-Check** | `feature/diffusion-gemma-v2` | `master` | `5a69c9743` | — | `master` |
| **CUDA Fast WHT** | `feature/cuda-fast-wht` | `master` | `a817a22bc`, `c1f1e28d2`, `192d8ae8b` | `a817a22bc` (Enum+CPU-WHT) wurde zuerst in `master` gepickt | `master` |

**Regeln:**
1. Jede Feature-Gruppe bekommt einen eigenen Branch (von `master` abgezweigt)
2. Abhaengigkeiten (z.B. `GGML_HINT_SRC0_IS_HADAMARD` Enum) muessen zuerst in `master`
3. Einzelne Commits pro Branch erleichtern Review und Rollback
4. Nach erfolgreichem Build + Test wird der Branch via PR/Merge in `master` integriert
5. Ergebnisse werden in [FORKS.md](FORKS.md) §5.x dokumentiert

**Warum nicht alles auf einem Branch?**
- Isolierte Cherry-Picks erleichtern Debugging (bisect)
- Kleinere PRs/Merges sind uebersichtlicher
- Rollback einer Feature-Gruppe ohne andere zu beeintraechtigen

---

**Remotes:** Siehe [FORKS.md §2](FORKS.md#2-remotes) fuer die vollstaendige Remote-Konfiguration.
