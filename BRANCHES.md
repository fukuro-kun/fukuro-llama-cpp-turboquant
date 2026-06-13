# Branch-Uebersicht

Dieses Repository enthaelt mehrere Branches mit unterschiedlichen Zwecken.

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

## Empfohlener Workflow

```bash
# Entwicklung:
git checkout master

# Sync mit AtomicBot pruefen:
git fetch upstream
git log --oneline upstream/feature/turboquant-kv-cache..feature/turboquant-kv-cache-sync

# PR erstellen:
git push origin feature/turboquant-kv-cache-sync
# → Auf GitHub: Create Pull Request gegen AtomicBot-ai/atomic-llama-cpp-turboquant
```
