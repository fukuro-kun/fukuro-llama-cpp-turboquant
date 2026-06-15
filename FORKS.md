# Fork-Abstammung und -Vergleich

> Zentrale Dokumentation der Repository-Lineage, der Unterschiede zwischen Fork-Ebenen und der Branch-Strategie.
> Siehe auch [BRANCHES.md](BRANCHES.md) fuer Branch-Details und [AGENTS.md](AGENTS.md) fuer Arbeitsvertraege.

---

## 1. Abstammungskette (Lineage)

```
ggml-org/llama.cpp
    |
    |  (Original: LLM-Inference in C/C++, ggml-Bibliothek)
    v
TheTom/llama-cpp-turboquant
    |
    |  (+ TurboQuant KV-/Weight-Kompression)
    v
AtomicBot-ai/atomic-llama-cpp-turboquant
    |
    |  (+ Gemma 4 MTP, Qwen NextN, UDT-Quantisierung, AtomicChat-Features)
    v
fukuro-kun/fukuro-llama-cpp-turboquant  (GitHub-Mirror)
    |
    |  (+ DiffusionGemma, Vulkan-Turbo3, lokale Anpassungen)
    v
codeberg.org/fukuro/fukuro-llama-cpp-turboquant  (PRIMARY REMOTE)
```

---

## 2. Remotes

| Remote | URL | Rolle |
|--------|-----|-------|
| **Primary** | `git@codeberg.org:fukuro/fukuro-llama-cpp-turboquant.git` | Haupt-Repository. Hier laeuft die Entwicklung. |
| **GitHub** | `git@github.com:fukuro-kun/fukuro-llama-cpp-turboquant.git` | Mirror und Sync-Punkt mit AtomicBot-ai. |
| **AtomicBot (Upstream-Fork)** | `git@github.com:AtomicBot-ai/atomic-llama-cpp-turboquant.git` | Direkter Upstream-Fork. PR-Ziel fuer saubere Aenderungen. |
| **TheTom (Origin-TurboQuant)** | `git@github.com:TheTom/llama-cpp-turboquant.git` | Ursprung der TurboQuant-Features. |
| **ggml-org (Root-Upstream)** | `git@github.com:ggml-org/llama.cpp.git` | Original-Upstream. Ca. 532 Commits ahead von unserem Stand. |

**Empfohlener Workflow:**
```bash
# Standard-Entwicklung
origin = codeberg.org:fukuro/... (push/pull)

# Sync mit AtomicBot
upstream = github.com:AtomicBot-ai/... (fetch only)
git fetch upstream
git log --oneline upstream/feature/turboquant-kv-cache..feature/turboquant-kv-cache-sync
```

---

## 3. Unterschiede pro Fork-Ebene

### 3.1 ggml-org/llama.cpp — Root-Upstream

| Aspekt | Stand |
|--------|-------|
| **Basis** | LLM-Inference Engine (C/C++), ggml-Tensor-Bibliothek |
| **Modelle** | 100+ Architekturen (Llama, Mistral, Qwen, Gemma, etc.) |
| **Backends** | CPU, CUDA, Vulkan, Metal, HIP, SYCL, etc. |
| **Quantisierung** | Q2_K–Q8_K, IQ-Formate, F16, BF16 |
| **Spekulativ** | EAGLE, classical draft-model |
| **Multimodal** | CLIP-Vision in llama-server |
| **Architektur** | Ab ~Mai 2026: `llama_model_base`-Klassenhierarchie |

### 3.2 TheTom/llama-cpp-turboquant

| Aspekt | Hinzugefuegt / Veraendert |
|--------|--------------------------|
| **TurboQuant** | WHT-rotierte KV-/Weight-Kompression |
| **Formate** | `TQ3_1S`, `TQ4_1S`, `turbo3` KV-Cache |
| **Backends** | Native TurboQuant-Kernels in CUDA, Metal (`TurboFlash`), Vulkan, HIP |
| **KV-Kompression** | ~4.3× KV-Cache-Einsparung via `-ctk turbo3 -ctv turbo3` |
| **Build** | Identisch zu upstream, zusaetzliche Quantisierungstypen |

### 3.3 AtomicBot-ai/atomic-llama-cpp-turboquant

| Aspekt | Hinzugefuegt / Veraendert |
|--------|--------------------------|
| **Gemma 4 MTP** | Multi-Token Prediction mit `gemma4_assistant`-Drafter. Single-context, cross-attention in target KV. +30–50 % Throughput. |
| **Qwen 3.x NextN** | Shared-model spekulative Decodierung. Kein zweites mmap. +24–36 % tps (MoE), +5–7 % (dense). |
| **UDT-Quantisierung** | Tensor-type Masks: NextN/MTP → `Q8_0`, attn Q/K → `Q6_K`, paart mit TurboQuant3 KV. |
| **Multimodal + Spec** | `--mmproj` gleichzeitig mit `mtp`/`nextn`/`eagle3` auf einem Slot. |
| **GGUF-Conversion** | Erweiterte `convert_hf_to_gguf.py` fuer Gemma4Assistant, NextN-Tensoren. |
| **Brand** | AtomicChat/AtomicBot-Erweiterungen, Model-Collections auf Hugging Face. |
| **Architektur** | Noch monolithisch (ein `llama_model` mit Switch-Statements), wie upstream vor Mai 2026. |

### 3.4 fukuro-llama-cpp-turboquant (unser Fork)

| Aspekt | Hinzugefuegt / Veraendert |
|--------|--------------------------|
| **DiffusionGemma** | Monolithischer Port von PR #24423 (block text-diffusion MoE auf Gemma-4-Backbone). Forward-Pass funktioniert, Diffusion-Decoding-Loop in Arbeit. |
| **Vulkan-Turbo3** | ~532 zusaetzliche Commits fuer Vulkan-Turbo3-Optimierungen (in `master`). |
| **Gemma 4 12B** | Assistant-Unterstuetzung fuer Gemma-4-12B-Modell. |
| **Primaries Remote** | Codeberg (Code-Hosting). GitHub als Mirror. |
| **DOX-Framework** | AGENTS.md-Vertraege in allen Verzeichnisbäumen fuer nachvollziehbare KI-Agenten-Arbeit. |

---

## 4. Feature-Matrix (Vergleich)

| Feature | ggml-org | TheTom | AtomicBot | fukuro |
|---------|----------|--------|-----------|--------|
| TurboQuant KV/Weights | ❌ | ✅ | ✅ | ✅ |
| Gemma 4 MTP | ❌ | ❌ | ✅ | ✅ |
| Qwen 3.x NextN | ❌ | ❌ | ✅ | ✅ |
| UDT-Quant-Masks | ❌ | ❌ | ✅ | ✅ |
| DiffusionGemma | ❌ | ❌ | ❌ | ✅ |
| Vulkan-Turbo3 | ❌ | ❌ | ❌ | ✅ |
| Multimodal + Spec | ❌ | ❌ | ✅ | ✅ |
| EAGLE / classical draft | ✅ | ✅ | ✅ | ✅ |
| FlashAttention | ✅ | ✅ | ✅ | ✅ |
| 100+ Modelle | ✅ | ✅ | ✅ | ✅ |

---

## 5. Branch-Strategie

| Branch | Remote | Zweck |
|--------|--------|-------|
| `master` | Codeberg (primary) | Hauptentwicklung. Alle Features (TurboQuant, MTP, NextN, DiffusionGemma, Vulkan-Turbo3). |
| `feature/turboquant-kv-cache-sync` | GitHub | Sauberer Sync mit AtomicBot-ai. Genau 2 Commits ahead (unsere Cherry-Picks). PR-Quelle. |
| `feature/turboquant-kv-cache` | GitHub | Obsolet. Alte Upstream-Merges. |

**Divergenz zu AtomicBot:**
- `master` ist **~532 Commits ahead** (Vulkan-Turbo3) plus DiffusionGemma plus unsere Fixes.
- `feature/turboquant-kv-cache-sync` ist **2 Commits ahead** (Gemma4Assistant + MTP tensor fixes).

Details: [BRANCHES.md](BRANCHES.md)

---

## 6. Sync- und PR-Workflow

### Saubere Aenderungen an AtomicBot-ai

1. Auf `feature/turboquant-kv-cache-sync` rebasen (nur unsere 2 Cherry-Picks).
2. Neue Aenderung auf einem Feature-Branch entwickeln.
3. Gegen `feature/turboquant-kv-cache-sync` rebasen.
4. PR an `AtomicBot-ai/atomic-llama-cpp-turboquant:feature/turboquant-kv-cache` erstellen.

### Upstream-Sync (ggml-org)

- Aktuell **nicht empfohlen** als grosser Merge. Unser Fork hat die alte monolithische Architektur; upstream ist auf Klassen-Hierarchie umgestellt.
- Einzelne Features (z. B. neue Modelle, Bugfixes) koennen gezielt cherry-picked werden.
- Langfristig: Gezielter Sync des Refactor-Commits, wenn DiffusionGemma stabil ist. Siehe [pocs/DIFFUSION_GEMMA_ENTSCHEIDUNG.md](pocs/DIFFUSION_GEMMA_ENTSCHEIDUNG.md).

---

## 7. Historie und Attribution

| Beitragender | Rolle | Repository |
|--------------|-------|------------|
| ggml-org / ggerganov | Original llama.cpp, ggml | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| TheTom / @TheTom | TurboQuant KV-Kompression | [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) |
| AtomicBot-ai / AtomicChat | Gemma 4 MTP, Qwen NextN, UDT, Multimodal+Spec | [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) |
| fukuro | DiffusionGemma-Port, Vulkan-Turbo3, DOX-Framework | [codeberg.org/fukuro/...](https://codeberg.org/fukuro/fukuro-llama-cpp-turboquant) |

---

*Letzte Aktualisierung: 2026-06-15*
