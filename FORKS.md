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
| **Architektur** | Ab Mai 2026 (Commit `994118a18`): `llama_model_base`-Klassenhierarchie mit `load_hparams`/`load_tensors` pro Modell-Klasse. 129 Dateien, ~20k Zeilen geaendert. |

### 3.2 TheTom/llama-cpp-turboquant

| Aspekt | Hinzugefuegt / Veraendert |
|--------|--------------------------|
| **TurboQuant** | WHT-rotierte KV-/Weight-Kompression |
| **Formate** | `TQ3_1S`, `TQ4_1S`, `turbo3` KV-Cache |
| **Backends** | Native TurboQuant-Kernels in CUDA, Metal (`TurboFlash`), Vulkan, HIP |
| **KV-Kompression** | ~4.3× KV-Cache-Einsparung via `-ctk turbo3 -ctv turbo3` |
| **Architektur** | Monolithisch (`struct llama_model` mit Switch-Statements). Refactor `994118a18` ist im Remote vorhanden, aber **nicht in `master` gemerged**. |
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
| **Architektur** | Monolithisch (`struct llama_model` mit Switch-Statements). Refactor **nicht** gemerged; MTP-, NextN- und UDT-Features greifen direkt in monolithische Code-Pfade ein. |

### 3.4 fukuro-llama-cpp-turboquant (unser Fork)

| Aspekt | Hinzugefuegt / Veraendert |
|--------|--------------------------|
| **DiffusionGemma** | Monolithischer Port von PR #24423 (block text-diffusion MoE auf Gemma-4-Backbone). Forward-Pass funktioniert, Diffusion-Decoding-Loop in Arbeit. |
| **Vulkan-Turbo3** | ~532 zusaetzliche Commits fuer Vulkan-Turbo3-Optimierungen (in `master`). |
| **Gemma 4 12B** | Assistant-Unterstuetzung fuer Gemma-4-12B-Modell. |
| **Primaries Remote** | Codeberg (Code-Hosting). GitHub als Mirror. |
| **DOX-Framework** | AGENTS.md-Vertraege in allen Verzeichnisbäumen fuer nachvollziehbare KI-Agenten-Arbeit. |

---

## 4. Architektur-Refactor: Stand und Bedeutung

### 4.1 Faktische Lage

| Repository | Refactor `994118a18` in `master`? | Architektur | `llama-model.h` Zeilen |
|------------|--------------------------------|-------------|----------------------|
| `ggml-org/llama.cpp` | ✅ Ja | Klassen-Hierarchie (`llama_model_base`) | 735 |
| `TheTom/llama-cpp-turboquant` | ❌ Nein | Monolithisch (`struct llama_model`) | 643 |
| `AtomicBot-ai/atomic-llama-cpp-turboquant` | ❌ Nein | Monolithisch (`struct llama_model`) | 598 |
| `fukuro-llama-cpp-turboquant` | ❌ Nein | Monolithisch (`struct llama_model`) | 688 |

**Wichtig:** Alle drei Forks haben den Commit `994118a18` in ihrer Git-Datenbank (weil `ggml-org` als Upstream-Remote konfiguriert ist und gefetched wird), aber **keiner** hat ihn in den eigenen `master` gemerged.

### 4.2 Warum der Refactor so einflussreich ist

Der Commit (Xuan-Son Nguyen, upstream PR #22004) verschiebt `load_hparams` und `load_tensors` aus der zentralen `llama-model.cpp` in **pro-Modell-Klassen**:

- Vorher: Ein `struct llama_model` mit gigantischem Switch-Statement fuer alle 100+ Architekturen.
- Nachher: Abstrakte `llama_model_base`-Klasse, von der jede Architektur erbt (`class llm_build_<arch> : public llama_model_base`).

Dies betrifft direkt unsere Fork-Features:
- **TurboQuant:** KV-Cache-Typ-Registrierung und Tensor-Mapping greifen in `load_tensors` ein.
- **Gemma 4 MTP:** `mtp_assistant` wird in `llama_model` gehalten; cross-attention in target KV ist monolithisch verankert.
- **Qwen NextN:** `nextn_predict_layers` und Shared-Model-Draft erfordern Modell-Struktur-Zugriff.
- **DiffusionGemma:** `canvas_length`, `pkv_k/pkv_v`, SC-Parameter — alles haengt am `llama_model`.

Ein Merge des Refactors wuerde **alle diese Features gleichzeitig** anfassen. Die geschaetzte Konfliktloesungszeit liegt bei **Wochen**, nicht Stunden.

### 4.3 Entwicklungs-Szenarien, die auf uns zukommen

#### Szenario A: TheTom merged den Refactor
- **Wahrscheinlichkeit:** Mittel (TurboQuant-Kernels sind backend-spezifisch und daher vom Refactor weniger betroffen als Modell-Loading).
- **Folge fuer uns:** AtomicBot muesste dann von TheTom mergen (oder den Refactor selbst einbauen). Unser `feature/turboquant-kv-cache-sync` waere inkompatibel.
- **Unsere Reaktion:** Entweder ebenfalls mergen (Wochen Arbeit) oder auf dem monolithischen Stand von AtomicBot bleiben (und langfristig Divergenz akzeptieren).

#### Szenario B: AtomicBot merged den Refactor
- **Wahrscheinlichkeit:** Niedrig bis Mittel. AtomicBot hat starke Bindung an die monolithische Struktur (MTP, NextN, UDT-Masks).
- **Folge fuer uns:** Wir waeren der einzige verbleibende monolithische Fork. Das ermoeglicht eine saubere Trennung: wir bleiben monolithisch, AtomicBot geht die Hierarchie-Richtung.
- **Unsere Reaktion:** Unser Sync-Branch (`feature/turboquant-kv-cache-sync`) waere obsolet. Wir muessten entscheiden, ob wir weiterhin PRs an AtomicBot senden koennen.

#### Szenario C: Keiner merged den Refactor (Status quo)
- **Wahrscheinlichkeit:** Kurzfristig hoch, langfristig sinkend.
- **Folge fuer uns:** Alle Forks bleiben kompatibel. DiffusionGemma kann problemlos als monolithischer Port existieren. Neue Modelle werden weiterhin als Switch-Case hinzugefuegt.
- **Risiko:** Der Refactor wird in upstream weiter ausgebaut (neue Modelle, Bugfixes). Je laenger wir warten, desto groesser wird die Divergenz.

#### Szenario D: Wir mergen den Refactor freiwillig
- **Wahrscheinlichkeit:** Nur wenn DiffusionGemma stabil und alle anderen Features migriert sind.
- **Aufwand:** Wochen an Konfliktloesung, Regressionstests auf Pascal/Ampere/Ada, Vulkan/CUDA/Metal.
- **Nutzen:** Langfristig einfachere Upstream-Syncs, sauberere Codebasis, bessere Modularitaet.

### 4.4 Empfohlene Strategie

**Kurzfristig (bis Ende 2026):**
- Monolithischen Stand beibehalten.
- DiffusionGemma zu einem funktionsfaehigen Diffusion-Decoding-Loop ausbauen.
- Gezielt cherry-picken: Neue Modell-Unterstuetzungen und Bugfixes aus upstream, die sich isolieren lassen.

**Mittelfristig (2027):**
- Beobachten, ob AtomicBot oder TheTom den Refactor mergen.
- Falls ja: Evaluieren, ob ein Mergen unsererseits sinnvoll ist — oder ob wir den monolithischen Zweig als eigenstaendige Linie pflegen.
- Falls nein: Weiterhin monolithisch bleiben; die Divergenz zu upstream akzeptieren und gezielt syncen.

**Langfristig:**
- Wenn der monolithische Pfad zu viel technische Schuld aufbaut, einen gezielten Refactor-Plan erstellen (nicht als grossen Merge, sondern schrittweise Modell-fuer-Modell).

---

## 5. Was uns von upstream fehlt

> Analyse vom **2026-06-15**.  
> Divergenz: **849 Commits** in `ggml-org/llama.cpp`, die nicht in unserem `master` sind.  
> TheTom und AtomicBot fehlen uns praktisch nichts — fast alle verpassten Verbesserungen kommen aus upstream.

### 5.1 Divergenz-Zahlen

| Vergleich | Commits, die uns fehlen | Richtung |
|-----------|------------------------|----------|
| TheTom → wir | 1 (HIP-FA-Pool-Retention) | Wir sind fast auf gleichem Stand |
| AtomicBot feature/tqc → wir | 0 | Wir sind *ahead* (50 Commits: DiffusionGemma + Vulkan-Turbo3) |
| ggml-org (upstream) → wir | **849** | Wir sind deutlich hinterher |
| ggml-org → TheTom | 1003 | TheTom ist noch weiter hinter upstream |

### 5.2 SYCL — Was ist das?

**SYCL** (Single-source heterogeneous programming with C++) ist Intels offener Standard fuer heterogene Berechnung. In llama.cpp wird er als **Intel-GPU-Backend** eingesetzt:

- **Zielhardware:** Intel Arc (A770 etc.), Intel iGPUs (Xe), Intel Data Center GPUs (Max/Ponte Vecchio)
- **Vorteil:** Ein C++-Code-Base laeuft auf CPU und GPU; keine separaten CUDA-Kernel
- **Nutzung in llama.cpp:** `-DLLAMA_SYCL=ON` statt CUDA/Vulkan
- **Relevanz fuer uns:** **Niedrig**. Unsere Hardware ist NVIDIA (CUDA) und AMD (Vulkan). Kein Intel Arc im Einsatz. SYCL-Verbesserungen interessieren uns nur als Referenzimplementierung fuer neue Ops.

### 5.3 Priorisierte Uebersicht (was uns fehlt)

#### 🔴 Hoch — sollten wir nachziehen

| Kategorie | Feature / Commit | Warum relevant |
|-----------|------------------|----------------|
| **AMD/Vulkan** | FlashAttention BFloat16 KV (`6e093b80e`) | BFloat16 wird auf modernen AMD-GPUs wichtig (RDNA3) |
| **AMD/Vulkan** | `v_dot2_f32_f16` in FA (`b4e3dc613`) | Schnelleres FlashAttention auf Vulkan |
| **AMD/Vulkan** | Walsh-Hadamard-Transform (`48e7078ee`, `e82beaa60`) | **Direkt fuer TurboQuant** — WHT ist Kernoperation |
| **AMD/Vulkan** | coopmat2 Feature-Check (`5a69c9743`) | Stabilere Vulkan auf Intel/AMD |
| **AMD/Vulkan** | `GL_NV_cooperative_matrix_decode_vector` (`b36eefc1b`) | Schnelleres MatMul auf NVIDIA-Vulkan |
| **CUDA** | Fast Walsh-Hadamard-Transform (`c1f1e28d2`) | TurboQuant-Performance auf CUDA |
| **CUDA** | Quantize KV-Cache Reservierung (`f8f0a47a5`) | Speicherverwaltung |
| **Neue Modelle** | **Llama 4** Scout / Maverick | Wichtige neue Meta-Modelle |
| **Neue Modelle** | **DeepSeek 3.2** | Wichtiger OSS-Modell-Trend |
| **Multimodal** | Video-Input-Support (`8f83d6c27`) | `llama-server` kann jetzt Videos verarbeiten |
| **Server** | Real-time Reasoning Interruption (`354ebac8c`) | Bessere Chat-UX |
| **Server** | mtmd Post-Decode Callback (`e3cab403b`) | Erweiterte Multimodal-Kontrolle |
| **Server** | Build-time gzip compression (`e8067a8b3`) | Kleinere Assets |
| **Server** | SSE Ping-Interval (`60130d18f`) | Verbindungsstabilitaet |
| **GGUF-Convert** | Fix Gemma 4 Unified conversion (`e8023568d`) | Wir haben Gemma 4 MTP — Audio-Fixes relevant |
| **GGUF-Convert** | Gemma 4 audio projector embedding size fix (`e3ba22d6c`) | Audio-Modality fuer Gemma 4 |

#### 🟡 Mittel — gezielt evaluieren

| Kategorie | Feature | Warum |
|-----------|---------|-------|
| Vulkan | Q3_K/Q6_K Block-Load (`19620004f`) | Quantisierungs-Performance |
| Vulkan | Fast path fuer Buffer-Transfers (`fdc3db9b6`) | Performance |
| Vulkan | Pipeline-Barriers (`3e7bd4f39`) | Korrektheit |
| Multimodal | HEIC/HEIF-Bilder (`5f04dc7ac`) | Format-Unterstuetzung |
| Multimodal | Frame-Merge fuer Qwen-VL (`31e82494c`) | Vision-Modelle |
| Neue Formate | MXFP4 / NVFP4 | Neue NVIDIA-4-bit-Formate |
| Server | Prompt-Logging (`1e912561d`) | Debugging |
| Server | PWA-Support (`f7ca93d12`) | Mobile UX |

#### 🟢 Niedrig / irrelevant

| Kategorie | Feature | Grund |
|-----------|---------|-------|
| **Metal** | *alle* Metal-Commits | Wir haben **keine** Apple-Hardware |
| **EAGLE3** | Spekulatives Decodieren (`88a39274e`) | Wir fokussieren auf **Gemma 4 MTP + Qwen NextN**, nicht Llama-EAGLE |
| **SYCL** | *alle* SYCL-Commits | Keine Intel Arc / oneAPI-Hardware |
| **Hexagon** | Qualcomm-Optimierungen | Keine Qualcomm-Hardware |
| **WebGPU** | Browser-Backend | Kein Web-Einsatz |
| **OpenCL** | Adreno-/Mobile-Optimierungen | Kein Mobile-Einsatz |
| **RISC-V** | RVV-Erweiterungen, Spacemit | Keine RISC-V-Hardware |
| **Tensor-Parallelism** | Multi-GPU (experimentell) | Wir haben nur Single-GPU-Systeme |

### 5.4 Neue Modell-Architekturen (upstream hat sie, wir nicht)

| Modell | Relevanz | Anmerkung |
|--------|----------|-----------|
| **llama4** | 🔴 Hoch | Llama 4 Scout / Maverick — wichtige Meta-Modelle |
| **deepseek32** | 🔴 Hoch | DeepSeek 3.2 |
| **deepseek2ocr** | 🟡 Mittel | OCR-Modus |
| **mistral4** | 🟡 Mittel | Mistral-Nachfolger |
| **eagle3** | 🟢 Niedrig | Draft-Modell — wir haben MTP/NextN |
| **mamba2** | 🟡 Mittel | SSM-Architektur |
| **cohere2-MoE** | 🟢 Niedrig | Nicht auf unserer Roadmap |
| **granite-moe** | 🟢 Niedrig | IBM — nicht prioritaer |
| **mellum** | 🟢 Niedrig | Nische |
| **nomic-bert / nomic-bert-moe** | 🟢 Niedrig | Embedding-Modelle |
| **jina-bert-v2/v3** | 🟢 Niedrig | Embedding |
| **glm-dsa** | 🟢 Niedrig | Nische |
| **lfm2moe** | 🟢 Niedrig | Liquid Foundation |
| **mimo2** | 🟢 Niedrig | Nische |
| **minicpm / minicpm3** | 🟢 Niedrig | Kleine Modelle |
| **nemotron-h-moe** | 🟢 Niedrig | NVIDIA — nicht prioritaer |

**Anmerkung:** `diffusion-gemma.cpp` existiert in upstream **nicht mehr** als eigenstaendige Datei — es wurde in die `llama_model_base`-Hierarchie integriert. Wir haben es als monolithischen Port.

### 5.5 Warum Cherry-Pick schwierig ist

Die meisten dieser 849 Commits **haengen voneinander ab**:

- **Vulkan-WHT** baut auf Refactor-Kernaenderungen auf (Op-Enum in `ggml.c`).
- **Video-Input** erfordert neue mtmd-APIs, die mit Server-Aenderungen gekoppelt sind.
- **Llama 4** nutzt die neue `llama_model_base`-Hierarchie fuer `load_hparams`.
- **MXFP4/NVFP4** erfordern neue GGML-Typen in `ggml.h` + `ggml-quants.c` + alle Backends.

**Strategie:** Nicht einzelne Commits, sondern **Ketten** cherry-picken. Zuerst die Backend-Verbesserungen (Vulkan, CUDA), dann die Modell-Dateien. Server-Features nur, wenn sie keine Architektur-Aenderung brauchen.

### 5.6 Empfohlene Cherry-Pick-Reihenfolge

1. **Vulkan: Walsh-Hadamard + v_dot2 + BFloat16 FA** — direkter TurboQuant-Nutzen
2. **CUDA: Fast WHT + KV-Reserve** — Performance auf unserer Hauptplattform
3. **Gemma 4 Audio-Fixes** — passt zu unserem MTP-Fokus
4. **Server: Reasoning Interruption + gzip + SSE ping** — UX-Verbesserungen
5. **Multimodal: Video-Input + HEIC** — wenn mtmd-Kompatibilitaet gegeben
6. **Llama 4 / DeepSeek 3.2** — als monolithische `case`-Erweiterung (nicht als Klasse)

---

## 6. Feature-Matrix (Vergleich)

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

## 7. Branch-Strategie

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

## 8. Sync- und PR-Workflow

### Saubere Aenderungen an AtomicBot-ai

1. Auf `feature/turboquant-kv-cache-sync` rebasen (nur unsere 2 Cherry-Picks).
2. Neue Aenderung auf einem Feature-Branch entwickeln.
3. Gegen `feature/turboquant-kv-cache-sync` rebasen.
4. PR an `AtomicBot-ai/atomic-llama-cpp-turboquant:feature/turboquant-kv-cache` erstellen.

### Upstream-Sync (ggml-org)

- Aktuell **nicht empfohlen** als grosser Merge. Unser Fork hat die alte monolithische Architektur; upstream ist auf Klassen-Hierarchie umgestellt. Siehe [§4](FORKS.md#4-architektur-refactor-stand-und-bedeutung).
- Einzelne Features (z. B. neue Modell-Unterstuetzungen, Bugfixes) koennen gezielt cherry-picked werden. Siehe [§5](FORKS.md#5-was-uns-von-upstream-fehlt) fuer die priorisierte Liste.
- Langfristig: Gezielter Sync des Refactor-Commits, wenn DiffusionGemma stabil ist. Siehe [pocs/DIFFUSION_GEMMA_ENTSCHEIDUNG.md](pocs/DIFFUSION_GEMMA_ENTSCHEIDUNG.md).

---

## 9. Historie und Attribution

| Beitragender | Rolle | Repository |
|--------------|-------|------------|
| ggml-org / ggerganov | Original llama.cpp, ggml | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| TheTom / @TheTom | TurboQuant KV-Kompression | [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) |
| AtomicBot-ai / AtomicChat | Gemma 4 MTP, Qwen NextN, UDT, Multimodal+Spec | [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) |
| fukuro | DiffusionGemma-Port, Vulkan-Turbo3, DOX-Framework | [codeberg.org/fukuro/...](https://codeberg.org/fukuro/fukuro-llama-cpp-turboquant) |

---

*Letzte Aktualisierung: 2026-06-15*
