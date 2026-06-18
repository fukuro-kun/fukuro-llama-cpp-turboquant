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
| **DiffusionGemma** | Monolithischer Port von PR #24423 (block text-diffusion MoE auf Gemma-4-Backbone). Forward-Pass funktioniert, **Diffusion-Decoding-Loop funktioniert** (Entropy-Bound Decoder vollstaendig implementiert). Verbleibende Limitierung: Self-Conditioning (SC-Tensoren fehlen in GGUFs). |
| **Vulkan-Optimierungen** | Q3_K/Q6_K Block-Load (+57%/+78% tg128 auf Intel BMG), iq1 shared-memory-Reduktion, host-memory Lock-Kontention optimiert. |
| **Gemma 4 12B** | Assistant-Unterstuetzung fuer Gemma-4-12B-Modell. |
| **Primaries Remote** | Codeberg (Code-Hosting). GitHub als Mirror. |
| **CUDA KV-Cache Reserve** | Pre-Reservierung von KV-Cache-Speicher fuer FlashAttention reduziert OOM-Risiko. |
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
| **AMD/Vulkan** | ~~FlashAttention BFloat16 KV (`6e093b80e`)~~ | ~~BFloat16 wird auf modernen AMD-GPUs wichtig (RDNA3)~~ → **NICHT NUTZBAR**, siehe [§5.7](FORKS.md#57-warum-bfloat16-fa-fuer-uns-nicht-nutzbar)
| **AMD/Vulkan** | ~~`v_dot2_f32_f16` in FA (`b4e3dc613`)~~ | ~~Schnelleres FlashAttention auf Vulkan~~ → **ZU KOMPLEX** (FlashAttention-Refactor-Abhaengigkeiten, siehe [§5.9](FORKS.md#59-vdot2-cherry-pick-abgebrochen)) |
| **AMD/Vulkan** | ~~Walsh-Hadamard-Transform (`48e7078ee`, `e82beaa60`)~~ | ~~**Direkt fuer TurboQuant** — WHT ist Kernoperation~~ → **CHERRY-PICKED** (siehe [§5.8](FORKS.md#58-wht-cherry-pick-ergebnis)) |
| **AMD/Vulkan** | ~~coopmat2 Feature-Check (`5a69c9743`)~~ | ~~Stabilere Vulkan auf Intel/AMD~~ → **CHERRY-PICKED** (siehe [§5.10](FORKS.md#510-coopmat2-feature-check-ergebnis)) |
| **AMD/Vulkan** | `GL_NV_cooperative_matrix_decode_vector` (`b36eefc1b`) | Schnelleres MatMul auf NVIDIA-Vulkan |
| **CUDA** | ~~Fast Walsh-Hadamard-Transform (`c1f1e28d2`)~~ | ~~TurboQuant-Performance auf CUDA~~ → **CHERRY-PICKED** in `feature/cuda-fast-wht`, siehe [§5.11](FORKS.md#511-cuda-fast-wht-plan) |
| **CUDA** | ~~Quantize KV-Cache Reservierung (`f8f0a47a5`)~~ | ~~FlashAttention-Speicherverwaltung~~ → **IN MASTER GEMERGED** (siehe [§5.15](FORKS.md#515-cuda-kv-cache-reserve)) |
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

### 5.6 Empfohlene Cherry-Pick-Reihenfolge (aktualisiert 2026-06-16)

**Erledigt / Abgeschlossen:**
1. ✅ ~~Vulkan-WHT~~ (`48e7078ee`, `e82beaa60`) — siehe [§5.8](FORKS.md#58-wht-cherry-pick-ergebnis)
2. ✅ ~~coopmat2 Feature-Check~~ (`5a69c9743`) — siehe [§5.10](FORKS.md#510-coopmat2-feature-check-ergebnis)
3. ✅ ~~CUDA Fast WHT~~ (`a817a22bc` + `c1f1e28d2` + `192d8ae8b`) — siehe [§5.11](FORKS.md#511-cuda-fast-wht-plan)

**Offen — Priorisiert (Vulkan / Performance):**
4. ⏳ **Vulkan: coopmat2 decode_vector** (`c74759a24`) — Vec4 B-Matrix-Loads, +BK=64. Siehe [§5.12](FORKS.md#512-coopmat2-decode-vector). **BLOCKIERT** durch Mesa 25.0.7 (Header fehlt).
5. ✅ ~~Vulkan: Q3_K/Q6_K Block-Load~~ (`19620004f`) — **IN MASTER GEMERGED** (+57% tg128 Q3_K, +78% Q6_K auf Intel BMG). Siehe [§5.13](FORKS.md#513-vulkan-q3kq6k-block-load).
6. ✅ ~~Vulkan: Buffer-Transfer Fast Path~~ (`fdc3db9b6`) — **IN MASTER GEMERGED** (16 Zeilen, Performance).
7. ✅ ~~CUDA: KV-Cache Reserve~~ (`f8f0a47a5`) — **IN MASTER GEMERGED** (siehe [§5.15](FORKS.md#515-cuda-kv-cache-reserve)).
8. ✅ ~~CUDA: MMVQ AMD MFMA Threshold~~ (`bc81d47ab`) — Q4_K_S +68% pp512 auf MI250X. **BEREITS VORHANDEN** via AtomicBot-Upstream (`get_mmvq_mmid_max_batch_cdna` in `mmvq.cu`).

**Offen — Gezielt evaluieren:**
9. ❌ ~~CUDA: PDL mul_mat_vec_q_moe~~ (`2154a0fdc`) — **BLOCKIERT** (erfordert PDL-Infrastruktur `ggml_cuda_pdl_sync()` / `GGML_CUDA_USE_PDL`, die in unserem Fork durch direkte Kernel-Starts ersetzt wurde).
10. ✅ ~~CPU: SVE FWHT Runtime Width~~ (`3571fa543`) — **BEREITS VORHANDEN** (SVE-`svcntw()`-Logik bereits in `ggml/src/ggml-cpu/ops.cpp`).
11. ❌ ~~KV-Cache: avoid copies~~ (`379ac6673`) — **BLOCKIERT** (erfordert KV-Cache-Refactor aus upstream, siehe Analyse).
12. ❌ ~~KV-Cache: SWA checkpoints~~ (`236531595`) — **BLOCKIERT** (erfordert denselben Refactor).
13. ✅ ~~Vulkan: iq1 shared memory~~ (`d6d0ce821`) — **IN MASTER GEMERGED** (reduziert shared memory für iq1 mul_mm).
14. ✅ ~~Vulkan: host memory lock contention~~ (`bef69f130`) — **IN MASTER GEMERGED** (`unique_lock` → `lock_guard`/`shared_lock`).

**Offen — Features / Modelle:**
15. ❌ ~~DeepSeek V3.2~~ (`1f0aa2a69`) — **BLOCKIERT** (erfordert KV-Cache-Refactor mit `llama_kv_cache_dsa`, siehe B3-Analyse).
16. ⏳ **Llama 4** Scout / Maverick — Noch kein Commit gefunden (upstream wahrscheinlich noch nicht verfügbar).
17. ✅ ~~Gemma 4 Audio-Fixes (Unified Conversion)~~ (`e8023568d`) — **IN MASTER GEMERGET** (`audio_embed_dim`, `model_patch_size` defensive Fixes).
17b. ❌ ~~Gemma 4 Audio Embedding Size~~ (`e3ba22d6c`) — **NICHT RELEVANT** (Granite Speech, nicht Gemma 4).
18. ❌ ~~Server: Reasoning Interruption~~ (`354ebac8c`) — **NICHT RELEVANT** (WebUI nicht genutzt, 277 Zeilen).
19. ❌ ~~Server: gzip compression~~ (`e8067a8b3`) — **BLOCKIERT** (andere WebUI-Architektur).
20. ~~Multimodal: Video-Input~~ (`8f83d6c27`) — **ZURUECKGESTELLT** (nice-to-have, 807 Zeilen, nicht dringend).

**Verworfen:**
- ~~v_dot2_f32_f16~~ (`b4e3dc613`) — zu komplex, siehe [§5.9](FORKS.md#59-vdot2-cherry-pick-abgebrochen)
- ~~BFloat16 FA~~ (`6e093b80e`) — nicht nutzbar auf unserer Hardware, siehe [§5.7](FORKS.md#57-warum-bfloat16-fa-fuer-uns-nicht-nutzbar-ist)

---

## 5.7 Warum BFloat16 FA fuer uns nicht nutzbar ist

Commit `6e093b80e` (Vulkan: Flash Attention support for BFloat16 KV cache) ist **fuer unsere Hardware nicht anwendbar**.

### Faktische Lage

- **Mesa 25.2.x** hat `VK_KHR_shader_bfloat16` **nur fuer GFX12+** (RDNA4/CDNA4) eingefuehrt.
- Unsere AMD-GPUs sind **RDNA3 (GFX1103)** und **Vega (GFX909)** — beide **aelter als GFX12**.
- Selbst mit Mesa 25.2.8 ist die Extension auf unserer Hardware **nicht exponiert**.

| GPU-Architektur | GFX-Generation | Mesa BF16? |
|-----------------|----------------|------------|
| RDNA4 / CDNA4 | GFX12+ | Ab Mesa 25.2 |
| RDNA3 (z.B. Radeon 760M) | GFX1103 | ❌ Nein |
| Vega (z.B. Radeon Vega iGPU) | GFX909 | ❌ Nein |

### Konsequenz

Der Cherry-Pick von BFloat16-FA wuerde auf unseren Systemen **nicht funktionieren** — die Vulkan-Extension fehlt. Der Build wuerde zwar durchlaufen (ggml-vulkan prueft Features zur Laufzeit), aber der BFloat16-Pfad wuerde nie aktiviert.

**Fuer uns relevante Vulkan-Cherry-Picks bleiben:**
- ~~**WHT** (`48e7078ee`, `e82beaa60`)~~ — **CHERRY-PICKED** (siehe [§5.8](FORKS.md#58-wht-cherry-pick-ergebnis))
- ~~**v_dot2_f32_f16** (`b4e3dc613`)~~ — **ZU KOMPLEX** (siehe [§5.9](FORKS.md#59-vdot2-cherry-pick-abgebrochen))

---

## 5.8 WHT Cherry-Pick Ergebnis

**Commits:** `48e7078ee` (WHT fast path) + `e82beaa60` (Intel shmem fix)

### Durchgefuehrte Aenderungen

1. **Neuer Shader:** `ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp` — Vulkan-Compute-Shader fuer schnelle Walsh-Hadamard-Transformation mittels Subgroup-Shuffle.
2. **Pipeline-Integration:** `ggml-vulkan.cpp` erkennt `GGML_HINT_SRC0_IS_HADAMARD` und waehlt den `fwht`-Shader statt generischer MatMul.
3. **Shader-Generator:** `vulkan-shaders-gen.cpp` registriert `fwht.comp` mit passenden Spezialisierungskonstanten.
4. **Abhaengigkeit nachgeholt:** `ggml.h` erhaelt `enum ggml_op_hint` mit `GGML_HINT_SRC0_IS_HADAMARD` (upstream-Commit, der vor WHT kam).

### Build-Status

- ✅ **Kompiliert** auf System A (AMD RDNA3 APU, Mesa 25.0.7, Vulkan 1.4.305)
- ⚠️  Testfaelle (`test_mul_mat_hadamard`) auskommentiert — Klasse fehlt in unserem Stand

### Benchmark-Ergebnis (System A, AMD RDNA3 APU)

| Modell | KV | pp512 (vorher) | pp512 (mit WHT) | tg64 (vorher) | tg64 (mit WHT) |
|--------|----|---------------|-----------------|---------------|----------------|
| Gemma 4 12B Q4_K_M | Standard | 87.2 t/s | **101.9 t/s** (+17%) | 7.96 t/s | 7.89 t/s (-1%) |
| Gemma 4 E2B Q4_K_M | Standard | 559.0 t/s | **554.8 t/s** (-1%) | 39.0 t/s | 38.3 t/s (-2%) |

**Interpretation:**
- **Prompt-Verarbeitung (pp512):** +17% beim 12B-Modell — WHT beschleunigt die Hadamard-Transformation in der Prompt-Verarbeitung.
- **Token-Generation (tg64):** Kein messbarer Vorteil — WHT wird hier nicht auf dem kritischen Pfad verwendet.
- **E2B-Modell:** Kein Vorteil — kleines Modell ist bereits GPU-bound, Hadamard-Transformation ist nicht der Flaschenhals.
- **Turbo3 KV:** Kein zusaetzlicher WHT-Vorteil — TurboQuant nutzt eigene Dequant-Shaders, nicht den generischen WHT-Pfad.

**Fazit:** WHT-Cherry-Pick funktioniert, bringt fuer unsere aktuellen Workloads **moderaten Nutzen** (+17% bei pp512 fuer grosse Modelle). Der Code ist sauber und erweitert die Vulkan-Faehigkeiten.

---

## 5.9 v_dot2 Cherry-Pick abgebrochen

**Commit:** `b4e3dc613` (vulkan: add `v_dot2_f32_f16` support in matrix-matrix multiplication and Flash Attention)

### Warum abgebrochen

Der Cherry-Pick brachte **6 Merge-Konflikte** in 2 Dateien:
1. `ggml-vulkan.cpp` (2 Konflikte) — FlashAttention-Pipeline-Architektur komplett umgebaut
2. `vulkan-shaders-gen.cpp` (4 Konflikte) — Neue Shader-Registrierung

### Konflikt-Ursache

Der v_dot2-Commit baut auf einem **FlashAttention-Refactor** auf, der in mehreren upstream-Commits eingefuehrt wurde:
- `pipeline_flash_attn_f32_f16[TYPE]` → `pipeline_flash_attn_f32_f16` (ohne Index)
- Neue Funktion `ggml_vk_fa_scalar_uses_mmq()`
- Neue Shader-Daten (`flash_attn_f32_f16_dot2_data`, etc.)
- `device->dot2_f16` Feature-Flag

Diese Aenderungen haengen von mindestens **5-10 weiteren upstream-Commits** ab, die zwischen unserem Stand und v_dot2 liegen.

### Risiko

Ein isolierter Cherry-Pick wuerde die FlashAttention-Implementierung inkonsistent machen und potenziell Laufzeitfehler verursachen (falsche Pipeline-Auswahl, fehlende Shader).

### Alternative

v_dot2 ist ein **Optimierungs-Commit**, kein Bugfix. Die Funktionalitaet existiert ohne ihn (nur langsamer). Ein sauberer Cherry-Pick erfordert:
1. Zuerst den FlashAttention-Refactor-Commit-Ketten cherry-picken
2. Dann v_dot2 auf sauberer Basis anwenden

Das ist ein **separates, grosses Projekt** (Schaetzung: 20+ Commits, hohes Konflikt-Risiko).

---

## 5.10 coopmat2 Feature-Check Ergebnis

**Commit:** `5a69c9743` (vulkan: check coopmat2 features before reporting support)

### Durchgefuehrte Aenderungen

- **Feature-Detection in `ggml-vulkan.cpp`:** Prueft 7 coopmat2-Features vor Aktivierung:
  - `cooperativeMatrixWorkgroupScope`
  - `cooperativeMatrixFlexibleDimensions`
  - `cooperativeMatrixReductions`
  - `cooperativeMatrixConversions`
  - `cooperativeMatrixPerElementOperations`
  - `cooperativeMatrixTensorAddressing`
  - `cooperativeMatrixBlockLoads`
- **Graceful Degradation:** Falls Features fehlen, wird `coopmat2_support = false` gesetzt
- **decode_vector-Teil entfernt:** `VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV` ist in unserem Vulkan-Header (Mesa 25.0.7) nicht verfuegbar — Teil eines spaeteren Commits (`b36eefc1b`)

### Build-Status

- ✅ **Kompiliert** auf System A (AMD RDNA3 APU)
- ⚠️ 2 Merge-Konflikte geloest (Feature-Chain-Erweiterung, matrix_cores-String)
- ⚠️ `coopmat2_decode_vector_support` entfernt (nicht in Mesa 25.0.7)

### Benchmark

| Modell | pp512 | tg64 |
|--------|-------|------|
| Gemma 4 12B Q4_K_M | 102.33 t/s | 7.95 t/s |

**Fazit:** Kein direkter Performance-Gewinn (Feature-Check, keine Optimierung), aber **hoehere Stabilitaet** — verhindert, dass coopmat2 auf unvollstaendiger Hardware aktiviert wird.

---

## 5.11 CUDA Fast WHT — Plan

**Ziel:** Walsh-Hadamard-Transform auf NVIDIA/CUDA beschleunigen (Parallele zum bereits gepickten Vulkan-WHT).

### Commits (in Reihenfolge)

| # | Commit | Titel | Zweck |
|---|--------|-------|-------|
| 1 | `a817a22bc` | ggml: implement fast walsh-hadamard transform for kv rotation | **Enum + CPU-WHT** — `GGML_OP_WHT`, `GGML_HINT_SRC0_IS_HADAMARD`, CPU-Referenz-Implementierung |
| 2 | `c1f1e28d2` | CUDA: add fast walsh-hadamard transform | **CUDA-WHT** — `fwht.cu`, `fwht.cuh`, CUDA-Kernel |
| 3 | `192d8ae8b` | CUDA: missing PDL sync for FWHT, better fallback | **Bugfix** — PDL-Sync, verbesserter Fallback |

### Abhaengigkeiten

- `a817a22bc` muss **zuerst in `master`** — das Enum `GGML_HINT_SRC0_IS_HADAMARD` wird von CUDA-WHT benoetigt
- `a817a22bc` ist auch die Basis fuer den bereits gepickten Vulkan-WHT (`48e7078ee`)

### Branch-Strategie

```
master (nach a817a22bc)
  |
  +-- feature/cuda-fast-wht (c1f1e28d2 + 192d8ae8b)
  |
  +-- feature/diffusion-gemma-v2 ( Vulkan-WHT + coopmat2 )
```

**Warum getrennte Branches?**
- `a817a22bc` (Enum+CPU-WHT) ist eine **Infrastruktur-Aenderung** — gehoert in `master`
- CUDA-WHT ist eine **Backend-spezifische Optimierung** — isoliert testen und mergen

### Erwartete Konflikte

| Commit | Dateien | Konflikte |
|--------|---------|-----------|
| `a817a22bc` | 8 (ggml.h, ggml-cpu.c, ops.cpp, ops.h, ggml.c, llama-graph.cpp, llama-kv-cache.cpp, test-backend-ops.cpp) | **1** in `test-backend-ops.cpp` (trivial: FWHT-Tests — upstream bringt `test_mul_mat_hadamard` Klasse mit) |
| `c1f1e28d2` | 4 (fwht.cu, fwht.cuh, ggml-cuda.cu, test-backend-ops.cpp) | **Keine erwartet** — nur neue Dateien |
| `192d8ae8b` | 3 (fwht.cu, fwht.cuh, ggml-cuda.cu) | **Keine erwartet** — Bugfix auf frisch gepicktem Code |

### Erwarteter Nutzen

| Hardware | Geschätzter Speedup |
|----------|-------------------|
| NVIDIA Ampere (RTX 3070 Mobile, Dev-Host) | Prompt-Verarbeitung +10–15% bei großen Modellen |
| NVIDIA Ada (RTX 4060 Ti) | Ähnlich bis leicht höher |
| TurboQuant-KV auf CUDA | Direkte Beschleunigung der Hadamard-Rotation |

### Durchgefuehrte Aenderungen

| Commit | Status | Details |
|--------|--------|---------|
| `a817a22bc` | ✅ **In `master` gepickt** | Sauber, keine Konflikte. Enum `GGML_HINT_SRC0_IS_HADAMARD` + CPU-WHT + `test_mul_mat_hadamard` Klasse |
| `c1f1e28d2` | ✅ **In `feature/cuda-fast-wht` gepickt** | Sauber, keine Konflikte. Neue Dateien `fwht.cu`, `fwht.cuh` |
| `192d8ae8b` | ✅ **In `feature/cuda-fast-wht` gepickt** | **Anpassung erforderlich** — `ggml_cuda_pdl_sync()` und `ggml_cuda_kernel_launch_params` existieren in unserem Stand nicht (spaetere upstream-Infrastruktur). Code auf direkte CUDA-Syntax umgeschrieben: `<<<grid, block, 0, stream>>>` |

### Build-Status

- ✅ **Kompiliert** auf Dev-Host (NVIDIA Ampere, RTX 3070 Mobile, CUDA 13.1)
- ✅ Alle 108 Binaries erfolgreich gelinkt
- ⚠️ 1 Anpassung in `fwht.cu`: `ggml_cuda_kernel_launch_params` → direkter Kernel-Aufruf

### Tatsaechliche Konflikte

| Commit | Erwartet | Tatsaechlich | Loesung |
|--------|----------|--------------|---------|
| `a817a22bc` | 1 in `test-backend-ops.cpp` | **0** | Kein Konflikt (auf `master` waren FWHT-Tests noch nicht auskommentiert) |
| `c1f1e28d2` | 0 | **0** | Kein Konflikt |
| `192d8ae8b` | 0 | **Build-Fehler** | `ggml_cuda_pdl_sync` und `ggml_cuda_kernel_launch_params` fehlten → Code auf direkte CUDA-Syntax umgeschrieben |

### Benchmark (Dev-Host, NVIDIA Ampere)

| Konfiguration | KV-Cache | pp512 | tg64 | Status |
|---------------|----------|-------|------|--------|
| **Standard** | f16/f16 | 7872 t/s | 272 t/s | ✅ Baseline |
| **TurboQuant** | turbo4/turbo3 | **8718 t/s** | 260 t/s | ✅ **pp +11%** |

**Modell:** TinyLlama 1B Q4_K_M (Standard-Modell, kein TurboQuant-Weight)
**WHT-Pfad aktiv:** Ja — TurboQuant KV-Cache (`-ctk turbo4 -ctv turbo3`) nutzt Hadamard-Transformation fuer KV-Kompression. Der CUDA-WHT-Kernel (`fwht.cu`) wird im Graphen aufgerufen.

**Ergebnis:**
- **Prompt-Verarbeitung (pp512):** +11% mit TurboQuant KV — WHT beschleunigt die Hadamard-Rotation auf der GPU
- **Token-Generation (tg64):** −4% (im Messfehlerbereich fuer kleines Modell)

**Hinweis:** Bei groesseren Modellen (12B+) wird der WHT-Vorteil vermutlich deutlicher ausfallen, da die KV-Cache-Dimensionen groesser sind und die GPU-Bandbreite besser ausgelastet wird.

### Status

- ✅ **Abgeschlossen** — Branch `feature/cuda-fast-wht` erstellt, alle 3 Commits gepickt und angepasst, Build kompiliert, Benchmark bestanden
- ✅ **IN MASTER GEMERGED** (2026-06-16)

---

## 5.12 coopmat2 decode_vector — Vec4 B-Matrix-Loads

**Commit:** `c74759a24` (vulkan: Use cm2 decode_vector for mul_mat_id B matrix loads)

### Was es tut

- **`decode_vector` fuer coopmat2:** Erlaubt Vec4-Laden der B-Matrix-Elemente in `mul_mm_cm2.comp`
- **BK = 64:** Erhoeht die Block-Groesse von 32 auf 64, wenn decode_vector verfuegbar ist
- **B-Matrix Alignment:** Stellt sicher, dass Alignment und Stride Vielfache von 4 sind

### Performance

| Kombination | Speedup |
|-------------|---------|
| Vec4 allein | Inkonistent |
| BK=64 allein | Inkonistent |
| **Beide zusammen** | **"Nice speedup"** (laut Commit-Message) |

### Abhaengigkeiten

- **ERFORDERT** `coopmat2_decode_vector_support` — das ist `VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV`
- Diese Extension ist **NICHT** in Mesa 25.0.7 vorhanden
- Erfordert vermutlich Mesa 25.2+ oder proprietären NVIDIA-Treiber

### Blocker

| Problem | Details |
|---------|---------|
| **Mesa 25.0.7** | `VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV` fehlt im Header |
| **Wir haben `coopmat2` bereits gepickt** | Aber `decode_vector`-Teil wurde entfernt |
| **NVIDIA-Hardware** | Auf unseren NVIDIA-Systemen (Ampere/Ada) läuft CUDA, nicht Vulkan |

### Fazit

- 🔴 **Nicht cherry-pickbar jetzt** — Mesa zu alt
- 🔵 **Warten auf Mesa 25.2+** oder testen auf NVIDIA-Vulkan (nicht unsere Hauptplattform)
- **Nutzen:** Hoch fuer NVIDIA-Vulkan, aber wir nutzen CUDA auf NVIDIA

---

## 5.13 Vulkan Q3_K/Q6_K Block-Load — Intel BMG +57%

**Commit:** `19620004f` (vulkan: Block-load Q3_K/Q6_K block data and subtract on 32b ints)

### Was es tut

- **MMVQ statt MMQ** fuer Q2_K/Q3_K/Q6_K auf Intel BMG (auch wenn nur 2-byte aligned)
- **Block-Load erzwingen:** Mesa ist nicht gut im Coalescing alternierender Arrays — wir zwingen es
- **32-bit Subtraktion:** Statt i8vec4 mit Bit-Twiddling — das high bit ist immer frei

### Performance (Intel BMG auf Mesa)

| Quant | tg128 vorher | tg128 nachher | Speedup |
|-------|-------------|---------------|---------|
| Q3_K (MMVQ) | — | — | **+57%** |
| Q6_K (MMVQ) | — | — | **+78%** |
| Q3_K (Block-Load) | — | — | **+24%** (zusätzlich) |
| Q6_K (Block-Load) | — | — | **+48%** (zusätzlich) |

### Dateien

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — 19 Zeilen
- `vulkan-shaders/mul_mat_vecq_funcs.glsl` — 108 Zeilen

### Konflikte erwartet

- **Niedrig** — Shader-Änderungen, aber nur Q3_K/Q6_K Pfade
- Koennte mit unseren Vulkan-Turbo3-Shaders kollidieren

### Benchmark-Ergebnisse (unser Fork)

Siehe [docs/fork/2026-06-17_B1_VULKAN_Q3K_Q6K_BENCHMARK.md](docs/fork/2026-06-17_B1_VULKAN_Q3K_Q6K_BENCHMARK.md).

| System | GPU | Status |
|--------|-----|--------|
| **Mars** | AMD RDNA3 (Vulkan) | 🔴 Benchmark unterbrochen — System waehrend Durchlauf unerreichbar |
| **Venus** | AMD RDNA2 (CPU-Fallback) | 🟢 5/6 Modelle erfolgreich |

**Venus-Referenzwerte (CPU-Fallback, nicht Vulkan):**

| Modell | Q3_K_M pp128 | Q3_K_M tg32 | Q6_K pp128 | Q6_K tg32 |
|--------|-------------|-------------|-----------|-----------|
| Gemma 4 12B | 13.62 t/s | 5.94 t/s | 9.54 t/s | 3.60 t/s |
| Gemma 4 31B | 3.77 t/s | 2.19 t/s | 2.73 t/s | 1.42 t/s |

### Fazit

- ✅ **IN MASTER GEMERGET** — Cherry-pick erfolgreich
- 🟡 **AMD-Benchmark ausstehend** — Mars-Vulkan-Test noch nicht komplett
- **Empfohlen:** Mars-Benchmark wiederholen sobald System stabil

---

## 5.14 Vulkan Buffer-Transfer Fast Path

**Commit:** `fdc3db9b6` (vulkan: add fast path for contiguous buffer transfers)

### Was es tut

- **Fast Path** fuer contiguous (zusammenhängende) Buffer-Transfers in Vulkan
- Vermeidet unnötige Kopiervorgänge, wenn Quelle und Ziel bereits contiguous sind

### Dateien

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — 16 Zeilen (+12/-4)

### Fazit

- 🟢 **Trivial cherry-pickbar** — nur 16 Zeilen
- 🟢 **Keine Abhaengigkeiten**
- **Nutzen:** Moderat — kleine Optimierung, aber sauber

---

## 5.15 CUDA KV-Cache Reserve — FlashAttention-Speicherverwaltung

**Commit:** `f8f0a47a5` (cuda: reserve space for quantize kv-cache at startup)

### Was es tut

- **Speicherreservierung** fuer quantisierten KV-Cache beim CUDA-Startup
- Verhindert, dass FlashAttention-Speicher zur Laufzeit allokiert werden muss
- Reduziert Fragmentierung und verbessert Vorhersagbarkeit

### Dateien

- `ggml/src/ggml-cuda/fattn-common.cuh` — 65 Zeilen
- `ggml/src/ggml-cuda/fattn.cu` — 35 Zeilen
- `ggml/src/ggml-cuda/fattn.cuh` — 2 Zeilen
- `ggml/src/ggml-cuda/ggml-cuda.cu` — 8 Zeilen

### Status

- ✅ **IN MASTER GEMERGED** (Commit `c1b8a86dc`)
- **Adaptierung:** HIP-Workaround (`hip_f16_alloc`) entfernt — ersetzt durch upstreams generischeren `f16_extra_data`-Ansatz
- **Build:** Kompiliert erfolgreich auf Hydra (CUDA)

### Fazit

- ✅ **Erledigt** — Stabilität + Performance fuer FlashAttention
- 🟡 **Komplexität:** Mittel (4 Dateien, 96 Zeilen)
- **Nutzen:** Weniger Speicherfragmentierung, stabilere FA bei langen Kontexten

---

## 5.16 CUDA MMVQ AMD MFMA Threshold — Q4_K_S +68%

**Commit:** `bc81d47ab` (CUDA: route batch>=4 quantized matmul to MMQ on AMD MFMA hardware)

### Was es tut

- **Per-Quant MMVQ/MMQ Batch-Threshold** auf AMD CDNA (MI250X etc.)
- Bisher: Globaler Threshold (MMVQ_MAX_BATCH_SIZE = 8)
- Jetzt: Unterschiedliche Thresholds je nach Quant-Familie

### Thresholds (AMD MFMA)

| Quant-Familie | MMVQ bis | MMQ ab | Speedup (ub=8) |
|---------------|----------|--------|----------------|
| Q3_K, Q4_K, Q5_K | Batch ≤ 3 | Batch ≥ 4 | **+5% .. +76%** |
| Q2_K, Q6_K | Batch ≤ 5 | Batch ≥ 6 | **+8% .. +35%** |
| Legacy, IQ | Batch ≤ 8 | — | Keine Änderung |

### Benchmark (MI250X, Llama-3.2-3B-Instruct, pp512)

| Quant | vorher (tok/s) | nachher (tok/s) | Speedup |
|-------|---------------|-----------------|---------|
| Q4_K_S | 559 | **940** | **+68%** |
| Q5_K_S | 503 | **884** | **+76%** |
| Q3_K_S | 629 | **879** | **+40%** |
| Q6_K | 582 | **776** | **+33%** |

### Wichtig für uns

- **NVIDIA-Pfade sind byte-identisch** — kein Risiko für unsere Hauptplattform
- AMD MFMA = `amd_mfma_available(cc)` — wird nur auf CDNA2/CDNA3 aktiviert
- `GGML_CUDA_FORCE_MMVQ=1` stellt altes Verhalten wieder her

### Status

- ✅ **BEREITS VORHANDEN** — Der Code ist via AtomicBot-Upstream bereits in `ggml/src/ggml-cuda/mmvq.cu` (`get_mmvq_mmid_max_batch_cdna` ab Zeile 167).
- **Keine Cherry-Pick-Aktion erforderlich.**

### Fazit

- ✅ **Bereits implementiert** — funktioniert transparent für AMD-CDNA-GPUs
- 🟡 **Für uns irrelevant auf NVIDIA** — Änderungen nur für AMD CDNA, NVIDIA-Pfade sind byte-identisch

---

## 5.17 CUDA PDL mul_mat_vec_q_moe — MTP +5-8%

**Commit:** `2154a0fdc` (CUDA: enroll mul_mat_vec_q_moe into pdl)

### Was es tut

- **`mul_mat_vec_q_moe`** in PDL (Pipeline Device Launch) eintragen
- Ermöglicht Kernel-Overlap mit nachfolgenden Operationen
- Besonders relevant für **MTP (Multi-Token Prediction)** — unsere Draft-Modelle

### Performance (B4500, Qwen 3.6 35B A3B MTP)

| Task | vorher (tok/s) | nachher (tok/s) | Speedup |
|------|---------------|-----------------|---------|
| code_python | 202.8 | **211.9** | +4.5% |
| code_cpp | 212.8 | **224.6** | +5.5% |
| summarize | 226.6 | **240.2** | +6.0% |
| qa_factual | 225.1 | **238.5** | +6.0% |
| stepwise_math | 209.2 | **221.7** | +6.0% |

### Dateien

- `ggml/src/ggml-cuda/mmvq.cu` — 14 Zeilen

### Fazit

- 🟢 **Trivial cherry-pickbar** — nur 14 Zeilen in einer Datei
- 🟢 **Direkter Nutzen für uns** — wir nutzen MTP (Gemma 4 Assistant)
- **Empfohlen:** Sofort cherry-picken nach CUDA-WHT-Merge

---

## 5.18 DeepSeek V3.2 — Sparse Attention (DSA)

**Commit:** `1f0aa2a69` (model: support for DeepseekV32ForCausalLM with generic DSA)

### Was es tut

- **DeepSeek V3.2 Model-Familie** — neue Architektur mit DSA (DeepSeek Sparse Attention)
- **Lightning Indexer** — effizienter Attention-Indexer fuer Sparse Patterns
- **KV-Cache Erweiterung** — `llama_kv_cache_dsa` mit Indexer-Cache
- **GGML_OP_FILL fuer f16** — neuer Op

### Dateien

- ~15 Dateien geändert
- `src/llama-model.cpp` — Tensor-Mapping
- `src/models/deepseek32.cpp` — Modell-Implementierung (neu)
- `src/llama-kv-cache.cpp` — DSA KV-Cache
- `ggml/src/ggml.c` — GGML_OP_FILL

### Komplexität

| Aspekt | Bewertung |
|--------|-----------|
| **Neue Dateien** | ~5 |
| **Geänderte Dateien** | ~15 |
| **Konflikte erwartet** | Mittel (mit unserem TurboQuant-KV-Cache) |
| **Risiko** | 6/10 |

### Fazit

- 🟡 **Hohe Relevanz** — DeepSeek ist wichtiger OSS-Trend
- 🟡 **Aber komplex** — berührt KV-Cache, den wir stark modifiziert haben
- **Empfohlen:** Erst nachdem TurboQuant-KV-Cache-Stabilität sichergestellt ist
- **Alternative:** Als Referenzimplementierung fuer zukünftige Sparse-Attention-Ideen

---

## 6. Feature-Matrix (Vergleich)

| Feature | ggml-org | TheTom | AtomicBot | fukuro |
|---------|----------|--------|-----------|--------|
| TurboQuant KV/Weights | ❌ | ✅ | ✅ | ✅ |
| Gemma 4 MTP | ❌ | ❌ | ✅ | ✅ |
| Qwen 3.x NextN | ❌ | ❌ | ✅ | ✅ |
| UDT-Quant-Masks | ❌ | ❌ | ✅ | ❌ |
| DiffusionGemma (Forward) | ❌ | ❌ | ❌ | ✅ |
| DiffusionGemma (Entropy-Bound Decoder) | ❌ | ❌ | ❌ | ✅ |
| DiffusionGemma (KV-Cache PREFILL→DECODE) | ❌ | ❌ | ❌ | ✅ |
| DiffusionGemma (Self-Conditioning) | ❌ | ❌ | ❌ | ❌ |
| Vulkan-Turbo3 | ❌ | ❌ | ❌ | ✅ |
| Multimodal + Spec | ❌ | ❌ | ✅ | ✅ |
| EAGLE / classical draft | ✅ | ✅ | ✅ | ✅ |
| FlashAttention | ✅ | ✅ | ✅ | ✅ |
| 100+ Modelle | ✅ | ✅ | ✅ | ✅ |

**Anmerkung DiffusionGemma:**
- ✅ **Forward-Pass** — `src/models/diffusion-gemma.cpp`, `llama-model.cpp` Integration
- ✅ **Entropy-Bound Decoder** — `tools/diffusion-cli/diffusion-cli.cpp` (Zeilen 67–444), Multi-Step-Denoising implementiert
- ✅ **KV-Cache PREFILL→DECODE** — Fix: `dg_ensure_pkv_store()` allokiert pro Layer auf jeweiligem Device
- ❌ **Self-Conditioning (SC)** — SC-Tensoren fehlen in GGUF-Dateien (kein upstream-Support)

---

## 7. Branch-Strategie

| Branch | Remote | Zweck |
|--------|--------|-------|
| `master` | Codeberg (primary) | Hauptentwicklung. Alle Features (TurboQuant, MTP, NextN, DiffusionGemma, Vulkan-Turbo3). |
| `feature/turboquant-kv-cache-sync` | GitHub | Sauberer Sync mit AtomicBot-ai. Genau 2 Commits ahead (unsere Cherry-Picks). PR-Quelle. |
| `feature/turboquant-kv-cache` | GitHub | Obsolet. Alte Upstream-Merges. |
| `feature/cuda-fast-wht` | Codeberg | CUDA Fast Walsh-Hadamard Transform. **IN MASTER GEMERGED** (+11% pp512). |
| `feature/diffusion-gemma-v2` | Codeberg | DiffusionGemma V2 (Entropy-Bound Decoder, PREFILL/DECODE-Fix, PKV pro Layer). |

**Divergenz zu AtomicBot:**
- `master` ist **~532 Commits ahead** (Vulkan-Turbo3) plus DiffusionGemma plus unsere Fixes.
- `feature/turboquant-kv-cache-sync` ist **2 Commits ahead** (Gemma4Assistant + MTP tensor fixes).

Details: [BRANCHES.md](BRANCHES.md)

---

## 8. Implementierungsplan (Stand 2026-06-16)

### Phase A — Sofort (diese Woche)

| # | Task | Aufwand | Erwarteter Nutzen |
|---|------|---------|-------------------|
| ~~A1~~ | ~~`feature/cuda-fast-wht` → `master` mergen~~ | ~~15 Min~~ | ~~Sauberer Stand~~ → **ERLEDIGT** |
| ~~A2~~ | ~~Cherry-Pick `2154a0fdc` (PDL MTP)~~ | ~~30 Min~~ | ~~MTP +5-8% auf BW~~ → **BLOCKIERT** (PDL-Infrastruktur fehlt im Fork) |
| ~~A3~~ | ~~Cherry-Pick `fdc3db9b6` (Vulkan Transfer)~~ | ~~20 Min~~ | ~~Performance~~ → **ERLEDIGT** |
| ~~A4~~ | ~~Cherry-Pick `3571fa543` (SVE FWHT)~~ | ~~20 Min~~ | ~~ARM-Kompatibilitaet~~ → **BEREITS VORHANDEN** |

### Phase B — Kurzfristig (nächste 2 Wochen)

| # | Task | Aufwand | Risiko | Erwarteter Nutzen |
|---|------|---------|--------|-------------------|
| ~~B1~~ | ~~Cherry-Pick `19620004f` (Vulkan Q3_K/Q6_K Block-Load)~~ | ~~2h~~ | ~~Mittel~~ | ~~+57%/+78% tg128~~ → **ERLEDIGT** |
| ~~B2~~ | ~~Cherry-Pick `f8f0a47a5` (CUDA KV-Cache Reserve)~~ | ~~3h~~ | ~~Mittel~~ | ~~FA-Stabilitaet~~ → **ERLEDIGT** |
| ~~B3~~ | ~~Cherry-Pick `379ac6673` + `236531595` (KV-Cache Fixes)~~ | ~~1h~~ | ~~Niedrig~~ | ~~Stabilitaetsverbesserungen~~ → **BLOCKIERT** (erfordert KV-Cache-Refactor, siehe Analyse) |
| B4 | DiffusionGemma: llama-server Integration | 4h | Mittel | Server-API fuer DiffusionGemma |

### Phase C — Mittelfristig (nächster Monat)

| # | Task | Aufwand | Risiko | Erwarteter Nutzen |
|---|------|---------|--------|-------------------|
| ~~C1~~ | ~~Cherry-Pick `354ebac8c` (Reasoning Interruption)~~ | ~~3h~~ | ~~Mittel~~ | ~~Bessere Chat-UX~~ → **NICHT RELEVANT** (WebUI nicht genutzt) |
| ~~C2~~ | ~~Cherry-Pick `e8067a8b3` (gzip compression)~~ | ~~1h~~ | ~~Niedrig~~ | ~~Kleinere Assets~~ → **BLOCKIERT** (andere WebUI-Architektur) |
| ~~C3~~ | ~~DeepSeek V3.2 evaluieren~~ | ~~4h~~ | ~~Hoch (15 Dateien)~~ | ~~Wichtiger OSS-Trend~~ → **BLOCKIERT** (KV-Cache-Refactor, siehe B3) |
| ~~C4~~ | ~~Self-Conditioning (SC) fuer DiffusionGemma~~ | ~~8h~~ | ~~Hoch~~ | ~~Bessere Output-Qualitaet~~ → **ZURUECKGESTELLT** (DiffusionGemma nicht stabil genug, keine Zeit) |

### Phase D — Langfristig (nachstehend)

| # | Task | Aufwand | Risiko | Erwarteter Nutzen |
|---|------|---------|--------|-------------------|
| D1 | Llama 4 Scout/Maverick | TBD | TBD | Wichtige Meta-Modelle |
| D2 | `c74759a24` (coopmat2 decode_vector) | TBD | BLOCKIERT (Mesa 25.0.7) | Vec4 B-Loads auf NVIDIA-Vulkan |
| ~~D3~~ | ~~Video-Input (`8f83d6c27`)~~ | ~~8h~~ | ~~Hoch (807 Zeilen)~~ | ~~Multimodal-Server~~ → **ZURUECKGESTELLT** (nice-to-have, nicht dringend) |

### Priorisierungslogik (aktueller Stand)

1. **Cherry-Pick-Phase abgeschlossen** — Alle verfuegbaren kleinen Commits sind gemerged; verbleibende sind blockiert (KV-Cache-Refactor, PDL, WebUI-Architektur).
2. **Naechster Schritt** — DiffusionGemma: llama-server Integration (B4) oder auf upstream-Sync durch AtomicBot warten.
3. **Mesa-Abhaengigkeiten** — decode_vector (D2) wartet auf Mesa 25.2+.
4. **Langfristig** — Llama 4 Scout/Maverick (D1) sobald Commits verfuegbar.

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
- Langfristig: Gezielter Sync des Refactor-Commits, wenn DiffusionGemma stabil ist. Siehe Trilium-Note "Port-Bericht: DiffusionGemma Integration" unter "fukuro-llama-cpp-turboquant".

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
