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
| **ggml-org (Root-Upstream)** | `git@github.com:ggml-org/llama.cpp.git` | Original-Upstream. |

**Empfohlener Workflow:**
```bash
# Standard-Entwicklung
origin = codeberg.org:fukuro/... (push/pull)

# Sync mit AtomicBot
upstream = github.com:AtomicBot-ai/... (fetch only)
git fetch upstream
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
| **KV-Kompression** | ~5.1× via `-ctk turbo3 -ctv turbo3`, ~3.8× via `-ctk turbo4 -ctv turbo4` |
| **Architektur** | Monolithisch (`struct llama_model` mit Switch-Statements). |

### 3.3 AtomicBot-ai/atomic-llama-cpp-turboquant

| Aspekt | Hinzugefuegt / Veraendert |
|--------|--------------------------|
| **Gemma 4 MTP** | Multi-Token Prediction mit `gemma4_assistant`-Drafter. Single-context, cross-attention in target KV. +30–50 % Throughput. |
| **Qwen 3.x NextN** | Shared-model spekulative Decodierung. Kein zweites mmap. +24–36 % tps (MoE), +5–7 % (dense). |
| **UDT-Quantisierung** | Tensor-type Masks: NextN/MTP → `Q8_0`, attn Q/K → `Q6_K`, paart mit TurboQuant3 KV. |
| **Multimodal + Spec** | `--mmproj` gleichzeitig mit `mtp`/`nextn`/`eagle3` auf einem Slot. |
| **GGUF-Conversion** | Erweiterte `convert_hf_to_gguf.py` fuer Gemma4Assistant, NextN-Tensoren. |
| **Architektur** | `llama_model_base`-Klassenhierarchie (sehe Sync-Merge 2026-06-19). |

### 3.4 fukuro-llama-cpp-turboquant (unser Fork) — Sync-Stand 2026-06-22

| Aspekt | Status |
|--------|--------|
| **Architektur** | ✅ `llama_model_base`-Klassenhierarchie (via Sync-Merge mit AtomicBot 2026-06-19). Die monolithische Aera ist beendet. |
| **DiffusionGemma** | ✅ `llama_model_diffusion_gemma : public llama_model_base` (`src/models/models.h` Zeile 851, `src/models/diffusion-gemma.cpp` 747 Zeilen). Forward-Pass und Entropy-Bound Decoder vollstaendig. Limitierung: Self-Conditioning (SC-Tensoren fehlen in GGUFs). **Offen:** gguf-py Registrierung (`MODEL_ARCH.DIFFUSION_GEMMA`, `add_diffusion_*` Funktionen) fehlt. |
| **Vulkan TurboQuant** | ✅ turbo3 (~5.1x) und turbo4 (~3.8x) KV-Cache mit FlashAttention auf Vulkan. TurboQuant FA-Pipelines werden via AtomicBot's generischem Shader-Ansatz behandelt. |
| **Vulkan-Optimierungen** | ✅ Q3_K/Q6_K Block-Load (+57%/+78% tg128 Intel BMG), iq1 shared-memory, host-memory Lock-Kontention. |
| **Vulkan APU GPU-Hang Fix** | ✅ `nodes_per_submit=10` für UMA-Geräte (Issue #21724). Behebt GPU-Hangs bei >188k Kontext und >16k Prompts. |
| **Gemma 4 12B Assistant** | ✅ Vollstaendig implementiert (`src/models/gemma4-assistant.cpp`). |
| **CUDA KV-Cache Reserve** | ✅ Via AtomicBot-Upstream (dynamische Allokation, kein expliziter Fork-Commit mehr). |
| **DOX-Framework** | ✅ AGENTS.md-Vertraege in allen Verzeichnisbaeumen. |

---

## 4. Architektur-Refactor: Stand und Bedeutung

### 4.1 Faktische Lage (Stand 2026-06-22)

| Repository | Refactor `994118a18` in `master`? | Architektur | `llama-model.h` Zeilen |
|------------|--------------------------------|-------------|----------------------|
| `ggml-org/llama.cpp` | ✅ Ja | Klassen-Hierarchie (`llama_model_base`) | 735 |
| `TheTom/llama-cpp-turboquant` | ❌ Nein | Monolithisch (`struct llama_model`) | 643 |
| `AtomicBot-ai/atomic-llama-cpp-turboquant` | ✅ Ja (sehe Sync 2026-06-19) | Klassen-Hierarchie (`llama_model_base`) | 598 |
| `fukuro-llama-cpp-turboquant` | ✅ Ja (via Sync-Merge 2026-06-23) | Klassen-Hierarchie (`llama_model_base`) | 688 |

**Wichtig:** Der Sync-Merge mit AtomicBot (2026-06-19) hat den Refactor in unseren Fork integriert. Alle Fork-Features (TurboQuant, MTP, NextN, DiffusionGemma) wurden auf die neue Klassenstruktur portiert.

### 4.2 Warum der Refactor so einflussreich ist

Der Commit (Xuan-Son Nguyen, upstream PR #22004) verschiebt `load_hparams` und `load_tensors` aus der zentralen `llama-model.cpp` in **pro-Modell-Klassen**:

- Vorher: Ein `struct llama_model` mit gigantischem Switch-Statement fuer alle 100+ Architekturen.
- Nachher: Abstrakte `llama_model_base`-Klasse, von der jede Architektur erbt (`class llm_build_<arch> : public llama_model_base`).

Unsere Fork-Features wurden erfolgreich portiert:
- **TurboQuant:** KV-Cache-Typ-Registrierung und Tensor-Mapping in `load_tensors` der jeweiligen Modell-Klasse.
- **Gemma 4 MTP:** `mtp_assistant` in `gemma4-assistant.cpp` als eigene Modell-Klasse.
- **Qwen NextN:** `nextn_predict_layers` und Shared-Model-Draft in `qwen35-nextn.cpp`.
- **DiffusionGemma:** `llama_model_diffusion_gemma` mit eigenem `load_arch_hparams`, `load_arch_tensors`, `build_arch_graph`.

### 4.3 Status

**Szenario B eingetreten:** AtomicBot hat den `llama_model_base`-Refactor übernommen. Unser Sync-Merge übernimmt ihn automatisch. Alle Fork-Features wurden portiert. Die monolithische Aera ist beendet.

### 4.4 Empfohlene Strategie — Update 2026-06-22

**Aktuell:**
- Sync-Merge Branch `feature/sync-atomicbot-2026-06-19` verifizieren und nach `master` mergen.
- DiffusionGemma gguf-py Registrierung fehlt (siehe §3.4).
- tg32-Tests auf AMD-APU laufen (Pipeline-Cache wird aufgebaut).

**Kurzfristig:**
- Nach Merge: DiffusionGemma Server-Integration (B4).
- Upstream-Features gezielt cherry-picken (siehe §5.2).

**Mittelfristig:**
- Llama 4 Scout/Maverick sobald Commits verfügbar.
- coopmat2 decode_vector sobald Mesa 25.2+ verfügbar.
- DeepSeek V3.2 erfordert KV-Cache-Refactor (komplex).

---

## 5. Was uns von upstream fehlt

> Update 2026-06-22: Nach dem Sync-Merge mit AtomicBot sind die meisten Features integriert. Verbleibende Differenzen zu upstream (ggml-org) sind in §5.2 gelistet.

### 5.1 SYCL — Referenz

**SYCL** ist Intels offener Standard fuer heterogene Berechnung, in llama.cpp als Intel-GPU-Backend eingesetzt. **Relevanz fuer uns: Niedrig** — unsere Hardware ist NVIDIA (CUDA) und AMD (Vulkan). Kein Intel Arc im Einsatz.

### 5.2 Noch fehlende Features (upstream)

| Feature | Status | Grund |
|---------|--------|-------|
| **Llama 4** Scout / Maverick | ⏳ Offen | Wichtige Meta-Modelle, noch kein Commit upstream |
| **DeepSeek V3.2** DSA | ❌ Blockiert | KV-Cache-Refactor erforderlich |
| **coopmat2 decode_vector** (`c74759a24`) | ⏳ Blockiert | Mesa 25.0.7 Header fehlt, warten auf 25.2+ |
| **GL_NV_cooperative_matrix_decode_vector** (`b36eefc1b`) | ⏳ Offen | Schnelleres MatMul auf NVIDIA-Vulkan |
| **Video-Input** (`8f83d6c27`) | ⏳ Offen | Server mtmd-APIs gekoppelt |
| **MXFP4 / NVFP4** | ⏳ Offen | Neue GGML-Typen + alle Backends |
| **Real-time Reasoning Interruption** (`354ebac8c`) | ⏳ Offen | Bessere Chat-UX |
| **Gemma 4 Audio-Fixes** (`e8023568d`, `e3ba22d6c`) | ✅ Via Sync | Audio-Modality fuer Gemma 4 |

### 5.3 Neue Modell-Architekturen (upstream hat sie, wir nicht)

| Modell | Relevanz | Anmerkung |
|--------|----------|-----------|
| **llama4** | 🔴 Hoch | Llama 4 Scout / Maverick |
| **deepseek32** | 🔴 Hoch | DeepSeek 3.2 — blockiert (KV-Cache-Refactor) |
| **deepseek2ocr** | 🟡 Mittel | OCR-Modus |
| **mistral4** | 🟡 Mittel | Mistral-Nachfolger |
| **mamba2** | 🟡 Mittel | SSM-Architektur |
| **eagle3** | 🟢 Niedrig | Draft-Modell — wir haben MTP/NextN |
| **cohere2-MoE** | 🟢 Niedrig | Nicht auf unserer Roadmap |
| **granite-moe** | 🟢 Niedrig | IBM |
| **mellum** | 🟢 Niedrig | Nische |
| **nomic-bert / jina-bert-v2/v3** | 🟢 Niedrig | Embedding-Modelle |
| **glm-dsa / lfm2moe / mimo2 / minicpm / nemotron-h-moe** | 🟢 Niedrig | Nischen-Modelle |

### 5.4 Cherry-Pick-Historie (abgeschlossen)

Alle Cherry-Picks wurden vor dem Sync-Merge abgeschlossen oder blockiert. Detaillierte Analyse in `docs/fork/`.

| Commit | Feature | Status |
|--------|---------|--------|
| `48e7078ee`, `e82beaa60` | Vulkan-WHT | ✅ In master (+17% pp512 12B) |
| `5a69c9743` | coopmat2 Feature-Check | ✅ In master (Stabilitaet) |
| `a817a22bc`+`c1f1e28d2` | CUDA Fast WHT | ✅ In master (+11% pp512 TurboQuant) |
| `19620004f` | Vulkan Q3_K/Q6_K Block-Load | ✅ Via Sync (+57%/+78% tg128 Intel BMG) |
| `fdc3db9b6` | Vulkan Buffer-Transfer | ✅ Via Sync |
| `f8f0a47a5` | CUDA KV-Cache Reserve | ✅ Via Sync (AtomicBot-Upstream) |
| `bc81d47ab` | CUDA MMVQ AMD MFMA | ✅ Via Sync (nur AMD CDNA) |
| `e8023568d`, `e3ba22d6c` | Gemma 4 Audio-Fixes | ✅ Via Sync |
| `6e093b80e` | BFloat16 FA | ❌ Nicht nutzbar (RDNA3/Vega, Mesa 25.2 nur GFX12+) |
| `b4e3dc613` | v_dot2_f32_f16 in FA | ❌ Abgebrochen (FA-Refactor-Abhaengigkeit) |
| `c74759a24` | coopmat2 decode_vector | ⏳ Blockiert (Mesa 25.0.7) |
| `2154a0fdc` | CUDA PDL mul_mat_vec_q_moe | ❌ Blockiert (PDL-Infrastruktur fehlt) |
| `1f0aa2a69` | DeepSeek V3.2 DSA | ❌ Blockiert (KV-Cache-Refactor) |
| `3571fa543` | CPU SVE FWHT Runtime Width | ✅ Bereits vorhanden |
| `379ac6673` | KV-Cache: avoid copies | ❌ Blockiert (KV-Cache-Refactor) |
| `236531595` | KV-Cache: SWA checkpoints | ❌ Blockiert (KV-Cache-Refactor) |
| `d6d0ce821` | Vulkan iq1 shared memory | ✅ Via Sync |
| `bef69f130` | Vulkan host memory lock contention | ✅ Via Sync |
| `20f5994`+`1163cb3`+`5f83fbb` | thecodacus MoE Pinning+Prefetch | ✅ In feature/thecodacus-pinning (+95% pp2048 MoE Offload, GTX 1070) |

### 5.5 Irrelevante Upstream-Bereiche

| Bereich | Grund |
|---------|-------|
| **Metal** | Keine Apple-Hardware |
| **SYCL** | Keine Intel Arc / oneAPI-Hardware |
| **EAGLE3** | Wir fokussieren auf Gemma 4 MTP + Qwen NextN |
| **Hexagon / WebGPU / OpenCL / RISC-V** | Keine entsprechende Hardware |
| **Tensor-Parallelism** | Nur Single-GPU-Systeme |

---

## 6. Feature-Matrix (Vergleich)

| Feature | ggml-org | TheTom | AtomicBot | fukuro |
|---------|----------|--------|-----------|--------|
| TurboQuant KV/Weights | ❌ | ✅ | ✅ | ✅ |
| Gemma 4 MTP | ❌ | ❌ | ✅ | ✅ |
| DiffusionGemma | ❌ | ❌ | ❌ | ✅ |
| Vulkan TurboQuant FA | ❌ | ✅ | ✅ | ✅ |
| Vulkan APU GPU-Hang Fix | ❌ | ❌ | ❌ | ✅ |
| Qwen 3.x NextN | ❌ | ❌ | ✅ | ✅ |
| UDT-Quantisierung | ❌ | ❌ | ✅ | ✅ |
| DOX-Framework | ❌ | ❌ | ❌ | ✅ |
| `llama_model_base`-Refactor | ✅ | ❌ | ✅ | ✅ (via Sync) |

---

## 7. Branch-Strategie

| Branch | Remote | Zweck |
|--------|--------|-------|
| `master` | Codeberg (primary) | Hauptentwicklung (vor Sync-Merge). |
| `feature/sync-atomicbot-2026-06-19` | lokal | Sync-Merge mit AtomicBot 2026-06-19. Sauber als Squash-Commit auf master neu aufgebaut (Original-Branch hatte korrupte Git-Objekte). Neuer Haupt-Branch nach Verifikation. |
| `feature/diffusion-gemma-v2` | Codeberg | DiffusionGemma V2 (Entropy-Bound Decoder, PREFILL/DECODE-Fix, PKV pro Layer). |

**Nach Sync-Merge:** `feature/sync-atomicbot-2026-06-19` wird nach `master` gemerged. Die alten Branches sind gelöscht.

Details: [BRANCHES.md](BRANCHES.md)

---

## 8. Sync- und PR-Workflow

### Saubere Aenderungen an AtomicBot-ai

1. Auf `feature/sync-atomicbot-2026-06-19` rebasen.
2. Neue Aenderung auf einem Feature-Branch entwickeln.
3. Gegen `feature/sync-atomicbot-2026-06-19` rebasen.
4. PR an `AtomicBot-ai/atomic-llama-cpp-turboquant` erstellen.

### Upstream-Sync (ggml-org)

- Der Sync-Merge mit AtomicBot 2026-06-19 hat den `llama_model_base`-Refactor integriert. Wir sind jetzt architektonisch kompatibel mit upstream.
- Einzelne Features (neue Modell-Unterstuetzungen, Bugfixes) koennen gezielt cherry-picked werden. Siehe [§5.2](#52-noch-fehlende-features-upstream) fuer die priorisierte Liste.
- Langfristig: Upstream-Syncs werden durch die Klassenhierarchie einfacher — neue Modelle koennen als eigene Klasse hinzugefuegt werden.

### Bekanntes Problem: AtomicBot Remote-Korruption

Die AtomicBot-Remote-Branches `feature/turboquant-kv-cache` und `sync/upstream-2026-06-19` haben fehlende Git-Objekte (Commit `cc59fcb6a` und 20+ weitere). Dies ist ein Problem auf AtomicBot's Seite (wahrscheinlich durch Rebase/Force-Push). Unser Sync-Merge wurde als sauberer Squash-Commit neu aufgebaut, um dies zu umgehen.

---

## 9. Historie und Attribution

| Beitragender | Rolle | Repository |
|--------------|-------|------------|
| ggml-org / ggerganov | Original llama.cpp, ggml | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| TheTom / @TheTom | TurboQuant KV-Kompression | [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) |
| AtomicBot-ai / AtomicChat | Gemma 4 MTP, Qwen NextN, UDT, Multimodal+Spec | [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) |
| fukuro | DiffusionGemma-Port, Vulkan-Turbo3, DOX-Framework | [codeberg.org/fukuro/...](https://codeberg.org/fukuro/fukuro-llama-cpp-turboquant) |

---

## 10. Git-Hygiene — Quota-Management

### Problem

Codeberg hat Repository-Quota-Limits (750 MiB Git-Storage gesamt). Der Fork brachte
~6660 Tags und große Binärdateien aus upstream mit, die nicht benoetigt werden und
massiv Quota verbrauchen.

### Aufraeum-Aktionen

#### 2026-07-02: Tag/Branch-Cleanup

- 6659 lokale Tags geloescht (upstream build-numbers, commit-hash tags, CI prebuilds)
- 11 obsolete Remote-Branches auf Codeberg geloescht
- `atomictemp` remote entfernt (historisch)
- Force-Push master (DFlash-Migration auf cherry-dflash-Basis)
- DFlash-Migration als `archive/cherry-dflash` Branch archiviert, master auf alte Basis zurueckgesetzt

#### 2026-07-04: History-Bereinigung (filter-repo + Repo-Recreate)

`git filter-repo` entfernte 5 Pfade die nur in der History existierten (nicht im HEAD):

| Datei | Kumulierte Groesse | Versionen |
|-------|---------------------|-----------|
| `tools/server/public/bundle.js` | 177.3 MB | 31 |
| `tools/server/public/index.html.gz` | 144.9 MB | 100 |
| `ggml-vulkan-shaders.hpp` (root) | 63.4 MB | 13 |
| `examples/server/public/index.html.gz` | 30.4 MB | 26 |
| `ggml/src/ggml-vulkan-shaders.hpp` | 16.8 MB | 2 |
| **Summe** | **~432 MB** | 172 |

Anschliessend Repo-Recreate (Loeschen + Neu-Anlegen auf Codeberg) fuer sofortige
Quota-Freigabe (ohne 30-Tage Grace-Periode).

### Quota-Entwicklung

| Zeitpunkt | Git gesamt | Repo-Groesse | Aktion |
|-----------|------------|--------------|--------|
| 2026-07-02 vor Cleanup | 527.6 MB | 426.7 MB | — |
| 2026-07-02 nach Tag/Branch-Cleanup | ~527.6 MB | ~426.7 MB | Tags waren nur lokal |
| 2026-07-04 nach Repo-Loeschung | 100.9 MB | 0 MB | Repo geloescht |
| 2026-07-04 nach Neu-Anlage | 228.3 MB | 127.4 MB | Bereinigte History gepusht |

**Einsparung: 527.6 MB → 228.3 MB = 299.3 MB (57% Reduktion)**

### Verbleibende Refs

| Ref | Typ | Zweck |
|-----|-----|-------|
| `master` | Branch | Hauptentwicklung (AtomicBot-Basis) |
| `archive/cherry-dflash` | Branch | DFlash-Migration (archiviert) |
| `archive/sync-atomicbot-2026-06-19` | Tag | Archiv-Checkpoint |
| `origin` | Remote | Codeberg (primary) |
| `github` | Remote | GitHub (mirror) |
| `tomtemp` | Remote | TheTom (fetch-only) |
| `upstream` | Remote | ggml-org (fetch-only) |

---

*Letzte Aktualisierung: 2026-07-04 (Git-Hygiene: filter-repo + Repo-Recreate)*
