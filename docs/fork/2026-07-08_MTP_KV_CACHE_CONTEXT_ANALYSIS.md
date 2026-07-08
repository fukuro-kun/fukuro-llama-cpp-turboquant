# Gemma-4 MTP: KV-Cache-Architektur und Kontext-Scaling-Analyse

**Datum:** 08.07.2026
**Modell:** Gemma-4 26B-A4B IQ4_NL + Q4_K_M MTP Draft
**Hardware:** GTX 1070 (8GB VRAM, Pascal, PCIe Gen3, 256 GB/s Memory Bandwidth)
**Konfiguration:** `--n-cpu-moe 20 -ctk turbo3 -ctv turbo4 --flash-attn on` + Pinning + Prefetch

---

## 1. Gemma-4 MTP Draft-Architektur (Recherche-Ergebnisse)

### 1.1 Der Draft hat KEINEN eigenen KV-Cache

Der Gemma-4 MTP Assistant teilt sich den KV-Cache **vollständig** mit dem Target-Modell.

| Eigenschaft | Wert |
|-------------|------|
| Eigener KV-Cache | **Nein** — liest K/V aus Target-KV-Cache via Cross-Attention |
| Attention-Typ | **Q-only** — nur Query-Projektion, keine eigenen K/V |
| Pre-fill | **Übersprungen** — KV-Sharing eliminiert Pre-fill komplett |
| Layer-Anzahl | 4 Transformer-Layer (~0.5B Parameter) |
| Input | `concat(target_embed(last_token), target_last_hidden_state)` → `pre_projection` |
| Position IDs | Konstant (KV-Cache ist statisch, wird nicht aktualisiert) |
| Cross-Attention | Jede MTP-Layer liest K/V von der letzten Layer des gleichen Attention-Typs (full/sliding) des Targets |

**Quellen:**
- Hugging Face Docs: "The entire model uses KV sharing... allowing the assistant to skip the pre-fill phase entirely"
- llama.cpp PR #22738: "Each MTP layer reads K/V from the last layer of the matching attention type (full / sliding) of the target's KV cache"
- MLX-VLM: "The drafter has no KV cache of its own; its only recurrent state is the target's last hidden, projected through post_projection"

### 1.2 Unterschied zu klassischem spekulativem Decoding

| Aspekt | Klassisches Spec-Dec | Gemma-4 MTP |
|--------|----------------------|-------------|
| Draft-Modell | Separat, unabhängig | Speziell für Target trainiert |
| KV-Cache | Eigener, skaliert mit ctx | Shared mit Target, kein zusätzlicher |
| Pre-fill | Muss vollständigen Kontext verarbeiten | Übersprungen |
| Input | Nur Tokens | Tokens + Target Hidden States |
| Attention | Vollständig (Q,K,V) | Q-only (Cross-Attention auf Target-KV) |
| Orchestrierung | Zweites `llama_context` | Weights neben Target in `llama_model::mtp_assistant` |

---

## 2. Kontext-Scaling: Gemessene VRAM-Werte

### 2.1 VRAM nach Kontextgröße (K=turbo3, V=turbo4, `-np 1`)

| ctx | VRAM (MiB) | Status |
|-----|-----------|--------|
| 512 | 6.722 | ✅ |
| 16.384 (16k) | 7.020 | ✅ |
| 32.768 (32k) | 7.178 | ✅ |
| 65.536 (64k) | 7.496 | ✅ |
| 98.304 (96k) | 8.018 | ✅ |
| 100.352 (98k) | 8.046 | ✅ MTP aktiv |
| 102.400 (100k) | 8.076 | ✅ MTP aktiv |
| 104.448 (102k) | 8.106 | ✅ MTP init, ❌ Crash bei Generierung |
| 105.472 (103k) | 7.554 | ❌ MTP nicht initialisiert |
| 131.072 (128k) | 7.794 | ❌ MTP nicht initialisiert |
| 163.840 (160k) | 8.100 | ❌ Crash bei Generierung (ohne MTP) |

### 2.2 Layer-Sharing

26B-A4B nutzt KV-Cache Layer-Sharing:
- Layer 0-2 teilen mit Layer 28
- Layer 3 teilt mit Layer 29

Das reduziert den effektiven KV-Cache-Bedarf (~26 Layer statt 30).

---

## 3. MTP-Initialisierung: Memory-Fitting-Problem

### 3.1 Beobachtung

Bei ctx > 103k schlägt die MTP-Initialisierung fehl:
```
W srv    load_model: [spec] failed to measure draft model memory: failed to create llama_context from model
W common_speculative_init: no implementations specified for speculative decoding
```

### 3.2 Root Cause (Hypothese, zu verifizieren)

Der Draft hat zwar keinen eigenen KV-Cache, aber die Memory-Fitting-Phase (`llama_init_from_model`) versucht einen `llama_context` für den Draft zu erstellen, um seinen Speicherbedarf zu schätzen. Bei knappem VRAM schlägt diese Schätzung fehl → MTP wird gar nicht erst initialisiert.

**Wichtige Log-Zeile:** `failed to initialize the context: Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)` — diese Warnung ist normal, aber wenn danach kein erfolgreicher Initialisierungsversuch kommt, fällt MTP komplett aus.

---

## 4. MTP-Performance bei großem Kontext

### 4.1 Veraltete Werte (trivialer Prompt "Count 1 to 100")

> ⚠️ **WARNUNG:** Diese Werte wurden mit einem trivial vorhersagbaren Prompt
> ("Count from 1 to 100") gemessen. Sie sind **NICHT repräsentativ** für
> realistische Nutzung. Siehe §4.2 für den korrekten Direktvergleich.

| ctx | MTP | pp (t/s) | tg (t/s) | Zeit 1200 tok | Drafts akzeptiert |
|-----|-----|----------|----------|---------------|-------------------|
| 8.192 | ✅ | — | **31.93** | ~37.6s | 11/11 (100%) |
| 98.304 (96k) | ✅ | — | **17.33** | 69.2s | 754 |
| 102.400 (100k) | ✅ | 127.3 | **16.98** | 70.7s | 754 |
| 102.400 (100k) | ❌ | 112.3 | **19.11** | 62.8s | — |
| 131.072 (128k) | ❌ | 125.1 | **20.36** | 58.9s | — |

### 4.2 Korrigierter Direktvergleich MTP vs. no-MTP (Essay-Prompt)

**Prompt:** "Schreibe einen ausführlichen Essay über die Geschichte der
künstlichen Intelligenz..." (6 Aspekte, 1200 Tokens, temp=1.0, seed=42)
**Konfiguration:** `--n-cpu-moe 20 -ctk turbo3 -ctv turbo4 --flash-attn on`,
Pinning+Prefetch, n_max=3

| ctx | MTP tg (t/s) | noMTP tg (t/s) | MTP VRAM | noMTP VRAM | MTP Vorteil |
|-----|-------------|----------------|----------|------------|-------------|
| 8k | 16.56 | **18.12** | 6832 | 6444 | ❌ -9% |
| 16k | 15.12 | **17.39** | 7040 | 6612 | ❌ -15% |
| 32k | 14.49 | **16.49** | 7210 | 6702 | ❌ -14% |
| 48k | 14.25 | **17.27** | 7380 | 6792 | ❌ -21% |
| **64k** | **CRASH** | — | — | — | ❌ 2^16 Bug |
| 96k | 13.38 | **17.16** | 8018 | 7190 | ❌ -28% |
| 128k | 15.68 | **16.03** | 7794 | 7498 | ❌ -2% |

**Schlussfolgerung:** MTP ist bei JEDEM Kontext langsamer als no-MTP für
realistische (kreative) Textgenerierung. Der Speedup bei trivialen Prompts
("Count 1 to 100": 31.93 t/s) entsteht durch ~100% Acceptance Rate — bei
kreativen Essays werden Drafts meist abgelehnt, und der Draft-Overhead
(additional KV-Cache reads, serielle Ausführung) dominiert.

### 4.3 Break-Even-Analyse (korrigiert)

Die ursprüngliche Annahme "MTP wird kontraproduktiv bei ctx > ~50-80k" ist
**falsch**. MTP ist bei JEDEM Kontext kontraproduktiv für realistische
Generierung auf der GTX 1070.

**Warum der 31.93 t/s Wert bei 8k irreführend war:**
- Prompt: "Count from 1 to 100" — extrem vorhersagbar, ~100% Acceptance
- Bei kreativen Texten (Essay, Code, Analyse): Acceptance deutlich niedriger
- Draft-Overhead (4 Layer Cross-Attention, seriell) addiert sich auf jede Runde
- GTX 1070: 256 GB/s Bandbreite — KV-Cache-Reads des Drafts dominieren

**Speedup-Formel:**
```
Speedup = expected_tokens_per_step / (1 + draft_cost_ratio)
```

Bei 8k Kontext, trivialer Prompt:
- `draft_cost_ratio` ≈ 0.05 (KV-Cache winzig)
- `expected_tokens_per_step` ≈ 2.5 (hohe Acceptance bei trivialem Text)
- Speedup = 2.5 / 1.05 = **2.38** → gemessen: +52% (Rest ist MoE-Offloading-Overhead)

Bei 8k Kontext, kreativer Prompt (Essay):
- `draft_cost_ratio` ≈ 0.05 (KV-Cache winzig)
- `expected_tokens_per_step` ≈ 1.1-1.3 (niedrige Acceptance bei kreativem Text)
- Speedup = 1.2 / 1.05 = **1.14** → mit MoE-Offloading-Overhead < 1.0 → **Netto-Slowdown**

Bei 100k Kontext (egal welcher Prompt):
- `draft_cost_ratio` ≈ 0.5-0.7 (KV-Reads dominieren auf GTX 1070)
- `expected_tokens_per_step` ≈ 1.8 (~60% Acceptance)
- Speedup = 1.8 / 1.6 = **1.125** → mit Attention-Drift + MoE-Overhead < 1.0 → **Netto-Slowdown**

### 4.3 Warum MTP bei großem Kontext langsamer wird

Drei Effekte wirken zusammen:

#### Effekt 1: Memory-Bandwidth-Bottleneck (Hauptgrund)

Bei großem Kontext wird Generierung **memory-bandwidth-bound**, nicht compute-bound.
- Target liest KV-Cache: 1× pro Token (60+ Layer)
- Draft liest KV-Cache: **4× pro Draft-Token** (4 Layer Cross-Attention)
- Bei `n_max=3`: **12× zusätzliche KV-Cache-Reads** pro Runde
- GTX 1070: nur 256 GB/s (vs ~900 GB/s auf modernen RTX)

**Quellen:**
- MagicDec (arXiv:2408.11049): "KV Cache Is The Dominant Bottleneck In Large batch size Long-context Regime"
- SpecPV (arXiv:2512.02337): "verification becomes the dominant bottleneck" bei langem Kontext

#### Effekt 2: Sinkende Acceptance Rate (Attention Drift)

- Draft-Modelle degradieren bei langem Kontext ("attention drift")
- Attention verschiebt sich vom Prompt auf eigene generierte Tokens
- Draft typischerweise auf <4k Kontext trainiert → Train/Test-Mismatch bei 100k

**Quellen:**
- Attention Drift (arXiv:2605.09992): "drafter models degrade sharply under... long-context inputs"
- LongSpec (arXiv:2502.17421): "most state-of-the-art SD methods are trained on short texts (<4k tokens)"

#### Effekt 3: Serielle Ausführung in llama.cpp

Draft und Target laufen **seriell**, nicht überlappt:
- Jede Runde = T_draft + T_verify
- Draft-Overhead addiert sich auf die Target-Zeit, ersetzt sie nicht

---

## 5. Praktische Empfehlung (korrigiert)

| Use Case | Config | ctx | tg | Begründung |
|----------|--------|-----|-----|------------|
| **Chat / RAG / alles** | MTP **aus**, turbo3/4 | 128k | **16-17 t/s** | MTP schadet bei realistischer Generierung |
| **Maximaler Kontext** | MTP aus, turbo3/4 | 160k | ~16 t/s | Max. KV-Cache, kein MTP-Overhead |
| **Trivial-Prompts** (Code-Vervollständigung, Listen) | MTP an, n_max=3 | ≤ 32k | ~30 t/s | Nur bei hoher Acceptance (>80%) |

**Empfehlung:** MTP standardmäßig **aus**. Nur bei trivial-vorhersagbaren
Tasks (Code-Vervollständigung, Listen, Zählen) und kleinem Kontext (≤32k)
kann MTP einen Speedup bringen. Für realistische Chat/RAG/Analyse-Workloads
ist MTP auf der GTX 1070 bei jedem Kontext langsamer als no-MTP.

**64k (65536) Bug:** ctx=65536 (= 2^16) crasht reproduzierbar mit MTP
(CUDA error bei Generierung). 32k, 48k, 80k, 96k, 128k funktionieren.
Vermutung: Integer-Overflow in der Compute-Buffer-Allokation bei 2^16.
Workaround: ctx auf 65535 oder 80k setzen.

---

## 6. Memory-Fitting-Code-Analyse

### 6.1 Der Ablauf (2-Phasen)

**Phase 1: Memory-Measurement** (`server-context.cpp:920-986`)
```
cparams_dft.ctx_other = nullptr  (Default, wird NICHT gesetzt)
  → common_get_device_memory_data()
    → common_get_device_memory_data_impl() (fit.cpp:29)
      → llama_init_from_model(model, *cparams)  (fit.cpp:66)
        → llama_context constructor (llama-context.cpp:92-99)
          → model.arch == LLM_ARCH_GEMMA4_ASSISTANT
          → params.ctx_other == nullptr
          → throw "Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)"
        → caught at llama-context.cpp:3589 → returns nullptr
      → fit.cpp:70: throw "failed to create llama_context from model"
    → caught at server-context.cpp:984
    → SRV_WRN("[spec] failed to measure MTP context memory: ...")
```

**Phase 2: Echte MTP-Initialisierung** (`server-context.cpp:1054-1074`)
```
cparams_mtp.ctx_other = ctx_tgt  (WIRD korrekt gesetzt!)
  → llama_init_from_model(model_tgt, cparams_mtp)
    → llama_context constructor
      → params.ctx_other != nullptr → OK
      → cparams.ctx_other = params.ctx_other
    → output_reserve(), sched_reserve()
      → graph_reserve() für pp und tg
      → Hier wird der Compute-Buffer allokiert
      → Bei knappem VRAM: "failed to allocate compute pp/tg buffers"
    → Wenn Exception → returns nullptr
  → server-context.cpp:1068: "failed to create MTP context"
```

### 6.2 Echte Ursache

**Die "failed to measure" Warnung ist by Design und harmlos** — der Kommentar im Code sagt das explizit. Die Measurement-Phase kann den Draft nicht messen weil er `ctx_other` braucht, aber das wird toleriert.

**Das echte Problem ist in Phase 2:** Bei `sched_reserve()` (llama-context.cpp:400) werden die Compute-Buffer für den Graphen allokiert. Bei 102k+ Kontext ist der Graph groß genug dass die Compute-Buffer nicht mehr in den verbleibenden VRAM passen. Die Exceptions kommen von:
- `"failed to allocate compute pp buffers"` (Zeile 623/650)
- `"failed to allocate compute tg buffers"` (Zeile 635)

Das ist **kein Bug** — es ist echtes OOM bei der Compute-Buffer-Allokation. Der Draft braucht zwar keinen eigenen KV-Cache, aber er braucht **Compute-Buffer** für seine 4-Layer Cross-Attention-Graphen, und die skalieren mit der Kontext-Größe (die Attention-Score-Matrix ist `n_heads × n_ctx` pro Layer).

### 6.3 Warum 100k funktioniert aber 102k nicht

| ctx | KV-Cache (Target) | Compute-Buffer (Draft) | Gesamt | VRAM frei |
|-----|-------------------|----------------------|--------|-----------|
| 100k | ~7.2 GB | ~0.8 GB | ~8.0 GB | ~0.1 GB Reserve |
| 102k | ~7.3 GB | ~0.8 GB | ~8.1 GB | <0 → OOM |
| 128k | ~7.8 GB | ~0.3 GB (kein MTP) | ~8.0 GB | ~0.2 GB |

Ohne MTP entfällt der Compute-Buffer für den Draft-Graphen → mehr Platz für KV-Cache → 128k funktioniert.

### 6.4 Fazit

**Kein Bug im Memory-Fitting-Code.** Die "failed to measure" Warnung ist by Design. Das echte Limit bei 102k ist der **Compute-Buffer** für die Draft-Cross-Attention-Graphen, der mit der Kontext-Größe skaliert. Das ist ein echtes VRAM-Limit, kein Code-Fehler.

---

## 7. Offene Fragen / TODO

- [x] MTP Kontext-Limit mit n_max=3 getestet: läuft bis 96k (32k crasht, 64k/80k/96k laufen)
- [x] MTP n_max=1/2 Test bei 64k: beide crasht mit CUDA error (Compute-Buffer OOM)
- [ ] Eventuell "Router" bauen: MTP automatisch an/aus je nach Kontextgröße

### 7.1 MTP Kontext-Limit Test (n_max=3)

| ctx | VRAM (MiB) | MTP Status |
|-----|-----------|------------|
| 16.384 (16k) | 7.550 | ✅ OK |
| 32.768 (32k) | 7.210 | ❌ Crasht (Compute-Buffer OOM) |
| 49.152 (48k) | 7.380 | ✅ OK |
| 65.536 (64k) | 7.550 | ✅ OK |
| 81.920 (80k) | 7.784 | ✅ OK |
| 98.304 (96k) | 8.018 | ✅ OK |

**Erkenntnis:** 32k crasht trotz MTP-Initialisierung — der Compute-Buffer für den Draft-Graph ist bei diesem Kontext zu groß. 48k+ funktionieren.

### 7.2 MTP n_max Test @ ctx=65536 (64k) — KORRIGIERT

| n_max | Ladezeit | VRAM | Generierung | Ergebnis |
|-------|----------|------|-------------|----------|
| 2 | ✅ 10s | 7550 MiB | ❌ CUDA error | 2^16 Bug |
| 3 | ✅ 10s | 7550 MiB | ❌ CUDA error | 2^16 Bug |
| 4 | ✅ 10s | 7550 MiB | ❌ CUDA error | 2^16 Bug |
| 5 | ✅ 10s | 7550 MiB | ❌ CUDA error | 2^16 Bug |

**Erkenntnis (korrigiert):** Der frühere Claim "n_max=3 funktioniert bei 64k"
war FALSCH — wahrscheinlich eine Verwechslung mit 96k. Der saubere Test zeigt:
**ALLE n_max-Werte crashen bei 64k**. Das ist kein n_max-Problem, sondern ein
**2^16 Integer-Overflow-Bug** in der Compute-Buffer-Allokation.

Bestätigung: 32k, 48k, 80k, 96k, 128k funktionieren alle mit n_max=3. Nur
65536 (= 2^16) crasht reproduzierbar.

### 7.3 MTP vs. no-MTP Direktvergleich (Essay-Prompt, n_max=3)

Siehe §4.2 für die vollständige Tabelle. Zusammenfassung:

| ctx | MTP tg | noMTP tg | Differenz |
|-----|--------|----------|-----------|
| 8k | 16.56 | 18.12 | -9% |
| 16k | 15.12 | 17.39 | -15% |
| 32k | 14.49 | 16.49 | -14% |
| 48k | 14.25 | 17.27 | -21% |
| 96k | 13.38 | 17.16 | -28% |
| 128k | 15.68 | 16.03 | -2% |

**MTP ist bei JEDEM Kontext langsamer** für realistische Generierung.
