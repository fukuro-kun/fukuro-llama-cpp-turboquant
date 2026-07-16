# E4B QAT + MTP Benchmark — Uranus (2026-07-12)

## Setup

| Komponente | Wert |
|------------|------|
| **Host** | Uranus (192.168.1.11) |
| **GPU** | 2× RTX 4060 Ti 16GB (32GB total) |
| **Target** | `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` (4.2GB, unsloth QAT) |
| **Drafter** | `mtp-gemma-4-E4B-it.gguf` (60MB, unsloth MTP Q4_0) |
| **KV-Cache** | turbo4 (K) + turbo3 (V), Target + Draft |
| **Context** | 131072 (128K) |
| **Flash-Attn** | ON |
| **GPU-Layers** | 99 (Target + Draft, voll auf GPU) |
| **Parallel-Slots** | 4 (auto) |

## Modelle

### Paar-Verifikation
Target und Drafter sind ein echtes Paar aus demselben unsloth Repo
(`unsloth/gemma-4-E4B-it-qat-GGUF`). SHA256 des Drafters: `b0005dc3...`.

### Wichtiger Hinweis: AtomicChat vs unsloth
Es gibt zwei Drafter-Quellen auf HF:
- `AtomicChat/gemma-4-E4B-it-assistant-GGUF` — 49 Tensors, Architektur `gemma4_assistant`
- `unsloth/gemma-4-E4B-it-qat-GGUF` (enthält `mtp-gemma-4-E4B-it.gguf`) — 49 Tensors, `gemma4-assistant`

Unser Fork lädt beide (Arch-Names `gemma4_assistant` und `gemma4-assistant` sind
beide registriert in `llama-arch.cpp:883-887`). Die AtomicChat-Datei mit 51
Tensoren (mit centroid LM head) wird aktuell **nicht** geladen — der Fork
erwartet 49 Tensors. Die unsloth-Datei funktioniert.

## MTP-Parameter-Tuning

### Ergebnis-Tabelle

| `--spec-draft-n-max` | Tokens | Gen t/s | vs. Baseline (58.6) | Bewertung |
|----------------------|--------|---------|---------------------|-----------|
| 16 | 435 | 45.2 | -23% | Viel zu aggressiv |
| 8 | 1685 | 38.3 | -35% | Noch schlechter — Verifikation zu teuer |
| 3 | 1892 | 54.6 | -7% | Fast Baseline |
| **2** | **1770** | **60.1** | **+3%** | **Sweet Spot** |
| (kein MTP) | 759 | 58.6 | baseline | Referenz |

### Erkenntnis
Bei `n_max >= 8` ist der Draft-Verifikation-Overhead höher als der Speedup
durch akzeptierte Draft-Tokens. `n_max=2` ist der Sweet Spot: minimaler
Overhead, hohe Accept-Rate, ~3% Speedup über Baseline.

### Empfohlene Konfiguration
```bash
./build/bin/llama-server \
  -m /media/fukuro/raid5/modelle/gemma-4-E4B-it-qat/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  -md /media/fukuro/raid5/modelle/gemma-4-E4B-it-qat/drafts/mtp-gemma-4-E4B-it.gguf \
  --port 8080 --host 0.0.0.0 \
  -ngl 99 -ngld 99 -fa 1 \
  -ctk turbo4 -ctv turbo3 -ctkd turbo3 -ctvd turbo3 \
  -c 131072 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0
```

## GPU-Speicher

| GPU | Belegt | Frei | davon llama-server | davon xtts-api |
|-----|--------|------|--------------------|-----------------|
| GPU 0 | 13.8 GB | 2.2 GB | 3.8 GB | ~10 GB |
| GPU 1 | 14.3 GB | 1.7 GB | 4.8 GB | ~9.5 GB |

KV-Cache für 128K Context mit turbo4/turbo3: ~2.6 GB gesamt.
E4B+MTP+KV: ~8.5 GB auf 2 GPUs. xtts-api: ~19.5 GB. Zusammen ~28 GB von 32 GB.

## Build-Hinweis

Auf Uranus fehlte `tools/ui/node_modules/` → `llama-ui-assets` Target schlug fehl.
Fix: `nvm install 22` (Node 21 ist zu alt für `@sveltejs/vite-plugin-svelte@6.2.1`),
dann `npm install` in `tools/ui/`. Danach läuft `cmake --build build --target llama-server` sauber.
