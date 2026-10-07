# Engine-Vergleich: Strata vs. OptLlama vs. eigener Fork — Qwen3.8-Flash-Next auf Dual-RTX-4060-Ti

**Datum:** 2026-10-07 · **Host:** uranus (2× RTX 4060 Ti 16 GB, i7-9700K 8 Kerne, 125 GB RAM, PCIe gemessen 6.2 GB/s H2D)
**Bezugsplan:** `docs/fork/plans/SESSION_PLAN_20261006_strata-optllama-uranus-eval.md` · **Trilium:** `5Ec9sW4aCN6S`

---

## 1. Executive Summary

| Engine | Modell | pp t/s @23k | pp t/s @5.8k | pp t/s @3k | Decode t/s | TTFT @23k |
|---|---|---|---|---|---|---|
| **Strata** (v0.1.4x, 82f46a8) | IQ3_XXS | **1728** | 987 | 610 | **55–71** | 13.4 s |
| **Strata** | UD-IQ4_XS | **1227** | 642 | 368 | 40–53 | 18.9 s |
| **OptLlama** (moe-cache, 167742d) | IQ3_XXS | 366 | 367 | 311 | 12.6–15.4 | — |
| **Fork** (master aa1976d) | UD-IQ4_XS | **267–279** | 263–279 | 259–278 | 11.5–12.5 | — |
| Fork beste Basis (F1, cache aus) | UD-IQ4_XS | 253 | 248 | 242 | 10.9–11.6 | — |

**Kernaussage:** Strata ist auf identischer Hardware und identischem Modell (UD-IQ4_XS)
**~4.3–4.6× schneller im Prefill** (1227 vs. ~267 t/s @23k) und **~4× schneller im Decode**
(~45 vs. ~11.5 t/s). Die Lücke wächst mit Promptlänge: bei ~800 Tokens noch ~1.5×, bei 23k 4.6×.

Der Abstand ist **nicht** durch unsere vorhandenen Features (moe-cache, ubp, prefetch) aufholbar:

- `moe-cache` (auto): **±0 %** — Cache deckt 2560/24576 Expert-Slots, kein messbarer Gewinn.
- `-ubp 8192`: **negativ** bei kurzen Prompts (69 vs. 106 t/s), neutral bei langen.
- `GGML_SCHED_PREFETCH_EXPERTS=1`: **Crash** (CUDA error in `ggml_cuda_op_mul_mat_cublas`, reproduzierbar).
- Hybrid `--n-cpu-moe 40/44`: **+10–17 %** — bester Fork-Wert (279 t/s @5.8k), OOM-Risiko ab ~23k.
- TurboQuant-KV (turbo4/turbo3): pp +4–8 %, Decode-Einbruch ~4 t/s @23k (nicht empfohlen für dieses Modell).

**OptLlama** ist ~30–40 % schneller als unser Fork in Prefill (366 vs. 250 t/s @23k, beide IQ3_XXS),
Decode +15–35 % mit Cache — deutlich hinter Strata. Sein Expert-Cache ist auf diesem Modell
fragil (Batch >2048 → MUL_MAT_ID-Compute-Fehler, Issue-#120-Klasse).

**Empfehlung (Matrix):** Der Abstand (≥2×, hier 4×+) gehört zur Kategorie
**"selektive Scheduling/Logistik-Adaption"**. Voller Port lohnt nicht; sinnvoll sind:
(1) Stream-all-Experts-Prefill-Pfad, (2) große Prefill-Chunks mit Transfer/Compute-Overlap,
(3) MTD/Experten-Pipeline-Asynchronität (echtes Double-Buffering statt Slot-Prefetch),
(4) Strata parallel via llama-swap als Sofortmaßnahme für Flash-Next-Workloads.
Dazu zwei **Fork-Bugs** fixen: Prefetch-Crash auf Qwen4Exp, GGUF-Reader-Offset-Bug
mit ISTA-DASLab GSQ-RCO IQ3_XXS.

---

## 2. Setup & Methodik

### 2.1 Getestete Builds

| Engine | Quelle | Commit/Version | Build |
|---|---|---|---|
| Strata | github.com/Niko1221/Strata | `82f46a8` (main) | lokal kompiliert sm_89 (CUDA 12.0), `setup.sh` |
| OptLlama | github.com/GenerelSchwerz/llama.cpp | `167742d` (`moe-cache`) | `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` |
| Fork | fukuro-llama-cpp-turboquant | `aa1976da98` (master) | `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` |

### 2.2 Modelle

| Quant | Dateien | Größe | Engines |
|---|---|---|---|
| `ISTA-DASLab GSQ-RCO IQ3_XXS` | 2 Shards | 70.6 GB | Strata, OptLlama |
| `unsloth UD-IQ4_XS` | 3 Shards | ~88 GB | Strata, Fork |

**Asymmetrie-Disclaimer:** Der Fork konnte IQ3_XXS nicht laden (GGUF-Offset-Fehler,
siehe §6) — Fork-Zahlen stammen von UD-IQ4_XS. Strata-UD-IQ4_XS liefert den
1:1-Modellvergleich Fork↔Strata; Strata-IQ3_XXS ist der schnellere Quant
(kleinere Expert-Blobs → mehr Expert-Cache-Abdeckung, 7483 vs. 4370 Slots).

### 2.3 Identische Bedingungen

- Kontext: `40960` überall. KV: Strata `int8` (empfohlener Pfad); llama.cpp-Engines `q8_0/q8_0` (nächstes Äquivalent); Fork zusätzlich `turbo4/turbo3`.
- Batch: llama.cpp-Engines `-b 8192`, physische ubatch `-ub 4096` (bzw. `-ubp`/`ub2048` siehe Configs). Strata: `--prefill auto` (intern große Chunks bis 8192).
- GPUs: `CUDA0,CUDA1` Layer-Split bei allen.
- Gleiche Prompt-Dateien (`docs/qwen38-eval/prompts/bench/bench-pp-*.txt`), `--vary`-Marker pro Run gegen Prompt-Caches, 3 Runs/Punkt (Median berichtet).
- **Aktuelle Tokenzahlen** aus `usage.prompt_tokens`/`timings`: die Dateien ergeben ~803 / ~2946 / ~5812 / ~23133 Tokens (Namen 1k/4k/8k/32k sind Oberlabels).
- Strata: OpenAI-`/v1/chat/completions` mit Streaming (TTFT = erstes Token-Chunk inkl. reasoning_content; Chat-Template zählt ~50 Tokens extra); llama.cpp: `/completion` (`timings.*`). pp t/s bei Strata ≈ prompt_n/TTFT.

### 2.4 Betriebszustand während Messung

- Produktiv-Services (`llama-server-uranus-qwen38-{standard,unc}`) gestoppt,
  `host-agent-reconcile.timer` (Soll/Ist-Watchdog, 120 s) ebenfalls gestoppt —
  **sonst hätte er die Server neu gestartet und Test-Engine-Prozesse als "verwaist" gekillt.**
  Beides nach Abschluss restauriert (Health 200/200 verifiziert).
- `ulimit -l` = 8192: Stratas PLE-Table-`mlock` schlug fehl (graceful, nicht blockierend);
  CUDA-Pinned-Allocs (cudaHostAlloc/Register) laufen am RLIMIT vorbei und funktionierten.

---

## 3. Ergebnisse im Detail

### 3.1 Strata (Referenz)

IQ3_XXS, 2 GPU Layer-Split (L0–24 / L25–47+Head), Expert-Cache auto (7483 Slots, 10.9 GiB VRAM GPU0 + 2205 slots/4 GiB CUDA1 borrow), 40 GiB Experten resident RAM, MTP spec 4/min-p 0.5, kv int8, ctx 40960:

| ~Prompt tokens | TTFT (ms) | pp t/s (≈n/TTFT) | Decode t/s |
|---|---|---|---|
| 854 | 2890 | 295 | 52–58 |
| 2997 | 4913 | 610 | 67–71 |
| 5863 | 5925 | 989 | 61–70 |
| 23184 | 13400 | **1730** | 65–68 |

UD-IQ4_XS (4370 Expert-Slots, 9.8 GiB; 55.4 GiB Experten im RAM):

| ~Prompt tokens | TTFT (ms) | pp t/s | Decode t/s |
|---|---|---|---|
| 854 | 5295 | 161 | 36–46 |
| 2997 | 8138 | 368 | 39–48 |
| 5863 | 9127 | 642 | 48–52 |
| 23184 | 18892 | **1227** | 40–53 |

- Startup: ~2.5–4 min (Experten-RAM-Load 40–55 GiB @ ~0.4 GiB/s, Cache-Füllung).
- Sanity vs. Issue #1062 (gleiche GPU-Klasse, UD-IQ4_XS): Issue meldet ~1930 pp @32k, 52–57 Decode — unsere 23k-Messung 1227 pp / ~45 t/s plausibel konsistent (Issue nutzte 128k Kontext und vermutlich warme KV-Pfade).
- Nicht-deterministische Elemente: adaptive Tiers, spec-decode acceptance; Messvarianz gering (<3 % pp, ~10 % decode).

**Single-GPU-Variante (nur CUDA0, GPU1 produktiv belegt, IQ3_XXS, sonst identisch):**

| ~Prompt tokens | TTFT (ms) | pp t/s | Decode t/s |
|---|---|---|---|
| 854 | ~4300 | 191–200 | 37–45 |
| 2997 | ~6260 | ~478 | 45–53 |
| 5863 | ~6700 | ~874 | 47–52 |
| 23184 | ~19950 | **1158** | 47–52 |

→ ~65–67 % der 2-GPU-Prefill-Rate @23k, Decode ~72 % — die zweite Karte kostet
nur ~1/3 der Leistung. **Auch auf einer einzigen 16-GB-Karte ist Strata ~4.3× pp /
~4× tg schneller als der beste Fork-Stand auf zwei Karten.** 5661 residente
Expert-Slots (vs. 7483+2205 auf 2 GPUs), 438 MiB frei — knapp, aber stabil.
Anmerkung: `--spec` ist fuer IQ-Packs Pflicht (T>=2); ein MTP-Abschalttest zur
Kernel/Spec-Isolation ist nicht moeglich — der MTP-Anteil (~2× Decode) bleibt
Schaetzung aus den publizierten Akzeptanzraten.

### 3.2 OptLlama (moe-cache)

| Config | pp @0.8k | pp @3k | pp @5.8k | pp @23k | Decode |
|---|---|---|---|---|---|
| A: `--cpu-moe`, kein Cache | 143 | 325 | 324 | 348 | 10.7–11.7 |
| B: `--moe-expert-cache-mib 9500,9500` + `--moe-early-router`, `-b8192 -ub4096` | **CRASH** | — | — | — | — |
| B3: Cache, kein early-router, `-b8192 -ub4096` | **CRASH** | — | — | — | — |
| B5: Cache, `-b8192 -ub2048` | 184 (kalt 86) | 311 | 367 | 366 | 12.5–15.4 |
| B4: Cache, Default-Batch 2048/512 | 117–158 | — | — | — | 10–16 |

- **Cache-Uplift vs. eigene Baseline:** pp ~+6–13 % (nur oberhalb ~5k sichtbar), Decode **+15–35 %** (@23k: 10.7 → 12.6; @5.8k: 11.3 → 14.8).
- **Stabilität:** Cache-Pfad bricht bei `-ub 4096` reproduzierbar mit
  `ggml_cuda_graph_evaluate_and_capture: op not supported ffn_moe_gate-23 (MUL_MAT_ID)` → `llama_decode ret=-3`.
  `GGML_CUDA_DISABLE_GRAPHS=1` ändert nichts (Fehler liegt im eigenen Capture-Pfad).
  Passt zu Issue #120 (grouped-MoE illegal access auf Flash-Next IQ3_S).
- Startup: ~8 s (mmap-Lazy; Experten on-demand aus Page-Cache/pinned Pool).
- `-fit` half nichts (`n_gpu_layers already set by user` — fit verweigert bei explizitem ngl).

### 3.3 Eigener Fork (master aa1976da98)

Alle: UD-IQ4_XS, `-dev CUDA0,CUDA1 -ngl 99 -c40960 -b8192 -t8 -fa on -np1`, KV `q8_0` außer F5:

| Config | pp @0.8k | pp @3k | pp @5.8k | pp @23k | Decode | Anmerkung |
|---|---|---|---|---|---|---|
| F1 `--cpu-moe`, `GGML_CUDA_MOE_CACHE=0`, `-ub4096` | 107 | 242 | 248 | 253 | 10.9–11.6 | Baseline |
| F2 `--cpu-moe`, Cache auto, `-ub4096` | 106 | 240 | 247 | 250 | 10.8–11.0 | **0 % Delta** — 2560/24576 Slots |
| F3a `-ub512 -ubp8192`, Cache auto | 69–71 | 165 | 248 | — | ~11 | ubp **negativ** bei kurz, neutral lang |
| F3b `GGML_SCHED_PREFETCH_EXPERTS=1` | **CRASH** | — | — | — | — | siehe §6 |
| F5 `turbo4/turbo3`-KV, Cache auto | 115 | 264 | 265 | 274 | **4.0 @23k** | pp +5–8 %, Decode-Kollaps lang |
| F6 `--n-cpu-moe 40` (8 L. Experten GPU) | 124 | 278 | **279** | **OOM-Crash** | ~12 | bester pp, instabil @23k |
| F7 `--n-cpu-moe 44` (4 L. Experten GPU) | — | 259 | 263 | 267 | 11.4–11.9 | stabil inkl. 23k |

moe-cache-Statistik (F2): Pools `dev0: 2048 slots ×1700KB` + `dev1: 512 slots` ≈ 4.25 GiB —
nur ~10 % der 24.576 Expert-Instanzen; 100 %-Hits der Cache-Sicht, aber die halbe
Expert-Arbeit bleibt ohnehin CPU (per-Layer-Roundtrip über PCIe 6.2 GB/s dominiert).

**Decode-EOS-Anomalie:** Bei langen Prompts beendet der Fork oft nach 1–10 Tokens
(predicted_n=1..18), wo Strata regulär 128 Tokens generiert. Decode-Werte <16 Tokens
sind deshalb teils aus kurzen Sequenzen — Trendwerte robust, kein korrektheitskritischer
Messfehler für pp.

---

## 4. Einordnung

### 4.1 Woher kommt der Strata-Vorsprung?

Faktoren, aus Log + Messung separierbar:

1. **Expert-Logistik (groß):** Strata lädt alle 47–55 GiB Experten resident in RAM
   (pinned + adaptiv), streamt Prefill in großen Chunks mit Transfer/Compute-Overlap,
   hybridisiert Misses (GPU-Kopie **oder** CPU-Berechnung je nach pcie_frac=0.17).
   Unser Pfad: mmap + per-Layer On-Demand → ~250 t/s-Plateau = PCIe-gebunden.
2. **MTP spec 4** (Decode ~2× auf 60+ t/s-Klasse beitragend): bei uns nicht nutzbar —
   das GGUF trägt keine `n_layer_nextn`-Layer, Strata lädt den MTP-Sidecar separat.
3. **Fused-Kernel** (GDN/QSA/HC): in der Time zusätzlich enthalten, nicht isoliert messbar.
   Wichtig: die sm_90+-Cluster-Pfade (QSA-Decode-Top-K, `qsa_select.cu`) sind
   **runtime-gated** (`ptxVersion >= 90`-Check, sonst `block_topk_reg_kernel`-Fallback)
   — unsere gesamte Messung lief auf sm_89-Fallbacks, d.h. der 4×-Vorsprung entsteht
   nachweislich *ohne* Hopper-Cluster-Features.
4. **Expert-Cache-Abdeckung:** Strata ~98 % der gerouteten Masse (13023/24576 Paare)
   vs. unsere ~10 % — aber selbst unsere 4 GiB konnten ≥keinen Gewinn bringen,
   weil die fehlenden ~70 GiB der Engpass sind, nicht die letzten Slots.

### 4.2 Obergrenze der Adaptions-Verbesserung

- **Decode:** Strata ~45–65 t/s vs. Fork ~11.5 → Faktor ~4–5×. Davon grob:
  MTP ~2×, Expert-Streaming/Overlap ~1.5–2×, Rest Kernel.
- **Prefill:** @23k 4.6×, @5.8k 2.6×, @0.8k 1.5×. Je länger der Prompt,
  desto mehr zahlt Amortisation (große Chunks + Stream-All) über per-Layer-Trips.
- Realistisch portierbar ohne Engine-Neubau (nur Scheduling/Buffering):
  grob **1.5–2.5× Prefill**, **1.5–2× Decode** — der Rest ist architekturverzahnt
  (fused GDN/QSA, Cluster-Kernels, Engine-internes PLE/MTP).

### 4.3 Entscheidungsmatrix (aus Plan, auf Messung angewandt)

| Schwelle | Messwert | Konsequenz |
|---|---|---|
| ≥2× | **erfüllt**: 4.3–4.6× pp, ~4× tg | Selektive Adaption lohnt sich |

Priorisierte Quick-Wins (steigender Aufwand):

1. **llama-swap-Integration:** Strata als Zweit-Engine für Flash-Next — null Fork-Arbeit,
   voller Gewinn. Sofort umsetzbar (Setup existiert jetzt auf uranus).
2. **Prefetch-Fix + Ring-Buffer-Prefetch:** unser Prefetch crasht derzeit; danach
   von Slot-Prefetch (2–3 Experten) auf kontinuierliches N+1-Layer-Streaming heben.
3. **Resident-RAM-Experten + Pinned-Pool-Budgetierung** (`--cpu-moe`-Pfad hebt mmap-
   Page-Faults → explizites Arena-Loading): mittlerer Aufwand, großer Hebel für pp.
4. **Stream-all-Experts-Prefill-Modus** für moe-cache (bei ≫ Expert-Instanzen zählt
   Cache-Hit-Mathe nicht mehr → Streaming-Policy statt LRU).
5. **MTP-Sidecar-Laden** (separate MTP-GGUF) — Architektur-Mehraufwand, ~2× Decode.
6. **n-cpu-moe-Auto-Fit-Fix:** `--n-cpu-moe N` + `-fit` sollte GPU-Expert-Lagen
   automatisch dosieren statt OOM (F6 zeigte den Gewinn, aber auch den Absturz).

Nicht empfohlen: OptLlama-Merge (Code-Divergenz, Instabilität auf diesem Modell,
geringerer Gewinn), Strata-Kernel-Übernahme (hart an dessen Engine gebunden).

---

## 5. Gefundene Bugs / Limitierungen (Fork)

1. **GGUF-Offset-Bug (P1):** ISTA-DASLab `GSQ-RCO IQ3_XXS` lädt nicht:
   `tensor 'blk.1.ffn_down_shexp.weight' has offset 2004206976, expected 1991099776`.
   OptLlama/Strata laden dieselbe Datei → unser `gguf_init_from_reader` ist zu strikt
   (vermutlich Shared-Expert-Tensor-Offsetbehandlung). Betrifft nur diese Quant-Familie.
2. **Prefetch-Crash (P1):** `GGML_SCHED_PREFETCH_EXPERTS=1` → `CUDA error: requested
   functionality is not supported` in `ggml_cuda_op_mul_mat_cublas` (ggml-cuda.cu:1792)
   beim ersten Request — Qwen4Exp-Pfad inkompatibel mit Expert-Prefetch-Staging.
3. **`-moe-cache`-Flag-Ordering (P2):** `--moe-cache 0` hinter `-ngl` greift zu spät —
   Backend-Registrierung läuft schon im `-ngl`-Parser (llama_supports_gpu_offload) und
   liest `GGML_CUDA_MOE_CACHE` bevor das Flag es setzt. Workaround: Env direkt.
4. **F6-OOM (P2):** `--n-cpu-moe 40` @`-b8192 -ub4096` ctx 40960 → device1 cudaMalloc
   2.3 GiB OOM bei ~23k Prompt. Fit-Logik berücksichtigt GPU-residente Expert-Lagen
   nicht beim Workspace-Headroom.
5. **TurboQuant-Decode-Einbruch (P2):** turbo4/3-KV halbiert Decode @23k (~4 vs. 11 t/s) —
   für dieses Modell nicht empfohlen; pp-Gewinn bleibt.
6. **MTP nicht nutzbar (Info):** UD-IQ4_XS-GGUF enthält keine NextN-Layer
   (`n_layer_nextn == 0`) — Strata lädt MTP aus dem Original-Checkpoint nach;
   unser Pfad bräuchte Sidecar-Support.

## 6. Betriebs-Notizen uranus

- `host-agent-reconcile.timer` (120 s): stellt Produktiv-Services wieder her und
  **kills "verwaiste" GPU-Prozesse** — bei jeder Bench-Session zuerst stoppen.
- Strata-Setup auf uranus: `~/Strata` (Engine kompiliert, Packs unter `~/Strata-data/`,
  `run-iq3_xxs.sh` / `run-ud-iq4_xs.sh` — PLE-mlock-Warnung wegen ulimit, unkritisch).
- OptLlama: `~/optllama` (Branch `moe-cache`, Build in `build/`).
- Bench-Rohdaten: `uranus:~/eval-flashnext/results/*.jsonl`
  (134 Runs inkl. Fehler/Crash-Protokoll), Prompts + `bench_measure.py` dort.

## 7. Vergleichs-Caveats

- Strata-Decode enthält MTP-Spekulation (~2×); Fork/OptLlama ohne Spec → ein Teil des
  Decode-Gaps ist Feature-Delta, nicht Kernel.
- IQ3_XXS (Strata/OptLlama) vs. UD-IQ4_XS (Fork) — der 1:1-Quervergleich ist
  Strata-UD vs. Fork-UD: **1227 vs. 267 pp t/s @23k bzw. ~45 vs. ~11.5 tg**.
- Strata nutzt PLE-Ngram-Table (32 GB) + adaptiven Draft — Decode-Variabilität höher.
- Einzelne Runs mit predicted_n <16 wurden für Decode-Median herausgefiltert
  (EOS-Kurzsequenzen); pp-Werte unbeeinflusst.
