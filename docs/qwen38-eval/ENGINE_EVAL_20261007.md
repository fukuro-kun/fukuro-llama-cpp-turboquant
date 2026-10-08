# Engine-Vergleich: Strata vs. OptLlama vs. eigener Fork — Qwen3.8-Flash-Next auf Dual-RTX-4060-Ti

**Datum:** 2026-10-07 · **Host:** uranus (2× RTX 4060 Ti 16 GB, i7-9700K 8 Kerne, 125 GB RAM, PCIe gemessen 6.2 GB/s H2D)
**Bezugsplan:** `docs/fork/plans/SESSION_PLAN_20261006_strata-optllama-uranus-eval.md` · **Trilium:** `5Ec9sW4aCN6S`

> **Mess-Epochen (Einordnung, Stand 2026-10-08):**
> 1. **Initiale Engine-Evaluation** (§1–§4, master `aa1976da98`, vor den
>    P1-Fixes): Prefetch-Crash, IQ3-Offset "unbekannter Ursache" und die
>    `-ubp`-Aussagen sind **Initialbefunde** — §5 beschreibt die seither
>    geloesten Fixes. Das initiale ~4.6×-Gap-Verhaeltnis ist historisch und
>    kein universelles Versprechen.
> 2. **Post-P1-Messungen** (§4.0): B0–B5 mit je drei Wiederholungen;
>    Konsolidierung und Kontrollen offen.
> 3. **Synthetischer pp8192/phys.-ub8192-Lauf** (§4.0 / Plan §10.7):
>    425.3 / 469.7 / 469.9, median 469.7 — **anderes Benchmark** (natives
>    Synthetic, nicht `/completion`-TTFT auf denselben Texten); keine neue
>    Gap-Zahl.
> 4. **MTP:** Graph/Sidecar-Smokes existieren; q8_0-KV-Spec-Degeneration und
>    das breite M-Gate bleiben **offen** (§4.0).

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

Der Abstand ist **nicht** durch unsere vorhandenen Features (moe-cache, ubp, prefetch) aufholbar
(**Initialbefund vor den P1-Fixes** — nachtraegliche Post-P1-Werte in §4.0):

- `moe-cache` (auto): **±0 %** — Cache deckt 2560/24576 Expert-Slots, kein messbarer Gewinn.
- `-ubp 8192`: **negativ** bei kurzen Prompts (69 vs. 106 t/s), neutral bei langen (Initialbefund; Post-P1 B2 lieferte 345,0 pp t/s @23k — siehe §4.0).
- `GGML_SCHED_PREFETCH_EXPERTS=1`: **Crash** (CUDA error in `ggml_cuda_op_mul_mat_cublas`, reproduzierbar) — **seit P1b behoben** (`3f3d272f69`, `819af39886`, §5.2).
- Hybrid `--n-cpu-moe 40/44`: **+10–17 %** — bester Fork-Wert (279 t/s @5.8k), OOM-Risiko ab ~23k.
- TurboQuant-KV (turbo4/turbo3): pp +4–8 %, Decode-Einbruch ~4 t/s @23k (nicht empfohlen für dieses Modell).

**OptLlama** lag im Prefill bei ~366 t/s (IQ3_XXS) gegen ~250–267 t/s des Forks
(UD-IQ4_XS); das ist **kein** Same-Quant-Vergleich. Der eigene Cache erhoehte
OptLlamas Decode in gueltigen Runs um ~15–35 % — deutlich hinter Strata. Sein Expert-Cache ist auf diesem Modell
fragil (Batch >2048 → MUL_MAT_ID-Compute-Fehler, Issue-#120-Klasse).

**Empfehlung (Matrix):** Der Abstand (≥2×, hier 4×+) gehört zur Kategorie
**"selektive Scheduling/Logistik-Adaption"**. Voller Port lohnt nicht; sinnvoll sind:
(1) Stream-all-Experts-Prefill-Pfad, (2) große Prefill-Chunks mit Transfer/Compute-Overlap,
(3) MTD/Experten-Pipeline-Asynchronität (echtes Double-Buffering statt Slot-Prefetch),
(4) Strata parallel via llama-swap als Sofortmaßnahme für Flash-Next-Workloads.
Dazu zwei **Fork-Bugs** fixen: Prefetch-Crash auf Qwen4Exp, GGUF-Reader-Offset-Bug
mit ISTA-DASLab GSQ-RCO IQ3_XXS. **Stand 2026-10-07: beide behoben und
gegatet** (siehe §5; Commits `0389e655e8`, `3f3d272f69`, `819af39886`).

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
damals unbekannter Ursache — **seit P1a behoben**: Typkollision Q2_0=42/TURBO2_0=47,
Fix `0389e655e8`, §5.1) — die untenstehenden Fork-Zahlen stammen von UD-IQ4_XS. Strata-UD-IQ4_XS liefert den
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
Anmerkung: Der getestete native IQ-Serve-Pfad verlangt `--spec` (T>=2);
der versuchte Abschalttest startete nicht. Fuer Isolation braucht es einen
geeigneten internen Diagnosepfad. Akzeptanzraten allein belegen keinen
~2x-Decode-Speedup; der MTP-Anteil wurde hier nicht separat gemessen.

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

### 4.0 Nachtraeglicher Implementierungs-/Methodikstand

Die Tabellen oben beschreiben die urspruengliche Engine-Evaluation,
nicht den aktuellen Stand nach P1-Fixes. Die Post-P1-Daten B0–B5
liegen mit je drei Wiederholungen vor: UD-IQ4_XS @23k B0 244,9 pp t/s,
B1 Prefetch 283,4 und B2 Prefetch + ubp8192 345,0; IQ3_XXS B5
320,7 pp t/s (343,0 ist das Maximum). Das lange Decode ist wegen
EOS-kurzer Laeufe noch nicht sauber vergleichbar. S1-Streaming und
S2-RAM-adaptive Expert-Arena sind **noch nicht implementiert**.

Der Qwen4Exp-MTP-Graph und Q8_0-Gewichts-Sidecar laufen; kurze f16-KV-
Smokes lieferten `B. Paris`. Die ersten n_max=2/4/8-Messungen (9,44 /
13,27 / 14,19 t/s) wurden aber mit verschiedenen Prompts und
83/300/200 erzeugten Tokens verglichen. Die vorherige Schlussfolgerung
+28 % und Optimum n=8 ist **nicht belastbar**; das Gesamtgate bleibt
offen. Hohe Akzeptanz und kohärentes, abgeschnittenes Reasoning ersetzen
weder Delivery- noch Logit-/Rollback-Paritaet. Der q8_0-KV-Spec-Fehler
ist eingegrenzt, nicht ursachenseitig gefixt; f16 ist ein vorlaeufiger
Workaround, kein Nachweis dass der ganze Pfad korrekt ist.

Stratas gemessene Config nutzt `--spec 4 --spec-min-p 0.5 --kv int8`.
`spec T` bedeutet ein feststehendes Token plus T−1 Drafts; im Fork
zaehlt `spec-draft-n-max` nur Drafts. Der vergleichbare erste Fork-Punkt
ist deshalb n_max=3. Stratas int8-KV besitzt eigene 64-Wert-Gruppen,
Skalen und Kernel, ist nicht identisch mit GGML-Q8_0. Kontrollierte
Text-/Laengen-/Kontext-Matrix und Diagnoseauftrag: Bezugsplan §10.7.

### 4.1 Woher kommt der Strata-Vorsprung?

Plausible Mechanismen; Zeitanteile sind noch nicht separat profiliert:

1. **Expert-Logistik:** Strata laedt Experten resident, verarbeitet grosse
   Prefill-Chunks und hat CPU-/GPU-/PCIe-Pfade. Der Fork hat bereits Host-
   Registrierung, `--no-mmap` und asynchronen Full-Tensor-Prefetch, aber
   dessen Qwen4Exp-Pfad crashte. Das ~250-t/s-Plateau allein beweist nicht,
   dass PCIe statt CPU-/GPU-Kernelzeit dominiert.
2. **MTP spec 4:** Strata verwendet einen vorbereiteten Head. Die Target-
   GGUF des Fork-Laufs hat keinen eingebetteten Head; externe Draftmodelle
   sind im Fork schon moeglich. Ein ~2x-MTP-Uplift wurde hier nicht isoliert.
3. **Kernel GDN/QSA/HC:** enthalten, nicht getrennt gemessen. Die sm_90+-
   QSA-Cluster-Variante ist gated; unsere sm_89-Messung benoetigt sie nicht.
   Daraus folgt nichts ueber den Beitrag der anderen sm_89-Kernel.
4. **Cache:** Slotzahl, geroutete Masse und reale Hit-Rate sind verschiedene
   Messgroessen. Aus 5661/24576 Slots oder gelabeltem `cache0` ohne Kontrolle
   der wirksamen Config keinen Anteil am Gesamt-Speedup ableiten.

### 4.1b Mechanismen-Matrix (OptLlama vs. Strata vs. Fork)

| Mechanismus | OptLlama | Strata | Fork |
|---|---|---|---|
| VRAM-Expert-Cache | Eigener Cache, +15–35 % tg in getesteten gueltigen Runs | Eigener VRAM-Tier; wirksame v2-Config nachpruefen | `--moe-cache`, F1/F2 hier ohne messbaren Gewinn |
| Early-Router vs. Gewichts-Prefetch | Early-Router-Option | "implizit" bisher nicht separat nachgewiesen | Full-Tensor-Prefetch vor `MUL_MAT_ID`, nicht gleich Early-Router; Crash auf Qwen4Exp |
| Resident-/Pinned-Host-Gewichte | Host-Pinned-Budget vorhanden; Arena-Gleichwertigkeit nicht belegt | Resident-Arena/mmap-A/B | `--no-mmap` laedt Host-Gewichte; optionale Registrierung von mmap-Pages vorhanden |
| Transfer/Compute-Overlap | Konkreter Pfad und Budget zu profilieren | Grosse Prefill-Chunks + Staging | Zweiter Backend-/Event-Pfad bereits vorhanden; keine nachgewiesene Strata-Gleichwertigkeit |
| CPU/GPU-Hybridexperten | Cache-Hits/Misses, genaue Zeitaufteilung offen | pcie_frac-/Pool-/GPU-Pfade | Cache dispatcht GPU-Hit-Rows und CPU-Miss-Rows; zusaetzlich statisches `--n-cpu-moe` |
| Grosser Prefill-Pfad | Batch-/Ubatch-Optionen | auto bis 8192 gemessen | `-ubp` plus bedingter Full-Tensor-Prefetch vorhanden; Qwen4Exp-Crash blockiert Vergleich |
| MTP | Nicht in diesem Lauf getestet | Vorbereiteter Head + spec 4 | Externe K2/Gemma-Sidecars vorhanden; Flash-Next-Graph/Artefakt noch nicht angebunden |
| Architektur-Kernel | Vergleich auf Op-/Profil-Ebene offen | Eigene GDN/QSA/HC-Pfade fuer sm_89 | GGML-/CUDA-Ops vorhanden; Fusion/Layouts nicht pauschal gleichwertig |

### 4.1c Strata-Ablation (1 GPU, IQ3_XXS, eingebaute A/B-Arme)

Strata bringt eigene Ablation-Flags mit — jede Zeile = Engine-Neustart + Bench
(3 Runs, Median; `strata-ablate.jsonl`):

| Variante | Was entfällt | pp @0.8k | pp @5.8k | pp @23k | tg |
|---|---|---:|---:|---:|---:|
| v0 baseline | — | 200 | 877 | 1154 | ~47 |
| v1 `--mmap-experts` | Resident-Arena → OS-Paging | 164 (−18 %) | 645 (−26 %) | 797 (−31 %) | ~28 (**−43 %**) |
| v2 `--expert-cache 0` | VRAM-Expert-Cache | 199 (−1 %) | 873 (−0 %) | 1162 (+1 %) | ~47 (±0 %) |
| v4 `--prefill 512` | grosse Chunks 8192→512 | 139 (−30 %) | 163 (**−81 %**) | 153 (**−87 %**) | ~48 (±0 %) |
| v7 ohne `--remote-expert-opt` | Remote-Expert-Decode-Opt | 199 (−0 %) | 875 (−0 %) | 1165 (+1 %) | ~50 (±0 %) |

Nicht über die Serve-API ablatierbar (Engine-Bisect-Tools für `strata generate`,
kein HTTP-Pfad): `--no-token-graph`, `--no-capture` (Serve braucht Token-Graph),
`--no-pool` (verify braucht VRAM-Expert-Tier, bei `auto` kein Platz),
`--no-fused-gr` (Session kam hoch, Modell registrierte sich aber nicht rechtzeitig).

**Interpretation und Grenzen der Ablation:**

1. **Grosse Chunks sind ein starker Prefill-Hebel innerhalb Stratas.** Mit
   `--prefill 512` sinkt die Rate auf 153–163 t/s. Das ist eine 81–87%ige
   *Durchsatzdifferenz*, kein Beweis fuer einen 81–87%igen Laufzeitanteil des
   Rings/Streamings: Chunkgroesse beeinflusst auch GEMM-Auslastung,
   Quant-Kernelwahl, Launch-Overhead und Speicherbedarf. Die Vergleichszahl
   267 des Forks stammt zudem von UD-IQ4_XS statt IQ3_XXS.
2. **Der mmap-A/B-Pfad ist langsamer:** ~28 statt ~47 t/s. Residenz,
   Page-Cache, Pinning und Transfer-Verhalten sind dabei nicht einzeln
   isoliert. Die Faktoren duerfen weder addiert noch mit einem ungemessenen
   ~2x-MTP-Anteil zu einer exakten Gesamtzerlegung multipliziert werden.
3. **`--expert-cache 0` zeigte hier keinen Effekt.** Vor einer Cache-
   Schlussfolgerung die *effektiven* Engine-Logs/Slots des v2-Laufs pruefen:
   Serve/Auto-Config kann Optionen ergaenzen; ein uebergebenes Flag beweist
   nicht, dass der Zielmechanismus wirklich aus war. Selbst ein bestaetigtes
   Nullergebnis waere auf diese Quantisierung, Hardware und Workload begrenzt,
   kein allgemeines Verbot von MoE-Caches oder Beweis fuer korrekten Fork-Code.
4. `--remote-expert-opt`: auf 1 GPU ohne messbaren Effekt; keine Aussage zum
   Multi-GPU-Pfad. Kernel-/MTP-Anteil bleiben ohne geeignete interne A/B-
   Messung offen. Umsetzungs-/Verifikationsplan: Bezugsplan Abschnitt 10.

### 4.2 Vergleichslatte, nicht zugesicherter Port-Ertrag

- **Same-Quant:** Strata UD-IQ4_XS ~1227 pp t/s @23k gegen Fork ~267
  (~4.6x); Decode bisher ~45 gegen ~11.5, wegen teils kurzer Fork-
  Sequenzen weniger belastbar. Kein IQ3_XXS/UD-IQ4_XS-Gemisch als Engine-Gain.
- **Nicht identifiziert:** getrennte Zeitanteile von MTP, Expert-Streaming,
  GDN/QSA/HC-Kerneln, CPU-Pool und Host-I/O. Daher keine belastbare
  multiplikative Zerlegung und keine gemessene 1.5–2.5x-Port-Prognose.
- Ziel ist die Strata-Vergleichslatte auf identischem Modell/Hardware
  anzunaehern. Die erreichbare Fork-Rate wird nach P1-Fixes und getrennten
  Streaming-/Residenz-/MTP-Gates gemessen; Strata ist weder ein bewiesenes
  Hardwaremaximum noch eine garantierte Obergrenze weiterer Optimierung.

### 4.3 Entscheidungsmatrix (aus Plan, auf Messung angewandt)

| Schwelle | Messwert | Konsequenz |
|---|---|---|
| ≥2× | **erfüllt**: 4.3–4.6× pp, ~4× tg | Selektive Adaption lohnt sich |

Priorisierung fuer den Folgeplan (Bezugsplan Abschnitt 10):

1. **P1-Blocker zuerst:** IQ3_XXS-Offset und Qwen4Exp-Prefetch-Crash
   reproduzieren, RCA und kleine Regressionstests vor dem Fix.
2. **Prefill:** grosse physische Chunks und budgetierte Expert-Transfer-
   Pipeline separat von bestehendem Prefetch messen; keine zugesagte 4–6x-Rate.
3. **Residenz:** zuerst vorhandenes `--no-mmap`, warmes mmap und Host-
   Registrierung vergleichen; nur bei Zusatznutzen eine RAM-budgetierte
   resident-pageable/pinned Expert-Arena implementieren.
4. **Flash-Next-MTP:** Der Server unterstuetzt schon externe Draftmodelle
   (`--spec-type draft-mtp --spec-draft-model`). Fuer Qwen4Exp fehlen
   kompatible Head-Artefakte, Mapping/Graph und Hidden-State-Anbindung;
   K2-Sidecar und Gemma-Async-Pipeline sind Wiederverwendungsreferenzen.
   Netto-Decode-Gewinn erst gegen target-only messen, kein belegtes ~2x.
5. **Cache und Kernel:** nicht pauschal verwerfen. Effekten nachgehen,
   wenn Profiling/effektive Config einen Engpass belegen; Strata-Source-
   Ports komponentenweise nach Layout-/Backend-/Parity-Pruefung.
6. **Betriebsoption:** Strata als Zweit-Engine via llama-swap bleibt eine
   moegliche separate Integrationsaufgabe, kein bereits deployter Zustand.

Kein blinder Gesamtmerge: OptLlama-Divergenz/Stabilitaet und Strata-
Layout-/Runtime-Unterschiede sprechen fuer komponentenweise, profiliert
begruendete Ports. Das ist **kein** generelles Verbot von Kernel-Adaption;
bei nachgewiesenem Restengpass sind gezielte Layout-/Kernel-Ports im Plan.

---

## 5. Gefundene Bugs / Limitierungen (Fork)

1. **GGUF-Offset-Fehler (P1, RCA gefunden 2026-10-07):** ISTA-DASLab
   `GSQ-RCO IQ3_XXS` lädt nicht:
   `tensor 'blk.1.ffn_down_shexp.weight' has offset 2004206976, expected 1991099776`.
   **Ursache: Typnummer-Kollision** — die Datei nutzt upstream `GGML_TYPE_Q2_0=42`
   (block 64, 18 B) fuer die `ffn_down_exps`/`shexp`-Tensoren; der Fork belegt
   42 mit eigenem `TURBO2_0` (block 128, 34 B) → systematische Offset-Drift
   (`n_el/64` pro Q2_0-Tensor). Datei ist wohlgeformt (alle Offsets sequentiell,
   Slack 0; Quell-Checksummen bestaetigt). Q2_0-Support fuer CPU/CUDA und
   Python implementiert; nur TURBO2 wird von 42 auf 47 verschoben, TQ3/TQ4
   bleiben kompatibel. **Gate P1a bestanden 2026-10-07:** CUDA-Backend-
   Paritaet 134/134 je RTX-4060-Ti, echte IQ3_XXS-Dual-GPU-Inferenz mit
   kohaerenter Ausgabe; IQ3_XXS ist damit als Same-Quant-Benchmark
   nutzbar. Nebenbefund: `fattn` lehnt Q8_0/F16-Misch-KV-Typen nicht
   sauber ab (pre-existing, kein Q2_0-Regressionsfund). Details: Plan §10.2.
2. **Prefetch-Crash (P1):** `GGML_SCHED_PREFETCH_EXPERTS=1` → `CUDA error: requested
   functionality is not supported` in `ggml_cuda_op_mul_mat_cublas` (ggml-cuda.cu:1792)
   beim ersten Request mit Expert-Prefetch. **Behoben 2026-10-07:** compute-
   sanitizer zeigte den eigentlichen Fehler — `mul_mat_q` (MMQ) liest wenige
   Bytes ueber das Tensorende hinaus (der Selektivpfad kopiert deshalb bis zu
   512 Extra-Bytes); die exakt `ggml_nbytes` grossen standalone Prefetch-Slots
   boten keinen Slack → unmapped Page. Fix `3f3d272f69`: Slot-Padding +512 B,
   Ende synchron genullt; `prefetch_max_size` scannt alle Split-Nodes.
   Nebenbefund `819af39886`: Segfault in `ggml_cuda_graph_check_compability`
   durch `node->src[0]->type`-Deref auf src-lose Ops (ARANGE). Gate bestanden:
   5 Prefetch-Laeufe ohne Crash, `test-backend-ops` 15840/15840, 3/3 Backends.
   Details: Plan §10.3.
3. **`-moe-cache`-Flag-Ordering (P2):** `--moe-cache 0` hinter `-ngl` greift zu spät —
   Backend-Registrierung läuft schon im `-ngl`-Parser (llama_supports_gpu_offload) und
   liest `GGML_CUDA_MOE_CACHE` bevor das Flag es setzt. Workaround: Env direkt.
4. **F6-OOM (P2):** `--n-cpu-moe 40` @`-b8192 -ub4096` ctx 40960 → device1 cudaMalloc
   2.3 GiB OOM bei ~23k Prompt. Fit-Logik berücksichtigt GPU-residente Expert-Lagen
   nicht beim Workspace-Headroom.
5. **TurboQuant-Decode-Einbruch (P2):** turbo4/3-KV halbiert Decode @23k (~4 vs. 11 t/s) —
   für dieses Modell nicht empfohlen; pp-Gewinn bleibt.
6. **Getesteter MTP-Aufruf nicht nutzbar (Info):** UD-IQ4_XS-GGUF enthält keine
   eingebetteten NextN-Layer (`n_layer_nextn == 0`); der Aufruf ohne externen
   Draft scheiterte. Externe Sidecars werden vom Fork schon unterstuetzt
   (K2/Gemma); fuer Flash-Next braucht es ein kompatibles Artefakt und
   Architektur-/Hidden-State-Anbindung. Strata bereitet seine Head-Dateien
   aus vorhandenen Checkpoint-Gewichten vor.

## 6. Betriebs-Notizen uranus

- `host-agent-reconcile.timer` (120 s): stellt Produktiv-Services wieder her und
  **kills "verwaiste" GPU-Prozesse** — bei jeder Bench-Session zuerst stoppen.
- Strata-Setup auf uranus: `~/Strata` (Engine kompiliert, Packs unter `~/Strata-data/`,
  `run-iq3_xxs.sh` / `run-ud-iq4_xs.sh` — PLE-mlock-Warnung wegen ulimit, unkritisch).
- OptLlama: `~/optllama` (Branch `moe-cache`, Build in `build/`).
- Bench-Rohdaten: `uranus:~/eval-flashnext/results/*.jsonl`
  (134 Runs inkl. Fehler/Crash-Protokoll), Prompts + `bench_measure.py` dort.

## 7. Vergleichs-Caveats

- Strata-Decode wurde mit MTP-Spekulation gemessen, Fork/OptLlama ohne Spec;
  Feature- und Kernel-Anteile am Decode-Gap sind nicht separat quantifiziert.
- IQ3_XXS (Strata/OptLlama) vs. UD-IQ4_XS (Fork) — der 1:1-Quervergleich ist
  Strata-UD vs. Fork-UD: **1227 vs. 267 pp t/s @23k bzw. ~45 vs. ~11.5 tg**.
- Strata nutzt PLE-Ngram-Table (32 GB) + adaptiven Draft — Decode-Variabilität höher.
- Einzelne Runs mit predicted_n <16 wurden für Decode-Median herausgefiltert
  (EOS-Kurzsequenzen); pp-Werte unbeeinflusst.
