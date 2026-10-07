# SESSION_PLAN: Strata + OptLlama A/B-Eval auf Uranus (Qwen3.8-Flash-Next)

**Erstellt:** 2026-10-06
**Status:** [x] Vergleich abgeschlossen (2026-10-07); Folgeplanung fuer Fork-Optimierungen in Abschnitt 10 (noch nicht implementiert)
**Bezug:** Trilium `TaWve7nWU3kF` (Strata-Notiz) + Subnote `5Ec9sW4aCN6S` (Eval-Bewertung); ROADMAP #99 Phase 9
**Ziel:** Empirische Obergrenze der MoE-Offload-Performance fuer Qwen3.8-Flash-Next auf Uranus-Hardware bestimmen — als Entscheidungsgrundlage, ob/welche Strata- oder OptLlama-Techniken in den Fork portiert werden

## 1. Kontext

- **Strata** (`Niko1221/Strata`, MIT, ~15,5k★): Single-Model-Engine fuer Qwen3.8-Flash-Next. Verifizierte Kernmechanismen: Prefill-Chunks bis 8192 Tokens, Layer-N+1 async Expert-Prefetch aus gepinntem RAM, Stream-all-Experts, adaptiver VRAM-Expert-Cache (~700 Exp./GB), hybride Miss-Berechnung (GPU + PCIe + CPU), fused GDN/QSA-Kernel, MTP (2,4–3,2 tok/pass). Multi-Backend (CUDA sm_61–sm_120, HIP, SYCL). Behauptete ~8x Prefill vs Stock-llama.cpp ist **nicht repo-belegt**.
- **OptLlama** (`GenerelSchwerz/llama.cpp`, Branch `moe-cache`, MIT): eigenstaendiger MoE-Expert-Cache — VRAM-Slots + LFRU-Eviction, Early-Router-Prefetch, Host-Budget-Accounting, grouped-decode MMQ/MMVQ, CUDA-Graph-kompatibel, saubere Backend-Schnittstelle (`ggml-backend-moe.h`). Eigen-Benchmarks ~2–3x Decode bei MoE > VRAM. **CUDA-only.**
- **Fork-Bestand:** thecodacus Pinning+Prefetch, `moe-cache.cu` (LRU/Heuristic/Set-Assoc/Workload), `-ubp`, NextN-MTP, `qwen4exp`-Arch (CPU-Baseline verifiziert). **Uranus wurde nie als moe-cache-Host vermessen** (Cache ist fuer Ampere+ mit viel VRAM designed).

## 2. Testgegenstand & Modelle

- **Modell:** Qwen3.8-Flash-Next — ~125B MoE, 512 Experten/Layer × 48 Layer = 24.576 Expert-Instanzen, 10 aktiv, GDN + QSA + Gated Residual + eingebettetes MTP, Vision-faehig (Qwen4-Preview).
- **GGUFs:**
  - `UD-IQ4_XS` (3 Shards, ~88 GB) — vorhanden auf Uranus (RAID5-Modellverzeichnis) und ganymed (`unsloth/`). Vor Test Existenz + Shard-Vollstaendigkeit verifizieren.
  - `GSQ-RCO IQ3_XXS` (2 Shards, 70,6 GB) + `mmproj-BF16` (~0,9 GB) — Download auf ganymed laeuft (2026-10-06); bei Bedarf rsync → Uranus.
- **Architektur-String im GGUF pruefen** (erwartet `qwen4exp`; Fork-#99-Port zielt darauf). `convert_hf_to_gguf.py`-Unterstuetzung nicht noetig (GGUFs fertig).

## 3. Hardware-Constraints (Uranus)

- 2× RTX 4060 Ti 16 GB (Ada, **sm_89**), PCIe 4.0 **x8** je Karte (~13–16 GB/s real) — exakt der Engpass, fuer den Strata optimiert ist → idealer A/B-Host.
- 128 GB RAM, 8-Kern-CPU — gepinnte 35–55 GB (Strata) problemlos; CPU-Miss-Compute begrenzt beide Engines.
- RAID5 24 TB — Modelle + ~29 GB ngram-Tabelle (Strata, optional) passen.
- Bekannt: Dauerlast-Instabilitaet (cublas-Crashes ~60–80 min) → Runs ≤60 min, Restart-Wrapper.
- Regeln: Remote-Host vor Arbeit auf aktuellen master-Stand bringen (AGENTS.md); kein paralleler Betrieb zweier GPU-Server pro Karte.

## 4. Messplan

### Phase A — Vorbereitung (offline moeglich)

- [x] IQ3_XXS-Download auf ganymed ✅ abgeschlossen 2026-10-06 23:41 (2 Shards, exakte Groessen verifiziert; + mmproj-BF16)
- [x] Benchmark-Prompts generiert: `docs/qwen38-eval/prompts/bench/bench-pp-{1k,4k,8k,32k}.txt` (deterministisch, gemischter Inhalt — Prosa/Code/Tabellen, kein naives Repeat-Pattern). **Messmethodik:** exaktes Token-Trimming unnoetig — `usage.prompt_tokens` aus der Response ist die x-Achse; alle Engines bekommen identische Dateien.
- [x] Strata-Build-/Betriebs-Anforderungen recherchiert (siehe Phase B — Setup via `setup.sh` ohne Source-Build moeglich)
- [ ] OptLlama-Build: Standard-CMake `-DGGML_CUDA=ON`
- [ ] Identische Sampling-Parameter definieren (temp 0.7, top_p 0.95 — Qwen3.8-Eval-Standard)
- [ ] memlock-Limit auf Uranus pruefen/setzen (`ulimit -l unlimited` bzw. `/etc/security/limits.conf`) — Strata pinned 35–55 GB

### Phase-A-Befund: Referenzwerte fuer identische Hardware existieren bereits

Strata Issue **#1062** misst exakt unsere Hardware-Klasse (2× RTX 4060 Ti 16 GB, 128 GB RAM, UD-IQ4_XS @131K):
**~900 / 1.930 / 2.230 tok/s Prefill @ 4K/32K/128K; ~52–57 tok/s Decode; TTFT @128K ~58 s; VRAM ~15,7 GB je Karte.**
→ Diese Zahlen sind die Sanity-Check-Latte: Liegt unsere Strata-Messung darunter, stimmt das Setup nicht; darueber waere ueberraschend.

**Strata-Betrieb (verifiziert gegen Repo-Doku/Issues):**

- **Kein Source-Build noetig:** `./setup.sh --family unsloth --model UD-IQ4_XS --context 131072 --gpus 0,1 --yes` laedt fertige Engine (pip, ~0,4 GB) + Modell. Source-Fallback: `cmake -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` (Default waere sm_120 — explizit setzen!). Eigene GGUFs: `./setup.sh --gguf-dir <dir>` (Shard-Originalnamen beibehalten).
- **Server:** Python `serve/` auf :8080, OpenAI `/v1/chat/completions` + `/v1/responses`, Anthropic `/v1/messages`; llama.cpp-kompatible `timings` in Response + `/metrics`, `/slots`. Config = `strata-<model>.json`.
- **PLE/ngram-Tabelle (~29 GB):** Pflicht-Asset auf SSD (`--ple-io direct` Default; auf HDD → Stall #605 → `--ple-io ram` nutzen, RAM reicht). `--no-ple` nur als A/B-Schalter (Qualitaetsverlust).
- **Multi-GPU:** Pipeline-/Layer-Split (kein TP, kein P2P noetig): `"gpu":[0,1]`, `"layer_split":"auto"` → typisch 0–22/23–47.
- **Determinismus (opt-in):** `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0 --adapt-every 0` + `STRATA_IQ_MT_MIN=1` (~1–3% langsamer). Default-Pfad ist absichtlich nicht byte-stabil.
- **Fallstricke:** `--kv k8v4`+`--kv-resident` kaputt auf 0.1.40 (#1188 → `int8` KV nehmen); Issue #1250 Host-OOM bei <64 GB RAM (bei 128 GB unkritisch); `--vram-reserve-mib 2048` falls Desktop auf einer GPU; `--parallel 2` macht alles langsamer → bei 1 Request bleiben; UD-IQ4_XS braucht Engine >=0.1.38.
- **Fertige Bench-Skripte:** `benchmark.py` + `monitor.py` in `bench/results/2026-09-30-community-rtx-5090/` (Community-Methodik: 3 Runs, 256 greedy Tokens, 4K/32K/128K-Sweep); `tools/needle_bench.py` fuer Needle-Tests.
- **llama-swap:** Referenz-Config in offenem PR #1132 (`docs/LLAMA_SWAP.md`); `allowed_hosts`/`STRATA_ALLOWED_HOSTS` setzen (403-Falle); Load ~80–105 s → `healthCheckTimeout: 300`.

### Phase B — Strata originaer (Referenz-Latte)

- Install: `git clone https://github.com/Niko1221/Strata && cd Strata && ./setup.sh --family unsloth --model UD-IQ4_XS --context 131072 --gpus 0,1 --yes` (Modell-Download ueberspringen/umbiegen auf lokale Shards — UD-IQ4_XS liegt bereits auf Uranus-RAID5; `--gguf-dir` nutzen). Disk-Bedarf ~80–120 GB frei halten (PLE-Tabelle ~29 GB kommt dazu).
- Optional zweiter Lauf mit GSQ-RCO IQ3_XXS von ganymed (rsync) — schnellerer Pfad laut Strata-Doku.
- Messpunkte: **pp t/s @ bench-pp-{1k,4k,8k,32k}.txt**, **tg t/s Decode** (>=200 Tokens), **TTFT**, RAM-/VRAM-Belegung, 3 Wiederholungen, Loadzeit. Via API (OpenAI-kompatible `timings`-Felder) — Community-Skripte `benchmark.py`/`monitor.py` aus `bench/results/` nutzen oder eigenes Skript.
- Sanity-Check gegen Issue-#1062-Werte (~900/1930/2230 tok/s pp @4K/32K/128K, 52–57 t/s tg).
- Determinismus-Modus separat (langsamer, nur fuer Qualitaetsvergleich).
- Qualitaet: `tools/needle_bench.py` @32k + Output-Sichtpruefung.

### Phase C — OptLlama `moe-cache`

- Build: Branch `moe-cache`, `-DGGML_CUDA=ON`, sm_89
- Flags: `--moe-expert-cache-mib` (Sweep), `--moe-expert-cache-host-pinned-mb`, ggf. `--spec-draft-moe-expert-cache-*`
- Gleiche Messpunkte wie Phase B
- **Achtung:** Issue #120 — grouped-MoE-Crash auf genau diesem Modell/Quant (UD-IQ4_XS); Fix-Stand vor Test pruefen

### Phase D — Fork-Baseline (Kontrollgruppe)

- Build: Fork `master`, CUDA sm_89 (Build-Verifikation gemaess AGENTS.md)
- Config-Matrix (kombinatorisch, abgestuft):
  - Offload: `-ngl`-Stufen + `--n-cpu-moe` Sweep
  - `GGML_CUDA_REGISTER_HOST=1`, `GGML_SCHED_PREFETCH_EXPERTS=1` (2 Slots)
  - `--moe-cache` + `-ncmoe` Budget-Sweep
  - `-ubp` Sweep {512, 2048, 4096, 8192}
  - KV: `f16` als Vergleichs-Baseline (Fairness), danach `turbo4/3` als Fork-Bonus separat
- Gleiche Messpunkte; entspricht #99 Phase 7 (CUDA-Baseline) + ersetzt/erweitert Phase 9 um externe Referenzen

### Phase E — Auswertung

- Tabelle: Engine × Config × {pp@1k/4k/8k/32k, tg, TTFT}
- **Obergrenze = bestes Strata-Ergebnis**; Delta = Strata ÷ Fork-Best-Config → Port-Entscheidung

## 5. Entscheidungsmatrix

| Delta (Strata vs Fork-Best) | Aktion |
|---|---|
| >= 2x pp | Scheduling-Ports priorisieren: Prefill-Streaming-Policy im moe-cache, Early-Router-Prefetch, Chunk-Scaling >512. Fused-Kernel-Ports als separates Tier-3-Item evaluieren |
| 1,2–2x | Quick-Wins: ubp-Tuning + Prefetch-Tiefe; Strata-Entwicklung beobachten; kein Kernel-Port |
| <= 1,2x | Fork-Stack ausreichend; ggf. Dual-Engine via llama-swap fuer Flash-Next-Workloads erwägen statt Port |
| Strata instabil/defekt auf Ada | OptLlama-Delta als Fallback bewerten (LFRU, Early-Router, Host-Budget als Ideen-Port) |

OptLlama-Delta separat: falls OptLlama > Fork-moe-cache signifikant → **Ideen-Port** (LFRU als 5. Policy in `moe-cache.cu`, Early-Router), **kein Merge** (Konfliktflaeche: gleicher Dateiname `moe-cache.cu`, Prefetch/Pinning-Env-Vars, `llama_model_params`).

## 6. Risiken & offene Fragen

- Strata nicht-deterministisch → Qualitaetsvergleich via KL/Needle, nicht Token-Equality.
- OptLlama Issue #120 (grouped-MoE-Crash auf diesem Modell/Quant) — Fix-Stand pruefen.
- 8-Kern-CPU drueckt CPU-abhaengige Pfade (beide Engines) → Ergebnis ist untere Schranke gegenueber staerkeren Hosts.
- Strata evtl. schneller mit GSQ-RCO-Quants (optimierte Pfade) → bei Zeitbudget beide GGUFs messen.
- MTP: Der getestete UD-IQ4_XS-GGUF hat keine eingebetteten NextN-Layer. Der Fork unterstuetzt externe Draftmodelle; Flash-Next-Artefakt/Graph-Anbindung separat pruefen (Abschnitt 10.7). Strata nutzt einen vorbereiteten Head; Akzeptanzraten sind nicht gleich End-to-End-Speedup.

## 7. Erfolgskriterium (Gate)

Abgeschlossen wenn: pp/tg-Matrix fuer >= 2 der 3 Engines gefuellt, Obergrenze quantifiziert, Port-Go/No-Go mit Begruendung in ROADMAP + Trilium-Subnote `5Ec9sW4aCN6S` eingetragen.

## 8. Doku-Ziele nach Abschluss

- Ergebnisse → Trilium Benchmark-Note (Wegweiser `8cIsjKEhz7Fd` beachten — nie in Modell-Info-Notes)
- ROADMAP-Item je nach Ergebnis anlegen/aktualisieren
- `docs/fork/ROADMAP.md` #99 Phase 9-Status verlinken

---

## 9. ERGEBNIS (2026-10-07, uranus) — ✅ durchgefuehrt

**Vollstaendiger Bericht:** `docs/qwen38-eval/ENGINE_EVAL_20261007.md`
(Rohdaten: `uranus:~/eval-flashnext/results/*.jsonl`, 134 Runs)

**Messung, Median der besten Configs, Prompt ~23k Tokens:**

| Engine | pp t/s | tg t/s | Gap vs. Fork-Best |
|---|---|---|---|
| Strata UD-IQ4_XS (gleiches Modell wie Fork) | 1227 | ~45 | **~4.6x pp / ~4x tg** |
| Strata IQ3_XXS | 1728 | ~65 | ~6.5x pp / ~5.6x tg |
| OptLlama IQ3_XXS Cache | 366 | ~15 | ~1.45x pp |
| Fork UD-IQ4_XS best (F7: --n-cpu-moe 44 + cache) | 267 | ~11.5 | 1.0 |

**Gate-Ergebnis: >= 2x erfuellt (4x+) → selektive Scheduling/Logistik-Adaption lohnt.**
Empfohlene Reihenfolge: llama-swap-Sofortmassnahme → Prefetch-Fix + Streaming-Prefill →
Resident-RAM-Experten-Arena → MTP-Sidecar → n-cpu-moe-Auto-Fit.

**Neu gefundene Fork-Bugs:** IQ3_XXS-GGUF-Offset (`ffn_down_shexp`),
Prefetch-Crash auf Qwen4Exp (`ggml_cuda_op_mul_mat_cublas`), `--moe-cache`-Flag-Ordering,
n-cpu-moe-40-OOM @23k, TurboQuant-Decode-Einbruch @23k (~4 t/s).

## 10. Folgeprojekt: reproduzierbare Fork-Optimierung (Umsetzung begonnen)

**Auftrag:** P1-Blocker zuerst; danach Prefill-Streaming, RAM-adaptive Expert-Residenz
und Flash-Next-MTP. Ziel ist *messbar bessere* Leistung des generischen Forks auf
Qwen4Exp und weiteren MoE-Modellen, nicht eine vorab zugesagte Strata-Paritaet.
Die abgeschlossene Engine-Evaluation (§9) bleibt unveraendert; dies sind neue
Implementierungs-Gates. Keine Produktions-Checkouts fuer Entwicklung aendern.

### 10.1 Was die Ablation belegt — und was nicht

- Strata, **eine** 16-GB-Ada-GPU, **IQ3_XXS**: `--prefill 512` statt auto
  senkte pp @23k von 1154 auf 153 t/s; `--mmap-experts` senkte pp auf
  797 und tg von ~47 auf ~28 t/s. Das sind *Differenzen innerhalb Stratas*,
  keine isolierten Laufzeitanteile von Ring, DMA, Pinning oder GEMM. Kleinere
  Chunks veraendern auch Launch-Overhead, Quant-MM-Auslastung und Cache-Verhalten.
  Effekte sind nicht additiv und nicht 1:1 auf ggml uebertragbar.
- `--expert-cache 0` hatte im gelabelten Strata-1-GPU-Lauf keinen
  messbaren Effekt; vor Interpretation wirksame Engine-Args, Cache-
  Slots und Serve-Autokonfiguration nachpruefen. Kein allgemeines
  Cache-Nullergebnis behaupten. Der Fork-Cache ist ein anderer
  Ausfuehrungspfad; Cache-Budget/Hit-/Miss-Zahlen mitkoprotokollieren.
- Ein separater Draft-Head ist **vortrainiert und wird beim Strata-Setup
  extrahiert/konvertiert**, nicht bei Inferenz neu erfunden oder trainiert.
  Die bisherigen ~2x-MTP-Beitraege sind nicht mit einem echten A/B isoliert.
- "GGUF-Reader zu strikt" ist durch die nachgewiesene Typkollision
  widerlegt (§10.2). Die genaue Prefetch-Crash-Ursache bleibt offen.
  Kein Deaktivieren der Offset-Pruefung und kein globaler Graph-Disable als Fix.

### 10.2 Phase P1a: IQ3_XXS-GGUF-Offset sicher diagnostizieren und beheben

**STATUS 2026-10-07 — Diagnose abgeschlossen, Ursache bewiesen:**

Der Fehler ist **keine** Reader-Striktheit und **keine** defekte Datei,
sondern eine **Typnummer-Kollision** im ggml-Enum:

- Die Datei verwendet **upstream `GGML_TYPE_Q2_0 = 42`** (block 64, 18 B
  → 0.28125 B/Element) fuer 38 einzelne `ffn_down_exps`-/`ffn_down_shexp`-
  Tensoren in Shard 1.
- Der Fork belegt Typ 42 mit **`GGML_TYPE_TURBO2_0`** (block 128, 34 B
  → 0.265625 B/Element) und kennt `Q2_0` gar nicht.
- Rechnung bestaetigt: Drift pro Q2_0-Tensor = `n_el/64` Byte; beim ersten
  betroffenen Tensor genau +12.5 MiB → der gemeldete Offset-Fehler.
- Mit `42 = Q2_0 (64/18)` sind **alle 1223+1 Offsets perfekt sequentiell**
  (Slack = 0, beide Shards wohlgeformt). Datei ist valide.

**Implementierung 2026-10-07 — lokale Tests bestanden, Modell-Gate noch offen:**

1. Nur die tatsaechlich kollidierende Nummer aendern: `Q2_0=42`,
   `TURBO2_0=47`, `GGML_TYPE_COUNT=48`. TURBO3/4 und die serialisierten
   TQ3_1S/TQ4_1S bleiben bei 43/44/45/46. Keine pauschale Migration auf
   >=64 und keine Garantie gegen kuenftige Upstream-Kollisionen.
2. Q2_0-Block/Traits, Quant-/Dequant-Referenz und CPU-Q2/Q8-Dot-Product
   nach Upstream `18b5f8b186` portiert; CUDA-Dequant, GET_ROWS, MMVQ und
   MMQ an die bestehende Backend-API angepasst. Python-Reader und
   Python-Quantisierung ergaenzt. Vulkan-TURBO2-Typkonstante synchronisiert;
   Weitere native Q2_0-Backend-Pfade sind nicht portiert oder validiert.
3. Typnummern sind Bestandteil von GGUF, KV-Session-State und RPC.
   Alle Bibliotheken/Clients gemeinsam neu bauen. Alte TURBO2-KV-Sessions
   werden vom bestehenden Typvergleich abgewiesen; TQ3-/TQ4-GGUFs behalten
   ihre Nummern. Alte manuell erzeugte TURBO2-GGUFs mit Wire-Typ 42 sind
   ohne Formatmarker nicht eindeutig von Q2_0 unterscheidbar und nicht
   automatisch migrierbar. Eine Dateinamensuche beweist keine Abwesenheit.
4. Regression: unabhaengig gepacktes Typ-42-GGUF mit zwei Tensoren scheitert
   vorher mit `offset 1152, expected 1088`; nachher `test-gguf` 88/88.
   Bestehende Negativfaelle unveraendert. CPU-Backend-Q2-Tests 160/160
   inklusive direkt gepacktem `+2d` fuer MMVQ-/MMQ- und Expert-ID-Faelle;
   Quant-Funktionen ohne Fehler; TurboQuant-Roundtrips 17/17; Python/C
   bitgleiche Quantisierung/Dequantisierung plus manuell gepackte Codes
   0..3. CPU-, CUDA-sm_89- und Vulkan-Build erfolgreich.
5. Echte IQ3-Shards und mmproj per `hf cache verify` gegen die Quelle
   geprueft (3 passende Checksummen; andere Quantisierungen lokal bewusst
   nicht vorhanden). Quellrepo `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`,
   Revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
6. Review-Runde 1: CPU/GGUF/Python bestaetigt; fehlender Quantize-CLI-
   Eintrag ergaenzt. Verdacht auf CUDA-`__byte_perm`-Fehlentpackung mit
   unabhaengigem sm_89-Probe ueber alle 65536 Zwei-Byte-Muster widerlegt:
   Original und zusaetzlich maskierte Version jeweils null Abweichungen.
   KV-Restore-Typvergleich vor Payload-Interpretation im Code bestaetigt.
   Runde 2 bestaetigt CLI und vier neue direkt gepackte Backend-Faelle:
   keine P0/P1, fuer den lokalen Implementierungs-Scope freigegeben.
7. Noch offen: Laden beider echter Shards und
   CUDA-Inferenz-/Logit-Sanity. Vorher keine Same-Quant-Performancezahl und
   keine P1a-Freigabe. Nach Entwicklungsrechner-Neustart Tests erneut
   ausfuehren; Logs dauerhaft im gitignorierten Build-Verzeichnis halten.

**Reproduktion (historisch):** denselben Zweiteiler mit *exakt* dem
bisherigen Fork-Commit gegen OptLlama/gguf-py testen; Fehlstelle in
`ggml/src/gguf.cpp:761-782`: `blk.1.ffn_down_shexp.weight`, offset
2004206976 statt 1991099776.

1. Dateigroesse/Hash der Shards gegen die Quelle vergleichen; GGUF-Header ohne
   70-GB-Kopie inspizieren. Fuer die ersten abweichenden Tensoren Typ, Dimensionen,
   `ggml_nbytes`, GGUF-Alignment, gespeicherte Offsets und reale Dateigrenzen
   ausgeben; mit unabh. Reader vergleichen. Zuerst den *vorigen* Tensor finden.
2. Ein kleines reproduzierbares Test-GGUF in `tests/test-gguf.cpp` erzeugen:
   dieselbe Layout-/Typ-/Offsetkonstellation inklusive Negativfaellen fuer
   Overlap, Overflow, unaligned Out-of-Range und abgeschnittene Daten.
   Erwartung gegen GGUF-Spezifikation und externe Reader abgleichen.
3. Ursache gezielt korrigieren: ggf. Quant-Blockgroesse, Writer-Offset oder
   wohldefinierte Datenluecke. Falls Luecken zulaessig sind, nur geordnete,
   korrekt ausgerichtete, nicht ueberlappende Bereiche innerhalb der Datei
   akzeptieren und `ctx->size` aus dem maximalen *validierten Ende* berechnen;
   untrusted GGUF niemals pauschal durchwinken.
4. Gate P1a: `test-gguf` inkl. Bad-Offset-Faellen gruen, beide IQ3-Shards
   vollstaendig ladbar; CPU-only Modell-Smoke bei verfuegbarem RAM, danach
   GPU-Inferenz + Logit-/Text-Sanity gegen funktionierenden Referenz-Reader.
   Erst dann ist IQ3_XXS als fairer Same-Quant-Benchmark nutzbar.

### 10.3 Phase P1b: Expert-Prefetch-Crash isolieren und fixen

**Ist-Pfad:** `ggml/src/ggml-backend.cpp:1585-1669,1705-1753,1924-1935,2045-2060`.
Er kopiert *ganze* Host-Tensoren fuer `MUL_MAT_ID` ueber einen zweiten Backend-
Stream in bis zu 8 max-tensorgrosse VRAM-Slots, bedingt auf erwartete Expert-
Abdeckung. Der Slot-Alias (`input_cpy->buffer/data`) wird fuer den Graph-Launch
provisorisch gesetzt, danach wiederhergestellt. Das ist **noch kein**
Strata-8192er-Chunk-/Token-Ring; mehr Slots allein loesen das nicht.

1. Mit dem funktionierenden UD-IQ4_XS auf Ada `prefetch=0/1`, `-ub`-Stufen,
   Cache aus/an, CPU-MoE/Hybrid und Graph-Capture an/aus vergleichen;
   fehlschlagende Konfiguration und CUDA-Callsite/Stack erfassen (ggf.
   Compute-Sanitizer). Ursprung der Meldung `requested functionality is not
   supported` verifizieren: cudaHostRegister, CuBLAS-Quantpfad, Graph-Capture,
   Tensor-Alias, Device/Stream oder Event-Lifetime; nicht raten.
2. Test mit synthetischem `MUL_MAT_ID`-Split anlegen (auch vorausgehendem
   SCALE-Node, unterschiedlichen Quanttypen, 1/2+ Slots, Wiederverwendung,
   Callback, 2 Devices). Vor/nach Prefetch dieselben Logits/Outputs; bei
   fehlendem Host-Buffer/Backend-Feature korrekt zur normalen Kopie fallen.
3. Fix nur an der nachgewiesenen Stelle, mit klarer Synchronisation von
   Copy-Ready und Slot-Free, Fehlerstatus, Graph-Tensor-Lebensdauer und
   Cleanup bei Early-Return/OOM. Keine fremden GPU-Prozesse beenden.
4. Gate P1b: >=3 wiederholte Prefill-/Decode-Runs mit aktivem Prefetch
   (kurz/5.8k/23k), kein Crash/NaN, Korrektheit gleich Baseline,
   CPU/CUDA-Build und Backend-Tests gruen; realer Uplift separat berichten.

**Reihenfolge:** P1a kann auf einem Entwicklungsrechner CPU-only anfangen;
P1b braucht CUDA und grosses Modell (16-GB-Ada-Host fuer finalen Gate).
Kleine CUDA-MoEs duerfen fuer schnelle Vorabtests auf freien Test-Hosts
laufen; produktiv belegte Maschinen sind keine Test-Hosts.

### 10.4 Messinfrastruktur und Baselines nach den P1-Fixes

- Gepinnte Builds/Modell-Hashes, gleiche Quantisierung und Modellrevision:
  zuerst UD-IQ4_XS Strata/Fork, danach IQ3_XXS beide (wenn P1a gruen).
  `-c40960`, KV-Typ, `-b/-ub/-ubp`, CPU-Threads, GPU-Belegung und
  Offload-Layer pro Lauf protokollieren. Nicht `/completion` gegen Chat-
  Streaming-TTFT direkt gleichsetzen: zusaetzlich gleiche API/Template-/
  Tokenzaehlung und servereigene pp/tg-Zeiten erfassen.
- Promptlaengen ~0.8k/3k/5.8k/23k, mindestens drei Wiederholungen und
  kontrollierte Warm-/Kaltlaeufe; fuer Decode >=256 tatsaechlich generierte
  Tokens mit festen Fortsetzungsprompts (keine EOS-Kurzsequenzen im Median).
  Median plus Streuung, TTFT, VRAM/RSS, Major-Faults, H2D-Bytes/
  Bandbreite, Event-Wartezeiten, GPU-Compute-/Copy-Zeit, CPU-Pool-Zeit
  und effektive Chunk-Groessen festhalten. Qualitaet via Logit-/Needle-Tests.
- Zuerst **vorhandene** Alternativen fair pruefen: mmap kalt/warm,
  `--mlock` sofern erlaubtes Limit, `--no-mmap`, vorhandenes
  `GGML_CUDA_REGISTER_HOST`, Prefetch nach P1b-Fix, `-ub/-ubp` sweep.
  `--no-mmap` laedt bereits Host-Gewichte explizit
  (`src/llama-model-loader.cpp:1562-1667`); bestehende Registrierungs-
  Logik pinnt gemappte Host-Gewichte (`:1693-1717`). Neue Arena nur,
  wenn ein nachweisbarer Restengpass uebrig bleibt.
- Der grosse Ada-Testhost bleibt bis zur gebuendelten GPU-Validierung aus; Build/CPU-
  Tests auf Entwicklungsrechner, kleinere CUDA-MoEs nur auf freien
  Entwicklungs-/Test-Hosts. Produktivsysteme nicht beanspruchen.

### 10.5 Phase S1: grosse Prefill-Chunks + echte Expert-Streaming-Pipeline

**Zielbild:** Ein *generischer*, opt-in Prefill-Pfad fuer MoE-Modelle bei
Teil-Offload, nicht hart auf 512 Experten/48 Layer codiert. `-ubp 8192`
allein ist noch keine Loesung: ggml teilt den Graph in CPU/GPU-Splits,
ermittelt IDs auf dem normalen Pfad erst nach dem Router und kann vor
Expert-Uploads synchronisieren (`ggml-backend.cpp:1762-1830`). Zuerst
profilieren, welche Schranke in der 23k-Konfiguration wirklich bindet.

1. **Instrumentierung:** pro Layer und Expert-Tensor Gate/Up/Down:
   UBatch-Tokens, Zahl benutzter Experten, CPU-Router-Wartezeit,
   H2D-Bytes/-Zeit, Event-Wartezeit, Kernel-/CPU-MoE-Zeit, Peak-VRAM;
   NVTX/CUDA-Events ohne dauerhaften Hot-Path-Logspam. Unterscheide
   GPU-residente Layer (`--n-cpu-moe`) von CPU-Experten, und moecache an/aus.
2. **Chunk-Sweep als Kontrollgruppe:** 256/512/1024/2048/4096/8192
   logische und physische Tokens unter konstanter Quant/KV/Offload-/
   Cache-Konfiguration. Pruefen, ob das Modell/QSA/GDN/Ring-Kontext
   grosse ubatches tatsaechlich verarbeitet (nicht intern wieder auf 512
   geteilt) und ob Workspace auf 16 GiB passt. Kurz-Prompts duerfen
   nicht zugunsten langer Kontexte massiv regressieren.
3. **Ausfuehrungspfad:** Bei grossem Prefill, hoher erwarteter Router-
   Abdeckung und ausreichendem Speicher die aufeinanderfolgenden
   Expert-Tensoren *vor* ihrem CPU/GPU-Split in budgetierte Staging-
   Buffers bringen; asynchroner Copy-Stream, Compute-Stream wartet nur
   auf benoetigtes Ready-Event. Slot erst nach letztem Verbraucher
   freigeben. Decode/kleine Batches bleiben selektiv (kein `512 ×
   grosse Tensoren` pro Token). Falls Router-Abdeckung niedrig: nur
   benutzte Experten bewegen statt gesamten Tensor. Kein impliziter
   Device-uebergreifender Pointer-/Event-Missbrauch.
4. **VRAM-Budget statt fester Ringzahl:** 2-3 Slots als Minimalprototyp,
   Anzahl aus `(aktuell_freies_VRAM - noch_nicht_reservierte_Peaks -
   Reserve)/max_tensor_bytes` abrunden; bereits allokierte KV-/Graph-
   Buffers nicht doppelt abziehen. Vergroessern nur bei gemessenem
   Copy-Stall und ausreichendem Headroom. Stratas Log-"384-slot ring" *nicht* mit 384 max-grossen
   ggml-Expert-Tensor-Slots gleichsetzen. OOM faellt atomar zurueck
   auf sicheren Baseline-Pfad, kein halbfertiger Graph.
5. **Graph-/Multibackend-Vertrag:** Tensor-Adressen fuer CUDA-Graph-
   Capture stabil halten bzw. Capture gezielt und messbar deaktivieren;
   Stream/Event-Abhaengigkeiten mit `ggml-backend`-API, sauberes
   Slot-Ownership bei Cancel, Kontextwechsel und Multi-GPU-Layer-Split.
   CUDA-zuerst, auf anderen Backends unveraenderter Fallback.
6. **Ablations-Gates:** `baseline → grosse ubatch allein → P1b-prefetch →
   budgetierter Pipeline-Prototyp → Pipeline+Arena`; auf gleichem
   Modell getrennt pp/tg messen. Gegenprobe auf einem kleineren MoE
   und einem dichten Modell (kein regressiver Effekt), 1/2 GPU,
   1/mehrere Server-Slots. Nur bei Korrektheit, Stabilitaet und
   statistisch reproduzierbarem netto pp-Gewinn aktivierbar.

**Ausbaustufe S1b, wenn ganze Tensor-Slots nicht ausreichen:** Statt immer
mehr max-grosse Slots zu reservieren, Expert-Tiles mit zusammengehoerigen
Gate/Up/Down-Gewichten und gruppierten Token-Listen streamen. Router-
Zuordnung auf dem Device erhalten, Tokens pro Expert buendeln, in
budgetierten Buffern kopieren/rechnen/scattern; nicht Expert-Aktivierungen
ueber den Host hin-/hertransportieren. Ein eigener MMID-/grouped-GEMM-
Backend-Pfad darf erforderlich sein; ggml-Split-Prefetch allein ersetzt
nicht Stratas Execution-Engine. Erst vorhandene CUDA-MMQ/MMID-Kernels
pruefen, neue Kernel/Buffer-Layouts nur mit CPU-/CUDA-Paritaet einfuehren.

**Kostenmodell fuer weitere MoEs:** Auswahl zwischen selektiver Kopie,
vollem Tensor-Upload, persistenten GPU-Hits und CPU-Miss-Compute nach
Expert-Belegungszahl, Tensor-Bytes, gemessener H2D-Bandbreite, CPU-/GPU-
Rechenzeit und freien Budgets. Keine fixe 512-Expert-Schwelle und keine
unabhaengige Verdreifachung von Ring, Cache und Workspace. Der bestehende
MoE-Cache kann bereits GPU-Hit-Rows/CPU-Miss-Rows hybrid ausfuehren;
diesen Pfad bewusst einbeziehen statt eine zweite konkurrierende Engine
zu erzeugen. Heuristik erst opt-in, Auto-Modus nach Cross-Model-Gate.

**Restgap-/Revisionsbedingung:** Wird Prefill weiterhin von QSA/GDN/HC-
oder CPU-Rechenzeit statt Transport limitiert, diesen Engpass profilieren.
Fuer die teuersten Ops gezielte Strata-Ideen-/Kernel-Ports pruefen:
Layout-/Quant-/Stride-Adapter, Dispatch, sm_89-Fallback, CUDA-Graph-
Vertrag, CPU-Oracle und Logit-/Kernellaufzeit-Paritaet. Es gibt **kein**
pauschales Kernel-Port-Verbot; Investition richtet sich nach gemessenen
Time-Shares, nicht nach vermuteten Anteilen. Nicht blind einen groesseren
Ring bauen und keine feste 4-6x-Zusage aus Stratas Ablation ableiten.

### 10.6 Phase S2: RAM-adaptive residente Experten

**Zielbild:** Wahl zwischen `mmap` (Default), resident-pageable und
resident-pinned **nur fuer CPU-/Streaming-Experten**. Vorher kontrollieren,
ob `--no-mmap`/Host-Registrierung denselben Effekt schon liefern. Die
Strata-Ablation trennt *Residenz, Pinning und I/O* nicht sauber.

1. **Inventar/Budget vor Allokation:** pro GGUF-Shard, Layer, Tensor und
   `ggml_type` Expert-Gewichte zaehlen; CPU-tatsaechlich benoetigte
   Gewichte plus Staging/temporare Kopien, PLE, KV, Page-Cache, GPU-
   Offload und Reserve einrechnen. `MemAvailable`, cgroup-Limit,
   optionalen Swap-/Commit-Status beruecksichtigen. Budget als
   Bytes/GiB oder Anteil *verfuegbaren* RAMs, kein "RAM insgesamt"-
   Automatismus. Bestehende Produktionsspeicher respektieren.
2. **Host-Buffer-Vertrag:** opt-in einmalige Allokation fuer ausgewaehlte
   Expert-Tensoren, alignment-/quantblock-korrekt, GGUF-Shard-Offsets
   beim Laden kopieren, ggml-Tensoren auf diese stabilen Host-Adressen
   binden; lifetime bis zum Modell-Unload, mmap nicht zu frueh
   freigeben. GPU-residente Gewichte nicht doppelt in die Arena
   laden; CPU-Kernels muessen dieselbe Bytes/Strides sehen.
   Inkrem. Konversion (pro Modellgruppe/Shard) statt zweitem
   Komplettmodell im RAM; Test gegen `--no-mmap`.
3. **Pinning-Tiers:** pinned Host-Buffer oder geprueftes
   `cudaHostRegister` nur fuer die innerhalb Host-/Treiberlimits
   passende Auswahl. `mlock`-Limit separat pruefen; CUDA-Registration
   ist kein Beweis fuer `mlock`. Fehlendes Pinning oder RAM-Budget
   ist kein Startup-Crash: transparenter Rueckfall auf pageable
   Residenz oder bestehendes mmap (mit geloggtem Grund). Kein
   stiller Swap-OOM. PLE/Shared-/Dense-Tensoren bleiben vorerst
   unveraendert.
4. **Backend-Portabilitaet:** Backend-Hostbuffer-API und CPU-Besitz
   nutzen; CUDA-spezifisches Pinning nur wenn vorhanden, Vulkan/
   CPU/UMA ohne CUDA-API und unveraendertem Default. Bei kleinen
   Modellen auf RAM-aermeren Hosts resident nur wenn Budget inkl.
   Reserve passt, sonst mmap.
5. **Tests/Gate:** Byte-/Quant-Paritaet zu mmap, Shard-Split,
   mehrfaches Laden/Entladen, knapper RAM, fehlendes Pinning,
   OOM-/Cancel-Pfad, ASan/Backend-Tests; RAM-, Swap- und Major-
   Fault-Zaehler vergleichen. Nur weitermachen, wenn abgeloester
   Page-Fault-/H2D-Engpass auch nach Warm-Cache-Messung besteht.

### 10.7 Phase M: Flash-Next-MTP-Sidecar korrekt und maximal nutzbar machen

**Korrektur zum alten Bericht:** Der Fork kann **bereits** externe
MTP-Artefakte laden: `--spec-type draft-mtp --spec-draft-model <GGUF>`
(unter anderem K2-Horizon-Head-only-Sidecar), und fuer Gemma 4 existiert
`--spec-type mtp --mtp-head` mit eigener asynchroner MTP-Scheduler-
Pipeline. Das UD-IQ4_XS-Target hat *keinen eingebetteten* NextN-Head;
**was fuer Flash-Next fehlt**, sind nachgewiesene kompatible Head-Gewichte
im GGUF, korrekte Architektur-Anbindung und ggf. ein Ziel-Hidden-State-
Hook. Stratas `--mtp`-Dateien sind kein automatisch ladbares GGUF.

**Vertragsanalyse 2026-10-07 (Strata `82f46a8c8`, noch kein MTP-Port):**

- Head: 31 `mtp.*`-Tensoren, vortrainierter einzelner Attention/MoE-Layer.
  `fc_embedding` und `fc_hidden` jeweils 2560x2560; Hidden-Norm 10240,
  HC=4, Mix-Low-Rank=320; 512 Experten, Gate/Up 512x1280x2560 und Down
  512x2560x640. Basis-Embedding/Output teilen, keine PLE-Tensoren duplizieren.
- Strata paart `R_i` (10240 rohe HC-Werte vor dem finalen Mixer) mit Token
  `i+1`, RoPE-Position `i`; der Head sagt Token `i+2` voraus. K2s
  RoPE-Position `i+1` nicht ungeprueft uebernehmen.
- Hidden-Norm ist ein eigener Paritaetspunkt: Strata normalisiert standard-
  maessig ueber 10240 Werte, bietet alternativ vier 2560er-Stream-Normen.
  Die HC-Mixer normalisieren je Stream. Gemma-Normgewichte werden als
  `1+w` aufbereitet; diese Transformation beim Export exakt einmal anwenden.
- Stratas Draft nutzt dichte Attention ohne Indexer-Zustand; oberhalb
  2051 Zellen ist das eine Draft-Approximation, kein exakter QSA-Graph.
  Zuerst exakte Ein-Schritt-Referenz, Approximation erst separat messen.
- Fork-Blocker: kein Qwen4Exp-MTP-Graph/`t_h_nextn`-Export; generischer Draft-
  Treiber prueft skalare `n_embd` statt HC-Breite. Qwen4Exp ist nicht fuer
  den vorhandenen partiellen Recurrent-State-Rollback freigegeben.
  Target-Checkpoint-Restore und separaten Draft-QSA-State gemeinsam testen;
  kein blosses Hinzufuegen des Arch-Namens zur Rollback-Whitelist.

1. **Asset-/Format-Gate:** identische Flash-Next-Checkpoint-Revision
   fuer Target und Head beschaffen/verifizieren, Strata-Setup-Extraktion
   und Original-HF-Tensorinventar/Shapes dokumentieren. Ein `conversion/
   qwen4exp.py`-Modul ist im aktuellen Checkout nicht vorhanden, obwohl
   die Architektur-Registry darauf verweist; Konverter zuerst pruefen.
   Qwen4Exp-MTP-Export ist nicht implementiert (`--mtp` bisher nur fuer
   Qwen3.5/3.6/Step3.5).
   Definieren, ob kompakte Head-only-GGUF wie K2 oder konvertierter
   kleiner Draft als separate `--spec-draft-model`-Quelle taugt.
   Keine Train-/"synthetischer Drafter"-Behauptung.
2. **Vertrags-Gate vor Code-Port:** Tokenizer/Vocab/IDs, Hidden-State-
   Dimension (HC-Strombreite vs. gemischter Output), genaue
   Aktivierungsebene, Norm/Projektion, Position/Rotary, PLE/QSA-
   Zustand und MoE/Draft-Gewichte gegen Referenz pruefen.
   `src/models/qwen4exp.cpp` exportiert derzeit nur `t_embd`, kein
   `t_h_nextn`; `common/speculative.cpp:817-934` erwartet die
   passende Ziel-Hidden-State-Breite und verwaltet `ctx_dft`.
   Erst isolierten Ein-Schritt-GGUF-Head mit Logit-/Top-k-
   Paritaet gegen Referenz durchsetzen; sonst keine Performancezahl.
3. **Wiederverwendung statt Duplikat:** K2-Head-only-Ladepfad in
   `src/models/k2-horizon.cpp`, `tools/server/server-context.cpp`
   (`--spec-draft-model`) und bestehende Target/Context-Hooks
   nutzen; nur Qwen4Exp-Hparams/Tensor-Mapping und Draft-Graph
   erweitern, wenn tatsaechlich erforderlich. Shared Target-Tensoren
   nicht als zweites grosses GGUF duplizieren. Target-Verifikation
   bleibt mathematisch autoritativ.
4. **Korrektheit/Grenzfaelle:** Verifikationspositionen,
   gruendliche Cache-/Recurrent-State-Rollbacks nach Rejection,
   Cache-Reset und Kontextwechsel, mehrere Requests/Slots,
   Prompt-Mischung Text/Coding und kurze/langsame Kontexte;
   Uncensored/anderes Target mit mismatched Checkpoint **ablehnen**.
   Greedy-Token-Sequenz muss target-only entsprechen;
   fuer Sampling Verteilungstest, nicht bloss Token-Gleichheit.
5. **Performance-Stufen:** ohne Draft gegen `draft-mtp` n=1..8,
   p_min-Sweep, Akzeptanz (drafted/accepted/pro Verifikations-
   Runde), Zeit fuer Head, Verify, Sync, Host-Transfers/CPU;
   Blocksize an Zeit pro *akzeptiertem* Token optimieren, nicht
   Draft-Anzahl. Danach Vergleich mit vorhandener Gemma-MTP-
   Async-Scheduler-/In-Graph-Sampler-Technik, soweit architektonisch
   kompatibel, plus optional N-Gram/Prompt-Lookup **nur** wenn
   End-to-End-tg und Output korrekt bleiben. Ressourcenbudget der
   Draft-Gewichte/Graphen auf 16-GB-GPU pruefen, CPU-/Split-
   Fallback bei zu wenig VRAM. Keine pauschale 2x-Zusage.
6. **Gate M:** korrektes Sidecar-Load, reproduzierbare Akzeptanz,
   >256 generierte Tokens je Prompt ohne EOS-Kurzartefakte,
   messbare netto tg-Verbesserung gegen *dieselbe* Target-Quant,
   Regressionstests fuer bestehende K2-, Gemma- und NextN-Drafts.
   Erst danach Default oder Integration in Produktivprofile erwaegen.

### 10.8 Integration, Fairness und Entscheidungspunkte

**Arbeits-Pakete (alle noch offen):**

| Paket | Primaere Code-Orte | Freigabe/Gate |
|---|---|---|
| P1a Offset | `ggml/src/gguf.cpp`, `tests/test-gguf.cpp`, ggf. Quant-Definition | Reproducer + sichere Reader-Paritaet |
| P1b Prefetch | `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`, Backend-Tests | korrekter CUDA-Pfad inkl. Slots/Events |
| Baseline/Profil | bestehende Bench-/API-Helfer, Context-Metriken | effektive Config, Token-/Timing-Paritaet |
| S1/S1b Streaming | Scheduler, `src/llama-context.cpp`, MMID/MMQ-CUDA nach Profil | Same-Quant-pp-Gain, keine Output-Regression |
| S2 Host-Residenz | `src/llama-model-loader.cpp`, `llama-model.*`, Hostbuffer-API | RAM-/Pinning-Fallback + nachgewiesener Zusatznutzen |
| M MTP | `conversion/qwen4exp.py`, GGUF-Registry, `src/models/qwen4exp.cpp`, `common/speculative.cpp`, Server | Head-Parity, Rollback, netto tg-Gain |

**Konfiguration (Entwurf, noch keine neuen Flags implementiert):** Host-Modus
`mmap|resident|auto`, Host-Budget/Reserve, Pinning-Budget, Prefill-Streaming
`off|on|auto`, Staging-Budget und Chunk-Decke als explizite gemeinsame Parameter
von CLI/Server/Library. Bestehende `--spec-draft-model`-/Draft-Parameter fuer
MTP behalten. Defaults vorerst wie heute; Overrides und *wirksame* Budgetwahl
im Startup-Log, keine stillen Backend- oder Quant-Wechsel. Ressourcenknappheit
ist ein gepruefter Rueckfallgrund, kein Anlass fuer System-Limit-Aenderungen.

- **Branch-/Transport-Vertrag:** eine gepinnte Fork-Basis pro Vergleich;
  Implementierung auf Feature-Branches/Worktrees mit getrennten P1-/
  Streaming-/Residenz-/MTP-Commits, keine Umstellung eines produktiven
  Checkouts. Remote-Verteilung ueber Git-Branches statt Patch-Dateien;
  noetige Pushes/Merges gemaess Freigabe, keine implizite Deployment-
  Freigabe durch diesen Plan.
- **Messziele, keine Vorhersage:** pro Phase reproduzierbarer Netto-
  Gain ausserhalb der Streuung bei korrekten Logits und keinen relevanten
  Kurzprompt-/Decode-Regressen. Nach kompletter Integration pp/tg als
  Anteil der neu gemessenen Strata-Rate je gleicher Quant, GPU-Zahl und
  Kontext ausweisen. Paritaet als Zielkorridor z.B. >=90 % dieser Rate,
  nicht als zugesichertes Ergebnis; bei Restgap die groessten gemessenen
  Zeitanteile abarbeiten oder eine weitere Investitionsentscheidung
  vorlegen. Kein "fertig", nur weil ein erster Prototyp 5 % gewinnt.
- **Sequenz der separaten Code-Changes:** P1a und P1b getrennt (je mit
  Regressionstest und eigener RCA) → Messbaseline → S1-Prototyp → S2
  nur bei nachgewiesenem Zusatznutzen → M-Asset/Parity → M-Optimierung.
  S2 kann technisch nach P1a/P1b parallel vorbereitet werden;
  fuer fairen A/B-Vergleich nie mehrere Schalter gleichzeitig aendern.
- **Stufenmatrix:** Baseline/P1-Fix, Prefetch, gross-ubatch, Streaming,
  resident-pageable, resident-pinned, MTP; ausserdem Kombinationen
  Streaming×Arena und Streaming×MTP fuer Interaktionen. Gleiche
  Target-Quantisierung, Stopp bei OOM/NaN/Qualitaetsverlust und
  reproduzierbarem 1k- oder Decode-Regress. Attribute nicht addieren.
- **Qualitaets-/Stabilitaets-Gate:** 1/2 GPU und 1/mehrere parallele
  Slots, Context 4k/40k, LoRA/CPU-Fallback sofern relevant,
  wiederholte Serverstarts und Soak auf Test-GPU. `test-gguf`,
  Backend-/MoE-/MTP-Tests, CUDA-Build, CPU-Build; DOX,
  ROADMAP #99, CHANGELOG und thematische Benchmark-Note
  nach Implementierungsphasen aktualisieren.
- **Stop-Kriterien:** Wenn volle Tensor-Streams bei kleinen/kalten
  Prompts H2D-Bytes oder TTFT verschlechtern, dynamischen
  Hybrid-Gate statt universellem Default einsetzen; wenn Arena
  gegen warmes `--no-mmap`/gepinntes mmap keinen Mehrwert hat,
  keine eigene Arena pflegen; wenn MTP-Head fehlt oder
  Akzeptanz/VRAM nicht reicht, Draft-Phase aussetzen statt
  Modellgewichte zu erfinden. Ein Strata-Gesamtfaktor ist eine
  Vergleichslatte, kein zugesicherter Port-Ertrag.
