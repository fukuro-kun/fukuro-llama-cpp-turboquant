# SESSION_PLAN: Strata + OptLlama A/B-Eval auf Uranus (Qwen3.8-Flash-Next)

**Erstellt:** 2026-10-06
**Status:** [x] Vergleich abgeschlossen (2026-10-07); [ ] Fork-Optimierung in Umsetzung — P1a-/P1b-Gates bestanden, Baseline B0–B5 gemessen; S1/S2/M-Gates offen (Stand 2026-10-08). [Aktueller Fortschritt und Checklisten](#100-statusuebersicht-und-aktualisierungsvertrag).
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

### 10.0 Statusuebersicht und Aktualisierungsvertrag

**Stand:** 2026-10-08. Abgeschlossene Teilaufgaben sind von noch offenen
Laufzeit-/Modell- und Gesamtfreigaben getrennt.

| Paket | Stand |
|---|---|
| P1a Offset/Q2_0 | **abgeschlossen** — Gate bestanden 2026-10-07 (echte IQ3-Inferenz Dual-GPU, kohärente Ausgabe, 134/134 CUDA-Backend-Paritaet je GPU) |
| P1b Prefetch | **abgeschlossen** — Gate bestanden 2026-10-07 (`3f3d272f69`, `819af39886`): 5 Prefill-/Decode-Laeufe ohne Crash, 15840/15840 Backend-Tests; Output-Vergleich durch inhaerente Kernel-Nichtdeterministik nur qualitativ |
| Baseline/Profil | B0–B5 jeweils 3 Reps vorhanden; Konsolidierung, no-mmap-Kontrollen und Zeitanteil-Profilierung **offen** (§10.4) |
| S1 / S1b Streaming | nicht implementiert — **offen**; vorhandener Full-Tensor-Prefetch ist kein abgeschlossenes Streaming-Gate |
| S2 Host-Residenz | RAM-adaptive Expert-Arena nicht implementiert — **offen**; vorhandene no-mmap-/Hostregister-Pfade zuerst kontrollieren |
| M MTP | Graph/Sidecar laufen; kurze f16-Smokes bestanden, q8_0-Spec-Fehler eingegrenzt, Ursachen- und kontrolliertes Performance-/Korrektheits-Gate **offen** (§10.7); separater Multistream-Fullstate-Restore-Bug **behoben** (f16-Fix/Test abgeschlossen vor Commit `ab6e9dfcea`, §10.9); q8_0-Paritaet und Gesamt-M-Gate **offen** |
| Integration | P1a-Branch verteilt (Remote-Stand seitdem nicht erneut verifiziert); P1b besitzt eigene Commits `3f3d272f69`/`819af39886` auf noch offenen Folge-Branches; Gesamtgates **offen** |

**Aktualisierungsvertrag fuer diese Checklisten:**

- Checkbox `[x]` nur fuer *verifiziertes* Ergebnis (Commit, Testlauf mit Count,
  Messung, Review-Freigabe) — niemals nur weil ein Schritt beschrieben ist.
- Direkt nach verifiziertem Fortschritt/Test/Commit aktualisieren und kurze
  Evidenz anhaengen (Commit-Hash, Testname + Zaehlerstand, Datum).
- Fehlgeschlagene, teilweise oder unbestaetigte Experimente nicht abhaken;
  Unbekanntes bleibt `[OFFEN]`.
- Bei Regression Haken zuruecknehmen und Grund kurz dokumentieren.
- Labels fuer nicht abgehakte Punkte: `[IN ARBEIT]`, `[OFFEN]`, `[BLOCKIERT]`.
- Ein Parent-Gate wird erst abgehakt, wenn alle seiner Kriterien erfuellt sind.
- ROADMAP und Ergebnisbericht bei Gate-Aenderungen synchronisieren;
  CHANGELOG und DOX nach jeder bedeutsamen Code-/Vertragsaenderung
  aktualisieren, auch wenn das gesamte Phasen-Gate noch offen ist.

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
  widerlegt (§10.2). Der Prefetch-Crash ist auf den MMQ-Overread ueber
  exakt bemessene Prefetch-Slots zurueckgefuehrt und gefixt (§10.3).
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

**Fortschritt P1a (Stand 2026-10-07, Commit `0389e655e8` auf
`feat/flashnext-p1a-q2-compat`, gepusht):**

- [x] Diagnose: Typnummer-Kollision bewiesen — Datei nutzt Q2_0=42,
      Fork belegte 42 mit TURBO2_0; nur TURBO2_0 auf 47 verschoben,
      43–46 unveraendert.
- [x] Reproducer + Negativfaelle: rohes gepacktes Typ-42-GGUF scheiterte
      vorher (87/88), nachher `test-gguf` 88/88.
- [x] Implementierung: Core/CPU/CUDA/Python + Quantize-CLI portiert.
- [x] Lokale Tests: CPU-Backend 160/160 (inkl. 4 neue gepackte Codec-
      Faelle), Quant-Funktionen 0 Fehler, TurboQuant 17/17, Python/C
      bitgleich + manuell gepackte Codes.
- [x] Builds lokal: CPU, CUDA sm_89, Vulkan erfolgreich.
- [x] Review: zwei unabhaengige Runden, keine offenen P0/P1.
- [x] CUDA-`__byte_perm`-Verdacht widerlegt (unabhaengige
      65536-Muster-Probe auf Ada sm_89: 0 Abweichungen).
- [x] Echte IQ3-Shards + mmproj per `hf cache verify` gegen Quellrevision
      `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (3 Checksums passend).
- [x] Isoliertes Remote-Worktree auf Commit `0389e655e8` auf Ada-Testhost
      erstellt; CUDA-12.0-Build fuer `test-gguf`, `test-backend-ops`,
      `llama-cli`, `llama-bench` und `llama-completion` erfolgreich
      (Produktiv-Checkout unangetastet).
- [x] CUDA-Fork-Backend auf CUDA0/CUDA1 zur Laufzeit getestet:
      `test-backend-ops -o MUL_MAT,MUL_MAT_ID,GET_ROWS,CPY,SET_ROWS`
      jeweils 134/134 inklusive direkt gepacktem `+2d`; Produktiv-
      dienste danach wieder HTTP-200 aktiv (`0389e655e8`).
- [x] Echte IQ3-Shards im Dual-GPU-Smoke geladen und erste Textausgabe
      fachlich korrekt erhalten (`The capital of France is Paris`).
      Der Smoke bricht danach am `fattn.cu:384`-Catch-all-Abort ab:
      `-ctk q8_0 -ctv f16` ist eine **pre-existing, nicht unterstuetzte
      Misch-KV-Kombination** (nur Turbo-Mischungen sind im Vec-Dispatch
      registriert) — kein Q2_0-Regressionsbefund. Validierung wird mit
      `q8_0/q8_0` und `-fa off` wiederholt.
- [x] Beide echten IQ3-Shards laden; Inferenz-/Text-Sanity: Dual-GPU
      (`--cpu-moe`) erzeugt kohärente, fachlich korrekte Fortsetzung
      ("The capital of France is Paris, …") mit FA q8_0/q8_0 und
      `-fa off`, je 32 Tokens, kein Fehler (Lauf `p1a-20261007-0389-r2`).
- [x] Gate P1a: IQ3_XXS als fairer Same-Quant-Benchmark freigegeben.
      Nebenbefund (pre-existing, kein Q2_0-Regressionsfund): `fattn`
      Vec-Dispatch akzeptiert keine Q8_0/F16-Misch-KV-Typen →
      Catch-all-Abort statt sauberer Ablehnung (separater Bug, offen).

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

**Fortschritt P1b (Stand 2026-10-07):**

- [x] Ist-Pfad gelesen: Scheduler-/Prefetch-Code in `ggml-backend.cpp`
      inspiziert (Slots, Stream, Slot-Alias).
- [x] Reproducer + RCA (2026-10-07, Commit `3f3d272f69`): Crash auf Ada
      reproduziert — `GGML_SCHED_PREFETCH_EXPERTS=1` + UD-IQ4_XS + `--cpu-moe`,
      exit 134 **auf Single- UND Dual-GPU** (fruehere Hypothese "Prefetch-Pool
      auf falschem Device" damit widerlegt). `CUDA_LAUNCH_BLOCKING=1`:
      illegaler Zugriff im `MUL_MAT_ID`-Kernel statt CUBLAS-Callsite.
      compute-sanitizer memcheck: `mul_mat_q` liest 2 Bytes **+3 hinter**
      einem exakt `ggml_nbytes`-grossen Prefetch-Slot (891 MiB). Grund:
      MMQ-Kernel lesen per Design ueber das Tensorende (Selektivpfad kopiert
      deshalb bis zu 512 Extra-Bytes); galloc-Tensoren haben impliziten
      Slack, Standalone-Slots nicht → unmapped Page + evtl. NaN-Bytes.
- [x] Kontrollierter Vergleich `prefetch=0/1` mit UD-IQ4_XS auf Ada:
      `prefetch0-dual=0`, `prefetch1-dual/single=134` vor Fix → `=0` nach Fix
      (`p1b-20261007-r1`, `p1b-20261007-fix1`).
- [x] Fix implementiert (`3f3d272f69`): Slots mit +512 B Padding allokiert
      (gleiche Marge wie Selektivpfad), Padding-Ende bei Allokation genullt
      (synchrones Buffer-Clear; H2D schreibt nur `nbytes`) → kein OOB, keine
      uninitialisierten/NaN-Bytes. Zusaetzlich `prefetch_max_size`: scannt
      nun alle Split-Nodes auf `MUL_MAT_ID` (Trigger sucht ebenfalls alle).
- [x] Zugehoeriger Pre-existing-Bug gefixt (`819af39886`): Segfault in
      `ggml_cuda_graph_check_compability` — `node->src[0]->type` wurde fuer
      src-lose Ops (z. B. `ARANGE`) dereferenziert; `is_tq_w` in den
      `MUL_MAT_ID`-Zweig verschoben (identisch auf master reproduziert,
      gdb-Backtrace).
- [ ] `[OFFEN]` Synthetischer `MUL_MAT_ID`-Split-Test angelegt und gruen.
      *(Nice-to-have: Szenario ist durch echten Modell-Repro +
      Sanitizer-Beweis + 15840 Backend-Tests abgedeckt.)*
- [x] Gate P1b (2026-10-07): 5 Prefetch-Laeufe ohne Crash/NaN —
      `p1b-20261007-fix1` (single + dual, 4k) und `p1b-20261007-gate`
      (dual: kurz/8k/256-tok-Gen), alle exit 0, kohärente Ausgabe.
      `test-backend-ops` 15840/15840, 3/3 Backends OK. **Hinweis
      Korrektheit:** bit-gleicher Output-Vergleich ist auf diesem Setup
      unerreichbar — zwei identische Baseline-Laeufe (`prefetch=0`,
      temp 0, seed 42) liefern bereits unterschiedliche Generierungen
      (inhärente Kernel-Nichtdeterministik, Atomics/Split-K); Vergleich
      nur ueber Kohaerenz/Qualitaet moeglich.

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

**Fortschritt Baseline/Profil (Stand 2026-10-08):**

- [x] Gepinnte Fork-Baseline mit UD-IQ4_XS gemessen (2026-10-08,
      `eval-flashnext/results/fork-p1fix.jsonl`, Configs B0–B5, je
      3 Reps, ~811/~2950/~5820/~23133 Prompt-Tokens):
      - B0 cpu-moe ub4096 (Referenz): pp 106/239/242/245 t/s, tg ~11 t/s
      - B1 +prefetch (P1b-Fix): pp 77/251/246/283 t/s — hilft ab ~23k
        (+16 %), bremst kurze Prompts (−28 % bei 0,8k)
      - B2 +prefetch +`-ubp 8192`: pp 75/249/244/**345** t/s —
        physische Chunk-Groesse ist der staerkste Hebel bei 23k (+41 %)
      - B3 hyb44 (`--n-cpu-moe 44` + prefetch): pp 84/270/264/302 t/s,
        tg ~11,6 t/s
      - B4 +GGML_CUDA_REGISTER_HOST: pp 72/262/256/296 t/s —
        gegen B1 etwa +4–5 % bei mittleren/langen Prompts, bei 0,8k
        leicht negativ; kein isolierter Beleg fuer den Arena-Nutzen
      - B5 IQ3_XXS + prefetch: pp 90/**320**/**311**/321 t/s —
        23k-Median 320,7; 343,0 ist das Maximum, nicht der Median.
        Quant-/Gewichtsvolumen-Effekt nicht separat vom Kernelpfad isoliert;
        tg ~10,4 t/s
      - Der 23k-Prompt terminiert in B0–B4 nach 1 Token (EOS) → tg
        dort unbrauchbar. B5 erzeugt 128/1/1 Tokens; nur ein langer
        Decode-Lauf reicht nicht fuer einen Medianvergleich. B5 weist
        zudem grosse Kalt-/Warm-Streuung auf; pp-Mediane sind Datenpunkte,
        keine abgeschlossene I/O-/Korrektheits-Ablation.
- [x] IQ3_XXS-Baseline gemessen (B5, siehe oben; „beider Engines"
      — Strata/OptLlama-Vergleich folgt bei der Final-Auswertung,
      Fork-Werte stehen).
- [~] Vorhandene Alternativen geprueft: mmap (Default), Prefetch
      nach Fix (B1/B2), `-ubp`-Sweep (8192), REGISTER_HOST (B4),
      hyb-Residenz (B3), IQ3 (B5). **Nicht gemessen:** `--no-mmap`,
      `--mlock` — markiert als Rest-Luecke, wenn noetig.
- [x] Messinfrastruktur-Protokoll etabliert: `p1-post-baseline.sh`
      (serverbasiert, `/v1/completions`, usage.prompt_tokens als
      x-Achse, 3 Reps, Aggregator `agg_baseline.py`); Configs und
      Ergebnis-Rohdaten unter `eval-flashnext/results/`.

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

**Fortschritt S1/S1b (Stand 2026-10-07):**

- [ ] `[OFFEN]` Instrumentierung pro Layer/Expert-Tensor implementiert
      (NVTX/CUDA-Events, kein Hot-Path-Logspam).
- [ ] `[OFFEN]` Chunk-Sweep 256..8192 als Kontrollgruppe gemessen.
- [ ] `[OFFEN]` Budgetierter Staging-/Copy-Stream-Pfad implementiert.
- [ ] `[OFFEN]` VRAM-Budget-/Slot-Auswahl statt fester Ringzahl.
- [ ] `[OFFEN]` Graph-/Multibackend-Vertrag eingehalten und getestet.
- [ ] `[OFFEN]` Ablations-Gates: reproduzierbarer Netto-pp-Gewinn,
      keine Output-/Kurzprompt-Regression.
- [ ] `[OFFEN]` S1b Expert-Tile-Streaming (nur falls Tensor-Slots nicht
      ausreichen; eigene MMID-/grouped-GEMM-Pfade ggf. erforderlich).
- [ ] `[OFFEN]` Kostenmodell/Heuristik fuer weitere MoEs; Auto-Modus
      erst nach Cross-Model-Gate.

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

**Fortschritt S2 (Stand 2026-10-07):**

- [ ] `[OFFEN]` Inventar-/Budget-Rechnung vor Allokation implementiert.
- [ ] `[OFFEN]` Host-Buffer-Vertrag (opt-in Residenz, korrekte Bindung,
      Lifetime) umgesetzt und gegen `--no-mmap` getestet.
- [ ] `[OFFEN]` Pinning-Tiers mit transparentem Rueckfall (kein
      Startup-Crash, kein stiller Swap-OOM).
- [ ] `[OFFEN]` Backend-Portabilitaet (Vulkan/CPU/UMA unveraendert).
- [ ] `[OFFEN]` Gate S2: Byte-/Quant-Paritaet, Grenzfaelle, nachgewiesener
      Zusatznutzen gegen warmes mmap/no-mmap.

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

**Fortschritt M (Port und kurze Laufzeittests vorhanden; Gesamtgate nach
Methodikpruefung wieder offen):**

- [x] Vertragsanalyse Strata `82f46a8c8`: 31 `mtp.*`-Tensoren/Shapes
      bekannt (HC 4x2560=10240, `fc_embedding`/`fc_hidden` 2560x2560,
      512 Experten, Gate/Up 512x1280x2560, Down 512x2560x640).
- [x] RoPE-Paarung dokumentiert: Strata nutzt Zelle `i` fuer Token `i+1`,
      Head sagt `i+2` voraus — K2s `i+1` nicht ungeprueft uebernehmen.
- [x] Fork-Blocker identifiziert: kein Qwen4Exp-MTP-Graph/`t_h_nextn`,
      skalare `n_embd`-Pruefung statt HC-Breite, kein Recurrent-Rollback
      fuer Qwen4Exp freigegeben, `conversion/qwen4exp.py` fehlt trotz
      Registry-Verweis; Strata-Manifest/Referenz-Inventare gesichtet.
- [x] Hidden-Norm-Paritaet entschieden (2026-10-08): `nextn.hnorm`
      liegt in den Dateien flach `[10240]` vor und wird im Graph als
      `[2560,4]`-View interpretiert (per-Stream-RMSNorm — numerisch
      identisch zum Upstream-Layout, kein `1+w`-Reweighting noetig).
- [ ] `[OFFEN]` Head-Checksum/Checkpoint-Match mit Target-Revision und
      Ein-Schritt-Orakel validiert (Inventare allein reichen nicht).
- [x] Asset-/Format-Gate (2026-10-08): unsloth non-shared Q8_0-Sidecar
      (`unsloth-mtp/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, 34 Tensoren,
      `blk.48.*` + `nextn.*` + eigene Embeddings/Tokenizer) laedt und
      laeuft als `-md` Draft. **Die `shared`-Variante ist inkompatibel**
      — ihr fehlt das vom Loader als Pflicht erwartete `token_embd`.
- [x] Vertrags-Gate vor Code-Port (2026-10-08, grob): Tokenizer aus
      Sidecar, Hidden-State-Breite 10240 (HC), Norm/Projektion-Shapes
      gegen Inventar verifiziert. Ein-Schritt-Orakel noch offen.
- [~] Qwen4Exp-MTP-Code implementiert (Commit `14fc22db93`): Graph,
      Hparams, Tensor-Mapping, `ctx_dft`-Anbindung; an Fork-QSA-Pfad
      adaptiert (aeltere `llm_graph_input_qsa`-API statt upstream
      `kpool`). Laeuft — Laufzeit-Fixes siehe unten; dichte Draft-
      Attention >2051 Zellen noch nicht separat gemessen.
- [~] Korrektheit/Grenzfaelle — Laufzeitbefunde 2026-10-08:
      - `[FIXED]` embd-Breiten-Vertrag: `llama_context::decode` nutzte
        `n_embd_inp()` (2560) fuer die Batch-Allocr-Stride; der
        MTP-Draft braucht aber HC-Breite 10240 → `set_input`-Memcpy
        ueberlas den Buffer (SIGSEGV). Fix: `n_embd_out()` bei
        `ctx_type == LLAMA_CONTEXT_TYPE_MTP` (upstream-Pendant:
        `llama_batch_ext_select_n_embd_inp`).
      - `[FIXED]` Double-Crop: der Trunk-Endmixer applizierte nach
        `crop_before_nextn` (res_hc bereits auf Output-Zeilen
        verengt) nochmals `get_rows(cur, inp_out_ids)` → OOB-Zeilen-
        Zugriff → Trunk-Logits korrupt ("isfdb"-Garbage) auch **ohne**
        Spec. Fix wie upstream: kein trailing `get_rows`.
      - `[EINGEGRENZT, NICHT GEFIXT]` Spec-Degeneration korreliert in
        den bisherigen Kontrolltests mit q8_0-Attention-KV. Der genaue
        Fehlermechanismus (Write/Read, Verify-Batch, Restore oder
        Rollback) ist **nicht bewiesen**; Graph-/Batch-Interaktionen
        bleiben moeglich. Beobachtete Dtype-Matrix:

        | attn ctk/ctv | idx-K | h_tgt | Ausgabe |
        |---|---|---|---|
        | f16 | f16  | sauber | korrekt (`B. Paris`) |
        | f16 | q8_0 | sauber | korrekt |
        | q8_0 | f16 | sauber | degeneriert |
        | q8_0 | q8_0 | NaN    | degeneriert |

        non-MTP + q8_0 ist sauber (`Paris`) → der Bug ist
        **spec-decode-spezifisch** (Draft+Verify+Checkpoint-Pfad), kein
        pauschaler Nachweis eines generellen q8_0-QSA-Problems.
        Indexer-Score-NaNs allein beweisen keine Output-Korruption;
        ihre Harmlosigkeit und das NaN-Verhalten von top_k sind aber
        ebenfalls **nicht bewiesen**. Endliches h_tgt garantiert keine
        korrekten Logits. Wiederholte Positionen 83/84 und wechselnde
        Kandidaten bei 85 wurden beobachtet; echter No-Progress-Livelock
        muss gegen normale Checkpoint-Replays/Cache-Aufholung geprueft werden.
        User hat einen weiteren Diagnoseblock mit 60-Minuten-Budget
        freigegeben: Mehrtoken-Forward gegen sequenziellen Forward,
        Byte-/Logit-Paritaet vor/nach Restore, K/V-Isolation und
        Commit-Fortschritt messen, statt weitere unkontrollierte Toggles.
      - `[FIXED]` Hybrid-Indexer-Restore (`5925020d35`): `mem_idx` wurde
        ueber eigenes `find_slot` restauriert und divergierte nach
        Checkpoint-Rewrites vom Attn-Layout. Neu: `state_read_sinfo`
        liefert das Attn-Slot-Layout, `mem_idx` restauriert gespiegelt.
        Notwendig, aber nicht hinreichend (q8_0-Attn bleibt broken).
      - `[TODO]` q8_0-Attn-Spec-Pfad weiter debuggen ODER explizit
        f16 forcieren/dokumentieren; Rollback-Retry-Cap gegen Livelock.
      - `[TODO]` Getrennt halten: (a) f16-Kontrollen laufen **jetzt**
        — Rollbacks nach Rejection, Mismatched-Checkpoint-Ablehnung,
        Greedy-/Sampling-Tests auf f16 sind nicht vom q8_0-Root-Cause-
        Fix blockiert; (b) q8_0 ist die gebundene **Negativkontrolle**
        — sie bleibt bis zum Fix erwartbar failing und dient der
        Diagnose, nicht als Gate fuer die f16-Arbeit.
- [ ] `[WIEDER OFFEN]` Performance-/Korrektheits-Gate M: Die ersten
      debug-freien f16-Einzelmessungen liefern Hinweise, aber keinen
      kontrollierten Speedup und kein Optimum:

      | Konfiguration | Prompt | tatsaechlich erzeugte Tokens | Akzeptanz | decode t/s |
      |---|---|---|---|---|
      | non-MTP | France + Erklaerung | 83, EOS, Content geliefert | — | 11.08 |
      | n_max=2 | France + Erklaerung | 300, length, Content leer | 0.910 | 9.44 |
      | n_max=4 | Primzahlen + Summe | 200, length, Content leer | 0.969 | 13.27 |
      | n_max=8 | Primzahlen + Summe | 200, length, Content leer | 0.931 | 14.19 |

      Gleiche Serveroptionen reichen nicht: verschiedene Texte und
      generierte Laengen machen die frueher angegebenen −15/+20/+28 %
      **nicht zu belastbaren MTP-Effekten**. n=4/8 haben denselben
      Prompt/Budget, aber je nur eine Messung. Ein kurzes `B. Paris`
      war ein bestandener Smoke, keine Gesamtfreigabe. Kohaerentes
      abgeschnittenes Reasoning ist kein bestandener Delivery-/Qualitaetstest.
      Hohe Akzeptanz beweist weder korrekte Zielverteilung noch Speedup.
      Draft-/Verify-/Checkpoint-Kosten sind noch nicht separat gemessen;
      Akzeptanzrate mal Draft-Obergrenze ist nicht die gemessene Yield.
      Keine Default-Empfehlung fuer n=8 bis zum kontrollierten Sweep.

      **Strata-Vertrag verifiziert (`82f46a8`):** `--spec T` zaehlt das
      ganze Verify-Fenster (ein feststehendes Token plus T−1 Drafts).
      Der gemessene UD-IQ4_XS-Lauf nutzte `--spec 4 --spec-min-p 0.5`
      und `--kv int8`; erste vergleichbare Fork-Obergrenze ist deshalb
      `--spec-draft-n-max 3`, nicht 4. Stratas int8-KV nutzt eigene
      64-Wert-Gruppen und separate FP16-Skalen, GGML-Q8_0 andere
      Block-/Layout-/Kernelpfade; beide sind quantisierte Attention-KV.

      **Verbindliche Nachmessung (User-Auftrag):**
      - [ ] Identische Prompt-Suite: Fakten/kurze Antwort, deutsche und
        englische Prosa/Dialog, Code, JSON/Extraktion, Mathematik/Reasoning.
      - [ ] target-only und n_max=1/2/3/4/6/8, mindestens 3 Reps;
        Sampling/Seed/Reasoning-Mode, Sidecar, KV, Offload, Cache und
        Server-Slots explizit identisch halten; effektive Config sichern.
      - [ ] Kurze/mittlere/lange Antworten und Durchsatzbudgets
        128/512/2048 Tokens getrennt testen. EOS/length, Content und
        Reasoning getrennt erfassen; gleiche Tokenstrecke fuer reine
        Kostenvergleiche nutzen, keine erzwungene Fortsetzung als
        Qualitaetsnachweis verkaufen.
      - [ ] Kontextkapazitaet und tatsaechliche KV-Belegung getrennt
        variieren (kurz, 4k, 16k/32k soweit stabil); VRAM und fuer den
        Expert-Cache verbleibendes Budget mitprotokollieren.
      - [ ] Zeitanteile Draft, Target-Verify, Hidden-State-Transfers,
        Checkpoint-Save/Restore und Replay sowie akzeptierte Tokens
        pro Target-Pass messen; Strata T=2/3/4 bei gleicher Textstrecke
        und effektiver Draft-/Suffix-Policy vergleichen.
      - [ ] Erst nach Logit-/Output-/Rollback-Paritaet, Delivery-Pruefung
        und wiederholtem Netto-Latenz-/tg-Gewinn Gesamtgate abhaken.
      Nebenbefund: Der untersuchte gdb-Backtrace zeigte gesunden
      `server_queue::start_loop`-Idle und accept4, keinen Init-Deadlock.
      Fehlende INFO-Zeilen koennen auch durch `-lv`-Filter entstehen;
      stderr-Buffering ist nicht pauschal bewiesen. Status via `/health`
      pruefen, nicht aus statischem RSS oder Log-Tail auf Deadlock schliessen.

### 10.8 Integration, Fairness und Entscheidungspunkte

**Arbeits-Pakete (P1a/P1b bestanden; Baseline-Konsolidierung und
Streaming-/Residenz-/MTP-Gesamtgates noch offen):**

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

**Fortschritt Integration (Stand 2026-10-07):**

- [x] P1a auf eigenem Feature-Branch commitiert und ueber `origin`
      verteilt (`0389e655e8`, `feat/flashnext-p1a-q2-compat`);
      Produktiv-Checkout unangetastet, Remote-Worktree isoliert.
- [x] P1b auf eigenen Commits (`3f3d272f69` Slot-Padding,
      `819af39886` ARANGE-src[0]-NULL) auf den noch offenen
      Folge-Branches; kein neuer Push-/Remote-Status-Anspruch.
- [ ] `[OFFEN]` Getrennte Commits/Branches fuer S1, S2, M gemaess
      Sequenz (Baseline → S1 → S2 → M).
- [ ] `[OFFEN]` Messziele pro Phase: reproduzierbarer Netto-Gain
      ausserhalb Streuung, korrekte Logits, keine Regresse.
- [ ] `[OFFEN]` Stufenmatrix inkl. Streaming×Arena und Streaming×MTP
      gemessen (gleiche Quant, Abbruch bei OOM/NaN/Regress).
- [ ] `[OFFEN]` Qualitaets-/Stabilitaets-Gate: 1/2 GPU, Slots, Kontexte,
      Serverstarts/Soak, Test-Suiten gruen; Abschlussstatus mit ROADMAP
      #99 und Benchmark-Note synchronisieren (inkrementelle DOX-/
      CHANGELOG-Pflege erfolgt bereits waehrend der Umsetzung).
- [ ] `[OFFEN]` Stop-Kriterien geprueft und Investitionsentscheidung
      bei Restgap dokumentiert.

### 10.9 Befund-Ledger (Code-Investigation) und angepasste Arbeitsplanung — Stand 2026-10-08

**Bewiesen (Restore-Bug) — zeitlich vor Commit `ab6e9dfcea`:**

- **Multistream-Fullstate-Restore-Bug (vor `ab6e9dfcea`):** der
  Whole-Context-Restore rief den KV-Cache-Clear (`clear(true)` in
  `llama-kv-cache.cpp`) einmal global UND erneut pro Stream innerhalb
  `state_read_meta` — frueher restaurierte Streams wurden gewiped. Test
  `tests/test-state-restore-fragmented.cpp::test_multistream_restore`
  (drei nicht-unifizierte Streams) schlug **vor** dem Fix auf CPU
  TinyStories `-ngl0` fehl: `stream0(f16) differs after full restore`.
  **Nach** Entfernen des per-Stream-Clears (in `ab6e9dfcea`):
  f16-Byte-Paritaet aller drei Streams und bestehender
  Fragmentierungs-/Decode-Test bestanden (Exit 0, frischer CPU-Lauf).
  TinyStories hat Head-Dimension 48 und erlaubt kein q8_0-KV;
  q8_0-Paritaet auf kompatiblem Modell noch offen, kein falsches PASS.
  Test prueft f16 und optional den explizit konfigurierten K-Typ.
  Betrifft nur den Fullstate-Pfad, **nicht** den partiellen
  Spec-Checkpoint-Pfad. Fix/Test auf `feat/flashnext-kv-restore-regressions`,
  Commit `ab6e9dfcea`, fokussierter Review ohne P0/P1, Diagnose-/
  Nichtleer-/Positionsguards nachgezogen und CPU-Test erneut bestanden;
  kein Claim einer Qwen-MTP-Reparatur.
- Qwen4Exp klemmt `n_rs_seq` auf 0 (`llama_context`-Konstruktion;
  `llama-arch` erlaubt den recurrent-State-Pfad nur fuer
  qwen35/moe).

**Widerlegt:**

- **CPU-Rawfloat→q8_0-Write-Hypothese:** `ops.cpp:5051` und `:5071`
  nutzen den `dst`-`from_float`-Quantizer — kein roher Float-Write
  in q8_0-KV.
- **Hybrididx-`init_batch` (Prae-Fix-Befund):** berechnete `n_rs_seq`,
  ignorierte dann den Vertrag gegen base — kann wegen des
  Qwen4Exp-`n_rs_seq=0`-Clamps den beobachteten Qwen-Lauf nicht
  erklaeren (Nicht-Kausalitaet bleibt bestehen). Die Vertrags-
  Korrektur selbst ist in `ab6e9dfcea` enthalten.

**Prefill-Quellenpruefung und erste Laufzeitprofile:**

- `--cpu-moe` legt Expert-Gewichte auf Hostspeicher; es erzwingt nicht
  automatisch CPU-Berechnung. `op_offload` kann unterstuetzte Expert-Ops
  auf CUDA dispatchen (`ggml-backend.cpp:963-970`). Tatsaechlichen
  Dispatch messen, nicht aus dem Flag auf CPU-Rechenzeit schliessen.
- CUDA besitzt MMVQ, MMQ und MMF vor dem synchronisierenden cuBLAS-
  Fallback (`ggml-cuda.cu:2940-2963`). Die pauschale Hypothese
  "grosse Batches haben nur sorted-cuBLAS, grouped-MMQ fehlt" ist
  widerlegt; der real verwendete Kernel und dessen Zeitanteil sind offen.
- Selektiver Upload benoetigt Router-IDs auf dem Host; Full-Tensor-
  Prefetch vermeidet diesen Wait bei hoher Abdeckung, ist aber noch
  kein expertweise vorauseilender Cross-Layer-Stager wie in Strata.
- Erster Nsight-Lauf (UD-IQ4_XS, zwei GPUs, synthetisches pp8192,
  Prefetch an, physischer ubp8192) **abgebrochen, CUDA-OOM**:
  QSA-Top-k ruft `argsort_f32_i32_cuda_cub` mit 8192x8192 auf;
  Allokation weiterer 256 MiB temporaerer Indizes scheitert.
  Keine neue Durchsatz-/Qualitaetszahl daraus ableiten. Workspace,
  physische Chunk-Grenze und Sortierpfad mitprofiliert untersuchen;
  Wiederholung mit ub4096 beendet; Profil zeigt MMQ, GDN und viele
  elementweise Ops, nicht nur sorted-cuBLAS. Wichtiger Runner-Fund:
  `llama-bench -dev CUDA0,CUDA1` erzeugt **zwei Single-GPU-Testfaelle**;
  echter Dual-GPU-Wert braucht `-dev CUDA0/CUDA1`. Erste Profilstatistik
  umfasst beide Single-GPU-Faelle und Loading, ist keine isolierte
  Dual-GPU-Zeitzerlegung. Bench meldet ausserdem alten Build-Metastand
  `3f3d272f69`; Target wird vor weiteren Messungen frisch gebaut.
  Nicht identisch zum serverbasierten B2-23k-Test; effektive Split-/Slot-
  Konfiguration abgleichen. Opt-in Workspace-Prototyp angelegt auf
  `feat/flashnext-topk-workspace`: zeilenweise CUB-Argsort-Batches statt
  temporarer Indizes/Keys fuer alle Prefill-Zeilen gleichzeitig.
  Default unveraendert; Build/Backend-Paritaet und netto Prefill-Gewinn
  bleiben bis zur echten Gegenprobe offen.

**Offen (unbewiesen, keine Gate-Freigabe):**

- Quant-multrow/`set_rows`/FA-Pfad, Logits nach Restore,
  `pending_h`/Tokenposition als Ursache der q8_0-Spec-Degeneration.
- Wiederholte Positionen (83/84, wechselnde Kandidaten bei 85) allein
  sind kein Livelock-Beweis — gegen normales Checkpoint-Replay/
  Cache-Aufholung abgrenzen.

**Angepasste Prioritaet (User-Vorgabe):** schnellstmoeglicher
*korrekter* Prefill Richtung Strata-Latte, nicht eine Arena um ihrer
selbst willen. Vergleichslatte niemals quant- oder GPU-uebergreifend
mischen: Strata same-UD dual @23k = 1227 pp t/s; Strata IQ3 dual
~1730, 1 GPU ~1154. Grosse physische Batches (`-ub/-ubp`, auch >8192
falls unterstuetzt) sind als Test-/Konfigurationsmassnahme erlaubt.
Gezielte Kernel-/Layout-/Scheduler-Adaptionen nur bei profiliertem
Restengpass, in spezialisierten Feature-Branches. Nach kleinen
Gewinnen weiteriterieren, nicht aufhoeren.

**Neue verifizierte Zwischenschritte (keine Gesamtfreigabe):**

- Frisches Dual-GPU-Profil mit Slash-Device-Liste (`CUDA0/CUDA1`),
  Build `cf86da5927`, synthetisches pp8192/ub4096: MMQ und GDN aktiv;
  Prozessprofil inkl. Laden zeigt ~128 GB H2D und ~21,5 s summierte
  H2D-Kopierzeit. Keine exakte Prefill-Zeitzerlegung daraus ableiten.
- Unprofiliertes pp8192/ub8192, gleiche UD-IQ4_XS-Datei, Dual-GPU,
  Prefetch an/Cache aus, drei Reps: 425,3 / 469,7 / 469,9 t/s.
  Median ~469,7; erste/warmere Laeufe getrennt betrachten. Noch kein
  Same-Text-Vergleich mit Strata und kein Qualitaetsgate. Passende
  unprofilierte ub4096-Gegenmessung gestartet.
- MTP-Runner implementiert, Mock-Tests und echte CPU-Server-Probe
  vorhanden. Main-Gegenpruefung fand trotz anfaenglich gruener Mocks
  entfernte CLI-Flags, falsche Thinking-Payload und zu schwache
  Delivery-/Identitaetschecks; korrigiert. Das Re-Review ist
  abgeschlossen (f2a6afc6) und hat verbleibende P2-Befunde
  identifiziert (Startup-/Request-Ctrl-C-Final-Metadata,
  Terminate-/Kill-Timeout, Fingerprint-Fehlerpfad — siehe
  scripts/AGENTS.md); diese Lifecycle-/Test-Punkte sind offen.
  GPU-Matrix erst nach dieser Verifikation. Kein Mock-PASS als
  Performance-PASS.

**Arbeits-Checkliste:**

- [ ] `[OFFEN]` q8_0-Spec-Diagnose gebunden auf 60-Minuten-Budget:
      nur kontrollierte Tests (Mehrtoken-Forward vs. sequenzieller
      Forward, Byte-/Logit-Paritaet vor/nach Restore, K/V-Isolation,
      Commit-Fortschritt). Ergebnis danach festhalten, weiter zum
      Prefill-Pfad.
- [ ] `[OFFEN]` Kontrollierte MTP-Matrix (§10.7): identische Prompts/
      Laengen/Kontexte, >=3 Reps, Stage-Kosten Draft/Verify/Checkpoint
      separat, Ein-Schritt-Orakel und Greedy-/Rollback-Paritaet **vor**
      jeder Performancezahl.
- [ ] `[OFFEN]` Baseline konsolidieren: `--no-mmap` warm und
      Host-Registrierung als fehlende Kontrollen (Rest-Luecke §10.4).
- [ ] `[OFFEN]` Prefill-Profil (NVTX/CUDA-Events pro Layer/Expert)
      plus Chunk-Sweep 256..8192, >8192 falls unterstuetzt und Budget
      reicht; danach den *gemessenen* Mechanismus implementieren,
      kein festes Arena-Rezept.
- [ ] `[OFFEN]` Iteration same-Quant auf 1 und 2 GPUs getrennt messen.

**Branch-Vorschlaege (nicht angelegt — Anlage/Verifikation beim
Main-Agent):**

- `feat/flashnext-prefill-profile` — Instrumentierung + Chunk-Sweep
- `feat/flashnext-s1-prefill-pipeline` — S1-Streaming-Mechanismus
- `feat/flashnext-kernel-<op>` — gezielte Kernel-/Layout-Ports nach Profil
- `feat/qwen4exp-q8-spec-fix` — falls aus der q8_0-Diagnose ein Fix folgt

### DOX-Audit-Checkpoint 2026-10-08

- [x] DOX-Audit durchgefuehrt und Audit-Record `docs/DOX_AUDIT.md`
      angelegt; Root-/Child-AGENTS-Kette, kanonische Docs und aktive
      lokale Verweise korrigiert (nur Doku, keine Builds/Tests).
- [ ] `[OFFEN]` Trilium-Notiz-Befunde aus `DOKU_BEOBACHTUNGEN.md`
      mit Live-Lesegriff pruefen (nicht als geloest markiert).
- [ ] `[OFFEN]` `test-speculative-mtp.cpp`: Legacy-Testquelle auf
      aktuelle API-Kompatibilitaet pruefen/portieren; erst danach eine
      Registrierung entscheiden.
- [ ] `[OFFEN]` Bench-Runner P2-Review: Ctrl-C-Final-Metadata,
      Terminate-/Kill-Timeout, Fingerprint-Fehlerpfad.
- [ ] `[OFFEN]` Topk-Worktree: Cross-Branch-Uebertragung der
      korrigierten Root-/Child-DOX; TOP_K-Row-Cap CUDA-Paritaetsgate.
- [ ] `[OFFEN]` Legacy-Docs mit Host-/Pfad-Details bei Bedarf
      bereinigen (keine globale Sauberung in dieser Runde).

### 10.10 Wiederaufnahme S1/S2 nach DOX-Commit

- [x] DOX-Hauptstand committiert: `284805e223`; getrennte
      Top-k-Dokumentation `b34b363e26`, unvalidierter Code nicht enthalten.
- [x] Ausstehende lokale ub4096-Rohantwort ausgewertet:
      `prefill-dual-4096-reps3-20261008.log`, synthetisches pp8192,
      Build `cf86da5927`, Dual-GPU, UD-IQ4_XS, q8_0/q8_0,
      drei Samples 318.462 / 335.751 / 335.616 t/s, Median 335.616.
      Vergleich zum bereits verifizierten ub8192-Median 469.724:
      Faktor 1.39959, etwa +39.96 %. JSON-Konfigurationen stimmen
      ausser physischer UBatch und Zeitstempel ueberein; Environment
      ist dort nicht gespeichert. Kein vollstaendiges A/B-Gate,
      keine Output-Paritaet und kein Same-Text-Strata-Vergleich.
- [x] Neuer isolierter Worktree/Branch `feat/flashnext-s1-lookahead`
      auf `284805e223` angelegt.
- [x] Scheduler-Pfad gelesen: voller Expert-Upload beginnt derzeit
      beim verbrauchenden Split; dessen Compute-Stream wartet
      unmittelbar auf das Ready-Event. Zusaetzliche Slots allein
      garantieren kein vorauseilendes Kopieren.
- [x] S1-Prototyp implementiert (`cdb42d8471`, Branch
      `feat/flashnext-s1-lookahead`), `ggml/src/ggml-backend.cpp`:
      Prefetch-State jetzt pro Backend/Device statt global einmalig;
      opt-in `GGML_SCHED_PREFETCH_ALL=1` staget jeden eligible
      Host-Weights-MUL_MAT_ID-Input eines Splits (Slot-Ring,
      Holder-Tracking gegen zirkulaere Event-Waits, 512B-MMQ-Padding
      unveraendert). Ready-Event-Waits gebundelt unmittelbar vor
      `ggml_backend_graph_compute_async`, damit generische Kopien
      späterer Inputs nicht hinter früheren Prefetch-Waits hängen.
      Eligible Inputs ohne freien Slot fallen unter PREFETCH_ALL auf
      generischen Async-Vollupload statt in den selektiven Pfad —
      der Routing-IDs-Rücklese-Sync (`ggml_backend_synchronize`)
      entfällt bei grossen Batches. CPU-Backend-Splits bleiben beim
      selektiven Pfad. Default ohne Env unveraendert.
- [ ] Lokale Scheduler-Regressionspruefung laeuft
      (`test-backend-ops` CPU, bisher alle Checks OK); danach
      CUDA-Paritaet und kontrolliertes Prefill-A/B auf dem
      Testsystem: PREFETCH_EXPERTS=1 vs. +PREFETCH_ALL=1 (3 und 8
      Slots) vs. PREFETCH_ALL allein, pp8192, 3 Reps, Median.
- [ ] S2 bleibt zweite Hauptsaeule: Warm-mmap/no-mmap/Hostregister
      fair vergleichen; Residenz/Pinning nur bei belegtem Zusatznutzen
      adaptieren, keine Arena als Selbstzweck.
- [ ] SSH-/Netzwerkfreigabe fuer die weitere CUDA-Gegenprobe
      angefragt; bis dahin nur lokale Arbeiten ohne GPU-Inferenz.
