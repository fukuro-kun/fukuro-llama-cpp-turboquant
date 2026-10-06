# SESSION_PLAN: Strata + OptLlama A/B-Eval auf Uranus (Qwen3.8-Flash-Next)

**Erstellt:** 2026-10-06
**Status:** [ ] Vorbereitet — Uranus offline (verifiziert 2026-10-06), wartet auf Host-Verfuegbarkeit
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
- MTP: GGUF hat eingebettete NextN-Layer → Fork `--spec-type draft-mtp`; Strata nutzt eigenes MTP (2,4–3,2 tok/pass) — Akzeptanzraten vergleichen.

## 7. Erfolgskriterium (Gate)

Abgeschlossen wenn: pp/tg-Matrix fuer >= 2 der 3 Engines gefuellt, Obergrenze quantifiziert, Port-Go/No-Go mit Begruendung in ROADMAP + Trilium-Subnote `5Ec9sW4aCN6S` eingetragen.

## 8. Doku-Ziele nach Abschluss

- Ergebnisse → Trilium Benchmark-Note (Wegweiser `8cIsjKEhz7Fd` beachten — nie in Modell-Info-Notes)
- ROADMAP-Item je nach Ergebnis anlegen/aktualisieren
- `docs/fork/ROADMAP.md` #99 Phase 9-Status verlinken
