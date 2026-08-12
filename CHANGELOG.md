# CHANGELOG — fukuro-llama-cpp-turboquant

Chronologische Auflistung aller bedeutsamen Änderungen. Solo-Session-Agenten tragen hier ihren Fortschritt ein.

Format: `YYYY-MM-DD — <type>: <Was> — <Warum>`

---

## 2026-08-12

### fix: Phobos Vulkan Performance-Regression — 3 Root Causes behoben

**RCA:** Phobos lief ~1,5 Tage auf CPU (3-7 t/s statt 22-27 t/s) nach einem
Rebuild am 10.08. Drei kombinierte Ursachen:

1. **`GGML_VULKAN=OFF` im Build-Cache** — Der Rebuild am 10.08. hatte
   Vulkan nicht aktiviert. Der Service startete normal mit CPU-Only-Warning.
   Fix: Rebuild mit `-DGGML_VULKAN=ON`, Build-Verifikation als DOX-Regel.

2. **TurboQuant Vulkan-Shader revertiert** — Commits `603f47105` und
   `9cbabbad8` revertierten die Vulkan TurboQuant Shader. Der f16-Fallback-
   Workaround (`9918d20e1`) wurde mit `cb3b1d571` entfernt. turbo3/4 auf
   Vulkan → Garbage Output / CPU-Fallback.
   Fix: Workaround in `llama-context.cpp` wiederhergestellt (Commit `973fa454a`).
   Start-Skripte auf q4_0 KV (Mars) / f16 KV (Venus) umgestellt.

3. **`fit_params` + `--mmproj` auf APU** — `fit_params` reserviert GPU-Speicher
   für mmproj in der Margin. Auf unified-memory APUs (AMD 760M) bleibt nichts
   für das Hauptmodell → `n_gpu_layers` auf 0 → CPU-Fallback.
   Fix: `-fit off` in allen Vulkan-Start-Skripten.

**Regression:** Mars von 256k/2 Slots/turbo3/4 auf 128k/1 Slot/q4_0.
Performance: ~27 t/s (tg), ~40 t/s (pp) — leicht über historisch 22-25 t/s.

**Commits:**
- `797c46b25` — Start-Skripte: f16 KV + -fit off + mmproj
- `9a64d8447` — --no-warmup für Vulkan-Server
- `b18be2f09` — Produktiv-Konfiguration Mars (q4_0 KV, 128k, 1 Slot)
- `973fa454a` — f16-Fallback-Workaround in llama-context.cpp

**DOX:** Build-Verifikations-Regel in AGENTS.md hinzugefügt.

---

## 2026-08-10

### fix: HTTPS-Support für llama-server (OpenSSL/libssl-dev) — alle 5 LAN-Hosts

- **Problem:** `LLAMA_OPENSSL=ON` ist CMake-Default, aber CMake baut stillschweigend ohne SSL weiter wenn `libssl-dev` fehlt (`OPENSSL_CRYPTO_LIBRARY-NOTFOUND`). Kein Build-Fehler, keine Warnung — HTTPS fehlt nur zur Laufzeit. HTTP 500 bei `image_url` mit HTTPS-URLs.
- **Fix:** `libssl-dev` auf allen Hosts installiert, CMake-Cache gelöscht (`rm build/CMakeCache.txt`), neu gebaut mit selben Backend-Flags.
- **Hosts:**
  - **hydra** (CUDA, Build-only): libssl-dev 3.5.5 installiert, glibc-Patch (rsqrt aus mathcalls.h entfernt), Rebuild. ✅ SSL_new=2 in libllama-server-impl.so, libssl.so.3 gelinkt.
  - **uranus** (CUDA, On-Demand): libssl-dev 3.0.13 bereits installiert, nur Rebuild. ✅ OpenSSL gefunden, SSL_new=2.
  - **styx** (CUDA, Dauer-Server): libssl-dev 3.0.13 installiert, Rebuild auf `/data/git/`. ✅ SSL aktiviert, HTTPS-Test OK. 26B-Service reaktiviert (VLM gestoppt wegen CUDA OOM — existierendes Problem, 8 GB VRAM zu klein für beide).
  - **phobos** (Vulkan, LXC auf mars): libssl-dev 3.5.6 installiert, Rebuild. ✅ Echte HTTPS-Downloads funktionieren (Logs: 30320 bytes von gstatic.com, 13504 bytes von google.com).
  - **venus** (Vulkan, Dauer-Server): libssl-dev 3.0.13 installiert, Rebuild. ✅ User-Service restarted, HTTPS-Test OK ("Failed to download image" statt "HTTPS is not supported"). Venus NICHT suspendiert (User-Wunsch).
- **Verifikation:** `strings build/bin/libllama-server-impl.so | grep -c SSL_new` >0 (SSL-Code in Shared Library, nicht im Binary). Funktions-Test: HTTPS-URL wird akzeptiert, Fehler ist "image input is not supported" oder "Failed to download image" — **nicht** "HTTPS is not supported".
- **Doku:** `AGENTS.md` Build-System-Sektion um libssl-dev als Build-Voraussetzung ergänzt.

## 2026-07-26

### Code-Cleanup: MoE-Cache toter Code entfernt (-358 Zeilen)

- **refactor:** `moe-cache.cu` und `llama-model.cpp` bereinigt. Experimenteller Code aus 3 ❌ NO-GO Experimenten (#71 Set-Associative, #18 Workload-Aware, #44 K-Override) entfernt.
  - **Entfernt:** `POLICY_SET_ASSOC_LRU`, `POLICY_WORKLOAD`, `workload_score`, `n_sets`/`n_ways`/`set_lru_*`, `window_count`, `set_assoc_ways`, `workload_wsize`, Set-assoc LRU-Helfer, Workload Reset-Logik, Env vars `GGML_CUDA_MOE_CACHE_SET_WAYS`/`GGML_CUDA_MOE_CACHE_WSIZE`, `LLAMA_MOE_K_OVERRIDE` Block.
  - **Behalten:** `POLICY_LRU` (default), `POLICY_HEURISTIC`, Grund-Cache-Infrastruktur, Prefetch/backfill/worker.
  - **Build:** ✅ Grün. MoE-Cache initialisiert korrekt mit LRU-Policy.
  - **Commit:** `04ed54f16`

### Research-Sweep #5: Monatliche Optimierungs-Recherche

- **research:** 4 parallele Subagents (Vulkan/AMD, CUDA/MoE, arXiv, Multi-GPU). 51 Items gesammelt, dedupliziert, Top-5 verifiziert.
  - **Verifikation:** 5/5 Quick-Wins bereits im Fork (#15524=#6, #16391=--cram, #19754=warmup, #25479=#3, #22887=K_PER_ITER).
  - **9 neue ROADMAP-Items:** #89-97 (davon #89✅, #90❌, #91✅ bei Verifikation).
  - **Fazit:** Fork ist saturiert — keine neuen umsetzbaren Quick-Wins verfügbar.
  - **Commits:** `311790c79`, `a13f4087c`

### #44 Alloc-MoE Phase 0 Quality-Benchmark — ❌ NO-GO

- **feat: `LLAMA_MOE_K_OVERRIDE` env var** — Override für `hparams.n_expert_used` zur Laufzeit. Erlaubt Reduktion der aktiven Experten pro Token (K) ohne Code-Änderung. Gelesen in `llama-model.cpp:load_hparams` nach GGUF-Parsing, vor Asserts. Validierung: 1 ≤ K ≤ n_expert. 0/unset = GGUF-Default.
  - **Datei:** `src/llama-model.cpp` (+15 Zeilen)
  - **Perplexity (WikiText-2 test, 20 chunks, ctx=2048):**
    - K=8 (Default): PPL = 493.96 ± 16.63
    - K=6: PPL = 560.69 ± 18.88 (**+13.5%**)
    - K=4: PPL = 607.11 ± 20.20 (**+22.9%**)
    - K=2: PPL = 822.51 ± 26.68 (**+66.5%**)
    - K=1: PPL = 1380.88 ± 44.41 (**+179.5%**)
  - **Speed (tg128, r=3, -ngl 0, 512MB MoE-Cache off):**
    - K=8: 3.87 t/s | K=6: 4.80 (+24%) | K=4: 5.33 (+38%) | K=2: 5.93 (+53%) | K=1: 6.43 (+66%)
  - **Go/No-Go >5% PPL-Drop → ❌ NO-GO.** Selbst K=6 (25% Reduktion) hat +13.5% PPL-Drop — 2.7× über Limit. Root Cause: Gemma-4-A4B mit K=8 hat moderate Expert-Diversität. Reduktion auf K=6 entfernt 25% der Diversität → signifikanter Quality-Drop. Alloc-L (per-Layer K) könnte besser sein, aber Basis-Drop bei K=6 zeigt zu kleinen Spielraum.
  - **Wichtige Korrektur:** Subagent-Analyse nahm K=2 als Default an → "17% Quality-Drop bei K=2". Tatsächlich ist K=8 der Default. K=2 hat +66.5% Drop (viel schlechter als angenommen).
  - **ROADMAP #44 → ❌.** Code (`LLAMA_MOE_K_OVERRIDE`) bleibt als Analyse-Tool.

## 2026-07-25

### #18 DALI Workload-Aware Cache Policy — ❌ NO-GO

- **feat: `POLICY_WORKLOAD` in `moe-cache.cu`** — DALI-inspired sliding-window workload accumulation. Windowed frequency pro Slot (workload_score), periodic reset nach wsize plan()-Calls, evict slot mit niedrigstem workload_score. Env-Vars: `GGML_CUDA_MOE_CACHE_POLICY=workload` + `GGML_CUDA_MOE_CACHE_WSIZE=N` (Default 32). Vollständig parallel zu LRU/Heuristic/Set-Assoc (A/B-Test-fähig).
  - **Datei:** `ggml/src/ggml-cuda/moe-cache.cu` (+~80 Zeilen: enum, slot.workload_score, pool.window_count, eviction path, hit recording, periodic reset, env-vars)
  - **Benchmark (hydra, RTX 3070, 8GB, 26B-A4B QAT, -ngl 0):**
    - Erster Benchmark (n=128, r=3): LRU tg128=2.19±1.04, Workload=3.97±0.41 (+81%) — **Artefakt** (Cache-Discovery-Phase verlangsamte LRU)
    - Verification (n=256, r=5): LRU tg256=4.42±0.20, Workload=4.53±0.09 (+2.5%), Heuristic=4.50±0.16 (+1.8%)
    - Final (n=512, r=5): 512MB: LRU tg512=4.45±0.11, Workload=4.40±0.13 (-1.1%); 1024MB: LRU=4.52±0.07, Workload=4.54±0.07 (+0.4% Rauschen)
  - **Go/No-Go >5% → ❌ NO-GO.** Root Cause: 128 Experten × 30 Layer = 3840 Entries → 160-320 Slots = 24:1 Oversubscription. Eviction-Policy nicht Bottleneck (PCIe dominiert). LRU ist "good enough".
  - **MoE-Cache-Thema für 128-Expert-Modelle erschöpft:** #69 Heuristic ❌, #71 Set-Assoc ❌, #18 Workload-Aware ❌ — alle im Rauschbereich oder schlechter als LRU.
  - **Code bleibt** als `POLICY_WORKLOAD` (Default OFF). Phase 2 (Greedy Assignment) nicht anwendbar (kein CPU MoE path). Phase 3 (Residual-Prefetch) optional für später.
  - **ROADMAP #18 → ❌.**

### #71 Set-Associative MoE Cache — ❌ NO-GO

- **feat: `POLICY_SET_ASSOC_LRU` in `moe-cache.cu`** — N-index M-way set-associative Cache als neue Eviction-Policy. Implementiert: `moe_cache_set`-Struktur (n_sets × n_ways Slots, LRU pro Set), Hash-basiertes Set-Mapping (`key % n_sets`), set-lokale LRU-Helfer, Env-Vars `GGML_CUDA_MOE_CACHE_POLICY=set-assoc-lru` + `GGML_CUDA_MOE_CACHE_SET_WAYS=M`. Vollständig parallel zum bestehenden fully-associative LRU/Heuristic-Pfad (A/B-Test-fähig). Backfill-Worker und Error-Recovery an set-associative angepasst.
  - **Datei:** `ggml/src/ggml-cuda/moe-cache.cu` (+136 Zeilen plan, +48 Zeilen backfill, +47 Zeilen LRU-Helfer, +39 Zeilen pool-init, +28 Zeilen env-var)
  - **Benchmark (hydra, RTX 3070, 8GB, 26B-A4B QAT, -ngl 0):**
    - 512MB Budget (160 slots): LRU tg64=1.90, Set-assoc 40×4=1.42 (**-25%**), 80×2=0.94 (**-50%**), 20×8=1.13 (**-41%**)
    - 1024MB Budget (320 slots): LRU tg128=1.36, Set-assoc 80×4=1.34 (-1.5%, Rauschen), 40×8=1.35 (-0.7%, Rauschen)
  - **Go/No-Go >5% → ❌ NO-GO.** Root Cause: 128 Experten × 30 Layer = 3840 Entries → 160-320 Slots = starke Oversubscription. Fully-associative LRU hat keine Conflict-Misses, Set-assoc restrictiert Placement. HashMap O(1) ist nicht Bottleneck (PCIe dominiert). Paper's 4.4× war RTX 4090 + SSD-Offloading (anderer Bottleneck).
  - **Code bleibt** als `POLICY_SET_ASSOC_LRU` (Default OFF) — potenziell nützlich für Modelle mit wenigen Experten (≤16), dokumentiert als ❌ für 128-Expert-Modelle.
  - **ROADMAP #71 → ❌**, SESSION_PLAN aktualisiert.

### /slots: progress + n_tokens_total Felder

- **feat: `progress` und `n_tokens_total` im `/slots` JSON-Endpoint** — Exponiert den Prefill-Fortschritt (`progress`, 0.0–1.0 geclampt) und die Gesamtzahl der Task-Tokens (`n_tokens_total`) pro Slot. Ermöglicht dem janus-Router während des Prefills zu pollen und zwischen "Backend arbeitet (SWA cache invalidation)" und "Backend hängt" zu unterscheiden. Spec: `docs/SPEC_slot_progress.md`.
  - **Datei:** `tools/server/server-context.cpp` (`server_slot::to_json()`, im `if (ptask)` Block nach `n_prompt_tokens_cache`)
  - **Edge Cases:** `ptask == nullptr` → Felder fehlen (if schützt); `n_tokens() == 0` → 0.0 ( guarded); `progress > 1.0` (Streaming-Input) → `std::min(1.0, ...)` clamp; Idle-Slot zeigt Endstand des letzten Tasks
  - **Review:** `review-swe` Subagent fand P1 (dreifacher `ptask->n_tokens()` Aufruf → Inkonsistenz bei Streaming-Input) → gefixt durch lokale Variable `n_tokens_total`
  - **Verifikation:** CPU-only Smoke-Test auf Dev-Host mit **Gemma-4-12B IQ4_NL** (nur fuer lokalen Compile-Check, *nicht* das Produktivmodell — Produktivstandard ist 26B-A4B QAT): `progress` steigt sauber 0.003 → 0.74 → 0.81 → 0.998 → 1.0, `n_tokens_total=2766` konsistent. Echte Verifikation mit 26B-A4B QAT auf phobos + styx nach Deployment (`progress=1.0, n_tokens_total=22`). Unit-Tests blockiert durch fehlendes `libssl-dev` (HTTPS fuer HF-Downloads) — Environment-Problem, nicht Code. Hinweis: E2B-Modell (ideal fuer CPU-Tests, 2.6 GB) liegt lokal unter `~/modelle/gemma-4-E2B-it/` und auf ganymed `/titan/topas/modelle/gemma-4-E2B-it/` — beim naechsten Smoke-Test verwenden statt 12B.

## 2026-07-23

### POST /cancel Review-Fix Loop (6 Runden, ship-ready)

- **fix: 6-Runden Code-Review mit SWE-1.7 bis ship-ready** — Der initiale `/cancel` Endpoint (`d7802c155`) hatte 4 P1 + 8 P2 Issues. Sechs Review-Runden mit `review-swe` Subagents (Pflicht-Loop seit code-review Skill Update) fanden insgesamt 2 P0, 8 P1, 15 P2 — alle gefixt. Zwei P0 in R3/R4 wären ohne Re-Review-Loop unentdeckt geblieben.
  - **R1** (`e3d9103c4`): 4 P1, 8 P2 → METRICS check für task_id, `device_mutex` lock + `active_count` refusal in `ggml_backend_cuda_device_reset`, `proxy_post` try/catch für empty body, dev_reset Warn-Log, Test-Skript Negative-Cases
  - **R2** (`33794496e`): 3 P1 → `start_time` in `slot.to_json()`, `dev_reset` sleep-mode tradeoff kommentiert, `cudaSetDevice` error logging via return value (nicht `cudaGetLastError`)
  - **R3** (`508f1c734`): **1 P0** (`t_start_process_prompt` uninitialised → UB in `to_json`), 1 P1 (`cudaSetDevice` noch falsch)
  - **R4** (`89a45eaf5`): **1 P0** (`spec` pointer uninitialised → UB via `can_speculate()` in `to_json`), 2 P1 (`id` uninitialised, `start_time=0` sentinel ambiguity)
  - **R5** (ship-ready): In-class initialisers `spec=nullptr`, `stop=STOP_TYPE_NONE`, `sampled=LLAMA_TOKEN_NULL`, `id=-1`, `start_time=-1` sentinel
  - **R6** (`7a8b0db74`): 1 P2 — `start_time=-1` sichtbar in `/slots` → nur ausgeben wenn `>=0`
  - **Deployment:** `7a8b0db74` auf allen 5 Hosts (styx, phobos, uranus+VLM, venus) deployiert + verifiziert. `/cancel` returns `{"cancelled":false,"error":"task not found"}` für non-existent task_id auf allen Hosts.
  - **Dateien:** `tools/server/server.cpp`, `tools/server/server-context.h`, `tools/server/server-context.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`, `common/speculative.cpp`, `scripts/test-cancel-endpoint.py`

## 2026-07-22

### POST /cancel Endpoint für laufende Requests

- **feat: `POST /cancel` HTTP-Endpoint** — Bricht laufende Requests ab und gibt den Slot sofort frei. Nutzt den bestehenden internen `SERVER_TASK_TYPE_CANCEL` Mechanismus, der `slot.release()` aufruft, den Slot auf `SLOT_STATE_IDLE` setzt und eine Error-Response an den Streaming-Client sendet (damit dessen HTTP-Handler terminiert). Der Endpoint postet einen Cancel-Task mit höchster Priorität an `queue_tasks`.
  - **Body:** `{"task_id": 12345}` oder `{}` (bricht den ersten laufenden Task ab)
  - **Router-Modus:** `{"model": "name", "task_id": 12345}` — `model` wird von `proxy_post` konsumiert, Rest geht an das Backend. `{}` ohne `model` funktioniert im Router-Modus nicht (welches Backend?).
  - **Response:** `{"cancelled": true, "task_id": 12345}` (fire-and-forget, idempotent — wenn Task schon beendet ist, ist Cancel ein No-Op) bzw. `{"cancelled": false, "message": "no running tasks"}`
  - **Validierung:** Negative `task_id` → 400. Ungültiges JSON → 400.
  - **Task-ID-Tracking:** Option B — `/slots` liefert bereits `id_task` pro Slot, kein zusätzlicher Code nötig. Router kann `/slots` pollen.
  - **Cancel-Handler erweitert:** `send_error(slot, "request cancelled")` vor `slot.release()` — sendet Error-Response an `queue_results` damit die `server_response_reader` des Streaming-Requests terminiert. Ohne dies würde der Streaming-HTTP-Handler forever blockieren.
  - **Test:** `scripts/test-cancel-endpoint.py` — 5 Tests: no-running-tasks, invalid-JSON, negative-task_id, cancel-by-task_id (Slot IDLE + Stream-Terminierung), cancel-first-running. Alle 5 grün auf Hydra (CPU-only).
  - **Dateien:** `tools/server/server-context.h` (`post_cancel` Field), `tools/server/server-context.cpp` (Handler-Lambda + Cancel-Handler `send_error`), `tools/server/server.cpp` (Endpoint-Registrierung + Router-Proxy), `scripts/test-cancel-endpoint.py` (Test).

## 2026-07-21

### M6 Tier-3 Tiefen-Evals KOMPLETT (11/11 Items) + Rebase-Audit Fix

- **fix: Chat-Template-Integration in `diffusion-cli.cpp` wiederhergestellt** — Rebase-Audit (Phase 1.1) fand: Der Chat-Template-Block (`common_chat_templates_apply`) wurde im AtomicBot-Sync-Squash (`394963e4f`) aus `tools/diffusion-cli/diffusion-cli.cpp` entfernt. In `88bd4f052` war der Block vorhanden, in master fehlte er — `prompt_text` wurde direkt tokenisiert statt via Chat-Template formatiert. Re-Applien + `try/catch` für Exception-Sicherheit (P1 aus Code-Review: `common_chat_templates_init/apply` können bei unparsebaren Templates werfen — ohne catch würde `std::terminate` das Programm abwürgen ohne das Modell freizugeben). Fallback auf Roh-Prompt bei Exception. Commit `506375f10`. Rebase-Audit Summary: keine weiteren Verluste gefunden (`common_speculative_*` refactored zu Multi-Impl-API, `llama_get_embeddings_pre_norm` umbenannt zu `_nextn` mit `masked`-Parameter, `ensure_sched_mtp` refactored, `gemma4_mtp_*` Helper in `graph::graph()` integriert, `diffusion-gemma-visual-server.cpp` war nie buildbar — dead code).

- **eval: #36 Auto Parameter Fitting TP → ❌ verworfen** — PR #22950 ist Draft-Status, seit 2 Monaten stuck (letzte Aktivität 2026-05-16). Reviewer (JohannesGaessler) Bedenken: meta backend memory reporting incorrect → `-fit` reduziert context nicht korrekt. TP funktioniert manuell (#79 ✅, +23-32% tg auf Uranus). Auto-fit ist Convenience, nicht worth Portierung eines buggy Draft-PRs. Commit `128243fdb`.

- **eval: 11 Tier-3 Tiefen-Evals (Phase 3, 3-Subagent-Wellen)** — Alle Tier-3 Items aus `docs/fork/ROADMAP.md` systematisch evaluiert (Paper-Recherche via arxiv-mcp, PR-Status via webfetch, Fork-Kompatibilität via grep, Hardware-Relevanz, Implementierungsaufwand). Ergebnisse:
  - **❌ Verworfen (4):** #73 CascadeInfer (Multi-Instance-Cluster-Scheduler, Architektur-Mismatch), #72 N4_0 (PR stale, kein Blackwell im Fleet), #74 Vulkan Bindless (redundant mit #85 Push Descriptors ✅), #22 GWQ (PTQ, Fork nutzt QAT Q4_K_XL)
  - **⏭️ Später (4):** #76 CPU Fusion (PR #20596 zeigt Regressionen auf Consumer-CPUs), #75 Pipeline Sched (PR #19922 closed, Konflikt mit #79 TP), #23 DuoServe (Bedingungen: #69 <5% UND #70 unzureichend), #43 SliderQuant (nur bei QAT-Lücke)
  - **☐ Machbar gestaffelt (3):** #71 CPU-GPU MoE (2 Wo, nach #69 Benchmark), #18 DALI (MVP 2-3 Wo POLICY_WORKLOAD), #44 Alloc-MoE (MVP 3-4 Wo Alloc-L, 17% Quality-Drop-Risiko bei K=2)
  - Aufwände revidiert (meist nach oben, da ROADMAP-Schätzungen zu optimistisch waren). Commits `db6cf13c2`, `7818943c1`, `b2a1f6b67`, `db31bd05c`.

- **ops: Styx auf `db31bd05c` aktualisiert** — `git pull`, CUDA-Build, Service restarted, health ok. LOKAL.md Deployment-Tabelle aktualisiert.

- **bench: #69 FlashMoE Heuristic Benchmark → ❌ NO-GO** — Styx (GTX 1070, 26B-A4B QAT, -ncmoe 20, turbo3/4, FA, Budget=512MB, Reserve=256MB): LRU tg128=28.16±0.92 t/s (39.7% hit, 320 slots), Heuristic tg128=27.28±0.75 t/s (39.9% hit, 320 slots) → **Heuristic -3.1% langsamer als LRU**, 5%-Speedup-Schwelle deutlich verfehlt. Hit-Rate identisch → Heuristic trifft dieselben Eviction-Entscheidungen wie LRU. Throttle=1 → Bail-out für beide Policies (Cache-Overhead > CPU-Path). Root Cause: α=0.7/β=0.3 Frequency korreliert stark mit Recency auf tg128-Runs. **Phase 2 (FlashMoE FFN) ❌ verworfen** — wenn schon einfache Heuristic kein Win, ist FFN skeptisch. Paper's 2.6× bezieht sich auf SSD-Offloading (nicht CPU-Offload wie Styx). #69 → ❌. **Voraussetzung für #71 und #18 erfüllt** (beide können jetzt starten). Siehe `docs/fork/2026-07-15_MOE_CACHE_PREDICTION_DESIGN.md`.

## 2026-07-19

### M5 (Coopmat2 + Multi-GPU) ✅ abgeschlossen — #21 PagedAttention verworfen

- **eval: #21 PagedAttention / Paged KV Cache → ❌ verworfen** — Tiefen-Recherche PR #22569 (ggml-org/llama.cpp): DRAFT, dirty (Merge-Konflikte), 2+ Monate inaktiv (zuletzt 2026-05-10), kein Maintainer-Feedback, 48 Dateien +4029 Zeilen. Der 2.5x-Durchsatz entsteht bei **247 concurrent sequences** (vLLM-Cloud-Serving-Design nach Kwon et al. 2023). Unser Edge/LAN-Setup fährt `-np 1` (Styx) bzw. `-np 2` (Uranus) — bei 1-2 sequences ist Paged **~3% langsamer** (479 vs 496 tok/s, reiner Overhead). CUDA-only Fokus (Author explizit), Mars (Vulkan/RDNA3) und Venus (Vulkan/GCN) würden nicht profitieren. TurboQuant (turbo3/turbo4, 3-4 bit KV-Kompression) ist die richtige KV-Optimierung für unseren Use-Case: komprimiert KV-Daten direkt statt nur Allokation zu managen. Konkurrenz-PRs #17579 (Collaborator, closed 2026-07-11) und #14070 (closed 2025-06-08) — upstream scheint nicht committed zu PagedAttention. **M5 → ✅ abgeschlossen** (#12✅, #20✅, #21❌).

### Adaptive MTP (#28) Live-Test auf Styx ✅

- **test: skip_streak Mechanismus auf Styx verifiziert** — Produktiver Service gestoppt, Test-Server mit `LLAMA_MTP_SKIP_STREAK_THRESHOLD=2` + `--spec-type draft-mtp` + Draft-Modell `drafts/gemma-4-26b-a4b-it-assistant.Q4_K_M.gguf` gestartet (ctx=32768, Vulkan/CUDA). Log bestätigt `skip_streak_threshold=2 (LLAMA_MTP_SKIP_STREAK_THRESHOLD)` im Konstruktor. Request "20 zufällige 6-stellige Zahlen" (300 tokens): **10+ `skip-streak triggered — returning empty draft (verify-only batch)` Events** in den Logs, MTP acceptance 40-63% (realistisch für 26B-A4B). skip_streak triggert korrekt bei zero-accept Serien und kehrt nach einem verify-only Batch zurück. Produktiver Service danach wiederhergestellt (196k ctx, health ok). Styx Commit: `7e39ea44d`.

### Adaptive MTP (#28) wiederhergestellt — Rebase-Verlust behoben

- **feat: `LLAMA_MTP_SKIP_STREAK_THRESHOLD` in `common/speculative.cpp` re-applien** — Der Adaptive MTP skip-streak Mechanismus war in Commit `88bd4f052` (2026-06-23, diffusion-gemma-v2 Squash) voll implementiert, ging aber bei dem AtomicBot-Sync-Squash (`394963e4f`, 2026-06-23) verloren und wurde im MTP 0% Fix (`4cff93d80`, 2026-06-24) nicht re-applien. Die Doku (`MTP.md`, `docs/speculative.md`, `README.md`) beschrieb weiterhin das Feature — ROADMAP #28 war als ✅ markiert, aber der Code fehlte. Re-Applien auf den neuen `common_speculative_impl_draft_mtp` Struct (multi-seq refactor seit dem Verlust): Member-Variablen (`prev_n_acc_drafts`, `zero_accept_streak`, `skip_streak_threshold`, `skip_streak_last_draft`), `getenv("LLAMA_MTP_SKIP_STREAK_THRESHOLD")` im Konstruktor, `mtp_would_skip_next_draft()` Helper, skip-check + streak-update in `draft()`, reset in `begin()`. Build verifiziert (llama-common + llama-server). ROADMAP #28 → ✅, M4 → ✅.

### Doku-vs-Code Prüfung (doku-pruefung Skill, 3 Subagents)

- **docs: AGENTS.md Styx 196k nachgetragen** — Kontext-Tabelle und Sektions-Überschrift standen noch auf 224k. Services-Tabelle um Venus + Uranus ergänzt, veraltete Skript-Namen (`run-gemma4-26b-a4b-*-server.sh`) durch aktuelle (`start-*-26b-server.sh`) ersetzt. EA Phase 3 (`src/llama-expected-attention.cpp`) in Schlüsseldateien-Tabelle aufgenommen. Qwen NextN-Referenz korrigiert (`qwen35.cpp`/`qwen35moe.cpp` sind Dense/MoE, nicht NextN). GPU-Datum aktualisiert.
- **docs: common/AGENTS.md NextN-Referenz korrigiert** — `qwen35-nextn.cpp`/`qwen35moe-nextn.cpp` existieren nicht, NextN-Logik ist in `qwen3next.cpp`.
- **docs: ROADMAP.md M3-Status bereinigt** — Item #6 fälschlich in M3 aufgeführt (gehört zu M2). 5 tote Plan-Links entfernt (SESSION_PLAN_*-Dateien nie erstellt, Features direkt integriert). **#28 Adaptive MTP: ✅→⏭️** — Doku beschreibt `LLAMA_MTP_SKIP_STREAK_THRESHOLD` env var, aber Code-Implementierung fehlt (grep in src/ common/ ggml/ findet keine Referenz). M4-Status entsprechend aktualisiert. **#88 MTP+TP-Test: ☐→⏭️** — Uranus durch vorleser-Training blockiert.
- **Subagent-Fehler korrigiert:** Subagent schlug `200704 (196k)` vor — 196k = 196608, nicht 200704. Subagent schlug vor alle LOKAL.md-Commit-Stände auf `5c9381a31` zu setzen — Remote-Hosts sind real noch auf älteren Ständen (styx: bd66d726c, phobos: b758dff3c, venus: 70a727dc5), LOKAL.md spiegelt Realität wider.

### Styx Kontext 224k → 196k (Stabilität)

- **fix: Styx Kontext von 224k auf 196k reduziert** — Bei 224k vollem Kontext würde RSS 31.4 GB > 31 GB RAM → Swap → CPU-I/O → verstärkt MoE-Bottleneck → 503 Service Unavailable (beobachtet 2026-07-19 13:58, Router failover-t zu phobos). 196k: RSS ~28.4 GB, 2.6 GB Reserve, Swap-Risiko MITTEL statt HOCH. tg/pp kaum beeinflusst (CPU-MoE dominiert, SWA liest nur Window). Trade-off: 28k Kontext weniger (12.5%), aber gratis Stabilität da Kontext meist <100k belegt ist (Router-Chat, Eval). Start-Skript `start-styx-26b-server.sh` aktualisiert, Service-Description aktualisiert, Styx neu gebaut + gestartet, `n_ctx=196608` verifiziert.

## 2026-07-16

### Prompt-Cache für Styx + Mars + Router-Fix

- **feat: `start-styx-26b-server.sh`** — Schlankes Start-Skript im Uranus-Stil. Prompt-Cache: `--cache-ram 16384` (16 GB), `--cache-reuse 256`, `--slot-cache-key-similarity 0.5`, `--slot-cache-key-min-prefix 64`. systemd `MemoryMax=12G` entfernt (hatte Cache verhungert). pandora-voice-service (Whisper) läuft parallel (2.2 GB RSS).
- **feat: `start-mars-26b-server.sh`** — Gleiches für Mars/phobos (Vulkan). Konservative 6 GB `--cache-ram` (unified memory APU, 28 GB LXC-RAM). Bei 2 Slots ist Cache-Reuse besonders wertvoll.
- **fix: InferenzQuelle Router (janus:8010)** — Styx-Endpoint korrigiert: `gemma-4-12b-it-IQ4_NL.gguf` → `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`. Router neu gestartet, beide Endpoints healthy.
- **docs: Trilium-Notes aktualisiert** — Styx-Note `Gd40xnFc79JK` (26B+Cache), phobos-Note `o6jGT8Qwqm4y` (Prompt-Cache Sektion). LOKAL.md Deployment-Tabelle aktualisiert.

## 2026-07-15 (Session 2)

### Doku-Aufräumarbeit + TURBO4 Bug-Analyse

- **refactor: docs/fork Aufräumarbeit** — 30→10 aktive Dateien, 35 archiviert. Struktur: `archive/rca/` (5 Bug-Analysen), `archive/benchmarks/` (3), `archive/sessions/` (7), `archive/research/` (4+Index), `archive/plans/` (14). ROADMAP: 21 Plan-Referenzen aktualisiert. AGENTS.md: Referenzen auf archivierte Dateien aktualisiert.
- **feat: llama-bench EA-Env-Var-Support** — `llama-bench` liest `LLAMA_ARG_EA_RATIO/FUTURE/SINK/LOCAL` env vars. Neues Skript `scripts/bench-ea-phase3.sh`.
- **refactor: EA Header-Cleanup** — Ungenutztes `#include <string>` entfernt, Phasen-Kommentare aktualisiert, Covariance-Hinweis ergänzt.
- **docs: TURBO4 Server Bug Root-Cause-Analyse** — Vulkan `fa_kv_ok()` unterstützt turbo4 (return true). Problem: Scheduler legt Input-Tensoren auf CPU (ggml-backend.cpp Zeile 937-940), bei `n_seqs > 1` (Server Default: 4) entsteht Mismatch. Drei Hypothesen, Test-Plan (`-np 1`), drei Fix-Vorschläge dokumentiert.
- **docs: EA Covariance Recherche** — arXiv:2510.00636 Ablation Table 4: Für Gemma 4 (QK-Norm) bringt Covariance <0.1% Gain bei 128× mehr Rechnung. Mean-only sufficient. Phase 4 depriorisiert.
- **fix: unused variable 'd' in test_head_aggregation_max** — Compiler-Warning beseitigt.
- **skill: solo-session Regel 15 + Fallstrick 12** — "Sinnvolle Parallelarbeit beim Warten" und "Active Looping" dokumentiert. Root Cause: Agent wartete 75+ min auf Benchmark und drehte sich im Kreis.

## 2026-07-15

### #19 Expected Attention Phase 3 — Intelligent EA Scoring

- **feat: Phase 3 — Score-basiertes KV-Cache Pruning** — Ersetzt oldest-first Heuristic durch echtes Expected Attention Scoring (arXiv:2510.00636). Pipeline: (1) Q-Capture in `build_qkv()` via `ggml_set_output` (MoE-Pattern). (2) Q-Extraction nach `graph_compute` in `extract_ea_qcur()` — synchrones `ggml_backend_tensor_get` (gleicher Pattern wie `track_moe_freq`). (3) Rolling Query Buffer `[n_layers][n_seq][n_head_kv]` mit Zirkular-Puffer (default 128 queries). (4) EA Scoring: Mean μ aus Rolling Buffer → RoPE-Transform μ' = R@μ (gemittelt über n_future Positionen) → Score = exp(K@μ'/√d) per Head → Max-Aggregation über Heads → nth_element Pruning der niedrigsten Scores. (5) Value-Norm Rescaling (optional): Score *= ||V||₂. (6) NeoX + Interleaved RoPE Konventionen unterstützt (Gemma/Qwen vs Llama/Mistral). 227/227 Unit-Tests grün. Smoke-Test auf Gemma 4 12B (CPU) ohne Crash.
- **fix: P0/P1 aus review-swe Code-Review** — (P0) Q-Capture async→sync (`ggml_backend_tensor_get` statt `_async`). (P0) K-Cache-Offset vertauscht: `head_stride`/`cell_stride` korrigiert, Offset-Formel `h * head_stride + cell_idx * cell_stride`. (P0) Rolling-Buffer head_dim per-Layer validiert (SWA-Safety). (P0) TurboQuant KV types deaktiviert (WHT-Domain-Mismatch K vs Q). (P1) RoPE-Position off-by-one fix. (P1) GQA Query-Head-Index `h * n_gqa` statt `h`. (P1) `ea_clear_queries` in `clear()`/`seq_rm()` integriert. (P1) `use_vnorm` implementiert (V L2-Norm aus Cache).
- **fix: P0/P1 aus review-swe Re-Review (Round 2)** — (P0) V-norm v_trans: V-Cache ist transposed bei `!flash_attn` → vnorm nur bei `!v_trans` aktiv (verhindert falsche Byte-Zugriffe). V head_dim auf `n_embd_head_v(il)` korrigiert. (P0) `nth_element` UB bei `ratio==1.0`: `n_to_prune` auf `n_eligible - 1` geclamped. (P1) `n_gqa` division-by-zero guard. (P1) `qcur_track` nur bei `compression_ratio > 0` (vermeidet Graph-Output-Overhead bei deaktiviertem Pruning).
- **feat: TurboQuant + EA WHT-Integration** — EA Scoring funktioniert jetzt mit TurboQuant KV-Cache (turbo2/3/4). WHT-Forward-Rotation wird auf `mu'` angewendet, bevor Scores gegen WHT-rotierte K-Werte berechnet werden. Pipeline: `mu → RoPE → pad(hd_eff) → WHT-forward → score gegen K(WHT-rotiert)`. WHT ist linear, daher `WHT(mean(q)) = mean(WHT(q))`. `ea_wht_forward()` in `llama-expected-attention.cpp` repliziert `turbo_cpu_fwht` (Sign-Arrays + Butterfly). InnerQ `scale_inv` wird vor WHT angewendet. 3 neue WHT-Tests (Identity, Linearity, Changes-Vector). Smoke-Test: Gemma 4 12B Q4_K_M + turbo3 KV + `--ea-ratio 0.3` → korrekte Ausgabe "Madrid", kein Crash.
- **fix: P0/P1 aus review-swe Re-Review (Round 3)** — (P0) `cell_stride` verwendete unpadded `n_embd_k_gqa` statt `k_tensor->nb[1]` → falsche K-Cache-Reads bei TurboQuant-Padding (head_dim nicht Vielfaches von 128). (P0) Stream-Dimension im Offset fehlend → `ea_compress()` bewertete stream-0 K-Cells für jede Sequenz in Multi-Stream-Caches. Offset-Formel korrigiert auf `s*stream_stride + cell_idx*cell_stride + h*head_stride`. Gleicher Fix für V-Tensor (vnorm-Pfad). (P1) WHT-Tests gestärkt: Reference-Equality (unabhängige Sign-Array-Kopie + Butterfly) und `scale_inv`-Pfad-Test. (P2) Redundantes `i % group_size` in `ea_wht_forward` entfernt. 614/614 Tests grün.
- **feat: llama-bench EA-Env-Var-Support** — `llama-bench` liest `LLAMA_ARG_EA_RATIO/FUTURE/SINK/LOCAL` env vars für EA-Benchmarks. Hilfe-Text aktualisiert. Neues Skript `scripts/bench-ea-phase3.sh` für systematische EA disabled-vs-enabled Vergleiche bei 4k/16k/64k/128k Kontextlängen.
- **refactor: EA Header-Cleanup** — Ungenutztes `#include <string>` aus `llama-expected-attention.h` entfernt. Phasen-Kommentare von "Phase 1 (CPU Math)" auf "Phase 3 (Intelligent EA Scoring)" aktualisiert. Covariance-Hinweis ergänzt (deferred to Phase 4, Gemma 4 QK-Norm → <0.1% Gain bei 128× mehr Rechnung).
- **docs: EA Covariance Recherche** — arXiv:2510.00636 Ablation Table 4: Für Gemma 4 (QK-Normalization) bringt Covariance <0.1% Accuracy-Gewinn bei O(d²)=16K vs O(d)=128 floats pro Head. Mean-only ist sufficient und empfohlen. Phase 4 priorisiert CUDA/Vulkan Backend-Optimierung statt Covariance.

## 2026-07-15

### #19 Expected Attention Phase 2d — Post-hoc KV-Cache Pruning

- **feat: Phase 2d — Post-hoc Pruning MVP** — `ea_compress_stub()` → `ea_compress()`: echtes KV-Cache Pruning. MVP-Heuristic: oldest-first Eviction (älteste nicht-geschützte Tokens werden entfernt, n_sink initiale + n_local recent Tokens geschützt). Integration in `llama_context::decode()` nach allen Ubatches. `dynamic_cast` unterstützt `llama_kv_cache` und `llama_memory_hybrid`. Nur aktiv während Decode (n_tokens <= n_ubatch). Phase 3: EA-Scoring-Mathematik (mean/covariance) statt oldest-first.

### #19 Expected Attention Phase 2a/b/c — Review-Fixes

- **fix: P0/P1/P2 aus review-swe Code-Review** — (P0) `--ea-no-covariance`/`--ea-no-vnorm` als `handler_void` registriert (vorher `handler_string` → frassen nächstes CLI-Argument). (P1) Validierung numerischer EA-Parameter: `n_sink`/`n_local`/`n_future` >= 0 in `arg.cpp`, `ratio` <= 1.0 Clamp in `llama_context`, `rolling_buffer_size` >= 1. (P1) `ea_compress_stub()` seq_id Bounds-Check + `n_to_prune` Clamp. (P2) `llama_ea_params` Default `compression_ratio` 0.5→0.0 (konsistent mit Public API). (P2) Bool-Layout in `llama_context_params` korrigiert (bools ans Ende). (P2) Env-Var `LLAMA_ARG_EA_NO_COV`→`LLAMA_ARG_EA_NO_COVARIANCE`.

### Tier 3 Eval: GRKV + CapKV

- **eval: #41 GRKV ❌** — Ridge-Regression-basiertes KV-Merging (arXiv:2605.31105). ABLEHNEN: Zu rechenintensiv für Pascal (keine Tensor Cores), numerische Instabilität mit 3-4 bit TurboQuant KV-Cache, kein Production-Referenzcode (nur Author-Repo), marginaler Gewinn bei 2-3 Wochen Aufwand.
- **eval: #42 CapKV ⏭️** — Capacity-aware KV Eviction (arXiv:2604.25975). VERSCHIEBEN auf Phase 3+: Redundant zu Expected Attention (beide eviction-basiert). NVIDIA kvpress Referenzcode vorhanden (Apache-2.0). 1-2 Wochen Portierung. Vergleich EA vs CapKV wenn EA nicht ausreicht.

### #19 Expected Attention Phase 2a/b/c — CLI-Parameter + KV-Cache Stub

- **feat: Phase 2a — Public API + cparams Integration** — `llama_context_params` (public API in `llama.h`): 7 neue EA-Felder. `llama_cparams` (internal): neue `struct llama_ea_params`. Default-Werte in `llama_context_default_params()`. Durchreichung `common_params → llama_context_params → llama_cparams`.
- **feat: Phase 2b — CLI-Parameter** — 6 neue CLI-Args: `--ea-ratio`, `--ea-future`, `--ea-sink`, `--ea-local`, `--ea-no-covariance`, `--ea-no-vnorm`. Env-Var-Äquivalente (`LLAMA_ARG_EA_RATIO` etc.). Default: `--ea-ratio=0.0` (disabled).
- **feat: Phase 2c — KV-Cache Pruning Stub** — `llama_kv_cache::ea_compress_stub()`: Validiert EA-Parameter, zählt populated cells, berechnet `n_to_prune`. Loggt via `LLAMA_LOG_DEBUG`. Kein echtes Pruning — Phase 2d implementiert `seq_rm()` + Attention-Mask-Update.

### #19 Expected Attention Phase 1 — CPU Math Implementiert

- **feat: Expected Attention KV Cache Compression Phase 1** — `src/llama-expected-attention.{h,cpp}`: Training-free KV-Pruning (arXiv:2510.00636). Orthogonal zu TurboQuant (Pruning vs Quantisierung, kombiniert ~10x Kompression). CPU-only Math: Rolling query buffer, mean/covariance, RoPE rotation matrix, E(A) score computation, value-norm rescaling, compression decision (sink+local protection). 12 Unit-Tests mit 135 Assertions, alle grün. Phase 2 (KV-Cache Integration) ausstehend.
- **eval: #17 HOBBIT ❌** — PCIe 3.0 Mismatch (9.93x war SSD-vs-DRAM, auf RTX 4090 nur 3.2-3.9x), kein Referenzcode (Paper-only, 10-14 Wochen), Fork hat bereits MoE Cache + thecodacus Prefetch.
- **eval: #24 HybriMoE ❌** — Architektur-Mismatch (CPU-GPU-Hybrid vs Fork's GPU-Cache), 1.33x/1.70x auf RTX A6000 (48GB), auf Styx (8GB, PCIe 3.0) Regression erwartet. Referenzcode auf kTransformers (nicht llama.cpp).

### MoE Cache Heuristic — Code-Review Fixes

- **fix: 5 P1 Issues aus review-swe Code-Review** — `moe-cache.cu` Heuristic Eviction: (1) `max_freq` wird jetzt bei Inserts aktualisiert (vorher degenerierte Heuristic zu LRU während Initial-Füllung), (2) `max_freq` wird bei Eviction des heißesten Slots recomputed (verhindert stale max_freq), (3) `g.tick` wird nur einmal pro `plan()` erhöht statt pro Hit/Insert (verhindert Iterations-Reihenfolge-Bias), (4) `freq` saturierend inkrementiert (verhindert uint32 Overflow nach ~4B Hits), (5) `GGML_CUDA_MOE_CACHE_POLICY` case-insensitive + Warning bei unbekannten Werten.

### #69/#70 MoE Cache Prediction — Phase 1 Heuristic Implementiert

- **feat: MoE Cache Heuristic Eviction Policy** — Neue Eviction-Policy in `moe-cache.cu` neben LRU. Aktivierbar via `GGML_CUDA_MOE_CACHE_POLICY=heuristic`. Algorithmus: `score = 0.7*(1/(age+1)) + 0.3*(freq/max_freq)`, evict slot mit niedrigstem Score. Inspiriert von FlashMoE (arXiv:2601.17063). Phase 1 des kombinierten FlashMoE+ST-MoE Design-Dokuments. Default bleibt LRU.
- **docs: MoE Cache Prediction Design** — `docs/fork/2026-07-15_MOE_CACHE_PREDICTION_DESIGN.md`. FlashMoE + ST-MoE kombiniert. Phase 1 (Heuristic, 2-3 Tage), Phase 2 (FlashMoE FFN, 2-3 Wochen), Phase 3 (THT Prefetch, 1-2 Wochen). CCT skipped wegen 6.6% Cross-Layer-Overlap.

### Code-Review Sweep Abschluss — alle 19 Items erledigt

- **fix: R1-9** — `server-context.cpp`: `update_batch` bekommt `n_batch_capacity` Parameter und limitiert `spec_draft` bei drohendem Batch-Overflow
- **fix: R2-2** — `ggml-vulkan.cpp`: Pipeline-Cache-Speicherung komplett unter `device->mutex` (Vulkan erfordert externe Sync für Pipeline-Cache-Objekte)
- **fix: R2-3** — `test-turbo-quant.c`: 17 Tests mit Assertions (turbo2/3/4, multi-block d=256/512, NaN/Inf edge cases). In CMake registriert.
- **fix: R2-6** — `ggml-turbo-quant.c`: `assert()` → `GGML_ASSERT()` (immer aktiv, auch in Release-Builds)

### Code-Review Sweep + Expected Attention Phase 1

- **review: 3x Code-Review (review-swe, review-kimi, review-swe)** — 3 parallele Review-Subagents prüften Spec-Decoding/MTP/Server, Vulkan/TurboQuant und expert-overlap Tool. Ergebnisse: 3 P0, 15 P1, 14 P2. Alle 3 P0 und 13/19 P1+P2 in dieser Session gefixst.
  - P0: expert-overlap ffn_moe_topk tensor read (View-Stride ignoriert), Vulkan turbo_wht dispatch (group_size!=128), Legacy turbo4 uninitialisiertes rnorm
  - P1: MTP ohne Assistant-Head abweisen (neues API `llama_model_n_layer_nextn`), turbo3/2 group_size<128 guard, Vulkan throw e;→throw; (Exception-Slicing), Vulkan pipeline leak fix, expert-overlap memory-leaks/CLI-Parsing/type-Map
  - P2: nullptr check embeddings_nextn_ith, last_n_drafted entfernt, n_batch<1 guard, speculative_process slot-release, Block-Größen-Kommentare, redundante extern, n_vocab einmalig, chunk_size=n_ubatch
  - Offen: R1-9 (batch-Allokation-Check), R2-2 (Pipeline-Cache-Race), R2-3 (Test-Abdeckung), R2-6 (assert→Laufzeit-Check)
- **feat: `llama_model_n_layer_nextn()` API** — Neuer öffentlicher Getter für MTP/NextN-Layer-Anzahl. Erlaubt Server-Check ob Modell MTP-Heads hat vor `--spec-type draft-mtp`.
- **docs: #19 Expected Attention Phase 1 Design** — `docs/fork/2026-07-15_EXPECTED_ATTENTION_DESIGN.md`. Expected Attention (arXiv:2510.00636) = training-freies KV-Pruning via Query-Statistik. Orthogonal zu TurboQuant (Pruning vs Quantisierung). Kombiniert ~10x Kompression (5.1x turbo3 * 2x pruning). Phase 1 (CPU + Mean-only) in 4-6 Wochen realistisch. ROADMAP #19 Status: ☐→🔬.

### ROADMAP Research-Sweep #5 (Tier 2 Completion)

- **eval: #67 MXFP4 Quantization für gpt-oss — ✅ bereits vollständig integriert** — Tiefen-Recherche bestätigt: `GGML_TYPE_MXFP4 = 39` in ggml.h, `LLAMA_FTYPE_MOSTLY_MXFP4_MOE = 38` in llama.h. Alle Backends (CUDA, Vulkan, Metal, SYCL, OpenCL, Hexagon, WebGPU, CPU) haben Implementierungen. Vulkan-Shader vollständig (types.glsl, dequant_funcs.glsl, mul_mat_vecq.comp, mul_mm_funcs.glsl). Quantize-Tool unterstützt `mxfp4` für MoE-Tensoren. Keine weitere Arbeit nötig.
- **eval: #83 IQ*_K Quantization mit Importance Matrix — ⏭️ SPÄTER** — PR #19726 (Portierung von ik_llama.cpp IQ2_K-IQ6_K) wurde CLOSED (nicht gemerged). Politische Kontroverse um ik_llama.cpp. CPU-only im PR, kein Vulkan-Support (ik_llama.cpp sagt explizit "do not enter Vulkan issues"). Vulkan-Portierung "Very High" Aufwand ohne Referenz. 40% Qualitäts-Verbesserung über Q4_K_S bei 4.5 bpw ist signifikant, aber Vulkan ist kritisch für Mars/Venus. Fork hat bereits IQ4_NL (Vulkan-tauglich) als Zwischenlösung.
- **eval: #81 PEARL: Parallel Speculative Decoding — ⏭️ SPÄTER** — PEARL's Hauptvorteil (Parallelisierung Draft+Target) benötigt echte Multi-GPU-Parallelität. Python-Ref nutzt `accelerate` mit 2 separaten GPUs. ggml/llama.cpp Spec-Decoding-Infrastruktur ist sequenziell (draft→verify), keine Multi-Device-Parallelität. Nur Uranus (2x RTX 4060 Ti) qualifiziert — Mars/Venus/Styx single-GPU profitieren nicht. 6-10 Sessions Aufwand für production-ready C++ Implementierung. 1.50x Speedup ist moderat vs Aufwand. AWS P-EAGLE zeigt 1.05-1.69x über vanilla EAGLE-3 in vLLM.
- **eval: #66 BucketServe: Dynamic Batching — ⏭️ SPÄTER** — 3.58× Speedup nur bei heterogenen Multi-Request-Workloads (hohe parallele Last). Fork dient als Inference-Engine für InferenzQuelle (Router) = low-concurrency. Continuous Batching (`--cont-batching`) bereits vorhanden und ausreichend. Kein öffentlicher Referenzcode (Paper-only, Juli 2025). Workload-Mismatch: Bei 2 Slots (Uranus) oder 1-2 Requests (Router) negligible.
- **eval: #39 Talon: Adaptive Token Trees — ⏭️ SPÄTER** — Training-free adaptive token tree expansion, komplementär zu EAGLE-3 (8-15% Speedup gain). Aber: Fork nutzt MTP (linear) als Haupt-SD, nicht EAGLE-3 (tree-based). Talon ist tree-based → Kompatibilität mit MTP unklar. Kein Referenzcode (Paper-only). 4-6 Wochen Aufwand bei Paper-only-Implementierung ist riskant.
- **verify: Uranus E4B Service** — Service läuft (PID 1141593, Port 8080). 2 Slots à 131072 ctx, MTP draft-mtp (n_max=2), turbo4 K / turbo3 V, FlashAttention on. Health OK, Test-Request "OK" erfolgreich (108ms prompt + 34ms für 2 Tokens).

### #79: NCCL Communication Optimization — ✅ evaluiert

- **eval: #79 NCCL Communication Optimization** — Evaluation in `docs/fork/2026-07-15_NCCL_EVAL.md`. NCCL ist bereits im Fork integriert (`GGML_CUDA_NCCL=ON`) und auf Uranus installiert (NCCL 2.30.7). Benchmark auf Uranus (2x RTX 4060 Ti, PCIe, kein NVLink): **Tensor Parallelism + NCCL gibt +23-32% tg Speedup** vs Layer Split, aber -11-21% pp Regression. NCCL ist +4-8% besser für pp (große AllReduce-Tensoren), Internal AllReduce ist +3-6% besser für tg (kleine Tensoren auf PCIe). `GGML_CUDA_ALLREDUCE=internal` env var für tg-heavy Workloads. 12B-Modell crasht mit TP (`GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)` in `ggml-backend-meta.cpp:541`) — Meta-Backend Split-Limitation. E4B und 26B-A4B funktionieren mit TP.

### #87: Cross-Layer Gate Expert Prediction — ❌ evaluiert

- **eval: #87 Cross-Layer Gate Expert Prediction (Fate, arXiv:2502.12224)** — Evaluation in `docs/fork/2026-07-15_CROSS_LAYER_GATE_EVAL.md`. Fate schlägt training-freie Expert-Prediction via Cross-Layer Gating vor (>83% Cosine-Ähnlichkeit der Gate-Inputs → 97.15% Prefetch-Accuracy). Paper: MIT-Lizenz, Code auf GitHub (FFFzy/Fate_open), training-frei. **Messung auf Mars (QAT Q4_K_XL, 155 Tokens):** Expert-Selection-Overlap zwischen benachbarten Layern = **6.6%** (nahe Random-Baseline 6.25% für 128 Experten + Top-8). 0% Exact Match. Multi-Step Overlap (N vs N+2, N+3) ebenfalls ~6%. **Schlussfolgerung:** ❌ Nicht viable für Gemma-4 26B-A4B. Die Gate-Funktion (Softmax + Top-8 aus 128) ist zu nicht-linear für Cross-Layer-Prediction. Fate wurde auf Modellen mit 8-16 Experten (Top-2) evaluiert, wo die Gate-Funktion wesentlich robuster ist. Neues Tool `llama-expert-overlap` entwickelt für diese Messung.
- **feat: llama-expert-overlap Tool** — Neues Profiling-Tool in `tools/expert-profile/expert-overlap.cpp`. Misst Cross-Layer Expert-Selection-Overlap für MoE-Modelle. Verwendet `ggml_backend_tensor_get` für GPU-Tensor-Zugriff. Build: `cmake --build build --target llama-expert-overlap`.

### MTP bei verschiedenen Kontextgrößen auf Mars — 256k Root Cause

- **bench: MTP Kontextgrößen-Scan auf Mars** — Benchmark in `docs/fork/2026-07-14_MTP_DRAFT_COMPARISON.md` (Nachtrag). Setup: QAT Q4_K_XL, Q4_0 Draft, "Hallo" Prompt (17 tokens), max_tokens=64, turbo3/turbo4 KV, FlashAttention on. **Ergebnisse:** MTP bringt bei 32k **+29.3%** tg (34.4 vs 26.6 t/s), bei 64k **+34.6%** (35.8 t/s), bei 128k **+15.4%** (30.7 t/s). Bei 160k neutral (26.1 t/s). Bei 256k **Timeout (>5min)** — 300x Slowdown in prompt processing (0.14 t/s für 8 tokens vs 43.5 t/s Baseline).
- **rca: 256k MTP Timeout — shared KV cache override** — Root Cause: MTP-Draft teilt sich den KV-Cache mit dem Backbone (`ctx_other`). Beim Laden wird der Draft-KV-Cache auf die Backbone-Größe überschrieben: `W llama_kv_cache: kv_size = 4096 overridden to 262144 to match the shared source cache`. Ein n_ctx-Limit-Fix für den Draft (Commits `108a804ed`, `98339dd7f`) wurde durch das KV-Cache-Sharing **überschrieben**. Die MTP-Attention operiert über 262k Positionen im shared KV-Cache. Bei 128k funktioniert MTP (innerhalb draft training context 131072), bei 256k nicht (überschreitet draft training context → RoPE-Skalierung/Performance-Einbruch). Fix-Versuch reverted (`7a76dcfd7`). Potenzieller zukünftiger Fix: MTP-Draft-Attention auf SWA-Fenster limitieren oder shared KV-Cache-Größe für MTP auf n_ctx_train limitieren.
- **empfehlung: MTP auf Mars bei ≤128k aktivieren** — Echter Speedup +15% bis +35% tg bei 32k-128k Kontext. Bei 256k Produktiv-Kontext deaktiviert lassen bis shared-KV-Cache-Problem gelöst ist.

## 2026-07-14

### MTP Draft-Vergleich mit QAT — Korrektur der Wochenrückschau

- **bench: MTP Draft-Vergleich Q4_K_M vs Q4_0 mit QAT auf Styx + Mars** — Benchmark in `docs/fork/2026-07-14_MTP_DRAFT_COMPARISON.md`. Setup: QAT Q4_K_XL, 8k Kontext, `-ngld 999` (Draft auf GPU), 200 tokens. **Styx (CUDA, MoE-Offload):** Baseline 26.65 t/s, Q4_0 Draft 24.29 t/s (-9%, 2.67 tok/decode), Q4_K_M Draft 20.97 t/s (-21%, 2.30 tok/decode). **Mars (Vulkan, voller GPU-Offload):** Baseline 25.79 t/s, Q4_0 Draft 26.70 t/s (+3.5%, im Rauschen, 2.70 tok/decode), Q4_K_M Draft 21.83 t/s (-15%, 2.30 tok/decode). Q4_0 ist der bessere Draft auf beiden Systemen. Auf Mars (voller GPU-Offload) ist Q4_0-MTP neutral — der Draft-Overhead ist praktisch null. Auf Styx (MoE-Offload) überwiegt der Draft-Overhead. **Korrektur:** Der 08.07. Wert 31.9 t/s (+50% MTP-Boost) gilt nur für IQ4_NL Config B, nicht QAT. "+150%" in Wochenrückschau war Config-A-vs-B-Vergleich (irreführend), korrigiert auf +50%. Bei Produktiv-Kontext (224k/256k) ist MTP ohnehin unmöglich (Draft-OOM auf GPU, CPU-Draft: -44%).

### Wochenrückschau 7.–14. Juli 2026

- **docs: Wochenrückschau — messbare Optimierungsergebnisse** — Vollständige Auswertung der Forschungswoche in `docs/fork/2026-07-14_WEEKLY_REVIEW.md`. Quellen: git-log (161 Commits), CHANGELOG, ROADMAP, SNAPSHOT, Solo-Session-Reports, Trilium-TTT (Tagesnotizen 07.–14.07.). **Live-Verifikation 14.07. 16:25 UTC:** Mars/phobos 25.74 t/s tg (TTT: 25.85, -0.4%), Styx 25.70 t/s tg (TTT: 26.02, -1.2%) — beide im Rauschen. **Netto-Wins der Woche:** Styx tg +102% über die Woche (12.7→25.7 t/s, IQ4_NL-ohne-Pinning → QAT-mit-Pinning+Prefetch); MTP +50% nur für IQ4_NL (21.3→31.9 t/s), mit QAT -9% bis -21%; Styx +64k Kontext (160k→224k) via QAT; Mars tg +16.6%, pp +10%, +76k Kontext (180k→256k) via QAT; E4B FA Crash-Fix (Uranus: Crash→103-112 t/s); xtts-api 3.20x Speedup (331s→108.3s). **Verworfen mit Messung:** #57 (-99.6% pp), #45 (-10.7% tg), #32 (-0.5%), #77 (±0.5%), #61 (-8.5% tg), MTP Q4_0 (-2.4%/-14% mit IQ4_NL, -9% mit QAT), #40 Phase 2 (-7% tg). **ROADMAP:** 22✅→25✅, 14❌→16❌, 7⏭️→10⏭️, 53☐→46☐. **Styx Crash-Loop behoben:** Zwei festgefahrene `llama-cli`-Test-Prozesse blockierten VRAM → Produktiv-Service OOM → gekillt + Neustart.

### #85: Vulkan Push Descriptors implementiert

- **feat: #85 Vulkan Push Descriptors (VK_KHR_push_descriptor)** — Deskriptoren werden direkt in den Command Buffer geschrieben statt Deskriptor-Sets zu allokieren und zu binden. Eliminiert `vkAllocateDescriptorSets`, `vkUpdateDescriptorSets`, `vkCmdBindDescriptorSets` pro Dispatch. Implementierung: Extension-Check in Device-Init (`GGML_VK_DISABLE_PUSH_DESCRIPTOR` zum Deaktivieren), `dsl_push` Layout mit `PUSH_DESCRIPTOR_BIT`, Pipeline-Layouts verwenden `dsl_push` wenn supported, `pushDescriptorSetKHR` in dispatch, Descriptor-Set-Allokation übersprungen. Mars bestätigt `push_desc: 1`. Benchmark 1B Q4_K_M: pp512 ±0.1%, tg128 ±0.2%, pp4096 ±0.3% — **kein messbarer Speedup** (RADV descriptor set allocation ist bereits sehr effizient mit pre-allocated pools).

### #78: Vulkan Pipeline Cache Disk Persistence — Code-Review Fixes

- **fix: #78 P0+P1 Code-Review Fixes** — (P0.1) `pipeline_cache` wird jetzt immer im Destructor zerstört, nicht nur wenn `pipeline_cache_path` gesetzt — sonst Resource-Leak wenn `GGML_VK_CACHE_DIR` unset (Vulkan verlangt destroy aller child objects vor `vkDestroyDevice`). (P0.2) Function-local statics (`unordered_set` + `mutex`) durch `device->mutex` + `bool pipeline_cache_saved` Flag ersetzt — static destruction order fiasco: statics wurden vor `vk_instance` zerstört → UAF in `~vk_device_struct()`. (P1.3) `fclose` return value prüfen, Windows `rename`-Fix (remove target first). (P1.5) Leerer `GGML_VK_CACHE_DIR` string abgefangen. Verifikation Mars: kein Crash, kein Leak, keine UAF.

### #77/#84: ROADMAP Items evaluiert und geschlossen

- **docs: #77 ❌ K-Quant MMVQ Path Fix** — Issue #21151 (10-15x Langsamkeit für Q4_K/Q5_K auf RDNA3) auf Mars nicht reproduzierbar. Benchmark: pp64 MMVQ 1168 t/s vs non-MMVQ 1174 t/s (±0.5%), tg1 MMVQ 67.84 t/s vs non-MMVQ 65.91 t/s (MMVQ 3% schneller). Vermutlich auf älteren RADV-Versionen behoben.
- **docs: #84 ❌ Wave32/Wave64 Subgroup Size Tuning** — PR #12087 merged (2025-03-17). Fork hat AMD_RDNA3 Architektur-Erkennung aber keine RDNA3-Pipeline-Konfiguration — fällt auf Wave64-Default zurück. Wave32 bricht Coopmat-Shader auf RDNA3. Vulkan Wave64 ist 20-22% schneller als ROCm Wave32 (Issue #20934). Aktuelle Konfiguration optimal.
- **docs: #80/#82 ⏭️ SPÄTER** — GEAR (KV-Cache Quant+LowRank+Sparse): TurboQuant (3-5x) ist bereits stärker als GEAR (2.29x), 6-8 Wochen Portierung. Fiddler (CPU-GPU MoE Orchestration): Lizenz unklar, AVX512_BF16 fehlt im Release, Mars (UMA) profitiert nicht, 4-6 Wochen Portierung.

### #78: Vulkan Pipeline Cache Disk Persistence implementiert

- **feat: #78 Vulkan Pipeline Cache Disk Persistence** — `GGML_VK_CACHE_DIR` env var steuert Cache-Verzeichnis. Pro-Device Cache-Dateien (`vk_pipeline_cache_{idx}.bin`) mit `pipelineCacheUUID`-Validierung (verwirft stale Cache bei Driver/GPU-Wechsel). Atomares Schreiben via temp+rename. Cache-Saving in `ggml_backend_vk_free()` mit once-per-device guard (`unordered_set` + mutex), da der `~vk_device_struct()` Destructor bei `exit()` nicht zuverlässig aufgerufen wird. Quelle: Perinban/llama.cpp commit 1b7250c. Benchmark Mars (llama-3.2-1b Q4_K_M, pp64): Kalt 1.161s → Warm 0.781s = **33% Startup-Speedup** (-380ms). Bei 26B MoE dominiert Modell-Laden (~11min), Cache-Effekt marginal.

### #62: MoE Expert Profiling & REAP Pruning implementiert

- **feat: #62 MoE Expert Profiling & REAP Pruning tools** — Portiert von PR #20454 (srossitto79). `tools/expert-profile/` (C++ Profiler) sammelt REAP-Saliency-Scores via ggml eval callback (ffn_moe_topk/weights/down). `tools/moe-pruning/` (Python GGUF-Pruner + Analyse-Tools). REAP Score = mean(gate_weight * ||expert_output||_2) pro Experte (arXiv:2510.13999, Cerebras Research). Auf Styx verifiziert: 30 MoE layers des 26B A4B QAT erfolgreich profiliert. Komplementär zu #40 MoE-Freq-Tracking.
- **fix: #62 remove debug logging from expert-profile callback** — Debug-Output aus wants()/on_tensor() entfernt.
- **fix: #62 diff-artefakte `++ b/` in `tools/moe-pruning/*` entfernt** — Die Python-Skripte, README und requirements.txt enthielten `diff`-Header als erste Zeile und waren nicht ausführbar; `python3 -m py_compile tools/moe-pruning/*.py` läuft jetzt sauber durch.
- **fix: #62 expert-profile: strided `ffn_moe_topk`-Kopierung korrigiert und restliche Debug-Ausgabe entfernt** — `ffn_moe_topk` ist ein `ggml_view_4d` des `ggml_argsort`-Outputs mit `nb[1] = n_expert*4`, nicht zusammenhängend. Der vorherige `memcpy` von `ggml_backend_tensor_get` war für `n_tokens > 1` fehlerhaft und hat falsche Expert-IDs geliefert. Jetzt wird `ggml_backend_tensor_get_2d` verwendet. Außerdem wurde die verbliebene `[debug-cb]` Ausgabe in `expert_eval_callback` entfernt.
- **fix: #62 gguf_prune.py: fehlende `ffn_gate_exps` und `ffn_gate_up_exps` Suffixe ergänzt** — Der Pruner erkannte Experten-Gate-Gewichte (separat oder fused) nicht und ließ sie ungeschnitten, was bei gemma4-ähnlichen MoE-Architekturen zu unvollständigem Pruning und Dateigrößen führen würde.
- **fix: #62 P1: intercept final weight/down variants for REAP correctness** — Gemma 4 nutzt `norm_w=true` → `ffn_moe_weights_norm` ist das korrekte Gate-Weight (nicht rohes `ffn_moe_weights`). Auch `ffn_moe_down_scaled` bei Per-Expert-Scales. Fix: intercept alle Varianten (`weights_norm`, `weights_softmax`, `down_scaled`), last-writer-wins, defer REAP-Komputation bis Batch komplett (next topk oder flush bei save). Verifikation Styx: v1 (buggy) never=0 für alle Layer, v2 (fixed) never=3-28 — korrekt für 128 Experten, 8 selected, 503 tokens.
- **docs: #59 Tensor Prefetching als nicht-viable markiert** — PR #21067 ist DRAFT mit dirty merge state, CUDA-only, erfordert --no-mmap, weniger effektiv für MoE. thecodacus Expert-Prefetch deckt MoE-Prefetch bereits ab.

### #61: Persistent VRAM Expert Cache implementiert

- **feat: #61 Phase 1 — ggml API + CPU mul_mat_id / scheduler integration** — Portiert von PR #24524 (leloch) Commit 1/3. Backend-agnostische Function-Table (`ggml_moe_cache`, zero-initialized) als Brücke zwischen ggml-cpu und CUDA-Backend. Thread 0 in mul_mat_id partitioniert hits/misses, dispatcht hit-rows als batched GPU launch, andere Threads berechnen miss-rows. SwiGLU glu_hits mask überspringt fused cache rows. Scheduler redirect_offer/finalize hooks für GPU-resident dst handoff. Ohne CUDA-Backend: alle Hooks null-check no-ops.
- **feat: #61 Phase 2 — CUDA MoE expert cache implementation** — Portiert von PR #24524 Commit 2/3. Vollständige CUDA-Implementierung (1771 Zeilen): Per-(expert_size,type) slot pools mit LRU-Eviction, async pinned-staging insert workers, idle-time prefetch backfill, hot-set persistence across runs. Paired gate+up pools mit fused gate+up+SwiGLU batched matvec. GPU-resident dst handoff für down projections. Decode-only fill, stable shape census, role-group budgeting. Baseline-sampled bail-out judge. CUDA errors degradieren zu CPU path. Aktiviert via GGML_CUDA_MOE_CACHE=1.
- **feat: #61 Phase 3 — CLI option + fit placement + LFRU cache removal** — Portiert von PR #24524 Commit 3/3 + Fork-Cleanup. `--moe-cache N` CLI flag (0=off, N=VRAM budget MiB, absent=auto). MoE-cache-aware fit placement bei heavily-spilling Modellen. LFRU Expert Cache dead code entfernt (expert_valid, tensor_copied, GGML_EXPERT_CACHE — 0% hit-rate). thecodacus Prefetch bleibt unangetastet.
- **bench: #61 Styx Benchmark** — GTX 1070 Pascal 8GB, Gemma-4 26B-A4B QAT, -ngl 999 --n-cpu-moe 20. Baseline: pp512=385 t/s, tg128=28.3 t/s. Mit Cache (512MB budget, 256MB reserve): 37.8% hit-rate, aber bail-out judge deaktivierte Cache korrekt (589us vs 512us pure-CPU per node — Pascal GPU zu langsam für kleine Expert-Matvecs). Cache designed für Ampere+ GPUs. Safety rails funktionieren wie designed: kein Crash, keine falschen Ergebnisse, korrekte Selbst-Deaktivierung.

## 2026-07-12

### LXC-Migration: llama-server bare-metal → LXC 240 (phobos)

- **feat: llama-server in LXC 240 (phobos) migriert** — Bare-metal systemd-Service auf Mars gestoppt und disabled. llama-server läuft jetzt im LXC 240 (phobos) mit systemd user-Service (loginctl enable-linger). LXC-Konfig: RAM 28GB, swap 8GB, rootfs 16GB, /jade/models read-only bind-mount. Build 7f5097b43 (9234) mit UI-Asset-Workaround (HF UI-Bucket unvollständig, loading.html fehlt). Port 18080. Performance identisch zu bare-metal: pp=41.8 t/s, tg=28.8 t/s (224k Kontext, solo).
- **fix: 188k-Performance-Klippe Root Cause geklärt** — Die scheinbare "188k-Klippe" war **kein Vulkan-Backend-Bug**, sondern ein **OOM-Artefakt bei konkurrierenden GPU-Prozessen**. Zwei llama-server gleichzeitig → GTT-Overflow (2×16 GB > 26 GB GTT) → OOM-Kill → GPU-Buffer-Eviction → 0.10 t/s. Solo-Betrieb mit 224k Kontext: 28-32 t/s. Die ursprüngliche Hypothese ("Code-Pfad-Wechsel im Vulkan-Backend") war falsch. Widerspruch zwischen Trilium `SWumEN7WOXBI` §5.8 ("KEIN VRAM-Bandbreiten-Problem") und `zYeLUsss9udM` ("VRAM-Thrashing") aufgelöst: Beides richtig aus verschiedenen Blickwinkeln — es ist GTT-Eviction, nicht Bandbreite. **Regel:** Niemals zwei llama-server gleichzeitig auf derselben GPU. 180k-Grenze obsolet. f16-Fallback-Workaround war nie nötig. Siehe `docs/fork/2026-06-20_VULKAN_LARGE_CONTEXT_PERF_CLIFF.md` RCA Update.
- **docs: Trilium-Doku umfassend aktualisiert** — 6 Notes aktualisiert: `SWumEN7WOXBI` §5.8 (RCA Update), `o6jGT8Qwqm4y` (LXC 240 phobos komplett überarbeitet), `6pWFJK57dEDu` (QAT-Standard Migration ergänzt), `IqF1CiABuNV9` (Mars LXC-Migration), `zYeLUsss9udM` (Widerspruch aufgelöst), `TMdG98nlAuwo` (als historisch markiert). Lokale Doku: `docs/fork/2026-06-20_VULKAN_LARGE_CONTEXT_PERF_CLIFF.md` (RCA Update), `AGENTS.md` (188k-Referenz korrigiert, 180k als obsolet markiert).
- **feat: Mars-Kontext 224k → 256k (262144) erhöht** — Das "256k OOM-killed" aus der alten Doku war mit IQ4_NL (14.7G) und/oder konkurrierenden Servern. Mit QAT (14.2G) und Solo-LXC-Betrieb funktioniert 256k problemlos. Produktiv-Service umgestellt: `Environment=CTX=262144` und `Environment=PARALLEL=2` im systemd-Service-File → 2 Slots à 128k (vorher 112k). Performance identisch: pp=41.8, tg=28.6. Das Modell-Maximum (262144/256K) ist jetzt voll ausgenutzt. Trilium: `o6jGT8Qwqm4y`, `6pWFJK57dEDu`, `IqF1CiABuNV9`, `SWumEN7WOXBI` aktualisiert.

## 2026-07-13

### Infra: phobos DNS-Registrierung + SSH-Config + mars DNS-Korrektur

- **fix: phobos bei pan (dnsmasq) registriert** — LXC 240 (phobos) war nicht im zentralen DNS-Server pan (LXC 249, saturn) eingetragen. A-Record + PTR für `phobos.eulenhorst.lan` → 192.168.1.240 hinzugefügt via `register-device.sh`. MAC: `BC:24:11:16:BA:17`. Vorher war phobos nur über hardcoded IP oder mars' FritzBox-DNS (`phobos.fritz.box`) auflösbar.
- **fix: mars /etc/resolv.conf korrigiert** — mars nutzte veraltet `search fritz.box` + FritzBox (192.168.1.1) als einzigen DNS-Server. Korrigiert auf `search eulenhorst.lan` + pan (192.168.1.249) primär + FritzBox Fallback — konsistent mit jupiter/saturn. Backup: `/etc/resolv.conf.bak.20260713`.
- **feat: SSH-config-Eintrag für phobos auf hydra** — `~/.ssh/config`: `Host phobos → HostName phobos.eulenhorst.lan`. Passwortloser Login mit id_ed25519 (Key war bereits in phobos' authorized_keys). `ssh phobos` funktioniert jetzt direkt.
- **docs: LOKAL.md, SNAPSHOT.md, Trilium aktualisiert** — LOKAL.md: phobos als eigener Host-Eintrag + DNS-Hinweis. SNAPSHOT.md: Neubau-Befehl vereinfacht (`ssh phobos` statt `ssh mars + lxc-attach`). Trilium: Netzwerkplan (`1IYS31C9CrSu`) Mars-Tabelle mit MAC ergänzt, phobos-Note (`o6jGT8Qwqm4y`) DNS-Eintrag dokumentiert, Mars-Note (`IqF1CiABuNV9`) DNS-Korrektur eingetragen.

### Solo-Session (Phase 3: Expert Prefetch + Buffer-Usage Investigation)

- **fix: thecodacus Expert Prefetch — MUL_MAT_ID Graph-Suche repariert** — Root Cause gefunden warum der selektive Kopierpfad für Expert-Weights nie aktiviert wurde: Der Code prüfte nur `split->graph.nodes[0]` auf `GGML_OP_MUL_MAT_ID`, aber bei Gemma-4 ist die erste Node im Split `GGML_OP_SCALE` (32), nicht `MUL_MAT_ID` (30). Fix: Suche im gesamten Split-Graph nach dem MUL_MAT_ID-Node der `input_cpy` als `src[0]` hat. Expert-Weights haben korrekt `WEIGHTS` Buffer-Usage (1) — der frühere Debug-Output mit `usage=2` (COMPUTE) bezog sich auf einen anderen Buffer (Compute-Staging-Buffer, nicht Weight-Buffer). Commits `aea4b0d7c`, `5f7a4ff0e`.
- **bench: thecodacus Prefetch 2-Slot Sweet-Spot — reiner Win auf Styx UND Mars** — Slot-Count-Analyse auf Styx (GTX 1070, 26B QAT, -ncmoe 20) und Mars (RDNA3, 26B Q4_K_M, -ncmoe 10): `GGML_SCHED_PREFETCH_SLOTS` env var hinzugefügt. **Styx**: 1-Slot +9.8% tg aber -11.9% pp (kein Overlap), **2-Slot: +28.9% pp512, +36.2% pp2048, +36.7% pp8192, +2.1% tg8** (Sweet-Spot), 3-Slot identisch zu 2-Slot. **Mars**: 3-Slot -35% tg128 (Queue-Konflikt auf RDNA3), **2-Slot: +8.8% pp512, +3.8% tg128** (reiner Win!), 1-Slot -14.4% tg. Beide Server-Scripts auf `GGML_SCHED_PREFETCH_SLOTS=2` umgestellt.
- **feat: Vulkan UMA Cached Host Memory (PR #23762) — +7% tg auf Mars** — Portiert von PR #23762. Bevorzugt `HostCached` statt write-combining memory auf UMA-Systemen. Fix: `memory_property_flags` wird jetzt auf die tatsächlich allokierten Flags gesetzt (vorher: angeforderte Flags). Non-coherent memory flush/invalidate handling hinzugefügt. Benchmark Mars (26B Q4_K_M, mit 2-Slot Prefetch): **tg128 16.73→17.90 (+7%)**, pp512 184.26→185.14 (+0.5%). Gesamt vs baseline: **tg128 +11%, pp512 +9.3%**. Decode profitiert massiv von cached reads.
- **❌: GGML_CUDA_DISABLE_MMQ_STREAM_K env var (PR #22170 stream-k disable)** — Deaktiviert MMQ stream-k decomposition via env var. **Benchmark auf Uranus (2x RTX 4060 Ti): E4B QAT pp512 -64%, 26B IQ4_XS tensor split pp512 2669→17.48 (-99.3%!), layer split pp512 3005→11.59 (-99.6%!).** Stream-k decomposition ist ESSENZIELL für MMQ-Performance auf Ada GPUs. PR #22170 war für spezifischen Edge Case (fixup-buffer Race Condition bei src1_ncols != ne11) gedacht. Env var bleibt im Code (Default: OFF = stream-k ON) für Debugging.
- **fix: UI-Assets loading.html Workaround** — `loading.html` fehlt im aktuellen HF llama-ui Release. Workaround in `scripts/ui-assets.cmake`: Wenn `loading.html` fehlt, wird sie aus `index.html` erstellt. Behebt llama-server Build-Fehler auf Mars und Styx.
- **❌: #37 Expert Upload Skipping — Postponed** — Tensor-Level Upload Skipping implementiert (`GGML_EXPERT_CACHE=1`), aber Cache-Hit-Rate = 0% weil Expert-Weights den selektiven Kopierpfad nicht durchlaufen (sie werden im generischen Kopierpfad kopiert, der keine Expert-Level-Granularität hat). Expert-Level-Cache im selektiven Kopierpfad implementiert aber crasht (Buffer-Recycling-Konflikt). Root Cause: `input_cpy` Buffer wird zwischen Forward-Passes recycelt, "valid" Markierungen werden ungültig. Postponed bis Buffer-Pinning oder Per-Expert Tensor-Splitting. Cache-Infrastruktur (`expert_valid`, `tensor_copied`, Hit/Miss-Stats) bleibt im Code, aktivierbar via `GGML_EXPERT_CACHE=1`.
- **bench: op_offload mit GGML_OP_OFFLOAD_MIN_BATCH=1 — -37% auf Pascal** — Mit Threshold=1 wird MUL_MAT_ID auf GPU offloaded (Cross-Backend-Copy der Expert-Weights). Auf Pascal (GTX 1070, PCIe 3.0) ist der Copy-Overhead massiv: tg8 26.4→16.6 t/s (-37%). CPU-Berechnung ist schneller als GPU + Copy. Default Threshold=32 ist optimal für Pascal — MUL_MAT_ID bleibt auf CPU, kein Cross-Backend-Copy nötig.

### Solo-Session (Phase 2: UBBoost + MoE Load Balancing)

- **feat: #34 UBBoost — implementiert, +20-41% PP, +19% TG auf Pascal** — Separate `n_ubatch_prefill` für Prefill-Phase. CLI flag `-ubp`/`--ubatch-prefill` (default 0 = use n_ubatch). Graph reserviert für `max(n_ubatch, n_ubatch_prefill)`. Decode wählt dynamisch `n_ubatch_prefill` (prefill) oder `n_ubatch` (decode) basierend auf Token-Count. SWA-Cache-Sizing-Fix: `max(n_ubatch, n_ubatch_prefill)` statt `n_ubatch` für ISWA/hybrid-ISWA Konstruktoren (verhindert OOM bei größeren Prefill-Batches). Pascal TILE-Fix: fp32 Config für 512/512 ncols=2 `nbatch_fa` 32→64 (static_assert auf Pascal). Benchmark Styx (GTX 1070, turbo3/4 KV): E4B ub=256/ubp=512: **+20% pp2048, +41% pp8192, +19% tg128** vs ub=512 baseline. 26B ub=256/ubp=512: **+18% pp2048, +10% pp8192, +4.5% tg128**. Optimal: ub=256, ubp=512. ubp>512 OOM auf 8GB. Commits `6eb8f9f5f`, `50685e399`, `0ab002883`.
- **feat: #40 MoE Expert Frequency Tracking — Phase 1 implementiert** — C API: `llama_set_moe_freq_track()` / `llama_get_moe_freq()` / `llama_model_n_expert()`. `build_moe_ffn` erzeugt `ggml_cont` copy der `selected_experts` (markiert als output, via `ggml_build_forward_expand` zum Graph hinzugefügt) — verhindert Scheduler-Buffer-Reuse über Layer hinweg. `track_moe_freq` liest copy-Tensoren nach Graph-Compute, akkumuliert Expert-Counts in `moe_freq[n_layer][n_expert]`. `llama-bench`: `LLAMA_MOE_FREQ_TRACK=1` env var aktiviert Tracking + gibt per-layer top-3/bottom-3 Report auf stderr. Validiert auf Styx (26B QAT, 30 Layer × 128 Experts, pp2048+tg128): korrekte Counts (pp: 98304/Layer = 24×4096, tg: 5128/Layer = 644×8). Heisse Experten identifizierbar (z.B. Layer 2: #83=12.4%, Layer 28: #107=10.3%). Viele Experten nie selektiert (0-Count) — Grundlage für datengetriebene Expert-Platzierung (Phase 2). Commits `1a087ddae`–`f47223589`.
- **fix: #40 Code-Review P1+P2** — (P1) GroveMoE double-counting: `build_moe_ffn` wird pro Layer zweimal aufgerufen (main + chunk experts) → `moe_freq_tracked_layers` Set verhindert doppeltes Tracking. (P1) `allow_reuse` ignorierte `moe_freq_track` → Graph-Reuse ohne Copy-Tensoren bei Mid-Stream-Aktivierung → `moe_freq_track` in `allow_reuse`-Check aufgenommen. (P2) `moe_freq_flat` als `mutable` deklariert (statt `const_cast`), Dead Code entfernt (`moe_topk_copies`, `get_moe_freq()`, No-Op-Kommentar). Commit `240e82a1b`.
- **feat+❌: #40 Phase 2 — Frequency-Guided Layer Offloading (NEGATIVERGEBNIS)** — `LLAMA_MOE_FREQ_EXPORT=<file>` exportiert Frequency-Daten als JSON. `LLAMA_MOE_FREQ_IMPORT=<file>` importiert JSON und nutzt Entropy-basierte Layer-Auswahl. Zwei Strategien getestet: (1) Pure-Entropy: kälteste N Layer auf CPU → tg128 -7% Regression. (2) Swap: hot CPU-Layer ↔ cold GPU-Layer → tg128 -6.8% Regression. Root Cause: kälteste Layer (höchste Entropy 0.844-0.858) sind späte Layer (23-27), die auf GPU bleiben müssen für Generation-Performance. Default "erste N Layer auf CPU" ist optimal. Per-Expert-Platzierung (statt per-Layer) wäre nötig für echten Benefit, erfordert Tensor-Splitting — tiefer GGML-Eingriff, verschoben. Frequency-Tracking-Infrastruktur + JSON Export/Import bleiben nützlich für Analyse. Commits `aa19aaac7`, `6c383f5a2`.

### Solo-Session (Phase 1: E4B+MTP Crash Fix)

- **fix: E4B+MTP FA Crash (head_dim=512) — gelöst** — Root Cause: E4B full-attention Layer haben `head_dim=512`, MMA-Kernel hat keine Template-Instanz für DKQ=512 (`fattn.cu:110` abort). TILE-Kernel hatte auch Lücken: kein Fallback für DV>256 ohne Mask, keine Config für 512/512 bei ncols=2. Drei Commits: (1) `fattn.cu`: Route head_dim=512 zu TILE-Kernel, (2) `fattn-tile.cuh`: Fallback für DV>256 mit gqa_ratio-basiertem ncols2, (3) `fattn-tile.cuh`: Neue TILE-Config für 512/512 bei ncols=2 in allen 4 Config-Funktionen. Verifikation auf Uranus (RTX 4060 Ti 16GB): E4B+MTP mit turbo4/turbo3 KV → 103 t/s, f16 → 112 t/s. Keine Regressionen. Code-Review: ship-ready, keine Issues. Commits `f9e7564bd`, `bd8ef5978`, `1a0af56dc`.
- **feat: #35 Row-Packing DMMV für RDNA3 — implementiert, +1% (erwartet +10-20%)** — Row-packing war bereits teilweise implementiert (NUM_ROWS=2 default, =4 auf GCN). Erhöhung auf RDNA3 (rm_stdq=1→2, rm_kq=2→4) brachte nur +1% auf E2B Q4_K auf Mars (pp128: 493→498, tg64: 40.0→40.4). DMMV ist nicht der Bottleneck für kleine MoE-Modelle. Keine Regression, Change beibehalten. Commit `acd8dfe3a`.
- **docs: Phase 3 Batch-Evaluation Tier 2 (8 Items)** — 4 parallele Subagents evaluierten #8, #15, #16, #34, #36, #37, #38, #40. Ergebnisse: #16 ✅ bereits integriert (shmem-staging, Commit 66e999ecc), #38 ⏭️ verschoben (Conf-KV konflikt mit TurboQuant), 6 Items ☐ offen implementierbar. ROADMAP-Status für alle 8 Items aktualisiert mit Eval-Ergebnissen.
- **❌ #45 CUDA Concurrent Streams QKV — bereits im Fork, Regression mit CUDA Graphs** — Feature vollständig implementiert (upstream PR #16991, `GGML_CUDA_GRAPH_OPT=1` env var). Benchmarks: (1) Uranus (RTX 4060 Ti 16GB, E4B Q4_K_XL, voll auf GPU): Mit CUDA Graphs tg2048 71.50→63.82 t/s (-10.7%, ±18.77 Varianz — Graph-Capture ineffizient durch interleaved Node-Order). Ohne CUDA Graphs tg512 72.24→72.99 (+1%, im Rauschen — QKV-Projektionen zu klein bei E4B). (2) Styx (GTX 1070 8GB, 26B QAT, MoE-Offload `--n-cpu-moe 20`): tg512 15.25±0.69 vs 14.25±0.12 — kein Effekt (im Rauschen ±6%), weil CPU-MoE-Offload Split-Buffers erzeugt → CUDA Graphs deaktiviert → `graph_optimize` returnt früh. Feature aktiviert gar nicht. Fazit: Nur nutzbar bei single-GPU + voller GPU-Offload + CUDA-Graphs-kompatibel — und dann Regression. ROADMAP #45 auf ❌ gesetzt.

## 2026-07-11

### Solo-Session (05:00–17:00)

- **feat: #3 Pascal CUDA MMVQ (PR #25479) — manuell portiert** — `MMVQ_PARAMETERS_PASCAL_DP4A` zum Enum hinzugefügt, Pascal CC 6.1/6.2 Detection, 2 Warps statt 4 für single-token decode (bandwidth-bound auf kleinen SMs). +14 -3 Zeilen in `mmvq.cu`. Code-Review: ship-ready, keine Issues. Build grün auf Hydra (CUDA) und Styx (CUDA CC 6.1). Commit `5c1884929`. **Benchmark auf Styx (GTX 1070, E2B MoE): Generation +8.9% (65.1 → 70.9 t/s), Prompt +0.7% (908.9 → 915.3 t/s).** Besser als die im PR versprochenen +3-6%.
- **❌ #13 Two-Tier Expert Cache (FR #20757) — nicht implementieren** — Alle 4 PRs (#21609, #21614, #23170, #24524) closed ohne Merge. thecodacus Memory Pinning im Fork deckt Tier 2 (pinned RAM) bereits ab. Geringer ROI für Pascal (compute-bound nicht PCIe-bound). RFC #24528 offen aber nicht umgesetzt.
- **❌ #10 UMA Zero-Copy (PR #22462) — nicht implementieren** — Verwandte UMA-PRs (#22455, #22930, #23770) wurden bereits getestet und revertiert. RCA-Masterplan zeigt: System-RAM ist langsamer als GTT für GPU-Compute auf Mars. PR #22462 ist open/unstable, verfolgt einen ähnlichen Ansatz.
- **✅ #6 MUL_MAT_ID Subgroup (PR #15524) — bereits integriert** — PR ist upstream gemerged (Aug 2025). Fork hat bereits `MatMulIdType::SUBGROUP`, `mul_mm_id_funcs.glsl` mit `subgroupBallot`, `matmul_id_subgroup_*` Shader werden generiert, Pipelines aktiv (`subgroup_ballot && subgroup_require_full_support && subgroup_min_size_16`). Vorheriger Revert war auf fehlerhaften Merge zurückzuführen, aktueller Stand ist sauber.
- **✅ #7 Vulkan FA Refactor (PR #19625) — bereits integriert** — PR ist upstream gemerged (Feb 2026). Fork hat Commit `66e999ecc`: `get_fa_tuning_params_scalar()`, `row_split`, `shmem_staging`, Q caching in registers, vendor-specific Br selection. 28 Referenzen im Code.
- **✅ #11 EAGLE-3 (PR #18039) — bereits integriert** — Commit `57774253c`, `LLM_ARCH_EAGLE3` in llama-arch.cpp, eagle3-Modell-Support, `--spec-type eagle3` verfügbar.
- **✅ #12 Coopmat2 (PR #19075) — bereits integriert** — coopmat2 Support in ggml-vulkan.cpp, `flash_attn_cm2.comp`, `mul_mm_cm2.comp`, `dequant_funcs_cm2.glsl`, SPV-Shader auf Mars generiert.
- **✅ #20 Tensor Parallelism (PR #19378) — bereits integriert** — Commit `d850df3f5`, backend-agnostic tensor parallelism, AllReduce in `ggml-cuda/allreduce.cu`, Meta-Device in `llama.cpp`.
- **✅ #28 Adaptive MTP (PR #22931) — eigene Implementierung** — PR wurde closed (Draft), wäre auch nicht kompatibel (upstream draft-mtp vs Fork gemma4-assistant). Fork hat bereits `LLAMA_MTP_SKIP_STREAK_THRESHOLD` env var (1-32) für adaptive MTP-Skipping.
- **⏭️ #14 LFU Caching — verschoben** — Recherche (5 Paper, vLLM Referenz): LFU allein nicht lohnenswert (3-4 Wo für 15-20% hit rate). Besser: LFRU (vLLM PR #37190) + thecodacus Pinning. SpecMD "Least-Stale" (arXiv:2602.03921) 85× besser als LRU. Für Styx (8GB) lohnenswert, für Mars (<5%) nicht. SESSION_PLAN mit Recherche erstellt.
- **feat: #2 Tensor Split Regex static (PR #24710)** — 29 std::regex Patterns von `const` auf `static const` in `src/llama-model.cpp`. Verhindert pro-Token Regex-Recompilation im Tensor-Split-Modus. Build grün auf Hydra (CUDA) und Mars (Vulkan). Commit `bc1eb0207`
- **❌ #1 MTP Logits Copy (PR #23198) — skipped** — 19 Merge-Konflikte in 10 Kern-Dateien (speculative.cpp, llama-context.cpp, qwen35.cpp, etc.). Fork hat `nextn`-Support, PR fügt `pre_norm`-Support hinzu — strukturelle Konflikte. MTP ist OFF in Produktion. Zu hoher Aufwand für geringen Nutzen.
- **❌ #6 MUL_MAT_ID Subgroup (PR #15524) — revertiert** — 23 Merge-Konflikte in 3 Vulkan-Dateien (18 in ggml-vulkan.cpp). Subagent löste Konflikte, aber Shader-Generierung wurde beschädigt (undefined references für `matmul_id_*` Shader). Erfordert manuelle Portierung mit tiefem Vulkan-Shader-Verständnis. Für spätere Session.
- **✅ #5 GTT Size Tuning — bereits konfiguriert** — Mars hat bereits `amdgpu.gttsize=26624` (26GB) in `/proc/cmdline`. GTT ist 26GB, VRAM ist 1GB (APU). Kein weiteres Tuning nötig.
- **⏳ #4 n-gram Decoding — verfügbar, kein Speedup auf E2B** — `--spec-type ngram-mod` auf Mars verfügbar und aktiviert (Log bestätigt `common_speculative_impl_ngram_mod`). Benchmark mit E2B-Modell (2.9GB) und repetitivem Counting-Prompt (256 tokens): **Baseline 39.2 t/s vs n-gram-mod 39.1 t/s** — kein messbarer Speedup. Erwartet für kleine Modelle mit nicht-perfekt-repetitiven Patterns. n-gram ist verfügbar für User die es nutzen wollen, aber kein Default-Speedup.
- **⏭️ #7 Vulkan FA Refactor (PR #19625) — verschoben** — Abhängig von #6 (MUL_MAT_ID), das revertiert wurde. Shader-Basis muss zuerst stabilisiert werden.
- **❌ #9 Vulkan Shmem-Staging (PR #20897) — PR closed** — PR wurde geschlossen ohne Merge (AI-generiert). Manuelle Portierung nötig, Risiko unklar.
- **docs: ROADMAP.md aktualisiert** — Status für #1, #2, #4, #5, #6, #7, #9 aktualisiert. M1 teilweise, M2 blockiert.

### Vorherige Änderungen

- **docs: ROADMAP-Workflow etabliert** — `docs/fork/ROADMAP.md` mit 30 Optimierungs-Ansätzen (M1-M6), Solo-Pläne für M1+M2 in `docs/fork/plans/`. Workflow in AGENTS.md dokumentiert.
- **docs: Commit-Format formalisiert** — `<type>: <Was> — <Warum>` (feat, docs, fix, security, refactor, bench)
- **docs: Optimierungs-Recherche (Web + arXiv, 30 Ansätze)** — 4 parallele Subagents, 50+ Quellen, 30 Ansätze kategorisiert nach Tier. Siehe `docs/fork/2026-07-11_OPTIMIZATION_RESEARCH.md`
- **fix: AGENTS.md Kontext-Tabelle korrigiert** — Styx "229k (lädt)" → "224k (lädt)" (229376 = 224×1024, nicht 229k)
- **feat: fork-speed-research Skill** — Projekt-spezifischer Skill in `.devin/skills/fork-speed-research/` für monatliche Recherche mit 4 parallelen Subagents

## 2026-07-10

- **feat: QAT Produktiv-Standard** — Services auf Mars und Styx von IQ4_NL auf QAT-UD-Q4_K_XL umgestellt. Kontextfenster von 180k/160k auf 224k erweitert. Mars 25.85 t/s, Styx 26.02 t/s. Commit `517ec94d1`
- **feat: Adapter für neue GGUF-Metadata-Keys** — QAT + MTP Q4_0 Draft Modelle laden korrekt. Commit `f9a3dfc62`
- **docs: Kontextfenster korrigiert** — Modell-Maximum ist 256K (262144), nicht 320k. Commit `0b62e3e1f`

## 2026-07-09

- **docs: Vulkan KV-Cache Benchmark** — turbo3/4 ist optimale Vulkan-Konfig (K=turbo3, V=turbo4). +31% schneller als turbo4/4 bei pp@96k-128k. Siehe `docs/fork/2026-07-09_VULKAN_KV_CACHE_BENCHMARK.md`

## 2026-07-08

- **feat: thecodacus MoE-Optimierungen** — Memory Pinning + Async Expert Prefetch. +72-106% pp, +30% tg auf GTX 1070. Siehe `docs/fork/2026-07-08_SOLO_SESSION_REPORT.md`
- **docs: FINALE EMPFEHLUNG aktualisiert** — QAT als Standard, Kontextfenster-Tests, MTP Q4_0 Ergebnisse
