# SESSION_PLAN: Strata/OptLlama-Eval → Fork-Optimierung (Qwen3.8-Flash-Next, Uranus)

**Komprimiert 2026-10-10.** Vollstaendige historische Fassung (RCA-Prosa, alle
Messreihen): `docs/fork/archive/plans/SESSION_PLAN_20261006_strata-optllama-uranus-eval_FULL.md`

## Ziel

Fork (`fukuro-llama-cpp-turboquant`) so schnell wie realistisch moeglich fuer
Qwen3.8-Flash-Next/Qwen4Exp auf Uranus machen — Strata als Referenz-Latte,
OptLlama als zweite llama.cpp-Basis. Prefill UND Generation, kein reines
Benchmarking.

## Setup

- **Uranus:** 2x RTX 4060 Ti 16 GB (Ada SM 8.9), PCIe 4.0 x8/GPU (~6.2 GB/s
  effektiv pro Link), 128 GB RAM, RAID5-Modellstorage.
- **Modell:** ~125B MoE, 512 Experten/Layer, 48 Layer, top-10, GDN+QSA+HC,
  eingebettetes MTP. Quants: UD-IQ4_XS (Referenz), IQ3_XXS.
- **Repos:** Fork-Worktree `/home/fukuro/git/flashnext-s1-lookahead`
  (Branch `feat/flashnext-s1-lookahead`, aktiv `fix/event-ordering`),
  Strata `/home/fukuro/Strata`, Daten `/home/fukuro/eval-flashnext/results/`.

## Ergebnis-Matrix (§9 FULL, 2026-10-07, 134 Runs, Prompt ~23k)

| Engine | pp t/s | tg t/s | Gap |
|---|---|---|---|
| Strata UD-IQ4_XS | 1227 | ~45 | 4.6x pp / 4x tg |
| Strata IQ3_XXS | 1728 | ~65 | — |
| OptLlama IQ3_XXS | 366 | ~15 | — |
| Fork UD-IQ4_XS (damals best) | 267 | ~11.5 | 1.0 |

Gate >=2x erfuellt → Port gerechtfertigt. Bericht: `docs/qwen38-eval/ENGINE_EVAL_20261007.md`

## Fork-Baselines B0–B5 (§10.4, `fork-p1fix.jsonl`, 3 Reps, pp @0.8k/3k/5.8k/23k)

| Config | pp t/s | tg |
|---|---|---|
| B0 cpu-moe ub4096 | 106/239/242/245 | ~11 |
| B1 +prefetch | 77/251/246/283 | — |
| B2 +prefetch +ubp8192 | 75/249/244/345 | — |
| B3 ncmoe44 hybrid | 84/270/264/302 | ~11.6 |
| B4 +REGISTER_HOST | 72/262/256/296 | — |
| B5 IQ3_XXS | 90/320/311/321 | ~10.4 |

## Abgeschlossene Fixes (verifiziert)

- **P1a** Q2_0/TURBO2_0-Typkollision → IQ3_XXS ladbar (2026-10-07).
- **P1b** Prefetch-Crash: MMQ-Overread + 512B-Padding (`3f3d272f69`, `819af39886`).
- **Event-Ordering** `89af4780`: Events immer allokieren + Wait auf schreibendem
  Stream → 0 Drains, **522.0 t/s** pp8192 (Plateau).
- **MoE-Cache-Floor** `8d549a7b`: min_expert 1 MB → 256 KB (IQ4_XS-Slices
  ~690-900 KB waren komplett exkludiert) → ~90 % Hit, **tg512 7.7→11.37 t/s**.
- **MTP-Korrektheit** (2026-10-10, `fix/event-ordering`, gepusht):
  `eff47330` PLE-Token-major-Layout + rollback-sichere (pos,tok)-History;
  `7e858065` rs_rollback-Plumbing (is_empty, seq_rm-Bounds, conv_state_at-Slots,
  n_rs_seq); `ba27fff8` QSA `no_fa` (all-masked FA-Tiles → NaN).
  Greedy-Output **byte-identisch** zu Target-only, Acceptance ~47-50 %.

## Gemessene Plateaus und gescheiterte Ansaetze (nicht wiederholen)

- **Prefill-Plateau 522 t/s** = Scheduler-/Dependency-Grenze, nicht Bandbreite.
- **Async-Staging x3 Varianten** (dedizierter Stream, Ping-Pong, 64-MB-Chunks):
  konsistent ~-5.5 %, revertiert. Host-Sync wirkt als Flow-Control.
- **Zero-Copy Demand-Read** (`GGML_SCHED_MOE_ZCOPY`): 250 t/s, latency-bound.
- **S1b Expert-Range-Streaming**: best 433 t/s = -15 % vs Baseline (H2D contendiert
  mit MMQ-Kernel). RANGES=1 crasht.
- **S2 Layer-Residenz** (`-ncmoe 46`): 2 Layer resident → 514.9 t/s Decke;
  VRAM limitiert auf ~2.6 GB/GPU frei → ~2 Layer max. TurboQuant-KV schafft
  keinen Headroom (Workspace frisst Ersparnis).
- **REGISTER_HOST=1** pinnt 89 GiB korrekt, nur ~+2-5 % — Bandbreite ist nicht
  der Engpass.

## Kernbefund (Phase A/B, nsys 2026-10-09)

- Fork bewegt **~63 GB/Pass** H2D bei 23k Tokens; Strata **~23 GB**
  (~2.7x weniger Bytes/Token). Strata-Mechanismus: **persistenter,
  feingranularer Expert-Cache**, der bei JEDEM MoE-Zugriff konsultiert wird —
  Prefill-Chunks wiederverwenden residente Experten (~63 % Hit ueber den Pass).
- Fork-Prefill laedt den **kompletten Expert-Tensor pro Ubatch** neu —
  Residenz wird nie abgefragt. Der MoE-Cache (90 % Decode-Hit) ist strukturell
  auf `n_tokens==1` beschraenkt: `max_batch=1` Default
  (`GGML_CUDA_MOE_CACHE_MAX_BATCH`, max 8), alle Fast-Paths `n_tokens==1`-gegated.
- Strata-Pipelining: ~10k kleine async Memcpys, 4.6 GB/s auf EINEM Link (73 %
  Duty). Fork: ~3.9 GB/s ueber beide Links (~32 %). Byte-Reduktion bleibt
  der groessere Hebel.

## MTP-Stand (2026-10-10)

- Korrekt und committet. Draft (4.1 GB Q8_0) laeuft komplett auf CUDA1,
  ~15 ms/Runde — **Draft-Residenz ist geloest** (Stratas ~700-MB-Vorteil
  entspricht hier normaler GPU-Residenz, kein Cache-Feature).
- **Perf:** MTP n_max=4 ≈ **11.2-11.7 t/s** vs Target-only **14.5-16.6 t/s**
  auf gleichem Build → aktuell Nettoverlust. Verify (5 Tokens) ~280 ms vs
  ~62 ms Decode — Multi-Token-Batches streamen Experten **uncached**.
- `GGML_CUDA_MOE_CACHE_MAX_BATCH=8`: Verify nutzt Cache → 11.2/13.6 t/s,
  noch unter Target-only (nicht-fused >1-Pfad zu schwach).

## Aktive Phase S3: Persistente Expert-Residenz fuer Prefill/Verify

**Hypothese:** Ubatch-uebergreifende Expert-Residenz + Miss-only-Upload
senkt H2D von ~63 GB Richtung Stratas ~23 GB → direkter pp- UND Verify-Gewinn.

### Stub-Befund: `upload_ids = used_ids` (`ggml-backend.cpp` ~1960, "#37")

Der Scheduler hat einen selektiven Upload-Pfad, der pro Ubatch nur die
**gerouteten** Experten hochlaedt (per-Expert-Ranges aus `used_ids`, Routing-
Readback). Der geplante Filter "skip already-resident" wurde **nie
implementiert**: `upload_ids = used_ids` bleibt ungefiltert, eine Validitaets-
Spur fuer `input_cpy`/Slot-Puffer existiert nicht. Konsequenz: derselbe
Expert wandert in jedem Chunk erneut ueber PCIe — exakt die ~63 GB/Pass.

**Status:** durch `96541837be` (Gather-API) teil-adressiert: Residenz wird
jetzt ueber den MoE-Cache-Pool abgefragt (D2D-Hits statt H2D-Reupload).
Der urspruenglich gedachte input_cpy-Validity-Filter (Wiederverwendung
innerhalb derselben Buffer-Generation) bleibt ein offener, separater Hebel.

### Strata vs. Fork vs. Soll-Zustand (Mechanismus)

- **Strata:** jeder MoE-Zugriff konsultiert den Expert-Cache. Prefill-Chunk k
  fuellt, Chunks k+1..N treffen (~63 % Hit ueber 23k-Pass → 23 GB statt 63).
- **Fork (vor Gather):** Decode n=1 → Cache (90 % Hit). Prefill/Verify →
  Voll-Tensor- bzw. ungefilterter Range-Upload, Residenz nie abgefragt.
- **Fork (mit `96541837be`, opt-in `GGML_SCHED_MOE_CACHE_GATHER=1`):**
  Gather-API assembliert den dst-Tensor pro Expert — Hit → D2D aus Pool-Slab,
  Miss → H2D + Admission in leere Slots (nie evicten; Eviction bleibt
  `plan()`/Decode vorbehalten, `valid` erst nach Sync auf dem Insert-Stream).
- **Soll:** Gather nach fairem A/B default-on, wenn pp-Gewinn reproduzierbar;
  Admission ggf. auf Hot-Prior beschraenken falls LRU-Churn dominiert.

### Modell-Abdeckung: MoE-generisch, nicht Qwen-spezifisch

Der Mechanismus greift ueberall dort, wo `MUL_MAT_ID` auf host-residenten
Expert-Tensoren laeuft (`moe_weights`-Pfad im Scheduler). Betroffen:
Qwen3/3.5/4-MoE-Familie, DeepSeek, Mixtral, gpt-oss, **Gemma 4**
(`src/models/gemma4.cpp` nutzt `build_moe_ffn` + `n_expert`-Tensoren) —
sowie jede kuenftige MoE-Arch. Dense-Modelle (gemma.cpp, qwen35.cpp dense)
nutzen den Pfad gar nicht — fuer sie aendert sich nichts. Der Cache ist
tensor-namen-/geometrie-basiert, keine Arch-Sonderlogik noetig.

**Gather-Implementierung (Stand 2026-10-10, `96541837be`, gepusht):**
- API `ggml_moe_cache.gather(name, src, dst, n_expert, expert_bytes, type,
  used_ids, dst_backend)`; Hits=D2D, Misses=H2D, Misses optional insert-bar.
- Scheduler: Routing-Readback vor den Prefetch-Branch gehoist, Gather in
  Voll-Upload + selektivem Pfad + Lookahead-Staging verdrahtet, Fallback
  auf Baseline-Upload bei Reject/OOM.
- Env: `GGML_SCHED_MOE_CACHE_GATHER=1` (Scheduler-Flag, NICHT
  `GGML_CUDA_MOE_CACHE_GATHER` — ein falsch benannter Env-Test lief ins Leere).

### Messstand S3 (Uranus, llama-server, Prompt 10560 tok, 2026-10-10)

| Arm | Env | pp t/s |
|---|---|---|
| PF-Envs (PF=1 ALL=1 SLOTS=8) + Cache | ncmoe=46 | 51.2/51.6/51.7 |
| gleiche Envs, Cache AUS | ncmoe=46 | 51.3 → **Cache unschuldig** |
| gleiche Envs + Gather | ncmoe=99, Budget 4096 | 64.9 → 94.5/94.8 |
| **keine PF-Envs, Cache an, Gather AUS** | ncmoe=46, Budget 3072 | 110.5/125.4 |
| **keine PF-Envs, Cache an, Gather AN** | ncmoe=46, Budget 3072 | 560.6/619.6/619.2 |

**Ergebnis: Gather liefert ~5x pp (125 → 620 t/s) und schlaegt das alte
522-Plateau.** Aufwaermtrend ueber Reps (560→620) = Pool-Fuellung durch
Admission-on-Miss; Saettigung ~619 bei 3072 MB Budget.

**Konfundierer aufgeklaert:** `GGML_SCHED_PREFETCH_ALL=1` ist im
llama-server-Regime giftig — laedt ALLE 512 Experten je Layer statt nur
die gerouteten (~10/512); ~51 t/s unabhaengig vom Cache. Die 522-Bench-
Zahl kam von llama-bench mit anderem Ubatch-Regime — NICHT auf Server-
Pfade uebertragen. Regel: PF-Envs nur im llama-bench-Kontext bewerten.

**Naechste Gates:** (a) Hits/Misses + H2D-Bytes je Phase instrumentieren;
(b) Verify-Pfad (`MAX_BATCH>=5`) unter Gather erneut messen; (c) Budget-
Sweep (groesserer Pool → mehr Hits?); (d) Gather default-on erwae gen,
sobald (a)-(c) stabil.

**Gates:** (a) Instrumentierung Hits/Misses/Bytes je Phase; (b) Verify-Hits
via MAX_BATCH>=5 schon messbar (getan: marginal); (c) Prefill-Gather erst bei
reproduzierbarem pp-Gewinn default-faehig, sonst opt-in; (d) Backfill/
Admission-Contention waehrend Prefill messen und ggf. waehrend pp drosseln.

## Offen

- S3 Implementierung (oben). n_max-Sweep unter neuem Verify-Pfad.
- Strata `--spec 4` ≙ Fork `n_max 3`-Semantik bei Vergleichen beachten.
- `dry`-Sampler verfaelscht Trajektorien — Correctness nur mit `top_k=1`.
- Kernel-Nichtdeterminismus: Trajektorien-Vergleich allein ist kein
  Correctness-Beweis (Bitgleichheit via ple_embd-Probe war der Nachweis).
- QSA-FA-Pfad bleibt ausgeschaltet bis Masken-Kompaktierung portiert ist.
