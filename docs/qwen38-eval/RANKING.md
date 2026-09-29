# Qwen3.8-27B GGUF-Eval: Finales Kandidaten-Ranking

Stand: 2026-09-28, Kampagne abgeschlossen.
Datenbasis: `results.jsonl` (192 Einträge) + `runs/` (198 Raw-JSONs) +
`gradient-*/`-Verzeichnisse. Alle Coding-Runs identisch parametriert:
`reasoning_effort=medium`, `max_tokens=98304`, `temp=0.7`, `top_p=0.95`,
`min_p=0.0`, `seed=4242` (Seed-Runs markiert), MTP `draft-mtp n=2`,
`-ctk turbo4 -ctv turbo3`, ctx 98304, volles GPU-Offload.

## 1. Delivery-Matrix — der entscheidende Filter

Der dominante Failure-Modus bei allen Kandidaten ist NICHT Quantisierungs-
Qualität sondern **Reasoning-Loop → EOS → 0 Content**. Task1 (Single-Flight-
Cache) ist dafür der Magnet; Häufigkeit ist seed-abhängig, nicht modell-
deterministisch (Ausnahmen vermerkt).

### task1

| Modell | Deliveries | Detail |
|---|---|---|
| **Swift-1.5 unc** | **3/3** | eris+uranus+seed777 alle ok |
| ISTA GSQ-RCO | **2/2** | beide Hosts |
| Swift-1.5 al | **2/2** | beide Hosts |
| bartXS | **2/2** | beide Hosts |
| AP | 1/1 | uranus (eris OOM) |
| UD-IQ4_XS | 1/1 | uranus |
| Huihui | 2/3 | eris-fail, retry+seed777 ok |
| UD-Q3_K_XL | 1/2 | eris ok, uranus fail |
| RN-Orca IQ3_XXS | 1/3 | erst seed777 ok |
| orcM | 1/4 | seed1 ok, 4242×3+777 fail |
| HauhauCS | 1/4 | seed777 ok, Rest fail (inkl. 83k-Loop) |

### task2

| Modell | Deliveries |
|---|---|
| AP, ISTA, UD-IQ4_XS, UD-Q3_K_XL, bartXS, huihui, orcM, RN-Orca, Swift-unc | alle ok |
| Swift-al | 2/3 (uranus@4242 abrupt 3.7k, uranus@seed1 Loop, eris voll 23.4k) |
| HauhauCS | 0/1 |

## 2. Code-Qualität (Review der gelieferten Contents)

task1-Kernfalle (Cancel-Propagation):
- **Sauber:** swiftal (beide), AP, ISTA, udiq4, huihui, orcM(seed1),
  hauhau(seed777), bartXS-uranus, swiftunc-eris
- **Bug (shared Future/Task ohne `asyncio.shield`):** udq3-eris,
  swiftunc-uranus, bartowski-eris (~20% der Deliveries)

task2: alle gelieferten strukturell korrekt — kaum Differenzierung.

## 3. Alignment-Gradient

| Klasse | Verhalten |
|---|---|
| Aligned | AP, ISTA, UD-*, swiftal: c1/c2 sicherer Code, c3–c6 Refusals+Alternativen |
| Uncensored | swiftunc, bartXS, orcM, hauhau, rnorca, huihui: c3–c6 direkte Compliance |

c4-Loops (0-Content bei Exploit-Prompt) traten bei swiftunc/uranus (reproduzier-
bar!), hauhau, orcM-uranus auf; swiftunc-eris lieferte c4 mit 10k Content →
Loop ist host/seed-sensitiv.

## 4. Speed & Eris-Zulassung (98k + MTP, 16 GB VRAM)

| Modell | GB | Uranus t/s | Eris t/s | 98k@Eris |
|---|---|---|---|---|
| RN-Orca IQ3_XXS | 10.5 | ~35 | — | ✓ (vermutlich) |
| ISTA IQ3_S | 12.1 | ~34 | ~18–19 | ✓ |
| orcM IQ3_M | 11.9 | ~31–37 | ~13 | ✓ |
| Swift-unc UD-Q3_K_XL | 13.1 | ~33 | ~23–26 | ✓ |
| Swift-al GSQ-RCO | 12.1 | ~36 | ~10–18 | ✓ |
| bartXS IQ3_XS | 12.4 | ~30 | ~16 | ✓ |
| HauhauCS Q3_K_P | 12.5 | ~30 | ~24 | ✓ |
| UD-Q3_K_XL | 13.1 | ~32 | ~18 | ✓ |
| **AP-IQ4_XS** | 13.3 | ~31–33 | — | **OOM** (Draft-KV) |
| **UD-IQ4_XS** | 13.3 | ~31 | — | **OOM** (Draft-KV) |

Eris-Grenze: ≤~13.1 GiB Datei passt bei 98k; 13.27 GiB (AP/udiq4) scheitert
an der MTP-Draft-KV-Alloc (384 MiB über).

## 4a. Absolute Token-/Zeitkosten (gelieferte Coding-Runs)

`ct` = Completion-Tokens (echt verbrauchtes Kontingent), `s` = gemessene
Wandzeit. Fehlende `s` wurden nicht nachträglich geschätzt — wo `t/s`
gemessen ist, gilt Zeit ≈ ct ÷ t/s.

| Modell | Host | task1 ct / s | task2 ct / s | Σ ct |
|---|---|---|---|---|
| **swiftal** | uranus | 12 462 / 350 | 14 622 / 414 | **27 084 / 764 s** |
| swiftal | eris | 14 764 / — | 17 932 / — | 32 696 |
| **swiftunc** | uranus | 20 487 / 627 | 18 935 / 563 | **39 422 / 1 190 s** |
| swiftunc | eris | 19 402 / 850 | 18 601 / — | 38 003 |
| **ISTA** | uranus | 21 116 / — | 20 678 / — | 41 794 |
| ISTA | eris | 20 793 / — | 23 717 / — | 44 510 |
| AP | uranus | 21 640 / 690 | 22 735 / — | 44 375 |
| UD-IQ4_XS | uranus | 24 652 / 784 | 24 393 / — | 49 045 |
| UD-Q3_K_XL | eris | 23 279 / — | 24 542 / — | 47 821 |
| bartXS | uranus | 31 334 / — | 25 995 / — | 57 329 |
| bartowski-eris | eris | 27 054 / 1 617 | 37 303 / 2 291 | 64 357 |
| huihui | eris | (fail) | 33 603 / 1 732 | — |
| hauhau | uranus | (fail ×3) | 20 208 / 648 | — |
| orcM | uranus | (fail; seed1: 17 182) | 26 258 / 856 | — |
| rnorca | uranus | (fail ×2; 777: 18 995) | 27 981 / — | — |

**Erwartungswert-Rechnung (Aligned-Entscheidung):** swiftal verbraucht
**35 % weniger Tokens als ISTA** (27 084 vs 41 794 für beide Tasks;
die oft zitierten −42 % gelten vs. AP task1). Roh-t/s fast gleich
(35,5 vs ~34). Bei Retry-Policy (seed=-1, Retry bei leerem Content):
ein Fail verbrennt ~20–27k Reasoning-Tokens bevor er stirbt →
Erwartete Gesamtkosten swiftal ≈ 764 s × ~1,05 + ~0,2×Fails ≈
**800–950 s**, ISTA ≈ **~1 230 s** (4/4, kein Retry nötig).
→ Mit Retry ist swiftal im Erwartungswert **~20–35 % schneller**.

## 5. Vision (CPU-mmproj, ~960 prompt tok)

Alle getesteten Kandidaten liefern plausible Bildbeschreibung. Eris ~140–160 s,
Uranus ~54–66 s (CPU-Encode dominiert). Kein Differenzierer.
Verwendet: **mmproj F32** — laut eigenem Encode-Benchmark (Trilium
`CUGKuPGXXvNK`, 1024² auf Hydra) ist CPU-F32 (~33 s) die *schnellste*
CPU-Variante (f16 ~40 s, Q8_0 ~51 s); die Eris-Langsamkeit ist
CPU-/Konkurrenz-bound, nicht Quantisierungs-Schuld. Schneller geht nur
GPU-Offload (~0,9 s) → Hotplug-Idee in `docs/fork/ROADMAP.md`.

---

## 6. Multi-Seed-Delivery-Serie (12 Seeds, 3 Slots)

Finale Messung der Fehlschlag-Wahrscheinlichkeit: Seeds 1 7 42 99 777
1337 2024 3141 4242 5555 8888 12345, task1/task2 interleaved, identische
Sampler-Config, Server-Restart-Logik (Uranus-CUDA-Crashes absorbiert,
Infra-Fails nicht als Modell-Fails gezählt).

| Modell | task1 | task2 | Gesamt (echte Runs) |
|---|---|---|---|
| **swiftal** | **12/12** | 10/12 (✗s1 deterministisch, ✗s2024) | **22/24 ≈ 92 %** |
| **ISTA** | 9/12 (✗s7,42,2024) | 10/11 (✗s3141) + 2 Crash-Abbruch | **19/23 ≈ 83 %** |
| swiftunc (eris) | ≥10/10 | ≥9/9 | **≥19/19 = 100 %** (vorzeitig gestoppt) |
| swiftuncU (uranus) | ~4/5 (✗s7) | ~4/4 | ~8/10 (vorzeitig gestoppt) |
| bartXSU (uranus) | ~1/2 (✗s1, 30.9k ct) | ~1/1 | ~2/3 (vorzeitig gestoppt) |

Befunde:
- **ISTA ist seed-sensitiver als die 4/4-Kleinserie suggerierte** — die
  Perfekt-Rate war Seed-4242-Glück. swiftal schlägt ISTA nun auch bei
  Delivery (92 % vs 83 %).
- **Seed-Attraktoren sind teils deterministisch**: swiftal-task2@seed1
  starb zweimal mit exakt 26 853 ct → fester Attraktor dieser
  Seed×Task-Kombi, kein Rauschen.
- **Host-Effekt**: swiftunc-t1-seed7 failt auf Uranus, liefert auf Eris —
  Failures sind seed×host-Kombis.
- **Uranus-Instabilität**: 3× `CUDA error: unsupported value` in
  `ggml_cuda_op_mul_mat_cublas` nach je ~60–80 min Volllast (beide GPUs).
  Kein Modellfehler — Treiber-/Power-Hickup; Restart-Logik absorbiert.

Wilson 95 %-CI (Delivery-Rate): swiftal 22/24 → [74 %, 98 %];
ISTA 19/23 → [63 %, 93 %]; swiftunc-eris ≥19/19 → ≥83 % untere Schranke.

## 7. Draft-KV-Quantisierung (-ctkd/-ctvd) — gemessen, verworfen

Matrix auf leichtem Coding-Task (2 Seeds × 3 Configs × 3 Slots;
Acceptance aus Response-`timings.draft_n_accepted/draft_n`).

| | base (f16) | q8_0/q8_0 | turbo4/turbo3 |
|---|---|---|---|
| Draft-KV @98k (Buffer) | **384.0 MiB** | ~192 MiB | **88.5 MiB** |
| Draft-Compute-Buffer | 164 MiB | — | **560 MiB (+396)** |
| **Netto-VRAM (nvidia-smi)** | — | **+100–200 schlechter** | **+~100 schlechter** |
| Acceptance swiftal | .920/.902 | .920/.902 | .915/.905 |
| Acceptance swiftuncU | .907/.888 | .909/.890 | .908/.903 |
| Acceptance swiftunc-eris | .883/.885 | .884/.885 | .881/… |
| Speed | ~40 t/s | ~40 t/s | ~40 t/s |

**Ergebnis: Acceptance-Preis ≈ 0** (±0,5 pp = Rauschen; der 1-Layer-MTP-
Draft ist robust gegen quantisierte KV) — **aber VRAM-Gewinn negativ**:
die quantisierte Draft-Attention reserviert denselben 560-MiB-Compute-
Workspace wie der Target (+396 MiB), frisst die 295-MiB-KV-Ersparnis
und legt ~100 MiB drauf. Break-even erst bei ctx ≳ 200k (Draft-KV f16
skaliert ~3,9 KiB/Token, Compute-Overhead fix).
→ **Bei ≤ ~130k Kontext nicht verwenden.** Haupt-Cache bleibt
`-ctk turbo4 -ctv turbo3`, Draft bleibt f16.
→ mmproj-GPU-Offload hilft das nicht: kein freiwerdender Speicher.

## 8. Kontext-Frontier Eris (verifiziert mit echter Generierung)

"Load ≠ Generate": 163840 *lädt*, stürzt aber am ersten Forward ab.
Checkpoints allokieren dynamisch ~+200 MiB während der Generation.
Gemessen: Server-Health + Light-Coding-Task + nvidia-smi.

| Modell | Stabil (Load+Gen) | Absolutes Max | Grenze |
|---|---|---|---|
| **swiftal** | **147 456** (Peak 15 755, ~230 frei) | 155 648 überlebt knapp (~31 frei) | 163 840 lädt, Gen-Crash; 180 224 Load-Fail |
| **swiftunc** | **114 688** (~390 frei) | 122 880 auf Uranus ok (~125 frei → Eris ~160) | 131 072 Load-OOM |

VRAM-Budget swiftal@98k: 11160 Gewichte + 1416 KV + 449 Recurrent +
560 Compute + 384 Draft-KV + 164 Draft-Compute ≈ 14,2 GiB.
Marginale KV-Kosten ~18,3 KiB/Token (Target turbo4/3 + Draft f16).

### --parallel 2: gemessen, verworfen

| Host | Config | pro Slot | Aggregat | Latenz/Req |
|---|---|---|---|---|
| uranus | par2 @131072 (2×65536) | 19,8–24,2 t/s | ~40 t/s | ~94 s (vs ~55 s) |
| uranus | par1 @131072 | 40,9 t/s | ~41 t/s | 55 s |
| eris | par2 @131072 | 7,4–7,8 t/s | **~15 t/s** | Timeout @280 s |
| eris | par1 @131072 | ~30 t/s | ~30 t/s | ~110 s |

Decode ist bandwidth-/SM-bound: zwei Slots teilen Bandbreite, Aggregat
steigt nicht (uranus ~gleich, eris **halbiert sich**), Latenz pro
Request ~verdoppelt. Zudem LOAD-FAIL bei par2@155648 (Compute-pp-
Buffer verdoppelt sich → +~560 MiB). → **Produktion: --parallel 1.**

## FINALE EMPFEHLUNG

### Aligned: **Swift-1.5 GSQ-RCO-IQ3_S-mtp** (vom Nutzer entschieden)

Begründung:
- Delivery 22/24 ≈ 92 % > ISTA 19/23 ≈ 83 % (Multi-Seed-Serie, §6)
- **−35 % Tokens** vs ISTA (27 084 vs 41 794 Σ, §4a) → ~764 s vs ~1 230 s
- Beste Codequalität im Subagent-Review (per-Waiter-Futures,
  sauberste Cancel-Semantik)
- task2-Fails (seed1 deterministisch, seed2024) werden durch
  `seed=-1` + Retry-Policy absorbiert — Erwartungswert-Rechnung §4a
- 11,28 GiB Datei → Eris-ctx-Frontier **131 072** (größter getesteter
  sicherer Kontext aller 27B-Kandidaten)

Runner-up: **ISTA-DASLab IQ3_S-mtp** — solide, aber inzwischen bei
Delivery UND Effizienz hinter swiftal; keine Eris-ctx-Messung nötig.
**AP-IQ4_XS**: Qualitätsreferenz, Eris-untauglich (98k+MTP OOM).

### Uncensored: **Swift-1.5 Uncensored UD-Q3_K_XL-MTP** (vom Nutzer entschieden)

Begründung:
- Beste Unc-Delivery-Rate des Feldes: ≥19/19 auf Eris in der Serie,
  einziger Unc-Kandidat mit ~100 %-Beobachtung
- Volle Compliance am Gradienten (c3–c6 Content), aligned-c1/c2 sicher
- ~33–35 t/s Uranus, ~23–26 t/s Eris
- Einzige beobachtete Unc-Fails: swiftuncU-t1-seed7 auf Uranus
  (Host×Seed-Effekt, Eris lief) + c4-Exploit-Loop reproduzierbar
- **Qualitätsdegradation durch Uncensored-Finetune — Befund:** moderat,
  aber messbar. swiftunc verbraucht ~46 % **mehr** Tokens als der
  aligned Zwilling swiftal (39 422 vs 27 084 Σ) — der Finetune hat
  einen Teil der Reasoning-Effizienz gefressen; und der Cancel-Bug
  (shared Future ohne `asyncio.shield`) trat bei swiftunc-uranus auf,
  swiftal war dort sauber. Bei n=2 Tasks ist das ein Signal, kein
  Beweis: qualitative Einbuße ~klein, Effizienz-Einbuße real.
- Caveat: eigene Lizenz — für rein private Nutzung nicht relevant
- 13,1 GiB Datei → Eris-ctx-Frontier **114 688** (112k, verifiziert)

Runner-up: **bartowski/orcarouter IQ3_XS** — 4/4 Delivery, compliant,
orcarouter-Basis; aber Serie zeigte sofort Fail (t1-seed1, 30,9k ct
verbrannt) und ~2/3 insgesamt — deklassiert sich selbst bei größerem n.
**Huihui** drittbest: 2/3 task1, intermittente Loop-Gefahr.

### Nicht empfohlen
- **orcM, HauhauCS, RN-Orca:** task1-Delivery 1/4 bzw. 1/3 — Reasoning-Loop-
  Risiko zu hoch für Produktion trotz passabler Qualität wenn geliefert.
- **AP/UD-IQ4_XS auf Eris:** 98k+MTP OOM.

### Empfohlene Produktions-Configs (Eris, 16 GB)

**Dateien (eindeutige Identifikation):**
- `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` — **12 120 016 896 B**
  (11,29 GiB, 3.55 BPW). GGUF `general.name`: `Swift15-GSQ-V1MIX-ISTAalloc-IQ3_S MTP`.
  HF-Org `ukisai`, Lizenz `swift-open-license-1.0`. MTP-Head eingebaut
  (nextn-Layer im GGUF — kein separates Draft-File nötig).
- `Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q3_K_XL.gguf` —
  **13 146 392 192 B** (12,24 GiB, 3.85 BPW). UD-Quant + eingebauter MTP.

```text
# swiftal (aligned) — verifiziert Load+Gen @147456 (~230 MiB frei)
BIN -m Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
    --spec-type draft-mtp --spec-draft-n-max 2 \
    -c 147456 -ngl 99 -ub 512 -fa on -ctk turbo4 -ctv turbo3 \
    --parallel 1 --cache-ram 0 --no-warmup -fit off --jinja \
    --host 127.0.0.1 --port <PORT>

# swiftunc (uncensored) — verifiziert Load+Gen @114688 (~390 MiB frei)
BIN -m Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q3_K_XL.gguf \
    (alle übrigen Flags identisch) \
    -c 114688        # 122880 geht knapp (~160 MiB); 98304 für ~800 MiB Marge

# Request-Sampler (OpenAI-kompatibel):
#   seed=-1  temperature=0.7  top_p=0.95  min_p=0.0
#   chat_template_kwargs.reasoning_effort="medium"   # low für leichte Tasks
#   max_tokens großzügig; RETRY bei content_chars==0 (Loop-Fail ~8 %)
```

Nicht verwenden: `-ctkd/-ctvd` bei ≤~130k (§7), `--parallel 2` (§8 —
kein Durchsatzgewinn, Latenz verdoppelt, eris-Aggregat halbiert),
ctx > empfohlene Werte (§8 — 163840 lädt, aber Generation crasht).

### Offene Risiken
- Delivery-Flakiness ist teils seed-sensitiv: Produktion sollte Retry-Logik
  oder `presence_penalty`/seed-Rotation vorsehen.
- **Seed-Empfehlung: `seed=-1` (randomisiert) + Retry bei leerem
  Content.** Kein fester Seed ist "lucky": 777 rettete hauhau, versagte
  aber bei orcM; 4242 lief bei manchen, loopt bei anderen. Ein fester
  Seed würde einen schlechten Attraktor *systematisch* treffen —
  Randomisierung macht Retries tatsächlich verschieden. Festen Seed nur
  für reproduzierbare Tests/Debugging.
- Uranus-Dauerlast-Instabilität (CUDA-cublas-Crashes ~60–80 min) —
  für Produktion-Eris irrelevant (3080 Laptop-GPU stabil), für Uranus-
  Batch-Runs Restart-Logik behalten.
- mmproj-GPU-Offload auf Eris: bei ≥98k ctx kein Headroom (mmproj F32
  ~1,8 GiB); nur via ctx ≤64k oder teilweises Offload möglich —
  separates Thema.
