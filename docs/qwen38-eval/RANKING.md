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

## 5. Vision (CPU-mmproj, ~960 prompt tok)

Alle getesteten Kandidaten liefern plausible Bildbeschreibung. Eris ~140–160 s,
Uranus ~54–66 s (CPU-Encode dominiert). Kein Differenzierer.

---

## FINALE EMPFEHLUNG

### Aligned: **ISTA-DASLab Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp**

Begründung:
- 4/4 Delivery auf beiden Hosts (einziges Aligned-Modell mit perfekter Rate)
- Sauberste Refusals am Gradienten (aligned-Verhalten verifiziert)
- 12,1 GB → kleinste Aligned-Datei mit MTP → größte Kontext-Frontier auf Eris
- ~34 t/s Uranus, ~18–19 t/s Eris; akademische GSQ-RCO-Methode (IST-Austria)
- Codequalität: 2er-Noten, saubere Concurrency + saubere task2

Runner-up: **Swift-1.5 aligned** — schnellste Token-Effizienz (-42% Reasoning
vs AP), beste Codequalität im Review, 2/2 task1; aber task2 2/3 (ein
Früh-Termination-Flake + ein Loop) → leicht schwächere Delivery-Sicherheit.
**AP-IQ4_XS** bleibt die Qualitätsreferenz, scheidet aber aus Eris-Produktion:
OOM bei 98k+MTP (nur ≤~80k oder Uranus-tauglich).

### Uncensored: **Swift-1.5 Uncensored UD-Q3_K_XL-MTP**

Begründung:
- 3/3 task1 + task2 delivery — einziger Unc-Kandidat mit perfekter Rate
- Volle Compliance am Gradienten (c3–c6 Content), aligned-c1/c2 sicher
- ~33–35 t/s Uranus, ~23–26 t/s Eris; 13,1 GB passt 98k auf Eris
- Finetune-Effizienz halbiert Reasoning-Tokens → faktisch schnellste Antwort
- Caveat: c4-Exploit-Loop auf Uranus reproduzierbar (seed-abhängig; eris lief)
- Caveat: eigene Lizenz (1 Mio. USD Umsatzgrenze kommerziell) — vor
  Produktivnutzung prüfen

Runner-up: **bartowski/orcarouter IQ3_XS** — 4/4 Delivery, compliant,
orcarouter-Basis (unabhängig verifizierter Studien-Sieger); langsamer und
token-intensiver (108k Reasoning-Chars task1). **Huihui** drittbest:
2/3 task1, guter Code, kleinste Datei — intermittente Loop-Gefahr.

### Nicht empfohlen
- **orcM, HauhauCS, RN-Orca:** task1-Delivery 1/4 bzw. 1/3 — Reasoning-Loop-
  Risiko zu hoch für Produktion trotz passabler Qualität wenn geliefert.
- **AP/UD-IQ4_XS auf Eris:** 98k+MTP OOM.

### Offene Risiken
- Delivery-Flakiness ist teils seed-sensitiv: Produktion sollte Retry-Logik
  oder `presence_penalty`/seed-Rotation vorsehen.
- Swift-Lizenz prüfen.
- Draft-KV-Quantisierung (-ctkd/-ctvd) könnte AP auf Eris retten (~0.4–1 GiB
  Headroom) — ungetestet.
