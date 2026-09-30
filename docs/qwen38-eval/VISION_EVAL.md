# Vision-Eval Qwen3.8-27B — Scores (Devin vs. ground_truth.md, 2026-09-29)

Skala: 0=falsch/leer/Halluzination dominant, 1=teils richtig/Schluesselfehler,
2=gut mit Fehlern, 3=korrekt+vollstaendig. Flags: h=hallucination.

## Haupteval nativ (Server-Default, aufloesungsgetrieben ~700–4100 Bild-Tokens)

| Bild | Prompt | qwen_al | qwen_unc | Bemerkung |
|---|---|---|---|---|
| iq_dashboard | standard | 3 | 3 | Titel + KPIs korrekt |
| iq_dashboard | special | 3 | 3 | Hosts/Profile/Services korrekt |
| iq_chat | standard | 3 | 3 | Dark-UI korrekt |
| iq_chat | special | 3 | 3 | 13 Bedienelemente vollstaendig |
| chart_bars | standard | 3 | 3 | Titel+Achsen korrekt |
| chart_bars | special | 3 | 3 | 4,4 t/yr, 33,5 Gt, ME-A 20,3 korrekt |
| tabelle_wiki | standard | 3 | 3 | Artikel+Layout korrekt |
| tabelle_wiki | ocr | 2 | 2 | 6/8 Zahlen exakt, identische Ziffernfehler (Istanbul ...602 statt ...082, Rom ...084 statt ...984) |
| code | standard | 3 | 3 | Editor+Zeilen korrekt |
| code | ocr | 3 | 3 | 17 Zeilen transkribiert |
| historisch | standard | 3 | 3 | Menge/Laternen/Gebaeude |
| handschrift | standard | 2 | 2 | Dokumentstruktur korrekt |
| handschrift | ocr | 1 h | 1 h | Ankerdaten halluziniert: al „Stuttgart 29.9.1836", unc „Loerrach 29.9.1836"; GT: Friedrichshafen 29. Okt. 1856 |
| infografik | standard | 3 | 3 | 21st Space Wing korrekt |
| infografik | special | 3 | 3 | alle Zahlen exakt ($1,27B, 15 WS, 9871+4238) |
| strassenschild | standard | 3 | 3 | NEDERLAND + 4 Stufen |
| strassenschild | special | 3 | 3 | 50/80/100/120 exakt |
| natur_makro | standard | 3 | 3 | Biene+Pollenkoerbchen |
| karte | standard | 1 h | 2 | al: „Nuerburgring" falsch benannt; unc: „Stadtdreieck" unspezifisch aber kategorisch korrekt |
| karte | special | 2 | 2 | Rennstrecke erkannt (GT: Edmonton City Airport) |
| zeitung | standard | 3 | 3 | Rauchwarenmarkt/Leipzig/1930 |
| zeitung | ocr | 3 | 3 | exzellente Fraktur-Transkription |
| formel | standard | 3 | 3 | Maxwell-Tabelle |
| formel | special | 3 | 3 | LaTeX-Gleichungen korrekt |
| multi | standard | 3 | 3 | beide Bilder korrekt |
| **Summe** | | **68/75 (2,72)** | **69/75 (2,76)** | |

## Granularitaets-Sweep (forciert min=max, 4 Kern-Bilder, standard)

| Bild | al g256 | al g512 | al g1024 | unc g256 | unc g512 | unc g1024 |
|---|---|---|---|---|---|---|
| iq_dashboard | **0 h** | FAIL | FAIL | 1 | 2 | 3 |
| chart_bars | FAIL | 2 | 3 | 2 | 2 | 2 |
| tabelle_wiki | 2 | 2 | 2 | 2 | 2 | 2 |
| zeitung | **3** | 3 | 3 | **3** | 3 | 3 |

Befund: Homogene Anker-Bilder (Zeitung-Titelkopf, Wikipedia-Titel) ueberleben
selbst 256 Tokens; heterogen-dichte Bilder (Dashboard, Chart) brechen ein —
dashboard@256 halluziniert komplett (erfundenes „Interfacial-Reference"-Layout).
Warnung des Servers belegt: <1024 problematisch fuer Grounding/Detail.

## Metriken

- Wall/Request: al Ø 188 s, unc Ø 194 s (decode-dominiert, GPU thermisch
  gedrosselt ~88 °C → tg 6,3–6,7 t/s in spaeten Runs vs ~13–15 t/s kalt)
- GPU-Encode: ~2,1 s @ ~2000 Bild-Tokens (F16-mmproj, GPU-Offload)
- ptok: nativ aufloesungsgetrieben 736 (code) bis 4124 (chart)
- Fehler: 3× 300-s-Client-Timeout (swiftal sweep), 1× CUDA-Crash
  (swiftunc @64k Encode-Peak) → bei ctx 40960 stabil

## Ops-Hinweis: n_ubatch-Constraint ist GEMMA-spezifisch

Auf amd-apu (Gemma 4) gilt `n_ubatch >= Bild-Tokens` — dort schaltet
`mtmd_decode_use_non_causal()` (tools/mtmd/mtmd.cpp) die Decoder-Attention
fuer Bild-Embeddings bidirektional (nur GEMMA3/GEMMA4V/GEMMA4UV), Micro-
Batching ist dann unmoeglich (Assert in llama-context.cpp). Qwen-VL faellt
in `default: return false` → Bild-Embeddings werden causal decodiert und
duerfen ueber U-Batches gesplittet werden — diese Eval lief komplett mit
`-ub 512` bei Bildern bis 4112 Tokens. Fuer Qwen-VLM-Profile muss `-ub`
daher NICHT auf image-max-tokens dimensioniert werden (nur ctx/VRAM).

Vorlaeufige Text-only-Messungen 29.09. (`llama-bench`, -r 1, b8192):
AMD-APU/Gemma: pp4096 187/219/239/250/245 t/s bei ub256/512/1024/2048/4096.
ub2048 statt 4096 ist ein Kandidat, keine statistisch belegte
Produktivbeschleunigung. Im Vision-Produktivprofil sind seitdem ub2048,
Bild-Tokens 512–1984 aktiv; Chart (1969 Prompt-Tokens) und Dashboard
(905) liefen mit GPU-mmproj durch. Die 64-Token-Marge beweist nicht,
dass alle Bildformate unterhalb der Chunk-Grenze bleiben.

## RTX3080-Laptop/Qwen3.8 Prefill ub-Sweep (29.09.)

Setup: llama-server, CUDA, Swiftal + GPU-offloaded F16-mmproj, batch 8192,
slot context 65536, `cache_prompt=false`, gemessener `cache_n=0`. Je ub gab es
zwei gepaarte Laeufe (ub64 vier); vor jedem Prompt wurde auf ≤72 °C und ohne
aktives Throttle abgekuehlt. 256/1024/4096/10240 Tokens wurden per
`/tokenize` exakt kalibriert. Gemessen wird der llama-server Prompt-Prefill
mit synthetischem, identischem Wortfragment — nicht eine reale Chat-Aufgabe.

| ub | 256 t/s | 1k t/s | 4k t/s | 10k t/s |
|---:|---:|---:|---:|---:|
| 32 | 308,7 | 332,9 | 311,7 | 284,9 |
| 48 | 358,8 | 416,3 | 398,0 | 363,7 |
| 64 | 431,8 | 492,3 | 472,5 | 424,5 |
| 80 | 417,3 | 513,6 | 487,2 | 444,9 |
| 96 | **475,2** | 557,7 | 526,2 | 476,6 |
| 112 | 461,4 | **561,8** | **557,8** | **501,4** |

Der getestete Bereich steigt fuer laengere Prefills bis ub112 an; ein Optimum
ist das nicht, da groessere Werte noch nicht vermessen wurden. Bei 256 Tokens
liegt ub96 vorn; bei 1k sind ub96/112 praktisch gleich. Qwen3.8 ist in dieser
Konfiguration ein Dense-Modell — die fruehere Erklaerung mit „kompaktem MoE-
Aktivierungs-Working-Set“ war sachlich falsch.

**Thermik:** Vor jedem Lauf waren Performance-Profil, Fan-Boost und ≤72 °C
Starttemperatur verifiziert. Bei 4k/10k erschien trotzdem wiederholt
`SW Thermal Slowdown` bei 86–89 °C; die waehrend-request 1-s Samples
lagen meist bei 90–140 W. Ein unmittelbarer 10k/ub48-Nach-request-Snapshot
zeigte 78,8 W bei aktivem Throttle; das war kein belegter Sustained-Wert
waehrend des Prefills. Das sind RTX3080-Laptop-Messwerte unter Performance-Profil,
aber keine „throttle-freien“ Idealwerte. Das erwartbare Erreichen von 86 °C und
die softwareseitige Slowdown-Anzeige werden getrennt protokolliert.

## Kontextfenster isoliert (ub112, gleiche Prompts und Batch)

Qwen3.8 + mmproj lief auch mit reserviertem `n_ctx=131072` (VRAM 14,9 GiB von
15,6 GiB). Zwei Durchlaeufe pro Promptlaenge, `cache_n=0`:

| Prompt | ctx 65536 | ctx 131072 |
|---:|---:|---:|
| 256 | 461,4 t/s | 462,1 t/s |
| 1024 | 561,8 t/s | 562,5 t/s |
| 4096 | 557,8 t/s | 559,8 t/s |
| 10240 | 501,4 t/s | 502,7 t/s |

Bei diesen Promptlaengen ist kein messbarer Prefill-Nachteil durch das
reservierte 128k-Fenster sichtbar. Die ub-Reihe bei 64k hielt das Kontextfenster
fest auf 65536 — deren ub-Trend kann also nicht durch unterschiedliche
Kontextallokationen erklaert werden. Das 128k-Ergebnis gilt fuer ub112, nicht
fuer jede ub-Stufe; ausserdem ist die VRAM-Reserve klein.

**Zwischenstand:** Der erste Sweep bis ub112 war unvollstaendig; siehe die
Erweiterungsstichprobe unten. Es gibt noch kein Produktionsoptimum und keinen
Speedup-Beleg fuer ein produktives Profil. AMD-APU-Werte nicht uebertragen:
anderes Modell/Backend; Gemma-Vision erzwingt zusaetzlich
`ub >= Bild-Chunk-Tokens`.

AMD-APU/Gemma Produktionsprofil (Vulkan, `-ub 2048`, 262144 total / zwei
Slots, echte Prompt-Requests bei zuvor idle Slots): 256/1024/4096/10240
Prompt-Tokens ergaben 210/295/252/190 t/s (`prompt_n` jeweils +1 BOS-Token,
`cache_n=0`). Der Durchsatz ist nicht monoton mit Promptlaenge; das laengste
Sample war langsamer. Das ist eine einzelne Laufserie und wegen anderer
Architektur/Backend kein direkter Zahlenvergleich mit RTX3080-Laptop/Qwen. Der aktuelle
Vision-Chunk-Grenzwert (1984) ist unabhaengig davon durch den Gemma-Assert
bestimmt.

## UB-Stichprobe bis 1024 auf RTX3080-Laptop und Ada-RTX4060Ti-GPU1 (29.09.)

Gleicher Qwen3.8-Swiftal GGUF + F16-mmproj, ctx65536, batch8192,
cache_n=0, Promptlaengen 10k/25k (je 2 Messungen); p50k wurde an ub256/512/1024
je 1× getestet. Start je Request bei ≤72 °C ohne Throttle. Ada-RTX4060Ti-Server lief
loopback-only auf der bereits reservierten GPU1-Portzuweisung; GPU0 blieb
unangetastet. Ein anfaenglicher parallel gestarteter Wrapper-Lauf wurde abgebrochen;
die Tabelle stammt ausschliesslich aus den danach separat gefahrenen, bereinigten
Ledgers (ohne die ueberlappenden Messungen).

#### RTX 3080 Laptop

| ub | 10k t/s (mean, range; n=2) | 25k t/s (mean, range; n=2) | 50k t/s (n=1) |
|---:|---:|---:|---:|
| 160 | 489.2 (460.5–517.8) | 415.0 (403.6–426.4) | — |
| 192 | 498.3 (496.1–500.6) | 443.3 (438.6–448.1) | — |
| 256 | 535.6 (533.2–538.1) | 473.3 (469.8–476.7) | 374.4 |
| 320 | 505.1 (503.8–506.3) | 447.6 (445.0–450.1) | — |
| 512 | 542.8 (542.3–543.3) | 484.2 (483.4–485.0) | 382.2 |
| 1024 | **555.7 (554.6–556.7)** | **499.8 (498.6–500.9)** | **398.5** |
| 1536 | 596.9 (560.8–633.0)* | 513.1 (497.2–529.1) | 404.5 |
| 2048 | 556.7 (553.4–560.0) | 496.2 (490.3–502.4) | 389.0 |
| 3072 | 559.7 (557.2–562.4) | 488.6 (486.2–491.3) | 372.1 |
| 4096 | 554.1 (553.0–555.3) | 475.2 (473.6–476.9) | 357.9 |

*ub1536 rep1 (633.0) war ein Kaltstart-Ausreisser (GPU startete bei 66 °C,
Vergleichslaeufe bei ≥72 °C); rep2 = 560.8. RTX3080-Laptop blieb bis ub4096
OOM-frei — der Compute-Puffer skaliert mit ub, reichte hier aber aus.

#### RTX 4060 Ti GPU 1

| ub | 10k t/s (mean, range; n=2) | 25k t/s (mean, range; n=2) | 50k t/s (n=1) |
|---:|---:|---:|---:|
| 160 | 712.5 (708.1–717.0) | 615.9 (613.5–619.3) | — |
| 192 | 757.8 (756.6–759.0) | 675.4 (674.8–678.5) | — |
| 256 | 819.8 (819.3–820.3) | 726.4 (726.0–726.9) | 602.8 |
| 320 | 771.3 (770.5–772.0) | 671.1 (671.1–671.2) | — |
| 512 | **824.7 (824.6–826.2)** | 756.0 (755.5–757.0) | 661.9 |
| 1024 | 819.1 (818.8–821.1) | **759.0 (758.6–760.5)** | **669.3** |
| 1536 | 810.1 (806.2–814.4) | 751.9 (749.3–754.5) | 667.1 |
| 2048 | 807.7 (805.8–808.7) | 755.4 (754.0–756.9) | 674.4 |
| 3072 | 794.4 (793.6–795.2) | 743.8 (743.0–744.5) | 661.5 |
| 4096 | 780.5 (780.2–780.7) | 730.5 (730.2–730.7) | 654.6 |

Durchsatz in t/s. Auf RTX3080-Laptop war `SW Thermal Slowdown` bei allen 10k/25k-Stichproben
aktiv; 50k-Max-Temperatur 89 °C, ein 50k/ub512-In-Request-Sample fiel auf
78,81 W. Ada-RTX4060Ti GPU1 blieb bei ≤75 °C ohne Thermal-Flag; 10k/25k liegen dort
im getesteten Band nahe ub512–1024, 50k steigt weiter bis ub1024.
Das spricht fuer hardwareabhaengige Sweetspots, aber **nicht** fuer eine
allgemeingueltige ub-Regel. Ein Ada-RTX4060Ti-CUBLAS-Fehler bei
50k/ub96 trat im separaten Wiederholungsversuch nach Position 49152 auf;
frische cache-disabled 50k-Laeufe bei ub256/512/1024 liefen.

## Ada-RTX4060Ti-GPU1 Feinsweep 512–3072 (29.09., throttle-frei)

128er-Raster 640–2944 (n=1 je Punkt) plus ±64-Verifikation 512–768 (n=2,
aufsteigend+absteigend). Gleiche Config wie oben; GPU1 ≤76 °C, nie Thermal-Flag.
Anker 512/1024/1536/2048/3072 aus der Stichprobe oben.

#### 10k-Prefill (t/s)

| ub | t/s | | ub | t/s | | ub | t/s |
|---:|---:|---|---:|---:|---|---:|---:|
| 256 | 820 | | 1024 | 819 | | 2176 | 795 |
| 320 | 771 | | 1152 | 812 | | 2304 | 797 |
| 512 | 825/831/823 | | 1280 | 816 | | 2432 | 797 |
| **576** | **776/774 ⚠** | | 1408 | 817 | | 2560 | 795 |
| **640** | **837/826/825** | | 1536 | 810 | | 2688 | 794 |
| **704** | **777/776 ⚠** | | 1664 | 804 | | 2816 | 796 |
| 768 | 820/815/816 | | 1792 | 804 | | 2944 | 791 |
| 896 | 811 | | 1920 | 802 | | 3072 | 794 |
| | | | 2048 | 808 | | 4096 | 781 |

#### 25k-Prefill (t/s, Auswahl)

| ub | 512 | 576 | 640 | 704 | 768 | 1024 | 1536 | 2048 | 3072 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t/s | 757 | 717 ⚠ | 757 | 709 ⚠ | 737 | 759 | 752 | 755 | 744 |

**Ergebnis:** Flaches Plateau 512–1536 (~810–825 t/s @10k), Optimum **ub 640**
(837/826/825 @10k, 765/757/756 @25k). Darunter echte, reproduzierbare Struktur:
**ub-Werte die keine Vielfachen von 128 sind** (576 = 4,5×128, 704 = 5,5×128)
fallen ~6 % ab — cuBLAS-Tile-Selektion, kein Rauschen (beide Reps identisch).
Praxisregel fuer Ada-RTX4060Ti/Qwen3.8: **ub ∈ {512, 640, 768, 1024, 1536},
d.h. Vielfache von 128**, Abfall erst ab ~2048–3072, 4096 noch -5 %.
Die ±64-Frage ist damit beantwortet: Zwischen 128er-Rasterpunkten gibt es
Dips; die Wahl muss auf das 128er-Gitter fallen. Keine Produktionsprofil-
Aenderung — Uranus-Produktion laeuft weiter auf GPU0/ub512, was innerhalb
des Optimum-Bands liegt.

**Vergleichsvorbehalt:** Die Hosts liefen nicht dieselbe Binärrevision:
RTX3080-Laptop `aa1976da9` / llama.cpp 9549 (GCC 15.2), Ada-RTX4060Ti `d5b96935a` /
llama.cpp 9559 (GCC 13.3). Innerhalb jedes Hosts ist der ub-Trend vergleichbar;
die absolute RTX3080-Laptop-vs-Ada-RTX4060Ti-Speeddifferenz ist daher kein isolierter Hardware-
Beweis. Ein streng kontrollierter Hardwarevergleich erfordert denselben Build.

## Produktionsupdate 2026-09-30

Auf Wunsch wurde der laufende Qwen3.8-Standardserver auf der CUDA-GPU mit
`-ub 640` neu gestartet (ctx131072, 1 Slot, MTP n=2, Turbo4/3,
kein mmproj in dieser bestehenden Instanz). `/health` und ein echter kurzer
Chat-Smoke-Test waren erfolgreich. Das ist eine bewusste Wahl des gemessenen
UB640-Peaks trotz kleinem Vorteil gegenüber UB512; der aktive Prozess wurde
verifiziert. Das lokale Startskript verwendet UB640 als Default (über
`UB` überschreibbar); der Remote-Checkout ist noch nicht synchronisiert.

Das zusätzliche Uranus-Profil P ist lokal vorbereitet, aber noch nicht
bereitgestellt/aktiviert: Standard-Vision auf Port 18084 (ctx98304, ein Slot)
und Swiftunc-Uncensored-Vision auf Port 18085 (ctx81920, ein Slot), jeweils
UB640, F16-mmproj, MTP aus. Beide Kontextpunkte bestanden einen Chart-Smoke;
größere Bildformate und Dauerlast bleiben zu validieren. Die Qwen-Endpunkte
verwenden keinen JEV-/parallel-decision-Modus.
