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

Auf phobos (Gemma 4) gilt `n_ubatch >= Bild-Tokens` — dort schaltet
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

## Eris/Qwen3.8 Prefill ub-Sweep (29.09.)

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
waehrend des Prefills. Das sind Eris-Messwerte unter Performance-Profil,
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

**Vorlaeufige Empfehlung:** ub96–112 als naechstes Kandidatenband fuer einen
Eris-Qwen-VLM-Profiltest; noch kein Produktionsoptimum. 112 war der groesste
gemessene Wert, kein bewiesener Knick/Saettigungspunkt. Auf Phobos nicht
uebertragbar: anderes Modell und Vulkan-Backend; Gemma-Vision erzwingt zusaetzlich
`ub >= Bild-Chunk-Tokens`.

Phobos/Gemma Produktionsprofil (Vulkan, `-ub 2048`, 262144 total / zwei
Slots, echte Prompt-Requests bei zuvor idle Slots): 256/1024/4096/10240
Prompt-Tokens ergaben 210/295/252/190 t/s (`prompt_n` jeweils +1 BOS-Token,
`cache_n=0`). Der Durchsatz ist nicht monoton mit Promptlaenge; das laengste
Sample war langsamer. Das ist eine einzelne Laufserie und wegen anderer
Architektur/Backend kein direkter Zahlenvergleich mit Eris/Qwen. Der aktuelle
Vision-Chunk-Grenzwert (1984) ist unabhaengig davon durch den Gemma-Assert
bestimmt.
