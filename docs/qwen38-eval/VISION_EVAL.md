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
