# tools/parallel-decision/AGENTS.md — Child-DOX

**Zweck:** Engine für parallele Entscheidungen über endliche, typisierte Kandidatenmengen; verwendet durch `llama-parallel-decision` und die Server-Endpunkte `/decision` sowie `/v1/decision`.

**Eigentümer:** fukuro + KI-Agent

**Geltungsbereich:** Alle Dateien unter `tools/parallel-decision/`.

---

## Lokale Verträge

- `decision-engine.*` ist die gemeinsame Implementierung für CLI und Server. Sie soll die öffentliche `llama.h`-API verwenden und keine direkte Abhängigkeit auf interne KV-Cache-Klassen einführen.
- Der Server aktiviert den Modus mit `--decision-seqs N` (`N = 0` oder `N >= 3`). Entscheidungssequenzen liegen oberhalb der normalen Slots; der Server vergrößert `n_seq_max` und erzwingt `kv_unified`, damit Sequenzzweige KV-Zellen ihres Stamms teilen.
- Rekurrente und hybride Modelle benötigen gleich lange Sequenzabschnitte pro Decode. Die Engine füllt Kandidatenzweige dort rechts auf und liest die Logits am letzten echten Token.
- Feldwerte sind auf endliche Kandidaten begrenzt. Struktur und Typen der Antwort werden programmatisch zusammengesetzt; semantisch falsche Entscheidungen bleiben möglich.
- Änderungen an Sequenzbelegung oder KV-Verhalten müssen mit TurboQuant-K/V-Typen, vereinheitlichtem Cache und den betroffenen Backends geprüft werden. Ein erfolgreicher Build allein belegt keine Laufzeitkompatibilität.

## Arbeitsanleitung

1. Bei Änderungen an der Engine Tokenisierung, kollidierende Kandidatenpfade sowie `tree`-/`greedy`-Auswertung berücksichtigen.
2. Anpassungen an `--decision-seqs` gemeinsam mit `common_params`, `n_seq_max`, Server-Output-Kapazität und Server-Task-Queue prüfen.
3. Änderungen an `decision-engine.*` sowohl über das Werkzeug als auch über `llama-server` verifizieren.

## Verifikation

- `cmake --build build --target llama-parallel-decision llama-server -j$(nproc)` erfolgreich.
- `/v1/decision` mit einem geladenen Modell testen; mindestens einen `tree`- und einen `greedy`-Fall sowie eine ungültige Anfrage prüfen.
- TurboQuant-KV, vereinheitlichten Cache und Hybrid-/rekurrente Modelle auf den jeweils verfügbaren Backends gezielt testen.

---

## Child-DOX-Index

Keine Child-AGENTS.md-Dateien.
