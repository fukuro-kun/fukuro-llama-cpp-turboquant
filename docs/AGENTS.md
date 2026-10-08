# docs/ — AGENTS.md (Child-DOX)

**Zweck:** Dokumentations-Verzeichnis des Forks. Enthaelt (a) die kanonischen Fork-Dokumente (ROADMAP, CHANGELOG, Statusberichte, Referenzen) und (b) erhaltene Upstream-/Vendor-Doku. Dieses Dokument legt fest, welche Datei welchen Stand fuehrt und wie Doku aktualisiert wird.

**Eigentuemer:** fukuro + KI-Agent

**Geltungsbereich:** Alle Pfade unter `docs/`.

---

## Lokale Vertraege

### Kanonische Dokumenten-Tabelle

Welche Datei fuehrt welchen Stand — andere Doku mit gleichem Thema verweist hierauf statt selbst Autoritaet zu sein:

| Datei | Zweck | Update-Trigger |
|-------|-------|----------------|
| `../CHANGELOG.md` (Root) | Aktive, fortlaufende Chronologie aller bedeutsamen Fork-Aenderungen | Nach jedem Feature/Bugfix/Doku-Schritt eintragen |
| `fork/ROADMAP.md` | Aktueller Optimierungs-Backlog (Abhakliste, M1-M6) | Statuswechsel eines Items, neue Findings |
| `fork/plans/SESSION_PLAN_*.md` | Aktive Umsetzungs-/Eval-Plaene mit Checklisten | Nach jedem verifizierten Fortschritt (siehe Root-Regel zu Gates) |
| `qwen38-eval/ENGINE_EVAL_20261007.md` | Gemessener Engine-Vergleichsbericht (Qwen Flash-Next) | Nach abgeschlossenen Mess-Epochen; eingefrorene verifizierte Eingaben beim Formatieren nicht still neu berechnen — gerechtfertigte Re-Analyse nur mit neuer Epoche/Erratum und erhaltenen Rohdaten |
| `speculative.md` | Aktuelle Referenz der nativen spekulativen Decodierung (Flags, Mechanismen, Grenzen) | Bei Aenderungen an `common/speculative.cpp`, `common/arg.cpp` Spec-Flags oder Server-Draft-Pfad |
| `../FORKS.md` | Fork-Lineage, Branch-Konventionen, Feature-Vergleich, Cherry-Pick-Status | Bei Remote-/Branch-Konventions-Aenderungen |
| `SNAPSHOT.md` | Historischer Juli-Status-Snapshot | Eingefroren — nur Banner/Markierung, kein Rewrite |
| `DOKU_BEOBACHTUNGEN.md` | Laufendes Audit-/Beobachtungs-Log | Neue Auffaelligkeiten und deren Disposition |
| `fork/CHANGELOG.md` | Historischer Port-/Verbesserungsbericht | Eingefroren — keine doppelte aktive Chronologie fuehren |
| `DOX_AUDIT.md` | Dauerhafter Hierarchie-/Link-Audit-Bericht (Scope, Evidenz, offene Punkte) | Bei erneuten Audits oder wenn offene Punkte abgearbeitet werden |

### Historisch vs. aktuell

- Datierte Berichte, Session-Plaene und Merge-Recherchen werden **nicht** zur aktuellen Autoritaet — sie bleiben erhalten, tragen aber einen `HISTORISCH`-Banner, der auf die kanonische Datei verweist.
- Keine Loeschung oder Archivierung allein wegen Alters — nur wenn eine Datei nachweislich durch eine neuere ersetzt wurde.
- Historische Dokumente duerfen entfernte Namen/APIs beschreiben (das ist ihr Inhalt), muessen aber kenntlich machen, dass sie keinen aktuellen Vertrag darstellen.

### Upstream-Doku

- Upstream-/Vendor-Doku unter `docs/` bleibt erhalten; Eingriffe nur fuer praezise, verifizierte Link-Reparaturen oder veraltete Fork-Einschuebe.
- Root-absolute Pfade (`/docs/...`, `/examples/...`) koennen im GitHub-Web-Kontext gueltig sein und gelten nicht automatisch als defekte lokale Links.
- Zitierte Pfade in forensischen Berichten (z.B. in geloggten Fehlermeldungen) sind keine aktiven Navigationslinks — nicht als Broken-Link-Funde zaehlen.

### Link-/Referenz-Pruefung

- Relative Links werden gegen die Ziel-Datei im Repo geprueft; Basename-Uebereinstimmung allein macht einen fehlerhaften relativen Pfad nicht gueltig (z.B. brauchen common-Links aus `docs/` `../`-Traversal).
- Verifikation ist mechanisch: `git diff --check`, Link-Scan, Quell-/CMake-Namenspruefung — keine "alles verifiziert"-Kurzform.
- Referenzen auf Tests oder Skripte nur angeben, wenn der Name in `tests/CMakeLists.txt` registriert bzw. die Datei on-disk existiert.

---

## Arbeitsanleitung

### Dokumentations-Workflow

1. Vor Edit: dieses Dokument + Root-AGENTS.md lesen; bei Unterverzeichnis-Spezifika das naechstgelegene AGENTS.md.
2. Neue Verfahrens-Doku: kanonisch unter der passenden Domain ablegen und in der Tabelle oben bzw. `fork/ROADMAP.md` referenzieren.
3. Nach Edit: Link-Pruefung auf die geaenderten Dateien (mechanisch), `git diff --check`.
4. Statusaenderungen nur mit Beleg (Commit, Messbericht) — keine unbelegten Gate-Schliessungen, keine erfundenen Testzahlen.
5. Verifizierte Mess-Eingaben nicht still neu berechnen (keine nachtraegliche Normalisierung beim Formatieren). Begruendete Re-Analysen/Korrekturen sind erlaubt, aber nur dokumentiert als neue Epoche oder Erratum mit erhaltenen Roh-Inputs — Rohdaten nie ueberschreiben.

### Was nicht hierher gehoert

- Host-spezifische Pfade, IPs, Host-Namen, Tokens → `LOKAL.md` / Trilium (siehe Root-Sicherheitsvertrag).
- Session-Rohergebnisse ohne Mehrwert → `fork/archive/` mit klarem Status.

---

## Verifikation

- [ ] Kanonische Tabelle konsistent (kein zweites Dokument fuehrt denselben Stand widersprechend)
- [ ] Neue/umbenannte Doku in der Tabelle oder einem Unter-Index referenziert
- [ ] `git diff --check` sauber; relative Links der geaenderten Dateien mechanisch geprueft
- [ ] Historische Dateien mit `HISTORISCH`-Banner versehen, nicht als aktuell referenziert
- [ ] Keine privaten Daten (Host-Namen, IPs, absolute lokale Pfade) eingefuehrt

---

## Child-DOX-Index

| Pfad | Zweck | Status |
|------|-------|--------|
| `docs/fork/` | Fork-Roadmap, -Berichte, -Plaene, -Archiv | [~] Kein eigenes DOX — durch dieses Dokument abgedeckt |
| `docs/qwen38-eval/` | Qwen3.8-Eval-Material (Testkonzept, Reports) | [~] Kein eigenes DOX — durch dieses Dokument abgedeckt |
| `docs/backend/` | Upstream-Backend-Doku | [~] Kein eigenes DOX — durch dieses Dokument abgedeckt |
| `docs/development/` | Entwicklungs-/Referenzdokumente (teils historisch) | [~] Kein eigenes DOX — durch dieses Dokument abgedeckt |

*Hinweis: Weitere Unterverzeichnisse ohne Index-Eintrag bleiben direkt von diesem Dokument abgedeckt.*
