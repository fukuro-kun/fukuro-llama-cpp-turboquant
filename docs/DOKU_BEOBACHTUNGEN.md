# Doku-Beobachtungen — fukuro-llama-cpp-turboquant

**Angelegt:** 2026-07-08 durch Doku-Engel (passiver Wächtermodus)
**Zweck:** Sammlung von Diskrepanzen, Widersprüchen und veralteten Beschreibungen
zwischen Doku und Code-Realität. Einträge werden **notiert**, nicht selbst behoben.

---

## Rundgang 1 — 2026-07-08

### [2026-07-08] — [widersprüchlich] — SNAPSHOT.md beschreibt veralteten DFlash-Zustand

**Fund:** `docs/SNAPSHOT.md` (untracked) beschreibt die DFlash-Migration als
aktuell: "DiffusionGemma entfernt", "DFlash primärer Pfad", "AtomicBot-Zwischenschicht
entfallen", Build `e2f1f261b` (cherry-dflash HEAD). Aber master wurde auf die alte
Basis zurückgesetzt — FORKS.md §10 bestätigt: "DFlash-Migration als archive/cherry-dflash
Branch archiviert, master auf alte Basis zurückgesetzt".

**Quelle A:** `docs/SNAPSHOT.md` (untracked), Zeilen 20-28, 51-59
**Quelle B:** `FORKS.md` §10 (Zeile 304-310), `git log master` (HEAD=d5a8cd14b,
keine DFlash-Commits in master-Historie), `archive/cherry-dflash` HEAD=e2f1f261b
**Vorschlag:** SNAPSHOT.md entweder auf aktuellen master-Stand aktualisieren
(AtomicBot-Basis, DiffusionGemma aktiv, kein DFlash) oder löschen falls nicht
mehr benötigt. Trilium-Momentaufnahme (Czii4MdFKb3i) enthält denselben
veralteten Inhalt und muss entsprechend behandelt werden.
**Status:** offen

---

### [2026-07-08] — [toter-link] — AGENTS.md referenziert gelöschte docs/fork-Dateien

**Fund:** AGENTS.md referenziert drei docs/fork-Dateien die nicht mehr existieren.
Diese wurden durch Commit `f86ac47a6` ("revert: 88bd4f052") gelöscht.

**Quelle A:** `AGENTS.md` — Referenzen auf:
- `docs/fork/2026-06-15_DIFFUSION_GEMMA_STATUS.md` (Zeile ~163, DiffusionGemma Status-Tabelle)
- `docs/fork/2026-06-15_STATUS_QUO_VULKAN.md` (Zeile ~175, Vulkan-Turbo3 Status)
- `docs/fork/archive/rca/2026-06-16_DEBUG_SESSION_PKV_FIX.md` (Zeile ~163, KV-Cache Path Fix)

**Quelle B:** `ls docs/fork/` — Dateien nicht vorhanden. `git log --diff-filter=D`
zeigt Löschung durch `f86ac47a6`.
**Vorschlag:** Entweder Dateien aus Git-Historie wiederherstellen
(`git show 88bd4f052:docs/fork/2026-06-15_DIFFUSION_GEMMA_STATUS.md > ...`)
oder Referenzen aus AGENTS.md entfernen. Die DiffusionGemma-Status-Tabelle in
AGENTS.md enthält weiterhin nützliche Informationen auch ohne die verlinkten Details.
**Status:** offen

---

### [2026-07-08] — [toter-link] — FORKS.md referenziert gelöschte BRANCHES.md

**Fund:** FORKS.md verweist an zwei Stellen auf `BRANCHES.md` für Branch-Details,
aber BRANCHES.md wurde durch Commit `f86ac47a6` gelöscht und existiert nicht mehr.

**Quelle A:** `FORKS.md` Zeile 4: "Siehe auch [BRANCHES.md](BRANCHES.md) fuer
Branch-Details" und Zeile 252: "Details: [BRANCHES.md](BRANCHES.md)"
**Quelle B:** `ls BRANCHES.md` → No such file. `git log --diff-filter=D -- BRANCHES.md`
zeigt Löschung durch `f86ac47a6`.
**Vorschlag:** Entweder BRANCHES.md neu erstellen (Branch-Übersicht für master
und archive/cherry-dflash) oder Referenzen aus FORKS.md entfernen.
**Status:** offen

---

### [2026-07-08] — [toter-link] — AGENTS.md referenziert nicht-existierende FORKS.md-Sections

**Fund:** AGENTS.md Vulkan-Optimierungen-Tabelle referenziert FORKS.md-Sections
§5.7, §5.9, §5.10, §5.11. Diese Sections existieren in FORKS.md nicht mehr —
FORKS.md wurde von 912 auf 288 Zeilen bereinigt (Commit `bfcfcdb81`) und enthält
nur noch Sections bis §5.5.

**Quelle A:** `AGENTS.md` Vulkan-Optimierungen-Tabelle:
- "Siehe [FORKS.md §5.9](FORKS.md#59-vdot2-cherry-pick-abgebrochen)"
- "Siehe [FORKS.md §5.7](FORKS.md#57-warum-bfloat16-fa-fuer-uns-nicht-nutzbar-ist)"
- "Siehe [FORKS.md §5.10](FORKS.md#510-coopmat2-feature-check-ergebnis)"
- "Siehe [FORKS.md §5.11](FORKS.md#511-cuda-fast-wht-plan)"

**Quelle B:** `FORKS.md` Section-Header — höchste Section unter §5 ist §5.5.
Sections §5.6-§5.11 wurden in der Bereinigung entfernt.
**Vorschlag:** Section-Referenzen aus AGENTS.md entfernen oder die relevanten
Inhalte (vdot2-Abbruch, BFloat16-Status, coopmat2-Check, CUDA-WHT-Plan) in
FORKS.md wieder aufnehmen falls noch relevant.
**Status:** offen

---

### [2026-07-08] — [veraltet] — Trilium Projekt-Hauptnote: conda/ellama vs. uv

**Fund:** Trilium Projekt-Hauptnote (eiba6WJDfTiq) erwähnt `conda activate ellama`
als Python-Umgebung für alle Skripte und Builds. Aber AGENTS.md hat eine
verbindliche Regel: "Python-Umgebungen: Ausschliesslich uv — Python-Pakete werden
nur via uv verwaltet."

**Quelle A:** Trilium-Note `eiba6WJDfTiq` — "Conda-Env: ellama — aktivieren mit
conda activate ellama", "conda activate ellama vor allen Python-Skripten und Builds"
**Quelle B:** `AGENTS.md` Lokale Verträge → "Python-Umgebungen: Ausschliesslich uv"
**Vorschlag:** Trilium-Note aktualisieren — conda-Referenzen durch uv ersetzen
oder klarstellen ob conda für C++-Builds noch verwendet wird während uv für
Python-Pakete gilt.
**Status:** offen

---

### [2026-07-08] — [veraltet] — Trilium Projekt-Hauptnote: BRANCHES.md als kritische Datei

**Fund:** Trilium Projekt-Hauptnote listet BRANCHES.md als kritische Datei, aber
BRANCHES.md existiert nicht mehr (gelöscht durch `f86ac47a6`).

**Quelle A:** Trilium-Note `eiba6WJDfTiq` — "Kritische Dateien: ... BRANCHES.md
(Branch-Uebersicht)"
**Quelle B:** `ls BRANCHES.md` → No such file
**Vorschlag:** BRANCHES.md aus der Liste kritischer Dateien in Trilium entfernen
(oder BRANCHES.md neu erstellen — siehe Beobachtung zu FORKS.md).
**Status:** offen

---

### [2026-07-08] — [veraltet] — Trilium Momentaufnahme (Czii4MdFKb3i) beschreibt DFlash-Zustand

**Fund:** Trilium Momentaufnahme-Note enthält denselben veralteten Inhalt wie
`docs/SNAPSHOT.md` — beschreibt DFlash-Migration als aktuell, aber master wurde
auf alte Basis zurückgesetzt.

**Quelle A:** Trilium-Note `Czii4MdFKb3i` — "DFlash primärer Pfad",
"DiffusionGemma entfernt", Build e2f1f261b
**Quelle B:** `git log master` — HEAD=d5a8cd14b, keine DFlash-Commits in master.
FORKS.md §10 bestätigt Reset.
**Vorschlag:** Trilium-Momentaufnahme als veraltet markieren oder aktualisieren.
Siehe auch Beobachtung zu SNAPSHOT.md.
**Status:** offen

---

## Zusammenfassung Rundgang 1

| # | Kategorie | Kurzbeschreibung | Status |
|---|-----------|------------------|--------|
| 1 | widersprüchlich | SNAPSHOT.md beschreibt veralteten DFlash-Zustand | offen |
| 2 | toter-link | AGENTS.md → 3 gelöschte docs/fork-Dateien | offen |
| 3 | toter-link | FORKS.md → gelöschte BRANCHES.md | offen |
| 4 | toter-link | AGENTS.md → 4 nicht-existierende FORKS.md-Sections | offen |
| 5 | veraltet | Trilium: conda/ellama vs. uv-Regel | offen |
| 6 | veraltet | Trilium: BRANCHES.md als kritische Datei | offen |
| 7 | veraltet | Trilium Momentaufnahme: DFlash-Zustand | offen |

**Wurzelursache für #2, #3, #6:** Commit `f86ac47a6` ("revert: 88bd4f052 —
versehentlicher Revert des AtomicBot Sync-Merges rückgängig") löschte versehentlich
Dateien die eigentlich behalten werden sollten. Der Revert-Mechanismus hat
Dateien gelöscht die im Original-Commit hinzugefügt wurden, aber der "Revert
des Reverts" hat sie nicht wiederhergestellt sondern im Gegenteil weitere
Dateien gelöscht. Die AGENTS.md-Reparatur (`4e4e5a9fe`) hat viele Schäden
behoben, aber nicht die verwaisten Datei-Referenzen.

**Wurzelursache für #4:** FORKS.md-Bereinigung (`bfcfcdb81`, 912→288 Zeilen)
entfernte Sections §5.6-§5.11, aber AGENTS.md wurde nicht entsprechend
aktualisiert.

---

## Rundgang 4 — 2026-07-08 (~04:20)

### [2026-07-08] — [sicherheitsverletzung] — AGENTS.md enthält Host-Namen (committed + uncommitted)

**Fund:** AGENTS.md enthält lokale Host-Namen ("Venus", "Mars", "Dev-Host", "Pascal-Host")
was die Sicherheitsregel "Keine privaten Daten in oeffentliche Repos" verletzt.
Die eigene Prüfregel in AGENTS.md (Zeile 96) listet genau diese Namen als verboten.

**Quelle A:**
- **Committed** (Zeile 182): "AMD-GCN (Vega) turbo3 MTP 0% → 59.7%, AMD-RDNA3 keine Regression"
- **Uncommitted** (Zeilen 106-112, neue GPU-Nutzungs-Regel): "Dev-Host" 5×, "Pascal-Host" 2×
  — z.B. "Auf Dev-Host (Dev-Host) wird die GPU von anderen Projekten genutzt",
  "Pascal-Host verwenden (ssh Pascal-Host, GTX 1070, Pascal)"

**Quelle B:** AGENTS.md Sicherheitsregel (Zeile 88-96):
"❌ Lokale Host-Namen" + Prüfbefehl `grep -ri "Dev-Host|uranus|mars|venus|Pascal-Host|..."`
**Vorschlag:** Host-Namen durch generische Beschreibungen ersetzen:
- "Dev-Host" → "Dev-Host (RTX 3070 Mobile)"
- "Pascal-Host" → "GPU-Test-Host (GTX 1070, Pascal)"
- "AMD-GCN (Vega)" → "AMD GCN (Vega)"
- "AMD-RDNA3" → "AMD RDNA3"
Die GPU-Architektur-Info ist erlaubt, die Host-Namen nicht.
**Status:** offen — **DRINGEND: vor Commit der uncommitted-Änderung bereinigen!**

---

### [2026-07-08] — [beobachtung] — Uncommitted: GPU-Nutzungs-Regel + .gitignore + LAN-Deployment

**Fund:** Aktive (uncommitted) Änderungen am Repo durch eine parallele Session:
1. **AGENTS.md**: Neue Sektion "GPU-Nutzungs-Regel (kritisch!) aktuell am 8.7.2026"
   + LAN-Deployment-Zeile. Referenzen auf LOKAL.md → "GPU-Nutzungs-Regeln" (existiert ✅)
   und "Fork-Deployment im LAN" (existiert ✅) sind gültig.
2. **.gitignore**: `HANDOFF.md` hinzugefügt (sinnvoll — Handoff-Dateien sollen nicht committet werden).

**Quelle A:** `git diff AGENTS.md`, `git diff .gitignore` (Stand 04:06 Uhr)
**Quelle B:** `LOKAL.md` Zeile 328 ("GPU-Nutzungs-Regeln"), Zeile 330 ("Fork-Deployment im LAN")
**Vorschlag:** Keine Aktion — Änderungen sind in Arbeit. Nur Host-Namen bereinigen (siehe separate Beobachtung).
**Status:** offen (beobachtet, nicht vom Doku-Engel zu committen)

---

## Zusammenfassung Rundgang 4

| # | Kategorie | Kurzbeschreibung | Status |
|---|-----------|------------------|--------|
| 8 | sicherheitsverletzung | AGENTS.md enthält Host-Namen (committed + uncommitted) | offen — DRINGEND |
| 9 | beobachtung | Uncommitted: GPU-Nutzungs-Regel + .gitignore + LAN-Deployment | offen (beobachtet) |
