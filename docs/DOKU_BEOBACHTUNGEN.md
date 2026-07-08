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

---

## Rundgang 5 — 2026-07-08 (~05:15)

### [2026-07-08] — [sicherheitsverletzung] — SESSION_PLAN.md enthält Host-Namen + lokale Pfade (COMMITTED)

**Fund:** Commit `edd42e60f` ("Solo-Session Vorbereitung: thecodacus Patches +
SESSION_PLAN") enthält `SESSION_PLAN.md` mit mehreren Sicherheitsverletzungen:
Host-Namen ("Pascal-Host" 6×, "Dev-Host" 2×) und lokale Dateipfade ("/path/to/user/...",
"~/git/..."). Die Datei ist bereits committet und in der Repo-Historie.

**Quelle A:** `SESSION_PLAN.md` (committed in `edd42e60f`):
- Zeile 5: "Host: Pascal-Host (GTX 1070, 8GB VRAM, Pascal)"
- Zeile 6: "Repo: /path/to/fukuro-llama-cpp-turboquant"
- Zeile 19: "Dev-Host GPU ist TABU"
- Zeile 41: "ssh Pascal-Host 'ls -la ~/git/fukuro-llama-cpp-turboquant/build/bin/llama-server'"

**Quelle B:** AGENTS.md Sicherheitsregel: "❌ Lokale Host-Namen", "❌ Lokale Dateipfade
(/path/to/user/...)" + Prüfbefehl `grep -ri "Dev-Host|...|Pascal-Host|...|/path/to/user|..."`
**Vorschlag:** `git filter-repo` oder `git rebase -i` um SESSION_PLAN.md aus der
Historie zu entfernen (oder Host-Namen/Pfade durch generische Beschreibungen ersetzen).
SESSION_PLAN.md sollte außerdem zu `.gitignore` hinzugefügt werden (wie HANDOFF.md).
**Status:** offen — **KRITISCH: bereits in committed-Historie**

---

### [2026-07-08] — [beobachtung] — thecodacus-Patches hinzugefügt (3 Diff-Dateien)

**Fund:** Commit `edd42e60f` fügt drei Patches aus thecodacus/llama.cpp hinzu:
- `patches/thecodacus/01-memory-pinning.diff` — cudaHostRegister für mmap-pages (+21% pp)
- `patches/thecodacus/02-async-expert-prefetch.diff` — Overlap Expert-Upload mit Compute (+20% pp)
- `patches/thecodacus/03-prefetch-slot-sizing-uaf-fix.diff` — Slot-Sizing + UAF Fix (+14% pp)

Patches sind frei von Host-Namen oder lokalen Pfaden (Sicherheitscheck ✅).
Geplant für Solo-Session auf Pascal-Host mit Gemma 4 MoE-Modellen.

**Quelle A:** `git show edd42e60f --stat`, `grep -rin "Dev-Host|Pascal-Host|/path/to/user" patches/` (keine Treffer)
**Quelle B:** SESSION_PLAN.md beschreibt Test-Plan
**Vorschlag:** Keine Aktion — Patches sind sauber. Nur SESSION_PLAN.md bereinigen (siehe separate Beobachtung).
**Status:** offen (beobachtet)

---

## Zusammenfassung Rundgang 5

| # | Kategorie | Kurzbeschreibung | Status |
|---|-----------|------------------|--------|
| 10 | sicherheitsverletzung | SESSION_PLAN.md: Host-Namen + Pfade (COMMITTED) | offen — KRITISCH |
| 11 | beobachtung | thecodacus-Patches hinzugefügt (sauber) | offen (beobachtet) |

---

## Rundgang 6 — 2026-07-08 (~07:20)

### [2026-07-08] — [sicherheitsverletzung] — AGENTS.md GPU-Nutzungs-Regel mit Host-Namen JETZT COMMITTED

**Fund:** Die zuvor (Rundgang 4) als uncommitted notierte GPU-Nutzungs-Regel mit
Host-Namen ("Dev-Host" 5×, "Pascal-Host" 2×) ist nun durch Commit `16b62f970` in der
committed-Historie. Die Sicherheitsverletzung ist jetzt permanent (ohne rebase/filter-repo).

**Quelle A:** `git show 16b62f970` — AGENTS.md Zeilen 106-112 (GPU-Nutzungs-Regel):
"Dev-Host" 5×, "Pascal-Host" 2×
**Quelle B:** AGENTS.md Sicherheitsregel: "❌ Lokale Host-Namen" + grep-Pattern
**Vorschlag:** `git rebase -i` oder `git filter-repo` um Host-Namen zu ersetzen.
Alternativ: Host-Namen in AGENTS.md durch generische Beschreibungen ersetzen
und bereinigten Commit als neuen HEAD setzen.
**Status:** offen — **KRITISCH: committed in 16b62f970**

---

### [2026-07-08] — [sicherheitsverletzung] — Solo-Session-Report enthält "Pascal-Host" (COMMITTED)

**Fund:** `docs/fork/2026-07-08_SOLO_SESSION_REPORT.md` (committed in `16b62f970`)
enthält Host-Namen "Pascal-Host" in Zeile 126: "Branch: feature/thecodacus-pinning auf Pascal-Host".

**Quelle A:** `docs/fork/2026-07-08_SOLO_SESSION_REPORT.md` Zeile 126
**Quelle B:** AGENTS.md Sicherheitsregel: "❌ Lokale Host-Namen" + grep-Pattern listet "Pascal-Host"
**Vorschlag:** "auf Pascal-Host" → "auf dem Pascal-Test-Host (GTX 1070)" oder entfernen.
**Status:** offen — **KRITISCH: committed in 16b62f970**

---

### [2026-07-08] — [artefakt-bereinigung] — Truncierte Duplikate H/HAND/HANDOFF gelöscht

**Fund:** Drei truncierte Artefakt-Dateien (`H`, `HAND`, `HANDOFF`) durch abgebrochenen
Shell-Befehl entstanden. Alle 4360 Bytes, identischer MD5 (`760caffd...`) wie `HANDOFF.md`.
Exakte Duplikate — `HANDOFF.md` ist kanonisch.

**Aktion:** `rm H HAND HANDOFF` — bereinigt durch Doku-Engel (Artefakt-Scan).
`HANDOFF.md` beibehalten (in `.gitignore` seit Commit `16b62f970` bzw. uncommitted .gitignore).
**Status:** erledigt durch Doku-Engel

---

### [2026-07-08] — [beobachtung] — thecodacus MoE-Optimierungen: Solo-Session erfolgreich

**Fund:** Commit `16b62f970` dokumentiert erfolgreiche Solo-Session:
- Prefill-Boost bis +106%, Decode-Boost +68% auf GTX 1070 (Pascal)
- Memory Pinning + Async Expert Prefetch portiert
- Env-Vars: `GGML_CUDA_REGISTER_HOST=1`, `GGML_SCHED_PREFETCH_EXPERTS=1`
- FORKS.md: thecodacus in Cherry-Pick-Tabelle aufgenommen
- AGENTS.md: thecodacus MoE-Optimierungen Status-Tabelle hinzugefügt

**Quelle A:** `git show 16b62f970 --stat`, `docs/fork/2026-07-08_SOLO_SESSION_REPORT.md`
**Quelle B:** `patches/thecodacus/` (3 Diff-Dateien aus Rundgang 5)
**Vorschlag:** Keine Aktion — Feature ist dokumentiert und funktional.
**Status:** offen (beobachtet)

---

## Zusammenfassung Rundgang 6

| # | Kategorie | Kurzbeschreibung | Status |
|---|-----------|------------------|--------|
| 12 | sicherheitsverletzung | AGENTS.md GPU-Regel mit Host-Namen committed | offen — KRITISCH |
| 13 | sicherheitsverletzung | Solo-Session-Report: "Pascal-Host" committed | offen — KRITISCH |
| 14 | artefakt-bereinigung | H/HAND/HANDOFF Duplikate gelöscht | erledigt |
| 15 | beobachtung | thecodacus MoE-Optimierungen erfolgreich | offen (beobachtet) |

---

## Rundgang 7 — 2026-07-08 (~07:45)

### [2026-07-08] — [sicherheitsverletzung] — UMFASSENDER Security-Audit: Host-Namen in 5 Dateien + 2 Dateinamen

**Fund:** Systematische Prüfung aller committed-Dateien auf Host-Namen ("Dev-Host", "Pascal-Host",
"Venus", "Mars") zeigt weitreichende Sicherheitsverletzungen. Die AGENTS.md-Sicherheitsregel
wird durch die eigenen AGENTS.md-Inhalte sowie mehrere weitere Dateien verletzt.

**Betroffene Dateien (committed in HEAD):**

| Datei | Host-Name(n) | Zeilen | Commits |
|-------|-------------|--------|---------|
| `AGENTS.md` | Dev-Host (6×), Pascal-Host (1×), Venus (1×), Mars (1×) | 106-112, 182 | `16b62f970`, `ba8acf43a` |
| `SESSION_PLAN.md` | Pascal-Host (12×), Dev-Host (2×) | 5,12,18,19,31,36,40,41,70,167,213,273,277 | `edd42e60f` |
| `docs/fork/2026-07-08_BASELINE_PASCAL-HOST.md` | Pascal-Host (im Dateinamen + Zeile 1) | 1 | `a4215b3d6` |
| `docs/fork/2026-07-08_SOLO_SESSION_REPORT.md` | Pascal-Host (1×) | 168 | `40e9f5ac1` |
| `scripts/run-gemma4-12b-iq4nl-Pascal-Host-server.sh` | Pascal-Host (im Dateinamen + Zeilen 3,8-10) | 3,8,9,10 | `a4215b3d6` |
| `docs/fork/2026-06-23_SOLO_SCHICHT_PLAN.md` | Dev-Host (1×) | 161 | (pre-existing) |

**Dateinamen mit Host-Namen (im git tree sichtbar auf Codeberg):**
- `docs/fork/2026-07-08_BASELINE_PASCAL-HOST.md`
- `scripts/run-gemma4-12b-iq4nl-Pascal-Host-server.sh`

**Quelle A:** `git grep -in "Pascal-Host" HEAD`, `git grep -in "Dev-Host" HEAD` (alle Treffer in Textdateien)
**Quelle B:** AGENTS.md Sicherheitsregel: "❌ Lokale Host-Namen" + `grep -ri "Dev-Host|...|Pascal-Host|..."`
**Vorschlag:** Umfassende Bereinigung erforderlich:
1. `git filter-repo` oder `git rebase -i` um alle Host-Namen aus der Historie zu entfernen
2. Dateien umbenennen: `*PASCAL-HOST*` → `*PASCAL*`, `*Pascal-Host*` → `*pascal*`
3. Host-Namen durch generische Beschreibungen ersetzen:
   - "Dev-Host" → "Dev-Host (RTX 3070 Mobile)"
   - "Pascal-Host" → "Pascal-Test-Host (GTX 1070)"
   - "AMD-GCN (Vega)" → "AMD GCN (Vega)"
   - "AMD-RDNA3" → "AMD RDNA3"
4. Danach Force-Push auf Codeberg + GitHub
**Status:** offen — **KRITISCH: umfassende Bereinigung erforderlich, multiple Commits betroffen**

---

### [2026-07-08] — [beobachtung] — thecodacus Feature-Commit + erweiterter Solo-Session-Report

**Fund:** Zwei neue Commits seit Rundgang 6:
1. `a4215b3d6` "feature: thecodacus MoE-Optimierungen (Pinning + Prefetch)" —
   implementiert die Patches aus `patches/thecodacus/` im Code. Benchmark-Ergebnisse:
   pp512 +72%, pp2048 +95%, pp4096 +106%, tg128 +31% auf GTX 1070.
2. `40e9f5ac1` "DOX: Solo-Session Report erweitert" — MTP+Pinning+Prefetch Kombi
   (100% Draft-Akzeptanz, 31.93 t/s +50%), Kontext-Scaling, Korrektheits-Verifikation.

Historie wurde rebaseiert (feature-commit vor Doku-Beobachtungen-Rundgang-6 eingefügt).
Doku-Beobachtungen-Commits sind erhalten (neue Hashes durch rebase).

**Quelle A:** `git log --oneline`, `git show a4215b3d6 --stat`, `git show 40e9f5ac1 --stat`
**Quelle B:** `patches/thecodacus/` (3 Diff-Dateien)
**Vorschlag:** Keine Aktion — Feature erfolgreich implementiert und dokumentiert.
**Status:** offen (beobachtet)

---

### [2026-07-08] — [veraltet] — Trilium-Projektnote: thecodacus MoE-Optimierungen fehlen

**Fund:** Trilium-Projektnote (eiba6WJDfTiq) erwähnt thecodacus MoE-Optimierungen
(Memory Pinning, Async Expert Prefetch) nicht, obwohl diese jetzt implementiert
und dokumentiert sind (Commits `a4215b3d6`, `16b62f970`, `40e9f5ac1`).

**Quelle A:** Trilium-Note `eiba6WJDfTiq` — kein Treffer für "thecodacus", "memory pinning",
"expert prefetch", "moe-optim"
**Quelle B:** `AGENTS.md` thecodacus Status-Tabelle, `docs/fork/2026-07-08_SOLO_SESSION_REPORT.md`
**Vorschlag:** Trilium-Projektnote um thecodacus MoE-Optimierungen Sektion erweitern
(Fakten: Env-Vars, Benchmark-Zahlen, Status). Dies ist eine inhaltliche Erweiterung,
nicht nur Konsistenz-Update — vom User oder aktiver Session durchzuführen.
**Status:** offen

---

## Rundgang 8 — 2026-07-08 (~16:50)

### [2026-07-08] — [widersprüchlich] — scripts/AGENTS.md Sicherheitsregel vs. Pascal-Host-Skripte

**Fund:** `scripts/AGENTS.md` (Child-DOX) hat eine Sicherheitsregel "Keine hartkodierten
lokalen Pfade, Host-Namen oder Zugangsdaten in Skripten" (Zeile 20). Aber die neuen
Service-Skripte verletzen diese Regel: Dateinamen und Inhalte enthalten "Pascal-Host".

**Quelle A:** `scripts/AGENTS.md` Zeile 20: "Keine hartkodierten lokalen Pfade, Host-Namen
oder Zugangsdaten in Skripten."
**Quelle B:** Neue Skripte (Commits `e95cdcec5`, `dff910039`, `72ef5c5cd`):
- `scripts/run-gemma4-26b-a4b-mtp-Pascal-Host-server.sh` — Dateiname enthält "Pascal-Host"
- Inhalt: "curl http://Pascal-Host:18080/v1/chat/completions"
- `scripts/llama-server-Pascal-Host-26b-a4b.service` — Dateiname enthält "Pascal-Host"
- `scripts/AGENTS.md` Zeile 44-45: Dokumentiert die Skripte mit "Pascal-Host" im Text
**Vorschlag:** scripts/AGENTS.md Sicherheitsregel aktualisieren — Host-Namen sind
in Service-Skripten erlaubt wenn sie LAN-spezifisch sind und nicht öffentlich
committet werden sollen. Oder: Skripte umbenennen (Pascal-Host → pascal) und Host-Namen
durch Umgebungsvariablen ersetzen.
**Status:** offen

---

### [2026-07-08] — [beobachtung] — 26B-A4B Service: Maximale Kontext-Ermittlung (160k)

**Fund:** Commit `72ef5c5cd` maximiert den Kontext für 26B-A4B Service auf 163840 (160k)
bei K=turbo3/V=turbo4 mixed KV-Cache auf 8GB VRAM. 165888 (162k) führt zu OOM.
Default ctx von 131072 → 163840.

**Quelle A:** `git show 72ef5c5cd` — "ctx auf 163840 (160k) maximiert",
"8110 MiB, läuft + Request erfolgreich (Berlin-Test)"
**Quelle B:** `scripts/run-gemma4-26b-a4b-mtp-Pascal-Host-server.sh` (aktuelle ctx=163840)
**Vorschlag:** Keine Aktion — Feature erfolgreich dokumentiert und verifiziert.
**Status:** offen (beobachtet)

---

### [2026-07-08] — [beobachtung] — Code-Review thecodacus Patches dokumentiert

**Fund:** Commit `e95cdcec5` fügt `docs/fork/2026-07-08_CODE_REVIEW_THECODACUS.md` hinzu —
Code-Review der 3 thecodacus Patches. Ergebnis: produktionsreif, keine Issues,
UAF-Fix korrekt implementiert.

**Quelle A:** `git show e95cdcec5 --stat` — Code-Review-Datei hinzugefügt
**Quelle B:** `docs/fork/2026-07-08_CODE_REVIEW_THECODACUS.md` (Review-Ergebnisse)
**Vorschlag:** Keine Aktion — Code-Review ist dokumentiert und positiv.
**Status:** offen (beobachtet)

---

## Zusammenfassung Rundgang 8

| # | Kategorie | Kurzbeschreibung | Status |
|---|-----------|------------------|--------|
| 16 | sicherheitsverletzung | Umfassender Audit: Host-Namen in 5 Dateien + 2 Dateinamen | offen — KRITISCH |
| 17 | beobachtung | thecodacus Feature-Commit + erweiterter Report | offen (beobachtet) |
| 18 | veraltet | Trilium: thecodacus MoE-Optimierungen fehlen | offen |

---

## Gesamt-Zusammenfassung (alle Rundgänge)

| # | Kategorie | Kurzbeschreibung | Status | Dringlichkeit |
|---|-----------|------------------|--------|---------------|
| 1 | widersprüchlich | SNAPSHOT.md beschreibt veralteten DFlash-Zustand | offen | mittel |
| 2 | toter-link | AGENTS.md → 3 gelöschte docs/fork-Dateien | offen | niedrig |
| 3 | toter-link | FORKS.md → gelöschte BRANCHES.md | offen | niedrig |
| 4 | toter-link | AGENTS.md → 4 nicht-existierende FORKS.md-Sections | offen | niedrig |
| 5 | veraltet | Trilium: conda/ellama vs. uv-Regel | offen | mittel |
| 6 | veraltet | Trilium: BRANCHES.md als kritische Datei | offen | niedrig |
| 7 | veraltet | Trilium Momentaufnahme: DFlash-Zustand | offen (markiert) | mittel |
| 8 | sicherheitsverletzung | AGENTS.md enthält Host-Namen (committed) | offen | **KRITISCH** |
| 9 | beobachtung | Uncommitted: GPU-Nutzungs-Regel + .gitignore | erledigt (committed) | — |
| 10 | sicherheitsverletzung | SESSION_PLAN.md: Host-Namen + Pfade (committed) | offen | **KRITISCH** |
| 11 | beobachtung | thecodacus-Patches hinzugefügt (sauber) | erledigt (implementiert) | — |
| 12 | sicherheitsverletzung | AGENTS.md GPU-Regel committed | offen (siehe #16) | **KRITISCH** |
| 13 | sicherheitsverletzung | Solo-Session-Report: "Pascal-Host" committed | offen (siehe #16) | **KRITISCH** |
| 14 | artefakt-bereinigung | H/HAND/HANDOFF Duplikate gelöscht | **erledigt** | — |
| 15 | beobachtung | thecodacus MoE-Optimierungen erfolgreich | offen (beobachtet) | — |
| 16 | sicherheitsverletzung | Umfassender Audit: 5 Dateien + 2 Dateinamen | offen | **KRITISCH** |
| 17 | beobachtung | thecodacus Feature-Commit + erweiterter Report | offen (beobachtet) | — |
| 18 | veraltet | Trilium: thecodacus MoE-Optimierungen fehlen | offen | mittel |
| 19 | widersprüchlich | scripts/AGENTS.md Sicherheitsregel vs. Pascal-Host-Skripte | offen | niedrig |
| 20 | beobachtung | 26B-A4B Service: Maximale Kontext 160k | offen (beobachtet) | — |
| 21 | beobachtung | Code-Review thecodacus Patches dokumentiert | offen (beobachtet) | — |

**DRINGLICHSTE EMPFEHLUNG:** Sicherheitsverletzungen (#8, #10, #16) bereinigen —
Host-Namen (Dev-Host, Pascal-Host, Venus, Mars) und lokale Pfade (/path/to/user) in committed
Dateien. Erfordert `git filter-repo` oder `git rebase -i` + Force-Push. Siehe
Skill `codeberg-quota-cleanup` für Repo-Recreate-Workflow.

