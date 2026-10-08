# DOX-Audit 2026-10-08

Audit der Dokumentations-Hierarchie (AGENTS.md-Kette, kanonische
`docs/`-Dokumente, lokale Markdown-Verweise). Basis-Worktree:
Branch `feat/flashnext-mtp-validation` @ `ab6e9dfcea`.
Zweitworktree: `feat/flashnext-topk-workspace` @ `62fbee475b`.

Der Audit ist **dokumentarisch** — keine Builds, keine Tests, keine
Benchmarks, keine Laufzeit-Probes wurden dafuer ausgefuehrt.

---

## 1. Umfang und Check-Matrix

| Bereich | Art der Pruefung | Status |
|---|---|---|
| Root-`AGENTS.md` | Vollstaendige inhaltliche Pruefung gegen aktuellen Code/Workflow | geprueft, refaktoriert |
| Alle Child-`AGENTS.md` (initiale 8 DOX-Vertraege: Root + 7 Children inkl. Grandchild `tools/parallel-decision/`; dazu neu angelegtes `docs/AGENTS.md` → 9 Dateien gesamt) | Inhaltliche Pruefung: Vertraege, Indizes, Test-/Skript-Namen, Verweise | geprueft, korrigiert |
| Kanonische Fork-Docs (`docs/speculative.md`, `FORKS.md`, `README.md`, `docs/fork/ROADMAP.md`, `CHANGELOG.md`) | Inhaltliche Pruefung der aktiven Behauptungen (Flags, Typen, Features) | geprueft, korrigiert |
| Historische/Archiv-Docs (`MTP.md`, `NEXTN.md`, die zwei konkreten Dateien `docs/development/pipeline-depth-2-pure-overlap.md` und `docs/development/gemma4-assistant-tensor-inventory.md`, `docs/SNAPSHOT.md`, `docs/fork/CHANGELOG.md`, `TURBOQUANT_UPSTREAM_MERGE.md`, `SESSION_PLAN.md`, `docs/qwen38-eval/TESTKONZEPT.md`) | Kontext-Markierung (`HISTORISCH`/Datums-Header), keine inhaltliche Neubewertung | markiert |
| Aktive Session-/Plan-Docs (`docs/fork/plans/SESSION_PLAN_20261006_*`, `docs/qwen38-eval/ENGINE_EVAL_20261007.md`, `docs/DOKU_BEOBACHTUNGEN.md`) | Konsistenz-Pruefung gegen git-Stand und bereits dokumentierte Befunde | korrigiert |
| Upstream-/Vendor-Markdown (222 getrackte `.md` gesamt) | **nur strukturell**: Link-Existenz, Child-Index-Aufloesung — **keine** semantische Vollpruefung der Inhalte | strukturell geprueft |
| Alle lokalen Markdown-Verweise | Datei-/Anker-Existenz, relative Aufloesung, Root-absolute GitHub-Pfade, Anker-Slugs | geprueft |
| Topk-Worktree-DOX | Teil-Korrektur (`ggml/AGENTS.md`, `tests/AGENTS.md`, Root-Hinweis) | teilweise, Follow-up offen |

**Wichtige Abgrenzung:** Die Pruefung aller upstream/historischen
Markdown-Dateien war rein strukturell (Verweise, Existenz). Sie ist
kein Nachweis, dass deren Inhalt zum aktuellen Code passt.

## 2. Korrigierte Befunde (Auswahl, file:line im Rohbericht)

- **Child-Index:** `src/AGENTS.md` markierte `src/models/` als `[x] Aktiv`
  ohne vorhandene DOX-Datei → als hier abgedeckt gefuehrt.
- **Root-`AGENTS.md`:** entfernte Host-/Deployment-Tabellen, Sessions-
  Status und tote Verweise auf geloeschte Docs/`FORKS.md`-Sektionen;
  `pkill`-Empfehlung durch gezieltes PID-Handling ersetzt; CPU-only-
  Verifikation explizit erlaubt; Remote-Sync auf gepinnten Commit/
  Feature-Branch-Worktree korrigiert.
- **Speculative-Docs:** `docs/speculative.md` komplett neu als aktuelle
  Referenz; alte Bezeichner (`mtp`, `nextn`, `eagle3` als Typen, `--mtp-head`,
  `--draft`) und nicht mehr existierende APIs aus aktiven Docs entfernt.
  Aktuelle Map: `none`, `draft-simple`, `draft-eagle3`, `draft-mtp`,
  `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`,
  `ngram-cache` (Quelle: `common/speculative.cpp`).
- **Test-Namen:** `test-backend-buffer`/`test-quantize`/`test-tq` (nicht
  existent) → `test-backend-ops`, `test-quantize-fns`, `test-turbo-quant`;
  `test-speculative-mtp.cpp` ist Quelle ohne CMake-Registrierung und
  wird nicht mehr als buildbares Target beworben.
- **MoE-Cache:** Policies sind `lru`/`heuristic` (Alias `0`/`1`);
  `set-assoc`/`workload` entfernt. `GGML_CUDA_MOE_CACHE` ist Default-aktiv.
- **Aktive tote Verweise repariert:** `tools/server/README.md`
  (Embeddings-Beispiel, `chat.mjs`/`chat.sh`-Abschnitt),
  `grammars/README.md`, `tools/server/README-dev.md`,
  `tools/completion/README.md`, `docs/backend/CANN.md`,
  `docs/qwen-udt/RUNBOOK.md`, `docs/autoparser.md` (Common-Root-Verweise
  brauchen `../`), `NEXTN.md`, `.pi/gg/SYSTEM.md`.
- **Historische Markierungen** ergaenzt: `MTP.md`, `NEXTN.md`,
  `docs/development/pipeline-depth-2-pure-overlap.md`,
  `docs/development/gemma4-assistant-tensor-inventory.md` (Stage-0-
  Inventar mit alten `mtp.pre_projection`-Namen; aktueller Loader nutzt
  `nextn_proj_pre`/`nextn_proj_post`), `SESSION_PLAN.md`,
  `TURBOQUANT_UPSTREAM_MERGE.md`, `docs/SNAPSHOT.md`,
  `docs/fork/CHANGELOG.md`; Datums-Kontext in
  `docs/qwen38-eval/TESTKONZEPT.md` und Epochen-Block in
  `docs/qwen38-eval/ENGINE_EVAL_20261007.md`.
- **Plan-Konsistenz** (`docs/fork/plans/SESSION_PLAN_20261006_*`):
  §10.0 M-Zeile (Restore-Fix abgeschlossen vor `ab6e9dfcea`, q8-Paritaet
  und Gesamt-M-Gate offen), §10.8 P1b als eigene Commits getrennt von
  S1/S2/M, §10.9 Restore-/Hybrididx-Befunde als Prae-Fix formuliert,
  f16-Kontrollen vs. q8_0-Negativkontrolle getrennt.

## 3. Link-Klassifikation

- Die Rohscan-Zahl "11 harte tote Verweise" enthielt auch **forensisch
  zitierte Pfade** (z. B. in `DOKU_BEOBACHTUNGEN.md` und
  `TURBOQUANT_UPSTREAM_MERGE.md`), nicht nur aktive Navigations-Links.
  Historische Zitate wurden als Inline-Pfade belassen bzw. deaktiviert.
- **Root-absolute GitHub-Pfade** (`/tree/...`, `/blob/...` relativ zum
  Repo) sind valide und wurden nicht als Fehler gezaehlt.
- **Basename-Re-Resolution ist kein Fix:** ein relativer `common/`-Link
  aus `docs/` ist kaputt, auch wenn eine Datei gleichen Namens
  existiert — korrekt ist `../common/...`.
- GitHub-Doppelueberschrift-Anker (`-1`-Suffix) sind valide
  GitHub-Slugs, keine Fehler.

## 4. Offene Befunde und Follow-ups

| Befund | Disposition |
|---|---|
| Trilium-Watcher-/Notiz-Befunde aus `DOKU_BEOBACHTUNGEN.md` | **nicht** als geloest markiert — kein Live-Trilium-Lesegriff in dieser Runde |
| `test-speculative-mtp.cpp` nicht registriert | Follow-up: Legacy-Testquelle auf aktuelle API-Kompatibilitaet pruefen/portieren; erst danach eine Registrierung entscheiden. |
| Bench-Runner P2: Ctrl-C-Final-Metadata, Terminate-/Kill-Timeout, Fingerprint-Fehlerpfad | offener Review-Punkt (in `scripts/AGENTS.md` vermerkt), kein Fix-Anspruch |
| Topk-Worktree: `GGML_CUDA_TOP_K_MAX_ROWS` registriert (`test-cuda-top-k-baseline`/`row-chunks`), CUDA-Ausfuehrung ausstehend | Gate offen, siehe `ggml/AGENTS.md` im Worktree |
| Cross-Branch-DOX-Drift (Topk-Root/`common` aeltere Hauptlinie) | Follow-up, kein Merge in dieser Runde |
| Legacy-Docs enthalten teils Host-/Pfad-Details | nicht global gesaeubert; neue Texte ohne private Daten |
| `BRANCHES.md` wurde in `f86ac47a66` geloescht | Verweise auf §7 von `FORKS.md` umgeleitet; forensische Zitate bleiben Inline-Pfade |
| q8_0-Spec-Degradation, M-Gesamtgate, S1/S2, B5-Konsolidierung | offen, siehe Plan §10.0/§10.9 und `ROADMAP.md` |
| Restliche upstream Link-Drifts in historischen Zitaten | intentional unveraendert (Kontextbeleg) |

## 5. Beweis und Reproduzierbarkeit

- Checks: `git ls-files '*.md'`, Python-Linkscanner
  (Existenz/Anker/relative Aufloesung), `git status`/`git diff --check`,
  CMake-Registrierungs-Grep fuer Test-Namen, Source-Grep fuer
  speculative Map/Flags/APIs.
- **Triage-Limitation:** die restlichen automatisierten Flags
  (Inline-Code-Referenzen in AGENTS-Dateien: ~109 Meldungen) wurden
  manuell triagiert/**stichprobenartig** geprueft, nicht zeilenweise
  verifiziert — sie bestehen ueberwiegend aus Endpunkt-Namen,
  Basename-Kurzformen, Kommandos und `<name>`-Platzhaltern. Einzelne
  echte Funde wurden korrigiert; eine Vollabdeckung wird nicht
  behauptet. Ebenso ist die Markdown-Pruefung strukturell — keine
  semantische Verifikation aller 222+ Docs gegen den Code.
- Roh-Artefakte und Skripte bleiben lokal erhalten; Pfade sind in
  `LOKAL.md` (gitignored) hinterlegt — bewusst nicht in dieser
  oeffentlichen Datei.
- Keine neue Messung, kein Test-Lauf, kein PR-/Remote-Status-Claim
  wurde in diesem Audit erzeugt.

## 6. Aktualisierungs-Trigger

Diese Datei wird angefasst, wenn: ein weiterer DOX-Audit- oder
Korrektur-Durchlauf stattfindet, ein offener Follow-up aus §4
geschlossen wird, oder die Check-Matrix um neue Bereiche erweitert
wird.
