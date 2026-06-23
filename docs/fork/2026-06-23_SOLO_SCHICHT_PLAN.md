# Solo-Schicht Plan: 2026-06-23 (8h)

**Erstellt:** 2026-06-23, auf hydra (RTX 3070, 8GB CUDA)
**Branch:** `feature/sync-atomicbot-2026-06-19` (sauberer Squash auf master)
**Mars:** tg32-Tests laufen im Hintergrund (llama-bench, 2h Timeout pro Test)

---

## Phase 1: Build-Verifikation des Feature-Branch (0-1.5h)

**Ziel:** Beweisen, dass der Squash-Branch kompiliert und funktioniert.

**Arbeiten:**
- `cmake --build build -j$(nproc)` auf hydra (CUDA)
- `./build/bin/test-chat` laufen lassen (Parser-Tests)
- `./build/bin/llama-bench -m <kleines Modell> -p 512 -n 32` als Smoke-Test
- Bei Fehlern: Fix auf dem Branch committen

**Verifikation:**
- [ ] Build exit code 0
- [ ] `test-chat` output: "All tests passed!"
- [ ] `llama-bench` output: t/s-Wert > 0
- [ ] `git log --oneline` zeigt saubere Historie ohne Abbrüche

---

## Phase 2: DiffusionGemma gguf-py Registrierung (1.5-3.5h)

**Ziel:** Das letzte bekannte Feature-Gap schließen. `MODEL_ARCH.DIFFUSION_GEMMA` und zugehörige Tensor-Mappings in gguf-py hinzufügen.

**Arbeiten:**
1. `gguf-py/gguf/constants.py`: `MODEL_ARCH.DIFFUSION_GEMMA` zum IntEnum + Namen-Mapping
2. `gguf-py/gguf/tensor_mapping.py`: DiffusionGemma-Tensoren mappen (`pkv_k`, `pkv_v`, `canvas_length` etc.)
3. `gguf-py/gguf/quants.py`: Falls spezielle Quantisierungs-Typen nötig
4. `convert_hf_to_gguf.py`: DiffusionGemma-Model-Klasse hinzufügen (wenn nötig)
5. Test: `python3 -c "from gguf import MODEL_ARCH; print(MODEL_ARCH.DIFFUSION_GEMMA)"`

**Verifikation:**
- [ ] `python3 -c "from gguf import MODEL_ARCH; print(MODEL_ARCH.DIFFUSION_GEMMA)"` funktioniert
- [ ] `grep DIFFUSION_GEMMA gguf-py/gguf/constants.py` zeigt den Eintrag
- [ ] Build mit den Änderungen: `cmake --build build` erfolgreich
- [ ] Commit auf Branch mit deutscher Nachricht

---

## Phase 3: Mars Test-Ergebnisse einsammeln + Trilium-Doku (3.5-5h)

**Ziel:** tg32-Scaling-Ergebnisse auf mars abholen, in Trilium dokumentieren.

**Arbeiten:**
1. Mars-Test-Log abholen: `/tmp/mars_tg32_bench_results.log`
2. Ergebnisse extrahieren (pp t/s, tg t/s für 180k, 188k, 196k, 262k)
3. In Trilium `qTcWDvZuCCsl` (LLM-Benchmarks — Mars) eintragen
4. §5.8 in `SWumEN7WOXBI` (Vulkan Performance-Klippe) aktualisieren
5. TTT-Eintrag für die Session erstellen

**Verifikation:**
- [ ] 4 tg32-Werte in Trilium-Tabelle (180k, 188k, 196k, 262k)
- [ ] TTT-Eintrag existiert mit Fazit
- [ ] `SWumEN7WOXBI` §5.8 zeigt aktualisierte Werte

---

## Phase 4: Sync-Merge nach master + Push (5-6.5h)

**Ziel:** Verifizierten Feature-Branch nach master mergen und pushen.

**Arbeiten:**
1. `git checkout master && git merge feature/sync-atomicbot-2026-06-19` (fast-forward)
2. Build-Verifikation auf master (quick smoke test)
3. `git push origin master` (Codeberg)
4. `git push github master` (Mirror)
5. FORKS.md §7 aktualisieren

**Verifikation:**
- [ ] `git log --oneline master` zeigt Sync-Commit als HEAD
- [ ] `git push` exit code 0
- [ ] Codeberg-Web-UI zeigt den neuen Commit
- [ ] `./build/bin/llama-server --version` funktioniert auf master

---

## Phase 5: Aufräumen + TTT (6.5-8h)

**Ziel:** Wissensstand sichern, offene Punkte dokumentieren.

**Arbeiten:**
1. `feature/diffusion-gemma-v2` als Squash neu aufbauen (korrupte Historie entfernen)
2. Obsolete Remote-Tracking-Branches aufräumen
3. `git gc` versuchen (sollte nach Reparatur funktionieren)
4. TTT-Eintrag: komplette Session dokumentieren
5. AGENTS.md aktualisieren, falls neue Erkenntnisse
6. Offene TODO-Liste für nächste Session erstellen

**Verifikation:**
- [ ] `git fsck` zeigt keine fehlerhaften Verknüpfungen mehr
- [ ] `git gc --prune=now` exit code 0
- [ ] TTT-Eintrag mit Fazit und offenen Punkten
- [ ] Alle Branches in `git branch -v` sind sauber

---

## Phase 6 (Stretch): DiffusionGemma Live-Test (falls Zeit übrig)

**Ziel:** DiffusionGemma-Modell mit echtem Text-Output testen — ca. 1000 Token Ausgabe zum Thema "KI".

**System:** Mars (AMD 760M, Vulkan, 32GB RAM — mehr VRAM via GTT) bevorzugt. Hydra (RTX 3070, 8GB CUDA) als Fallback.

**Arbeiten:**
1. Auf mars: DiffusionGemma GGUF laden (falls verfügbar)
2. `llama-cli -m <diffusion-gemma.gguf> -p "Schreibe einen Aufsatz über Künstliche Intelligenz" -n 1000`
3. Falls auf mars nicht möglich (kein Modell): hydra mit kleinstem verfügbaren DiffusionGemma-Modell
4. Output prüfen: Ist der Text kohärent? Entspricht er dem Diffusion-Decoder-Pattern?
5. Ergebnisse in Trilium dokumentieren (unter DiffusionGemma-Note)

**Verifikation:**
- [ ] `llama-cli` startet ohne Crash
- [ ] Modell lädt korrekt (alle Layer)
- [ ] Output: >100 Token generiert (Ziel: 1000)
- [ ] Text ist kohärent (nicht random tokens)
- [ ] t/s-Wert dokumentiert
- [ ] Trilium-Eintrag unter DiffusionGemma vorhanden

**Risiko:** DiffusionGemma benötigt möglicherweise spezielle GGUF-Dateien, die noch nicht konvertiert wurden (hängt von Phase 2 ab). Falls keine GGUF verfügbar: Phase 6 entfällt, stattdessen mehr Zeit für Phase 5.

---

## Risiko-Bewertung

| Phase | Risiko | Mitigation |
|-------|--------|------------|
| Build | Kompilierungsfehler durch Squash | Falls Fehler: gezielt fixen, nicht abbrechen |
| gguf-py | API-Änderungen in gguf-py | Erst `constants.py` lesen, dann minimal-invasiv ändern |
| Mars | Tests timeout (>2h pro Test) | Wenn timeout: kleinere Kontextgrößen als Fallback |
| Merge | master könnte neue Commits haben | `git fetch origin` vor Merge, rebase falls nötig |
| Push | Codeberg/GitHub nicht erreichbar | Push kann auf nächste Session verschoben werden |
| Diffusion | Keine GGUF verfügbar | Phase 6 entfällt, mehr Zeit für Phase 5 |

## Was NICHT getan wird

- Keine `git push --force` auf master
- Keine Löschung von `feature/diffusion-gemma-v2` ohne Squash-Backup
- Keine Änderungen an AGENTS.md-Verträgen ohne DOX-Durchlauf
- Keine Commits mit Host-Namen oder privaten Daten

---

*Plan erstellt: 2026-06-23*
*Gültig für: Solo-Schicht 2026-06-23*
*Nächste Session: Verifikation der Checkboxen, offene Punkte übernehmen*
