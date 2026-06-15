# DiffusionGemma Integration — Entscheidungsprozess und Gedankengang

> Datum: 2026-06-15
> Entscheider: fukuro + KI-Agent
> Kontext: Integration von PR #24423 (DiffusionGemma) in fukuro-llama-cpp-turboquant

---

## Ausgangslage

PR #24423 implementiert DiffusionGemma (block text-diffusion MoE auf Gemma-4 Backbone) fuer llama.cpp. Der PR ist noch im Draft-Status, wurde aber von unsloth als funktionierendes Beispiel bereitgestellt.

### Unser Fork
- `fukuro-llama-cpp-turboquant` — Fork mit TurboQuant KV-Kompression, Gemma4 MTP, Qwen NextN
- Stand: ~6 Wochen hinter upstream (ca. 532 Commits Unterschied)
- **Kritisch:** Unser Fork hat die alte **monolithische Architektur** (ein `llama_model` mit Switch-Statements), waehrend upstream seit dem 4. Mai 2026 auf eine **Klassen-Hierarchie** (`llama_model_base`) umgestellt hat.

---

## Betrachtete Optionen

### Option A: Refactor zuerst, dann PR

**Ansatz:** Den upstream-Architektur-Refactor (Commit `994118a18`, 4. Mai 2026) zuerst mergen, dann PR #24423 1:1 uebernehmen.

**Argumente dafuer:**
- Sauberster technischer Weg
- Kein Portierungsaufwand fuer DiffusionGemma
- Zukuenftige Upstream-Syncs werden einfacher
- Keine "technische Schuld"

**Argumente dagegen:**
- Refactor aendert 9200 Zeilen in `llama-model.cpp` allein
- 100+ Dateien werden umgeschrieben/umbenannt
- Unsere Fork-Features (TurboQuant, MTP, NextN) greifen direkt in die aenderten Code-Pfade ein
- Cherry-Pick-Test schlug sofort fehl (Konflikte mit uncommitted DiffusionGemma-Aenderungen)
- **Schaetzung:** Wochen an Konfliktloesung + Regressionstests

**Bewertung:** Theoretisch ideal, praktisch zu riskant und zeitintensiv.

---

### Option B: Monolithischer Port (gewaehlt)

**Ansatz:** Den DiffusionGemma-Graph-Builder und die unterstuetzenden Strukturen in unsere monolithische Architektur portieren. Die Logik bleibt identisch, nur die "Verpackung" (Klassen-Hierarchie vs. Switch-Statements) wird angepasst.

**Argumente dafuer:**
- Kontrollierbarer Aufwand (4-5 Stunden geschätzt)
- Isoliertes Risiko — bei Fehlern ist nur DiffusionGemma betroffen
- Schnelles Ergebnis — DiffusionGemma läuft heute
- Keine Gefahr fuer existierende Features (TurboQuant, MTP, NextN)
- Rückgängig machbar

**Argumente dagegen:**
- Technische Schuld — jeder Upstream-Sync erfordert manuelles Mergen
- Nicht "sauber" im architektonischen Sinne
- Erfordert späteren Refactor, wenn upstream stabil ist

**Bewertung:** Pragmatisch, sicher, schnell. Der beste Kompromiss fuer unsere Situation.

---

### Option C: Minimaler Merge (Lobotomie)

**Ansatz:** Nur die Infrastruktur behalten (Konvertierungsskript, GGUF-Definitionen, CLI-Flags), aber den Graph-Builder (der eigentliche Inference-Code) weglassen.

**Argumente dafuer:**
- Minimaler Aufwand (1 Stunde)
- Kein Risiko

**Argumente dagegen:**
- DiffusionGemma läuft nicht — man kann nur GGUFs erstellen, nicht inferieren
- "Lobotomie" — die Haelfte der Funktionalitaet fehlt

**Bewertung:** Nicht akzeptabel — wir wollen das Modell verwenden.

---

## Der entscheidende Moment

Der KI-Agent tendierte zunaechst zu Option A (Refactor zuerst), weil es der "saubere" Weg ist. fukuro stellte die entscheidende Frage:

> "Vielleicht macht eine umgekehrte Reihenfolge mehr Sinn ... erst die Architektur-Aenderung, dann der straight-forward merge?"

Diese Frage fuehrte zu einer tieferen Analyse. Der KI-Agent testete den Cherry-Pick des Refactor-Commits und fand:

- 9200 Zeilen Aenderung in einer einzigen Datei
- 100+ Dateien betroffen
- Datei-Umbenennungen (gemma4-iswa.cpp → gemma4.cpp, etc.)
- Direkte Konflikte mit unseren uncommitted DiffusionGemma-Aenderungen

**Die Erkenntnis:** Der Refactor ist nicht nur gross, sondern **strukturell destruktiv** fuer unseren Fork. Er wuerde Wochen an Arbeit erfordern, um TurboQuant, MTP und NextN wieder zum Laufen zu bringen.

---

## Die Entscheidung

**Wir waehlen Option B (Monolithischer Port).**

**Begruendung:**
1. **Zeit:** fukuro moechte DiffusionGemma *jetzt* testen und verwenden, nicht in Wochen.
2. **Risiko:** Der monolithische Port ist kontrollierbar. Der Refactor wuerde unseren Fork fuer Wochen instabil machen.
3. **Pragmatismus:** Der PR ist noch Draft. Wenn er final wird und upstream stabil ist, koennen wir immer noch den Refactor nachholen.
4. **Zukunft:** Wir dokumentieren den Port so, dass er spaeter auf die neue Hierarchie migriert werden kann.

---

## Implementierungsplan (Option B)

### Phase 1: Struktur vorbereiten
1. `src/models/models.h` — Forward declaration `llm_build_diffusion_gemma`
2. `src/llama-model.cpp` — 4 `case LLM_ARCH_DIFFUSION_GEMMA:` einfuegen
3. `src/CMakeLists.txt` — `diffusion-gemma.cpp` kompilieren

### Phase 2: Graph-Builder portieren
4. `src/models/diffusion-gemma.cpp` — `llama_model_diffusion_gemma` Klasse entfernen
5. Alle `dynamic_cast` durch direkten Zugriff auf `model` ersetzen
6. Diffusion-spezifische Felder (`canvas_length`, `pkv_k`, `pkv_v`, etc.) in `llama_model.h` ergaenzen

### Phase 3: Build & Test
7. CMake-Build
8. Debuggen
9. CPU-Offload-Test auf hydra

---

## Offene Punkte

- [ ] `llm_graph_input_sc` — Self-conditioning Input
- [ ] `llm_graph_input_attn_diffusion_decode` — Decode-Phase Mask
- [ ] `dg_ensure_sc_embT` / `dg_ensure_sc_dev` — SC Helper-Funktionen
- [ ] C-API Funktionen: `llama_diffusion_set_sc`, `llama_diffusion_set_device_sc`

---

## Langfristige Perspektive

Wenn upstream den PR finalisiert und die Architektur stabil ist:
1. Gezielter Sync des Refactor-Commits `994118a18`
2. Migration von DiffusionGemma auf `llama_model_diffusion_gemma : public llama_model_base`
3. Entfernung des monolithischen "Shim"-Codes

Bis dahin: **Monolithischer Port, getestet, dokumentiert, funktionsfaehig.**

---

## Signatur

Entscheidung getroffen von: fukuro + KI-Agent
Datum: 2026-06-15
Verbindlich fuer: DiffusionGemma Integration in fukuro-llama-cpp-turboquant
