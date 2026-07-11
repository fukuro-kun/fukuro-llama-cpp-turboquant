# SESSION_PLAN 2026-07-13 — E4B-Crash + Row-Packing + Tier 2 Eval

**Erstellt:** 2026-07-12 (Pre-Solo Planung)
**Solo-Session:** geplant für nächste 14h Schicht
**Skills:** solo-session, fork-speed-research, code-review, ttt

---

## Session-Ziel

1. E4B+MTP 8GB Crash untersuchen und lösen (oder Workaround finden)
2. #35 Row-Packing DMMV implementieren (+10-20% DMMV auf Mars)
3. Batch-Evaluation offener Tier 2 Items
4. #45 CUDA Streams QKV (wenn Zeit bleibt)

---

## Entscheidungen (Interview-Protokoll)

| Frage | Entscheidung | Begründung |
|-------|-------------|------------|
| Fokus-Reihenfolge? | E4B-Crash → #35 → Batch-Eval → #45 | User-Vorgabe: erst Crash lösen, dann Code, dann Doku, dann weiterer Code |
| Server stoppen? | Services deaktivieren (systemctl --user disable/stop), später reaktivieren | User: "die services einfach für den testzeitraum deaktivieren, sonst starten sie sich immer wieder neu" |
| Uranus offline? | NEIN — Uranus ist online | User-Korrektur: "uranus ist online!" → TP-Items nicht skippen, E4B auf 16GB testbar |
| Code-Review? | Pflicht bei nicht-trivialen Änderungen | User: "A" (Ja, Pflicht) — code-review Skill laden, Subagent prüft vor Commit |
| Abbruch zwischendrin? | NEIN — "du bist eine Maschine, kein Mensch!" | User will keine Zwischenstopps, alle Phasen durchziehen |

---

## Phase 1: E4B+MTP Crash Untersuchung (~3-4h)

### Problem
E4B's PLE-Architektur (Per-Layer Embeddings) erzeugt variable V-Embedding-Größen pro Layer.
Das bricht FlashAttention (fattn.cu:110), turbo KV-Cache benötigt FA, ohne turbo → OOM auf 8GB.

### Schritte
1. ☐ **fattn.cu:110 lesen** — verstehen was genau crasht (PLE variable V-Embeddings)
2. ☐ **E4B Modell-Struktur analysieren** — welche Layer haben welche V-Embedding-Größen?
3. ☐ **Test auf Uranus (16GB RTX 4060 Ti)** — E4B+MTP mit f16 KV, kein turbo, -c 8192
   - Wenn genug VRAM → Workaround gefunden (16GB minimum)
4. ☐ **Test auf Hydra (8GB) mit minimalem Kontext** — -ngl 99 + f16 + -c 2048
   - Vielleicht reicht minimaler KV-Cache ohne turbo
5. ☐ **FA-Fix Versuch** — PLE-Layer pad/truncate vor FA (wenn machbar)
   - fattn.cu modifizieren: variable V-Embeddings auf max-Layer-Größe padden
   - Build + Test auf Hydra
6. ☐ **Code-Review** für FA-Fix (falls implementiert)
7. ☐ **Doku aktualisieren** — E4B_MTP_8GB_CRASH.md mit Ergebnissen

### Verifikation
- Vorher: E4B+MTP crasht auf 8GB mit FA+turbo
- Nachher: Entweder Fix (E4B+MTP läuft) oder dokumentierter Workaround (16GB minimum / minimaler Kontext / Ollama)

### Defaults bei Blockern
- FA-Fix zu komplex → Workaround dokumentieren (16GB oder Ollama)
- Uranus-Test schlägt fehl → auf Hydra minimalen Kontext testen
- Alle Tests fehlschlagen → Root Cause dokumentieren, als "known limitation" markieren

---

## Phase 2: #35 Row-Packing DMMV (~4-6h)

### Problem
DMMV (Dequantize-MatVec) Shader verarbeiten eine Row pro Workgroup. Zwei Rows pro Workgroup
könnten Workgroup-Count halbieren und Akkumulator-Footprint verdoppeln.

### Schritte
1. ☐ **mul_mat_vec_base.glsl lesen** — aktuelle Shader-Struktur verstehen
2. ☐ **mul_mat_vec_q4_k.comp analysieren** — spezifischer K-quant Shader
3. ☐ **Row-Packing Prototyp** — zwei Rows pro Workgroup in mul_mat_vec_base.glsl
4. ☐ **Build auf Mars** — Vulkan-Shader kompilieren
5. ☐ **Benchmark auf Mars** — pp128/pp512 mit Q4_K Modell, vor/nach Row-Packing
6. ☐ **Code-Review** — Subagent prüft Shader-Änderung
7. ☐ **Commit + ROADMAP aktualisieren**

### Verifikation
- Vorher: DMMV mit 1 Row/Workgroup → t/s Baseline
- Nachher: DMMV mit 2 Rows/Workgroup → t/s Vergleich, Ziel +10-20%

### Defaults bei Blockern
- Shader kompiliert nicht → auf 1-2 Quants beschränken (Q4_K zuerst)
- Kein Speedup → als ❌ markieren, Erkenntnis dokumentieren
- Regression → revert, als ❌ markieren

---

## Phase 3: Batch-Evaluation Tier 2 (~3-4h)

### Items zu evaluieren
| # | Item | Methode |
|---|------|---------|
| 8 | Mixed Precision KV Cache | git log, commit e889fbd prüfen |
| 15 | PipeShard | arXiv:2604.26334 lesen, Kompatibilität prüfen |
| 16 | Vulkanised shmem-staging | upstream Status prüfen |
| 34 | UBBoost | Discussion #23262 lesen, ubatch-Code analysieren |
| 36 | Auto Param Fitting TP | PR #22950 prüfen, Uranus-Kompatibilität |
| 37 | LFRU Expert Caching | vLLM commit 71ed1fc, thecodacus-Integration prüfen |
| 38 | Conf-KV | arXiv:2605.24786, TurboQuant-Kompatibilität |
| 40 | MoE Load Balancing | Konzept-Analyse, thecodacus-Erweiterung |

### Schritte
1. ☐ **Pro Item:** git log/grep (bereits im Fork?) → ~10-15 min
2. ☐ **Bei "nicht im Fork":** PR/Paper analysieren (Subagent) → ~20-30 min
3. ☐ **ROADMAP-Status aktualisieren** pro Item
4. ☐ **Commit**

### Verifikation
- Alle 8 Items haben ✅/❌/⏭️ Status mit Begründung

---

## Phase 4: #45 CUDA Streams QKV (Restzeit, ~2-3h)

### Schritte
1. ☐ **GGML_CUDA_GRAPH_OPT recherchieren** — was existiert bereits?
2. ☐ **QKV-Projection Code analysieren** — wo werden Q/K/V getrennt?
3. ☐ **Prototyp: CUDA Streams für Q/K/V** — falls machbar
4. ☐ **Build + Benchmark auf Styx** — falls Prototyp kompiliert
5. ☐ **Code-Review + Commit**

### Defaults
- Zu komplex für Restzeit → Recherche dokumentieren, als SESSION_PLAN_45.md speichern

---

## Recherche-Strategien

| Problemtyp | Recherche-Strategie |
|------------|---------------------|
| Unbekannter Code-Pfad (fattn.cu) | `read` + `grep` + Subagent (`subagent_explore`) |
| Vulkan Shader-Struktur | `read` mul_mat_vec_base.glsl, vergleiche mit coopmat2 Shadern |
| PR-Status verifizieren | `git log --all --oneline \| grep` + webfetch GitHub API |
| arXiv-Paper | arxiv-mcp Server (Subagent) |
| Uranus-Verfügbarkeit | `ssh uranus` testen |

## Recherche-Fallbacks

| Block | Mögliches Problem | Recherche-Fallback |
|-------|-------------------|-------------------|
| E4B FA-Fix | PLE-Architektur zu komplex | Workaround dokumentieren, Ollama-Empfehlung beibehalten |
| Row-Packing Shader | Shader kompiliert nicht | Auf Q4_K beschränken, andere Quants später |
| Tier 2 Item unklar | PR existiert nicht mehr | Als "unverifiziert" markieren, webfetch versuchen |
| Uranus nicht erreichbar | SSH-Key/Netzwerk-Problem | Auf Hydra ausweichen (8GB) |

## Offene Fragen (nicht im Interview geklärt)

- E4B auf Uranus: Ist das QAT-Modell dort verfügbar? (muss getestet werden)
- Row-Packing: Welche Quants profitieren am meisten? (Q4_K zuerst, dann Q5_K/Q6_K)
- CUDA Streams: Ist GGML_CUDA_GRAPH_OPT bereits aktiv im Fork? (muss geprüft werden)
