# SESSION_PLAN: DERISKED Retest mit max_tokens=8192 (Auto-Fortsetzung)

**Erstellt:** 2026-09-27 ~06:00 (Auto-Fortsetzung der K2-Eval-Session)
**Ziel:** Prüfen ob DERISKED-IQ3_Ms 6/11 leere Antworten ein Artefakt des
4092-Token-Caps waren. Nutzer hatte Plan auf 8184 korrigiert — Lauf nutzte 4092.

## Schritte

- A) 6 leere DERISKED-Prompts (a1,a4,c2,c4,c5,c6) mit max_tokens=8192 neu fahren,
  Config ngl=42/ctx32k wie gehabt. Erwartung bei echten Loops: auch bei 8192 leer.
- B) Kontrollprompt a2 (vorher gut) bei 8192 → Regression-Check.
- C) Ergebnis in Trilium-Modellnotiz nachziehen (Zahl ändern falls anders).

## Defaults

- Eris, eris-chat bleibt pausiert bis fertig, danach restaurieren.
- Abbruch: Load-Fail → ngl 41 retry.
EOF
