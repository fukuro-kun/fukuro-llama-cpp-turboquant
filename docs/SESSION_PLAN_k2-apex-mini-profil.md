# SESSION_PLAN: K2-APEX-mini InferenzQuelle-Profil + Langzeit-Stabilität

**Erstellt:** 2026-09-27 (nach K2-Varianten-Eval, Session-Plan-Nachfolge)
**Voraussetzung/User-Entscheid:** Empfehlung APEX-mini angenommen? Falls User andere
Variante wählt, Profil-Werte analog übertragen.

## Ziel
APEX-mini als produktives InferenzQuelle-Profil vorbereiten und die Stabilität
unter realistischer Last verifizieren — inkl. Fork-spezifischem `/v1/decision`.

## Blöcke

**A. Stabilitäts-Langlauf auf Eris (~45 min)**
- Server: `APEX-mini -c 32768 -ngl 44 -ub 512 -fa on -ctk turbo4 -ctv turbo3`
  (produktives Profil; alternativ `-c 131072 -ngl 34` für Long-Context-Variante)
- 20 sequentielle Requests gemischter Länge (wie Qual-Suite, Prompts in
  `/tmp/k2-qual-prompts/`), VRAM + t/s + Fehler loggen
- Erfolg: 0 Crashes/OOMs, t/s-Drift <10 %, keine leeren Antworten

**B. Decision-Endpoint-Kompatibilität (~15 min)**
- `POST /decision` + `/v1/decision` gegen den APEX-mini-Server testen
  (Fork-Feature, muss mit K2-Architektur funktionieren)
- Escalation-Pfad: Entscheid → voller Chat-Request

**C. InferenzQuelle-Profil-Entwurf (~20 min)**
- `hosts.json`-Entwurf für Janus: Modell-ID, Eris:18088-Endpoint, Parameter
  (temp, top_p, max_tokens ≥2048 wegen Thinking; bei DERISKED zwingend
  `repeat_penalty=1.15`)
- **Nicht deployen ohne User-Freigabe** — nur Entwurf committen

## Defaults bei Unklarheit
- temperature=0.7 (K2, nicht Gemma-Regel)
- eris-chat pausieren für Tests, danach restaurieren + /health prüfen
- Bei Instabilität: ngl −2 probieren, dann abbrechen und dokumentieren

## Risiken/Abbruch
- VRAM-Konflikt mit eris-chat → Service stoppen/starten (getestetes Verfahren)
- Decision-Endpoint inkompatibel mit K2 → Bug dokumentieren, nicht im Session
  fixen
