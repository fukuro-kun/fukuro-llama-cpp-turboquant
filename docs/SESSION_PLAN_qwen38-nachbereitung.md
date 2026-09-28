# SESSION_PLAN: Qwen3.8-Eval Nachbereitung + Resttests

> **Ziel:** Die abgeschlossene Kampagne (`docs/qwen38-eval/RANKING.md`)
> nachbereiten: Doku in Trilium verankern, die zwei offenen
> Hardware-/Modellfragen klären, Produktions-Policy aus den Befunden
> ableiten.
>
> **Stand der Empfehlung:**
> Aligned = ISTA GSQ-RCO IQ3_S-mtp · Uncensored = Swift-1.5 UD-Q3_K_XL-MTP.
> Diese Session ändert die Empfehlung nur, wenn ein Resttest sie widerlegt.

## Block A — Doku (P1)

- [ ] Trilium: Benchmark-Ergebnisse eintragen. Wegweiser-Note
      `8cIsjKEhz7Fd` lesen, passende Modell-/Eval-Note finden oder
      thematische Note "Qwen3.8-27B-Eval" anlegen — NICHT in
      Modell-Info-Notes schreiben.
- [ ] Inhalt: Ranking-Tabelle, Delivery-Raten, Eris-Grenze ~13,1 GiB
      @98k+MTP, xhigh-Failure-Mode, Seed-Abhängigkeit.
- [ ] TTT-Eintrag für die Session (war signifikant: ~13h Kampagne,
      192 Runs, 2 Hosts).

## Block B — -ctkd/-ctvd Test (P2, könnte AP retten)

- [ ] Frage: Rettet Draft-KV-Quantisierung (`-ctkd turbo4 -ctvd turbo3`,
      im Fork implementiert, `server-context.cpp:1124`) AP-IQ4_XS auf
      Eris @98k+MTP? OOM war Draft-KV-Alloc (384 MiB über Limit).
- [ ] Ablauf: eris-chat pausieren → AP mit `-ctkd/-ctvd` + sonst
      identischen Flags laden → task1 medium laufen lassen → MTP-
      Acceptance vergleichen (Draft-KV-Quant könnte Acceptance senken).
- [ ] Erfolg = Start + task1 delivered + Acceptance-Verlust < ~5pp.
      Ergebnis in `results.jsonl` + RANKING.md-Caveat aktualisieren.
- [ ] eris-chat.service danach wiederherstellen + verifizieren.

## Block C — Orcarouter-Gate (P2, braucht User)

- [ ] `RentedNoodle/Qwen3.8-27B-OrcaRouter-…` ist auf HF gated —
      Account auf Ganymed hat "Access denied". **User muss Freigabe
      beantragen**, dann IQ3_M (12,77 GB) laden und Block laufen lassen.
- [ ] Falls Freigabe ausbleibt: bartowski-IQ3_XS (gleiche Gewichtsbasis,
      4/4 delivered) bleibt der faktische Orca-Kandidat — im RANKING
      steht er bereits als Runner-up.

## Block D — Produktions-Policy (P3)

- [ ] Seed-Rotation/Retry-Policy dokumentieren: Befund = Reasoning-Loop
      → 0 Content ist seed-abhängig (orcM 1/4, hauhau 1/8, rnorca 1/3).
      Für Coding-Agenten: bei `finish_reason=stop` + `content_chars=0`
      automatisch Retry mit anderem Seed. In
      `docs/qwen38-eval/RANKING.md` als Deployment-Note festhalten.
- [ ] Optional: JonathanColetti IQ4_XS (15,3 GB) — nur Uranus-tauglich,
      keine Eris-Perspektive. Nur testen wenn Langeweile, nicht blockend.

## Abbruch-Kriterien / Defaults

- `-ctkd/-ctvd` Start schlägt fehl trotz Flag → Caveat "ungetestet"
  stehen lassen, kein Debugging des Features in dieser Session.
- Eris thermisch >90 °C → Block pausieren, Uranus kann B nicht
  übernehmen (Eris-Frage ist Eris-spezifisch).
- Jede Änderung an RANKING.md nur mit neuem Ledger-Eintrag belegen.
