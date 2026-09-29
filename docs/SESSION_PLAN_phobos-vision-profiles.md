# SESSION_PLAN: phobos Vision-Profile + inferenzquelle-Integration

**Stand:** 2026-09-29, Folgeplan zur TTM/nogttspill/Vision-Session.

## Ziel

Die heute aktivierte Vision-Config auf phobos ist produktiv live, aber noch nicht
als sauberes inferenzquelle-Profil modelliert. Ausserdem sind kleine
Nachtests offen (image-token-Tuning, venus-Routing-Aufräumung).

## Offene Punkte

| # | Aufgabe | Details |
|---|---------|---------|
| A | `model_profiles` für phobos in hosts.json | Analog uranus: gemma-prod (aktiv), ggf. qwen38-balanced als alternatives Profil mit eigener Start/Stop-Konfiguration. Skripte auf phobos vorhanden? — ggf. start-phobos-qwen38-server.sh anlegen. |
| B | venus-Vision-Routing prüfen | venus läuft Granite ohne Vision seit 24.08. — Router-Fallbacks/Tags auf phobos konsistent? `venus-vlm` Endpoint enabled=false? In hosts.json prüfen + Doku. |
| C | image-max-tokens A/B | 512 vs 1024: Encode-Zeit (~19s bei 1024) vs OCR-Qualität. Nur wenn Zeit — Default 1024 ist ok. |
| D | TTM-Stabilität beobachten | /etc/modprobe.d/ttm.conf auf mars greift nach Reboot — beim nächsten geplanten Neustart verifizieren dass 24 GiB persistieren. |

## Defaults bei Unklarheit

- Keine Änderung an der laufenden Produktiv-Config ohne Messgrund.
- Weniger ist mehr: wenn ein Profil nicht klar nützt, nicht eintragen.
- GPU-Exklusivität auf phobos einhalten (Guard-Reihenfolge).

## Verifikation

- Router `/health` zeigt phobos mit supports_vision:true.
- Bei Profil-Änderungen: `POST /reload` bzw. SIGHUP + Endpoint-Check.
- Kein Server-Wechsel ohne GTT-Release-Verifikation.

## Abbruch-Kriterien

- Host-OOM-Verdacht → sofort stoppen, nur einen Server aktiv halten.
- Vision-Requests >3× Fehler → mmproj-Config zurück auf bewährten Stand (§5.18 turbo4/3 ohne mmproj).
