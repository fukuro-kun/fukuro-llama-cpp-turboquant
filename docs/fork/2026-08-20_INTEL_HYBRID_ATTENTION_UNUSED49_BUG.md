# Intel Vulkan Hybrid-Attention: `<unused49>` Garbage-Tokens

**Datum:** 2026-08-20
**Status:** BEOBACHTET — transient, nicht reproduzierbar
**Betroffen:** Hyperion (Intel UHD G4, ANV/Mesa, Vulkan)
**Commit:** `918eb40d8` (Intel Hybrid-Attention — Prefill unfused, Decode FA)

---

## Symptom

Hyperion generiert gelegentlich 50 × `<unused49>` Placeholder-Special-Tokens
als `content` statt sinnvollem Text. `finish_reason: "stop"` (Modell generiert
selbst EOS nach den Garbage-Tokens, kein Cutoff).

Beispiel-Antwort (rohes JSON, gekürzt):
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "<unused49><unused49><unused49>...(50×)"
    },
    "finish_reason": "stop"
  }],
  "usage": {"completion_tokens": 50, "prompt_tokens": 531}
}
```

## Kontext

- **Client:** Mystil/RPG-Framework "Helferlein" (Rule- oder Lore-Helferlein)
- **Request-Parameter:** `max_tokens=50`, `thinking_budget_tokens=1` (force_no_thinking),
  `temperature=1.0`, `top_p=0.95`, Streaming
- **Prompt-Größe:** 531 Tokens (System + User), `cached_tokens=161`
- **Router-Log:** `Stream fertig: hyperion ttft=3922ms total=11996ms chunks=54 tok=531→50`
  — Router sah den Request als erfolgreich (50 Tokens generiert)

## Diagnose

### Ursache: Intel Vulkan Hybrid-Attention KV-Cache-Inkompatibilität

Commit `918eb40d8` führte Hybrid-Attention für Intel ANV ein:
- **Prefill** (>1 Token/Sequence) → unfused Attention (skalare FA auf Intel ANV
  ist "inkorrekt" für viele Query-Zeilen, siehe Commit-Message)
- **Decode** (1 Token) → Flash Attention (unfused zu langsam für Decode)

Die Übergangsstelle zwischen unfused Prefill und FA Decode introduziert eine
KV-Cache-Kompatibilitätslücke: wenn die KV-Cache-Werte aus unfused Prefill und
die FA-Decode-Attention nicht numerisch kompatibel sind, produziert die
Attention falsche Scores → falsche Logits → `<unused49>` Garbage-Tokens.

### Beweise

1. **Nur auf Hyperion (Intel Vulkan)** — nie auf Uranus (NVIDIA CUDA) oder
   Phobos (AMD Vulkan) bei identischem Request
2. **`<unused49>` sind Placeholder-Special-Tokens** (Gemma-4 Vokabular) →
   falsche Logits aus korrupter Attention, nicht zufällige Token-Auswahl
3. **`finish_reason: "stop"` nicht `"length"`** → Modell hat selbst EOS
   generiert nach den Garbage-Tokens, nicht durch max_tokens abgeschnitten
4. **Nicht reproduzierbar** — 5/5 Retry-Tests mit identischem Prompt
   produzierten korrekte Antworten ("STATUS: INCOMPLETE\nSCORE: 0.3...")
5. **Commit-Message sagt selbst:** "skalare Flash-Attention-Pfad auf Intel ANV
   ohne coopmat2 bleibt bei Prefill mit vielen Query-Zeilen **inkorrekt**"
6. **Nur sehr kurze Responses (max_tokens=50)** — längere Responses
   (100, 400 Tokens) nicht betroffen in heutigen Logs

### Häufigkeit (2026-08-20)

5 Vorkommnisse heute, alle vom selben Mystil-Helferlein-Client:
```
07:01:09  tok=556→50  chunks=54
07:01:32  tok=453→50  chunks=54
13:49:32  tok=738→50  chunks=54
14:41:25  tok=501→50  chunks=54
15:29:06  tok=531→50  chunks=54
```

Alle mit `chunks=54` und `→50` — identisches Generierungsverhalten.
16 hyperion-Streams heute gesamt → 5/16 = 31% der 50-Token-Requests betroffen.

## Was NICHT die Ursache ist

- **NICHT max_tokens:** `max_tokens` ist nur eine Abschneide-Grenze, das LLM
  sieht sie nicht bei der Generierung. Auf Uranus funktioniert derselbe
  Request mit `max_tokens=50` einwandfrei.
- **NICHT force_no_thinking:** Mit `thinking_budget_tokens=1` und
  `max_tokens=500` funktioniert Hyperion korrekt. Nur die Kombination aus
  Hybrid-Attention +特定 Cache-Zustand triggert den Bug.
- **NICHT der Router:** Router leitet korrekt weiter, `thinking_budget_tokens=1`
  wird korrekt gesetzt. Der Router kann nicht erkennen dass der Inhalt Müll ist.

## Mitigation-Optionen (falls es schlimmer wird)

1. **FA komplett deaktivieren auf Intel** — Revert zu `1241af89a` (FA global
   aus für Intel ANV). Unfused Attention für alles. Langsamer Decode aber
   korrekt. Beseitigt die Übergangsstelle.
2. **Garbage-Detection im Router** — Router prüft ob generierte Tokens in
   `<unused>` Range fallen → Request als Fehler markieren + Retry auf anderem
   Endpoint. Erkennt das Problem aber behebt nicht die Ursache.
3. **Root-Cause Fix im Fork** — Tiefenanalyse der KV-Cache-Kompatibilität
   zwischen unfused Prefill und FA Decode auf Intel Vulkan. Numerische
   Präzisionsanalyse der Attention-Scores.

## Entscheidung (2026-08-20)

**Beobachten.** Ist transient, betrifft nur `max_tokens=50` Helferlein-Requests,
und Haupt-Requests (400 Tokens) sind nicht betroffen. Wenn es häufiger wird
oder längere Responses betrifft → Option 1 (FA deaktivieren) als Sofortmaß,
Option 3 als Proper Fix.

## Querverweise

- Fork-Commit: `918eb40d8` — `vulkan: Intel Hybrid-Attention — Prefill unfused, Decode FA`
- Vorgänger-Commit: `1241af89a` — `vulkan: Disable Flash Attention for Intel ANV (scalar path)`
- Fork-AGENTS.md → "Intel iGPU (ANV/Mesa)" Sektion
- Fork-CHANGELOG.md → 2026-08-17 Eintrag
- InferenzQuelle Router-Log: `/home/fukuro/.local/share/inferenzquelle/router.log` auf Janus
- Hyperion llama-server Log: `/home/fukuro/llama-server-hyperion.log` auf Hyperion
