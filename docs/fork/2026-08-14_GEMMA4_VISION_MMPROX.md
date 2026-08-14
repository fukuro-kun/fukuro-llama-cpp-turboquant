# Gemma-4 26B-A4B Vision (mmproj) — Konfiguration und Performance

**Datum:** 2026-08-14
**Modell:** Gemma-4 26B-A4B-it QAT-UD-Q4\_K\_XL (14.2 GB)
**Vision-Encoder:** SigLIP ~550M, mmproj-Q6\_K (~806 MB)

## Überblick

Gemma-4 26B-A4B-it unterstützt Multimodalität via `--mmproj`. Der Vision-Encoder
(SigLIP) verarbeitet Bilder und embeddet sie als Token in den Prompt-Kontext.
Der llama-server akzeptiert OpenAI-kompatible `image_url`-Requests mit
`data:image/...;base64,...`-URLs.

## Start-Skripte mit --mmproj

| Skript | GPU-Architektur | KV-Cache | Kontext | Bemerkung |
|--------|----------------|----------|---------|-----------|
| `scripts/start-venus-26b-server.sh` | Vulkan GCN (AMD Vega/Renoir) | f16 | 256k (2×128k) | Produktion. `-fit off` verhindert fit\_params-Reset bei mmproj auf APU |
| `scripts/start-mars-26b-server.sh` | Vulkan RDNA3 (AMD 760M) | turbo3/3 | 161792 (2×79k) | mmproj seit 14.08.2026 **aktiviert** — Kontext auf 161792 reduziert (RADV-Limit bei ≥163840, siehe Trilium SWumEN7WOXBI §5.12). 32 t/s tg, 38 t/s pp. |

### Wichtige Flags

| Flag | Grund |
|------|-------|
| `--mmproj <path>` | Vision-Encoder laden (SigLIP Q6\_K) |
| `-fit off` | Verhindert `fit_params`-Reset auf APUs (Unified Memory). Ohne `-fit off` reserviert `fit_params` GPU-Speicher für mmproj und reduziert `n_gpu_layers` auf 0 → CPU-Fallback |
| `--no-warmup` | Verhindert langen Warmup-Hang (RADV Pipeline-Kompilierung beim Start) |

### Bekannte Probleme

1. **HTTPS-Bild-URLs:** llama-server ist ohne SSL gebaut (`libssl-dev` fehlt).
   Externe HTTPS-Bild-URLs geben HTTP 500. **Nur base64 data-URLs funktionieren.**
   Clientseitige Lösung: Bild → base64 → `data:image/png;base64,...`

2. **RADV + mmproj + n_ctx ≥ 163840 (RDNA3):** Auf AMD RDNA3 (760M)
   mit RADV/Mesa 26.1.2 löst die Kombination mmproj + n_ctx ≥ 163840
   extreme Shader-Kompilierung aus (0.07 t/s, 640s Startup). Bei
   n_ctx ≤ 161792 funktioniert mmproj einwandfrei (32 t/s, 60s Startup).
   **Fix:** Kontext auf 161792 (2×79k) reduzieren. Auf GCN (Vega) tritt
   dieses Problem nicht auf (keine coopmat, f16 KV-Cache). Siehe Trilium
   SWumEN7WOXBI §5.12 für vollständige Diagnose.

3. **fit\_params + mmproj auf APU:** `--mmproj` reserviert GPU-Speicher in der
   `fit_params`-Margin. Auf Unified-Memory APUs reduziert `fit_params`
   `n_gpu_layers` auf 0 → CPU-Fallback. **Fix:** `-fit off`.

## Vision-Request-Format (OpenAI-kompatibel)

```json
{
  "model": "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Was siehst du auf diesem Bild?"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}}
    ]
  }],
  "max_tokens": 800,
  "temperature": 1.0,
  "stream": false
}
```

### ⚠️ Wichtige Hinweise

| Thema | Detail |
|-------|--------|
| **Modellname** | Muss exakt mit dem geladenen Modell übereinstimmen. Der InferenzQuelle-Router matched `ep.model == requested_model` (kein Substring). Kurzname `gemma-4-26B` → Router fällt in VLM-Kaskade → HTTP 503 |
| **max\_tokens** | **≥800 empfohlen.** Gemma-4 generiert standardmäßig `<\|think\|>`-Tokens (Reasoning) vor der Antwort. Bei `max_tokens=300` wird nur Thinking generiert, `content` bleibt leer (`finish_reason: length`). Thinking erscheint in `reasoning_content`, Antwort in `content` |
| **Thinking deaktivieren** | `thinking_budget_tokens: 1` im Request-Body (zuverlässig). `chat_template_kwargs.enable_thinking: false` allein wird in Praxis ignoriert. Siehe AGENTS.md → "Gemma-4 Thinking-Deaktivierung" |
| **temperature** | **IMMER 1.0** für Gemma-4. Keine Ausnahmen. Gemma-4 ist darauf kalibriert |
| **Bildformat** | Nur `data:image/...;base64,...` (keine HTTPS-URLs, siehe oben) |

### Response-Struktur

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Das Bild zeigt ...",
      "reasoning_content": "* Thinking-Schritte hier *"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 115,
    "completion_tokens": 620,
    "prompt_tokens_details": {"cached_tokens": 7}
  }
}
```

## Performance (Vulkan GCN, AMD Vega iGPU, 14.08.2026)

Messung auf einem AMD Ryzen 5 5560U (6C/12T) mit Radeon Vega iGPU (GCN/Renoir),
62 GB DDR4-3200, Vulkan via RADV (Mesa 25.2.8). UMA/shared-memory Architektur,
keine Matrix Cores.

| Metrik | Wert | Vergleich Text-Only |
|--------|------|---------------------|
| Wall-Zeit (800 max\_tokens) | ~48-76s | ~16-44s (1k-10k Prompt) |
| Prompt-Processing (pp) | ~19 t/s | ~50 t/s |
| Token-Generation (tg) | ~9 t/s | ~10.6 t/s |
| Prompt-Tokens | ~100-115 (Text + Bild-Embeddings) | ~1k-10k |

### Erklärung der Performance-Differenz

- **pp ~19 t/s vs ~50 t/s:** Der SigLIP-Vision-Encoder muss das Bild verarbeiten
  (Conv-Layer, Attention, Projection). Diese Operationen laufen auf der GPU
  (Vulkan) und erzeugen zusätzliche Shader-Dispatches und GTT-Buffer-Operations.
- **tg ~9 t/s vs ~10.6 t/s:** Der mmproj-Encoder belegt GPU-Ressourcen (GTT,
  Shader-Cache), die dem LLM-Decoder nicht mehr voll zur Verfügung stehen.
- **Wall-Zeit:** Dominiert durch Thinking-Modus (~350 Tokens Reasoning vor der
  Antwort). Ohne Thinking wäre die Wall-Zeit ~15-20s.

### Architektur-Bedingungen (GCN/Vega)

- **Keine Matrix Cores:** Die Vega iGPU hat `matrix cores: none` (verifiziert
  via `vulkaninfo`). Alle MatMuls laufen auf Standard-Shader-Units (ALU).
  Trotzdem ist die GPU schneller als CPU bei allen Prompt-Längen, weil 7 CUs
  die Batch-Verarbeitung parallelisieren.
- **UMA (Unified Memory):** CPU und GPU teilen sich DDR4-3200 (~51.2 GB/s
  dual-channel). Kein schneller dedizierter VRAM. GTT (~31 GB auto) hält
  Modell + KV-Cache + mmproj-Buffer.
- **f16 KV-Cache:** Auf GCN ist turbo3/4 bei PP 35-54% langsamer als f16
  (scalar FA fallback, Dequant-Overhead). Daher f16 KV-Cache auf Venus.

## Test-Verifikation (14.08.2026)

Getestet mit synthetischen Test-Bildern (geometrische Formen, Text) via
`curl -w` über den InferenzQuelle-Router (Janus):

| Test | Prompt-Tokens | Completion-Tokens | Wall (s) | Antwort korrekt |
|------|---------------|-------------------|----------|-----------------|
| Bild 1 (rotes Rechteck + Text) | 77 | 288 | ~48s | ✅ Form, Farbe, Text erkannt |
| Bild 2 (Kreis + Rechteck + Linie + Text) | 100 | 452 | ~48s | ✅ Alle Formen/Farben erkannt |
| Bild 3 (Stern + Kreis + Text) | 113 | 549 | ~48s | ✅ Form, Farbe, Text erkannt |

> **⚠️ Messartefakt urllib:** Python `urllib.request.urlopen` bei chunked
> HTTP-Responses liefert falsche wall-Zeiten (0.1s statt 48s). Für
> Performance-Messungen **immer `curl -w`** verwenden.

## Referenzen

- Start-Skripte: `scripts/start-venus-26b-server.sh`, `scripts/start-mars-26b-server.sh`
- InferenzQuelle Router-Doku: `docs/LLAMA_SERVER_LAN.md` → "Vision-Routing"
- Thinking-Deaktivierung: `AGENTS.md` → "Gemma-4 Thinking-Deaktivierung"
- Vulkan KV-Cache Benchmarks: `docs/fork/2026-07-09_VULKAN_KV_CACHE_BENCHMARK.md`
- Vulkan Large-Context Perf: `docs/fork/2026-06-20_VULKAN_LARGE_CONTEXT_PERF_CLIFF.md`
