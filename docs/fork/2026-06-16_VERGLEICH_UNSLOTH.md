# Vergleich: Unsere Implementierung vs. Unsloth Referenz (PR #24423)

> Analyse-Datum: 2026-06-15
> Update: 2026-06-16 — Alle Kernalgorithmen (EB, Schedule, Konvergenz, EOS) jetzt identisch
> Stand: Nach vollstaendiger EB-Decoder-Implementierung

---

## 1. Canvas-Initialisierung

| Aspekt | Unsloth (Referenz) | Unsere Implementierung | Status |
|--------|-------------------|------------------------|--------|
| **Init-Methode** | `vocab_dist(rng)` — uniform random über Vokabular | Identisch | ✅ Korrigiert |
| **Vorher** | N/A | `<pad>` Token (falsch!) | ❌ Behoben |
| **Begründung** | Diffusion startet mit "Rauschen", nicht leer | — | — |

**Urteil:** Wir haben das inzwischen korrigiert. ✅

---

## 2. Sampling-Strategie (KRITISCH)

### Unsloth: Entropy-Bound Decoder (EB)

```cpp
// Pro Position: argmax, entropy, multinomial sample
for (int32_t pos = p0; pos < p1; pos++) {
    const float * row = logits + (size_t) (logit_off + pos) * n_vocab;
    float m = -INFINITY; int32_t amax = 0;
    for (int32_t v = 0; v < n_vocab; v++) {
        const float z = row[v] * temp_inv;
        if (z > m) { m = z; amax = v; }
    }
    float Z = 0.0f;
    for (int32_t v = 0; v < n_vocab; v++) {
        Z += expf(row[v] * temp_inv - m);
    }
    const float target = u[pos] * Z;
    float cum = 0.0f, H = 0.0f;
    int32_t sampled = n_vocab - 1; bool picked = false;
    for (int32_t v = 0; v < n_vocab; v++) {
        const float e = expf(row[v] * temp_inv - m);
        const float p = e / Z;
        if (p > 0.0f) { H -= p * logf(p); }
        cum += e;
        if (!picked && cum >= target) { sampled = v; picked = true; }
    }
    entropy[pos]       = H;        // Shannon-Entropie der Softmax-Verteilung
    argmax_canvas[pos] = amax;     // bester Token (für Output)
    denoiser[pos]      = sampled;  // Multinomial-Sample (für Canvas-Update)
}
```

Dann: **Acceptance nach Entropie-Bound**:
```cpp
// Sortiere Positionen nach Entropie (niedrigste zuerst)
std::iota(order.begin(), order.end(), 0);
std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    return entropy[a] < entropy[b];
});

// Akzeptiere die niedrigst-entropischen Positionen, bis die kumulative
// Entropie den Bound überschreitet
std::vector<char> accepted(C, 0);
double cumE = 0.0;
for (int32_t k = 0; k < C; k++) {
    const int32_t pos = order[k];
    cumE += entropy[pos];
    if (cumE - entropy[pos] <= params.entropy_bound) {
        accepted[pos] = 1;
    }
}

// Renoising: akzeptierte -> sampled, rest -> fresh random
for (int32_t pos = 0; pos < C; pos++) {
    current_canvas[pos] = accepted[pos] ? denoiser[pos] : renoise[pos];
    output_tokens[n_input + pos] = argmax_canvas[pos];  // Output ist argmax!
}
```

### Unsere Implementierung

```cpp
// Per-Position: argmax, entropy, multinomial sample (identisch zur Referenz)
auto worker = [&](int32_t p0, int32_t p1) {
    for (int32_t pos = p0; pos < p1; pos++) {
        const float * row = logits + (size_t) (logit_off + pos) * n_vocab;

        // Argmax
        float m = -INFINITY; int32_t amax = 0;
        for (int32_t v = 0; v < n_vocab; v++) {
            const float z = row[v] * temp_inv;
            if (z > m) { m = z; amax = v; }
        }

        // Softmax + Entropie + Multinomial
        float Z = 0.0f;
        for (int32_t v = 0; v < n_vocab; v++) {
            Z += expf(row[v] * temp_inv - m);
        }

        const float target = u[pos] * Z;
        float cum = 0.0f, H = 0.0f;
        int32_t sampled = n_vocab - 1; bool picked = false;
        for (int32_t v = 0; v < n_vocab; v++) {
            const float e = expf(row[v] * temp_inv - m);
            const float p = e / Z;
            if (p > 0.0f) { H -= p * logf(p); }
            cum += e;
            if (!picked && cum >= target) { sampled = v; picked = true; }
        }

        entropy[pos]       = H;
        argmax_canvas[pos] = amax;
        denoiser[pos]      = sampled;
    }
};

// Acceptance: Sortiere nach Entropie, akzeptiere niedrigste innerhalb Bound
std::iota(order.begin(), order.end(), 0);
std::sort(order.begin(), order.end(),
          [&](int32_t a, int32_t b) { return entropy[a] < entropy[b]; });

std::vector<char> accepted(C, 0);
double cumE = 0.0;
for (int32_t k = 0; k < C; k++) {
    const int32_t pos = order[k];
    cumE += entropy[pos];
    if (cumE - entropy[pos] <= params.entropy_bound) {
        accepted[pos] = 1;
    }
}

// Renoising + Output
for (int32_t pos = 0; pos < C; pos++) {
    current_canvas[pos]          = accepted[pos] ? denoiser[pos] : renoise[pos];
    output_tokens[n_input + pos] = argmax_canvas[pos];  // OUTPUT = argmax!
}
```

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **Sampling** | Argmax + Entropie + Multinomial | Identisch |
| **Acceptance** | MI-bounded (niedrigste Entropie zuerst) | Identisch |
| **Renoising** | Nur nicht-akzeptierte Positionen | Identisch |
| **Output** | Argmax-Canvas | Identisch |
| **Begründung** | "Akzeptiere vertrauenswürdige Positionen, renoise den Rest" | Identisch |

**Impact:** Keine. Die Implementierung ist identisch. Der EB-Decoder funktioniert korrekt.

---

## 3. Phase-Handling

### Unsloth

```cpp
// CACHED path (optimiert)
if (params.kv_cache) {
    // PREFILL: prompt einmalig, schreibe K,V Store
    llama_diffusion_set_phase(model, 1, n_input);
    // DECODE: nur Canvas, lese cached K,V
    llama_diffusion_set_phase(model, 2, n_input);
}
// UNIFIED path (fallback)
else {
    // Beides zusammen in einem Batch
    llama_diffusion_set_phase(model, 0, 0);
}
```

### Unsere Implementierung

```cpp
// CACHED path (implementiert und getestet)
if (params.kv_cache) {
    llama_diffusion_set_phase(model, 1, n_input);  // PREFILL
    llama_diffusion_set_phase(model, 2, n_input);  // DECODE
}
// UNIFIED path (Fallback)
else {
    llama_diffusion_set_phase(model, 0, 0);
}
```

**WICHTIG:** Der PREFILL→DECODE-Pfad war ursprünglich **broken** auf partieller GPU-Offload (`-ngl 8`).
**Root Cause:** `dg_ensure_pkv_store()` allokierte den gesamten PKV-Store auf `dev_layer(0)`. Bei CPU/GPU-Split war das CPU, aber `Kcur`/`Vcur` der GPU-Layer waren CUDA-resident → Cross-Backend `ggml_cpy` schlug fehl → leerer PKV-Store → `cut=0`.
**Fix:** PKV wird jetzt pro Layer auf dem Buffer-Type des jeweiligen Layer-Device (`m.dev_layer(il)`) allokiert. Alle Operationen werden dadurch intra-Backend.
**Test-Ergebnisse:** `-ngl 8`: cut=14, "Paris" ✅; `-ngl 0`: cut=8 ✅.

**CLI-Verhalten:** Die `llama-diffusion-cli` nutzt aktuell `kv_cache=false` (UNIFIED-Modus) als Default für maximale Stabilität. Der PREFILL→DECODE-Pfad ist verfügbar, aber nicht aktiviert.

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **KV-Cache** | Optional (PREFILL→DECODE) | Implementiert und getestet (nach PKV-Fix) |
| **Effizienz** | O(C) pro Step | O(C) pro Step (nach erstem PREFILL) |
| **PKV-Allokation** | Global auf dev_layer(0) | Pro-Layer auf jeweiligem Device |
| **CLI-Default** | kv_cache=true | kv_cache=false (UNIFIED) |

**Impact:** Gering. Code-Pfad existiert und funktioniert, CLI nutzt ihn aus Vorsicht nicht als Default.

---

## 4. Temperature-Schedule

### Unsloth

```cpp
// Lineare Abkühlung: t_max (heiß) → t_min (kalt)
for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
    const int32_t step_idx = S - cur_step;
    const float t = params.t_min + (params.t_max - params.t_min) * ((float) cur_step / (float) S);
    const float temp_inv = 1.0f / t;
}
```

### Unsere Implementierung

```cpp
// Identische lineare Abkühlung
for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
    const int32_t step_idx = S - cur_step;
    const float t = params.t_min + (params.t_max - params.t_min) * ((float) cur_step / (float) S);
    const float temp_inv = 1.0f / t;
}
```

**Default-Werte:** `t_min = 0.4f`, `t_max = 0.8f`, `max_denoising_steps = 48`

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **Schedule** | Linear abnehmend | Identisch |
| **Start-Temp** | 0.8 | 0.8 |
| **End-Temp** | 0.4 | 0.4 |
| **Begründung** | Anfangs etwas Rauschen erlauben, dann stabilisieren | Identisch |

**Impact:** Keine. Identische Implementierung.

---

## 5. Konvergenz-Erkennung

### Unsloth

```cpp
// Zwei Kriterien müssen BOTH erfüllt sein:
// 1. Stabilität: argmax Canvas bleibt N Steps gleich
held = (prev_argmax == argmax_canvas) ? held + 1 : 0;
const bool stable = (held >= params.stability_threshold);

// 2. Konfidenz: Durchschnittliche Entropie unter Schwellwert
const bool confident = (entropy_sum / (float) C) < params.confidence_threshold;

if (stable && confident) { finished = true; }
```

### Unsere Implementierung

```cpp
// Identische dual-criterion Konvergenz
held = (prev_argmax == argmax_canvas) ? held + 1 : 0;
const bool confident = (entropy_sum / (float) C) < params.confidence_threshold;
if (held >= params.stability_threshold && confident) {
    finished = true;
}
```

**Default-Werte:** `stability_threshold = 1`, `confidence_threshold = 0.005f`

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **Stabilität** | Argmax-Canvas gleich für N Steps | Identisch |
| **Konfidenz** | Entropie-basiert | Identisch |
| **Frühstop** | Ja (adaptive) | Identisch |
| **Begründung** | "Stabil UND konfidenz" verhindert vorzeitiges Stoppen | Identisch |

**Impact:** Keine. Identische Implementierung.

---

## 6. Self-Conditioning (SC)

### Unsloth

```cpp
// SC = softmax(previous step's logits / previous t)
// Gated off on first step (kein vorheriger Step)
llama_diffusion_set_sc(model, dev_sc ? nullptr : sc_buffer.data(),
                       step_idx == 0 ? 0.0f : 1.0f, prev_temp_inv, true);
```

### Unsere Implementierung

```cpp
// SC bewusst deaktiviert
llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);
```

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **SC Aktiv** | Ja (ab Step 2) | Nein (hardcoded disabled) |
| **Begründung** | Bessere Kohärenz zwischen Steps | SC-Tensoren fehlen in verfügbaren GGUFs |

**Impact:** Mittel. SC-Infrastruktur existiert (Tensoren optional mit `TENSOR_NOT_REQUIRED`), aber die verfügbaren GGUF-Modelle enthalten keine SC-Tensoren. `diffusion-gemma-server` versucht SC, falls Tensoren vorhanden sind.

---

## 7. EOS-Handling

### Unsloth

**Während Diffusion-Loop:** EOS wird **nicht** blockiert. Der Sampler kann EOS wählen, aber:
- Die **Output** ist der Argmax-Canvas
- EOS wird erst beim **Trimmen** berücksichtigt

```cpp
// trim_canvas: schneide bei erstem EOS/EOT
const size_t cut = trim_canvas(vocab, canvas, (size_t) canvas_length);
```

### Unsere Implementierung

```cpp
// EOS während Loop NICHT blockiert
// Output ist Argmax-Canvas
// Trimmen nach Loop bei erstem EOS/EOT
static size_t trim_canvas(const llama_vocab * vocab,
                          const llama_token * canvas, size_t C) {
    llama_token eos = llama_vocab_eos(vocab);
    llama_token eot = llama_vocab_eot(vocab);
    for (size_t i = 0; i < C; i++) {
        if ((eos >= 0 && canvas[i] == eos) || (eot >= 0 && canvas[i] == eot)) {
            return i;
        }
    }
    return C;
}
```

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **Während Loop** | EOS erlaubt (Sampler entscheidet) | Identisch |
| **Nach Loop** | trim_canvas schneidet bei EOS | Identisch |
| **Begründung** | EOS ist valide Vorhersage; Trimmen später | Identisch |

**Impact:** Keine. Identische Implementierung.

---

## 8. GPU-Optimierungen

### Unsloth

```cpp
// Stage-1: On-device sampling (spart 268 MB D2H pro Step)
const bool gpu_reduce = dev_sc && device_sample_ok;
if (gpu_reduce) {
    llama_diffusion_device_sample(model, u.data(), argmax_canvas.data(),
                                  entropy.data(), denoiser.data(), C, temp_inv);
}
```

### Unsere Implementierung

```cpp
// Keine GPU-Optimierungen
// Alles auf Host
```

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **Device-Sampling** | Optional (CUDA) | Nicht implementiert |
| **SC on Device** | Optional (spart Upload) | Nicht implementiert |
| **Begründung** | Performance bei großem Vokabular (262144) | Einfacher, aber langsamer |

**Impact:** Niedrig-Mittel. Bei unserer Hardware (RTX 3070) nicht kritisch, aber nett.

---

## Zusammenfassung: Status (Update 2026-06-16)

| # | Aspekt | Status | Datum |
|---|--------|--------|-------|
| 1 | **Entropy-Bound Decoder** | ✅ Implementiert und getestet | 2026-06-15 |
| 2 | **Argmax-Output** | ✅ Implementiert | 2026-06-15 |
| 3 | **Renoising (nicht-akzeptierte)** | ✅ Implementiert | 2026-06-15 |
| 4 | **Temperature-Schedule (linear)** | ✅ Implementiert | 2026-06-15 |
| 5 | **Konvergenz: Stabilität + Konfidenz** | ✅ Implementiert | 2026-06-15 |
| 6 | **KV-Cache (PREFILL/DECODE)** | ✅ Implementiert und getestet | 2026-06-16 |
| 7 | **PKV Cross-Backend Bugfix** | ✅ Per-buft PKV-Allokation | 2026-06-16 |
| 8 | **EOS-Handling** | ✅ Erlaubt + trim_canvas | 2026-06-15 |

### Verbleibende Unterschiede

| Aspekt | Status | Begründung |
|--------|--------|-----------|
| **Self-Conditioning (SC)** | ❌ Deaktiviert | SC-Tensoren fehlen in verfügbaren GGUFs |
| **GPU-Sampling (Stage-1)** | ❌ Nicht implementiert | Host-basiertes Sampling; bei RTX 3070 akzeptabel |
| **SC on Device** | ❌ Nicht implementiert | Erfordert SC-Tensoren + CUDA-Kernel |

### Fazit

**5 von 6 Kernalgorithmen sind identisch mit der Unsloth-Referenz.** Der DiffusionGemma-Decoder funktioniert vollständig mit Entropy-Bound, Temperatur-Schedule und dualer Konvergenz-Erkennung. Die verbleibenden Unterschiede (SC, GPU-Sampling) sind optional und beeintrrächtigen die Funktionalität nicht.
