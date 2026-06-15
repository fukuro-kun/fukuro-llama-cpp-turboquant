# Vergleich: Unsere Implementierung vs. Unsloth Referenz (PR #24423)

> Analyse-Datum: 2026-06-15
> Stand: Nach erstem funktionierenden Diffusion-Loop

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
// Nur Temperature-Sampling + EOS-Blocking
std::vector<float> masked_logits(n_vocab);
memcpy(masked_logits.data(), row, n_vocab * sizeof(float));
llama_token eos = llama_vocab_eos(vocab);
if (eos >= 0) masked_logits[eos] = -1e30f;

llama_token new_tok = sample_temp(masked_logits.data(), n_vocab, temp);
new_canvas[j] = new_tok;
```

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **Sampling** | Argmax + Entropie + Multinomial | Nur Temperature-Multinomial |
| **Acceptance** | MI-bounded (niedrigste Entropie zuerst) | Keine — alle Positionen werden gesampelt |
| **Renoising** | Nur nicht-akzeptierte Positionen | Alle Positionen werden neu gesampelt |
| **Output** | Argmax-Canvas | Gesampelte Tokens |
| **Begründung** | "Akzeptiere vertrauenswürdige Positionen, renoise den Rest" | Einfacher Ansatz, keine Konvergenz-Optimierung |

**Impact:** Hoch. Der Entropy-Bound ist DER Kern-Algorithmus von DiffusionGemma. Ohne ihn konvergiert der Canvas nicht zu einem stabilen Ergebnis.

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
// NUR UNIFIED — kein KV-Cache für Prompt
llama_diffusion_set_phase(model, 0, 0);
// [prompt | canvas] wird jeden Step komplett neu decoded
```

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **KV-Cache** | Optional (PREFILL→DECODE) | Nicht implementiert |
| **Effizienz** | O(C) pro Step (statt O(P+C)) | O(P+C) pro Step |
| **Begründung** | Prompt ändert sich nicht → cachen | Einfacher, aber langsamer |

**Impact:** Mittel. Bei kurzen Prompts vernachlässigbar, bei langen Prompts signifikant.

---

## 4. Temperature-Schedule

### Unsloth

```cpp
// Lineare Abkühlung: t_max (heiß) → t_min (kalt)
for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
    const int32_t step_idx = S - cur_step;  // 0-based
    const float t = params.t_min + (params.t_max - params.t_min) * ((float) cur_step / (float) S);
    const float temp_inv = 1.0f / t;
}
```

**Default-Werte:** `t_min = 0.4f`, `t_max = 0.8f`, `S = 48`

### Unsere Implementierung

```cpp
// Konstante Temperatur
float temp = 1.0f;  // CLI-Parameter, default
// Keine Schedule — immer gleich
```

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **Schedule** | Linear abnehmend | Konstant |
| **Start-Temp** | 0.8 (relativ kalt) | 1.0 |
| **End-Temp** | 0.4 (kalt) | 1.0 |
| **Begründung** | Anfangs etwas Rauschen erlauben, dann stabilisieren | Kein Schedule → möglicherweise zu viel Rauschen |

**Impact:** Mittel. Die Schedule hilft bei der Konvergenz, aber konstante Temperatur kann auch funktionieren.

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

**Default-Werte:** `stability_threshold = 1`, `confidence_threshold = 0.005f`

### Unsere Implementierung

```cpp
// Nur ein Kriterium: Keine Änderungen überhaupt
if (!changed && step > 2) {
    fprintf(stderr, "  -> Canvas converged at step %d\n", step + 1);
    break;
}
```

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **Stabilität** | Argmax-Canvas gleich für N Steps | Alle Tokens gleich |
| **Konfidenz** | Entropie-basiert | Nicht vorhanden |
| **Frühstop** | Ja (adaptive) | Ja (simpler) |
| **Begründung** | "Stabil UND konfidenz" verhindert vorzeitiges Stoppen bei lokalen Minima | Einfacher, aber möglicherweise unzuverlässig |

**Impact:** Mittel. Unsloths Ansatz ist robuster.

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
// SC deaktiviert (use_sc = false)
llama_diffusion_set_sc(model, sc_cache.data(), use_sc ? 1.0f : 0.0f,
                       use_sc ? 1.0f / prev_temp : 1.0f, use_sc);
// use_sc ist hardcoded auf false
```

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **SC Aktiv** | Ja (ab Step 2) | Nein |
| **Begründung** | Bessere Kohärenz zwischen Steps | Noch nicht aktiviert |

**Impact:** Mittel. SC verbessert die Qualität, aber der Unterschied ist subtil.

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
// EOS wird WÄHREND des Loops geblockt (falsch!)
llama_token eos = llama_vocab_eos(vocab);
if (eos >= 0) masked_logits[eos] = -1e30f;
```

**Unterschiede:**

| Aspekt | Unsloth | Wir |
|--------|---------|-----|
| **Während Loop** | EOS erlaubt (Sampler entscheidet) | EOS blockiert |
| **Nach Loop** | trim_canvas schneidet bei EOS | Kein Trimming im Loop |
| **Begründung** | EOS ist valide Vorhersage des Modells; Trimmen später | Blockierung verhindert, dass Modell EOS lernt |

**Impact:** Mittel. Blockieren ist nicht fatal, aber nicht authentisch.

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

## Zusammenfassung: Was müssen wir ändern?

| # | Änderung | Aufwand | Impact | Priorität |
|---|----------|---------|--------|-----------|
| 1 | **Entropy-Bound Decoder implementieren** | 3-4h | 🔥🔥🔥🔥🔥 KRITISCH | **SOFORT** |
| 2 | **Argmax-Output statt gesampelte Tokens** | 30min | 🔥🔥🔥🔥 Hoch | **SOFORT** |
| 3 | **Renoising nur für nicht-akzeptierte** | 30min | 🔥🔥🔥🔥 Hoch | **SOFORT** |
| 4 | **Temperature-Schedule (linear)** | 1h | 🔥🔥🔥 Mittel | Kurzfrisitg |
| 5 | **Konvergenz: Stabilität + Konfidenz** | 1h | 🔥🔥🔥 Mittel | Kurzfrisitg |
| 6 | **SC aktivieren** | 30min | 🔥🔥 Niedrig-Mittel | Kurzfrisitg |
| 7 | **KV-Cache (PREFILL/DECODE)** | 2h | 🔥🔥 Niedrig-Mittel | Optional |
| 8 | **EOS nicht blockieren** | 10min | 🔥 Niedrig | Optional |

---

## Empfehlung

**Die größte Diskrepanz ist der Entropy-Bound Decoder.** Das ist nicht nur ein Optimierung, sondern der **zentrale Algorithmus** von DiffusionGemma. Ohne ihn haben wir keinen echten Diffusion-Decoder.

Meine Empfehlung:
1. **Jetzt:** Entropy-Bound Decoder implementieren (Punkte 1-3)
2. **Danach:** Temperature-Schedule und Konvergenz (Punkte 4-5)
3. **Später:** SC, KV-Cache, GPU-Optimierungen (Punkte 6-8)

Soll ich mit der Implementierung des Entropy-Bound Decoders beginnen?
