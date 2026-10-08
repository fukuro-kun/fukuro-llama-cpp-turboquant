# Spekulative Decodierung (aktueller Fork-Stand)

Referenz fuer die **aktuelle** Implementierung der spekulativen Decodierung in diesem Fork. Historische Implementierungsberichte (alte Flag- und API-Namen, alte Benchmarks) stehen in [`MTP.md`](../MTP.md) und [`NEXTN.md`](../NEXTN.md) — beide sind als `HISTORISCH` markiert und gelten nicht als Betriebsvertrag. Quelle der Wahrheit fuer Flags und Defaults ist [`common/arg.cpp`](../common/arg.cpp); die registrierten Typen stehen in [`common/speculative.cpp`](../common/speculative.cpp).

## Registrierte Mechanismen

`--spec-type` akzeptiert eine kommagetrennte Liste der in `common/speculative.cpp` registrierten Typen:

| Typ | Mechanismus |
|-----|-------------|
| `none` | Keine Spekulation (Default) |
| `draft-simple` | Externes Draft-Modell (`-md`/`--spec-draft-model` Pflicht) |
| `draft-eagle3` | EAGLE-3 Draft-Modell (Head im Draft-GGUF) |
| `draft-mtp` | Multi-Token-Prediction-Head — eingebettet im Target-GGUF oder als separater Assistant/Sidecar |
| `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache` | Modellfreie n-gram-Varianten |

Die alten Namen `mtp`, `nextn` und `eagle3` sind **nicht** registriert; sie treten nur noch in historischen Dokumenten auf.

## Drei Draft-Pfade

1. **Modellfrei (ngram-\*)**: Kein Draft-Modell noetig; Drafts aus der Token-Historie bzw. einem Lookup-Cache. Eignet sich fuer repetitive Texte (Code, Reasoning, Zusammenfassungen).
2. **Eingebetteter MTP/NextN-Head (`draft-mtp` ohne `-md`)**: Das Target-GGUF traegt MTP/NextN-Schichten (`llama_model_n_layer_nextn(model) > 0`). Der Server baut einen zweiten `llama_context` mit `LLAMA_CONTEXT_TYPE_MTP` auf demselben bereits geladenen Modell — hier findet **kein** erneutes Laden des Modells statt (siehe [`tools/server/server-context.cpp`](../tools/server/server-context.cpp)). Explizit denselben Pfad per `-m` und `-md` zu uebergeben ist dagegen der Sidecar-Pfad und laedt das Modell separat als `model_dft`.
3. **Explizites Draft-/Sidecar-Modell (`draft-mtp` mit `-md`)**: Eine separate Assistant-GGUF wird als `model_dft` geladen und bekommt einen eigenen MTP-Kontext. Die Sidecar muss zur Target-Architektur passen (Tensor-Shape und Tokenizer-Kompatibilitaet sind Pflichttests, nicht angenommen).

Beispiel (Target f16-KV und Draft f16-KV, max. 3 Drafts pro Runde):

```sh
llama-server -m model.gguf -md assistant.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 --spec-draft-n-min 0 \
  -ctk f16 -ctv f16 -ctkd f16 -ctvd f16
```

`--spec-draft-n-max N` ist die **Obergrenze der Draft-Tokens pro Runde** (nicht die Blockgroesse). Andere Implementierungen zaehlen anders — z. B. bedeutet Strata `T=4` einen Verify-Batch aus 1 Target- + bis zu 3 Draft-Tokens; das entspricht hier `n-max 3`, nicht `4`. Vergleiche nur mit aequivalenten Zaehlweisen.

## n-gram-Varianten und ihre Flags

Alle n-gram-Typen sind modellfrei; sie brauchen kein `-md`.

| Typ | Spezifische Flags (`common/arg.cpp`) |
|-----|--------------------------------------|
| `ngram-simple` | `--spec-ngram-simple-size-n` (Laenge des Lookup-n-gram), `--spec-ngram-simple-size-m` (Laenge des Draft-m-gram), `--spec-ngram-simple-min-hits` |
| `ngram-map-k` | `--spec-ngram-map-k-size-n`, `--spec-ngram-map-k-size-m`, `--spec-ngram-map-k-min-hits` |
| `ngram-map-k4v` | `--spec-ngram-map-k4v-size-n`, `--spec-ngram-map-k4v-size-m`, `--spec-ngram-map-k4v-min-hits` |
| `ngram-mod` | `--spec-ngram-mod-n-match` (Lookup-Laenge), `--spec-ngram-mod-n-min`, `--spec-ngram-mod-n-max` (min./max. genutzte Draft-Tokens) |
| `ngram-cache` | Lookup-Cache-Dateien: `-lcs`/`--lookup-cache-static` (nicht durch Generierung aktualisiert), `-lcd`/`--lookup-cache-dynamic` (durch Generierung aktualisiert); Cache-Werkzeuge unter [`examples/lookup/`](../examples/lookup) (`lookup-create`, `lookup-merge`, `lookup-stats`) |

Beispiel:

```sh
llama-server -m model.gguf --spec-type ngram-simple
```

## Relevante Flags (Auswahl)

| Flag | Bedeutung |
|------|-----------|
| `--spec-type` | kommagetrennte Typ-Liste (s.o.) |
| `--spec-draft-model`, `-md`, `--model-draft` | Pfad zur Draft-/Assistant-GGUF |
| `--spec-draft-n-max` / `--spec-draft-n-min` | max./min. Draft-Tokens pro Runde |
| `--spec-draft-device`, `-devd` | Geraet(e) fuer den Draft-Kontext |
| `--spec-draft-ngl`, `-ngld` | GPU-Layer des Drafts |
| `--spec-draft-threads`, `-td` | Threads des Drafts fuer die Generierung |
| `--spec-draft-threads-batch`, `-tbd` | Threads des Drafts fuer den Prompt-/Batch-Pfad |
| `-ctkd`, `-ctvd` | KV-Cache-Typen des Drafts |
| `--spec-draft-override-tensor`, `-otd` | Tensor-Buffer-Overrides des Drafts |
| `-cmoed`, `-ncmoed` | CPU-Anteil der MoE-Experten des Drafts |
| `--spec-draft-p-split` | Split-Wahrscheinlichkeit des Drafts |
| `--spec-draft-p-min` | minimale Draft-Konfidenz bei Greedy-Annahme |
| `--spec-draft-backend-sampling` / `--no-spec-draft-backend-sampling` | Draft-Sampling im Backend ein/aus |

`--spec-draft-p-min` und `--spec-draft-p-split` sind **Draft-seitige** Schwellen (Konfidenz des Draft-Tokens bzw. Split-Wahrscheinlichkeit), keine generischen Target-Akzeptanz-Schwellen. Weitere Draft-Affinitaets-Flags (`--spec-draft-cpu-mask*`, `--spec-draft-cpu-range*`, `--spec-draft-cpu-strict*`, `--spec-draft-prio*`, `--spec-draft-poll*`) und Defaults siehe `common/arg.cpp`.

Entfernte Flags: `--draft`, `--draft-max`, `--draft-min`, `--mtp-head`, `--draft-block-size`, `--spec-type mtp`/`nextn`/`eagle3`. Alte Skripte/Dokumente, die sie benutzen, sind historisch oder kaputt — nicht stillschweigend mitschleppen.

## Bekannte Grenzen und offene Gates

- **Qwen3.8-Flash-Next (Architektur `qwen4exp`), getestete Konfigurationen:** q8_0-KV auf Attention **plus** MTP degradiert die Ausgabe (Root Cause offen). f16-KV ist die derzeit getestete Kontrollgruppe/Workaround — **kein** Beweis des vollen Pfads. Der Befund gilt fuer die vermessenen Qwen4Exp-Konfigurationen; eine Verallgemeinerung auf andere Architekturen (z.B. Qwen3.5, 27B-Modelle) ist nicht gemessen. Der M-Gate im Plan [`fork/plans/SESSION_PLAN_20261006_strata-optllama-uranus-eval.md`](fork/plans/SESSION_PLAN_20261006_strata-optllama-uranus-eval.md) bleibt **offen**; Stand der Messungen: [`qwen38-eval/ENGINE_EVAL_20261007.md`](qwen38-eval/ENGINE_EVAL_20261007.md).
- **Kein generischer Speedup-Anspruch:** Gewinne sind modell-, quant-, backend- und lastabhaengig. Ein Claim braucht identische Prompts/Tokenbudgets, Wiederholungen und Korrektheits-/Delivery-Pruefung (siehe Root-AGENTS: Benchmark-Fairness). Akzeptanzrate, endliches Reasoning oder ein einzelner t/s-Wert sind keine Gesamtfreigabe.
- **Sampling-Fairness:** Vergleiche nur bei gleichem Sampler; fuer Gemma-4 `temperature=1.0` (Root-Vertrag) und ausreichendes Tokenbudget (`max_tokens >= 500`, sonst endet die Ausgabe evtl. in `reasoning_content` mit `finish_reason: length` ohne `content`).
- **Kontext — Kapazitaet und Belegung sind getrennte Achsen:** `-c` reserviert KV-/Buffer-Kapazitaet und wirkt damit auf das Speicher-/Expert-Cache-/Workspace-Budget, auch wenn der Kontext nie gefuellt wird. Fuer belastbare Vergleiche sind **beide** zu kontrollieren und zu loggen: die konfigurierte Kapazitaet **und** die tatsaechlich belegte Prompt-/Generierungslaenge.
- **Multimodal (`--mmproj`):** Die fruehere Whitelist (`common_speculative_is_mtmd_safe`) und der per-Turn `skip_draft_mtmd`-Fallback existieren im aktuellen Code nicht mehr. Die Kombination Spec-Decoding x mmproj ist ein **offenes Gate** — keine Kompatibilitaets- oder Speedup-Zusicherung treffen, bis der Pfad erneut verifiziert ist.
- **Interne API:** [`src/llama-ext.h`](../src/llama-ext.h) (z.B. `llama_set_embeddings_nextn`/`llama_get_embeddings_nextn_ith`) ist ein Fork-Staging-Interface, **kein** stabiles oeffentliches API. Alte APIs (`llama_model_load_mtp_from_file`, `llama_decode_mtp_async`/`_wait`, `llama_set_nextn`, `common_speculative_state_mtp/nextn`) existieren nicht mehr — nur in historischen Dokumenten.

## Statistiken

Der Server meldet pro Implementierung Akzeptanz- und Token-Zaehler im Log (`draft acceptance rate`, `#gen drafts`, `#acc drafts`, `#acc tokens`) sowie `draft_n`/`draft_n_accepted` in den Response-Timings. Sie dienen der Diagnose, nicht als Gate-Ersatz; ein eigener Draft-Akzeptanz-Export ueber `/metrics` ist nicht verifiziert und gilt nicht als verfuegbar.

## Siehe auch

- [`NEXTN.md`](../NEXTN.md) — historischer NextN-Implementierungsbericht (HISTORISCH)
- [`MTP.md`](../MTP.md) — historischer MTP-Bericht (HISTORISCH)
- [`fork/plans/SESSION_PLAN_20261006_strata-optllama-uranus-eval.md`](fork/plans/SESSION_PLAN_20261006_strata-optllama-uranus-eval.md) — aktiver Eval-/Implementierungsplan
- [`qwen38-eval/ENGINE_EVAL_20261007.md`](qwen38-eval/ENGINE_EVAL_20261007.md) — gemessener Engine-Vergleich
