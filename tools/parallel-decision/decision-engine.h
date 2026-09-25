#pragma once

// Engine für parallele, eingeschränkte Entscheidungen mit endlichen JSON-Schemata;
// gemeinsam genutzt von llama-parallel-decision und dem /decision-Endpunkt des Servers.
//
// Jedes Schemafeld besitzt endlich viele zulässige Werte. Nach einem gemeinsamen Kontext
// werden die Tokenpfade der Kandidatenwerte verzweigt und in einem gebündelten llama_decode
// ausgewertet. Kleine Felder bewerten alle Verzweigungsknoten des Token-Tries parallel und
// liefern eine exakte Verteilung über die zulässigen Werte; größere Felder folgen dem Trie
// schrittweise nach dem jeweils wahrscheinlichsten Token.

#include "llama.h"
#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>

struct common_chat_templates;

namespace llama_decision {

using tokens_t = std::vector<llama_token>;
using decision_json = nlohmann::ordered_json;

// Ein Feld aus Sicht des Scorers: Text vor dem Feldwert und die zulässigen Werttexte.
struct field_input {
    std::string              suffix;     // z. B. '  "fire": '
    std::vector<std::string> candidates; // Kandidaten ohne den gemeinsamen Präfix im Suffix
};

struct options {
    std::string mode           = "auto"; // auto: bis tree_max Werte tree, sonst greedy; tree; greedy
    size_t      tree_max       = 128;
    bool        split_boundary = false;  // veraltete Option: Suffix und Werte getrennt tokenisieren
    bool        allow_cache    = true;   // passendes statisches Präfix wiederverwenden
};

struct field_result {
    int                winner       = -1;
    float              path_score   = 1.0f;
    int                scored_nodes = 0;
    bool               tree         = false;
    std::vector<float> probs;            // tree-Felder: Wahrscheinlichkeit jedes zulässigen Werts
};

struct result {
    std::vector<field_result> fields;
    bool   cache_hit      = false;
    size_t shared_tokens  = 0;
    size_t context_tokens = 0;
    int    rows           = 0;
    int    rounds         = 0;
    double prefill_ms     = 0;
    double scoring_ms     = 0;
};

// Mehrere Kontexte werden mit demselben Schema und gecachten Präfix ausgewertet.
// Die Einträge enthalten Felder, Kontext-Token und Zeilen; Zeiten und Cache-Status
// gelten für den gesamten Batch.
struct batch_result {
    std::vector<result> items;
    bool   cache_hit     = false;
    size_t shared_tokens = 0;
    int    rows          = 0;
    int    rounds        = 0;
    double prefill_ms    = 0;
    double scoring_ms    = 0;
};

// Wertet Entscheidungen in einem bestehenden Kontext mit Sequenz-IDs
// [seq_base, seq_base + n_seqs) aus: Eine Sequenz hält das statische Präfix,
// die übrigen bilden pro gleichzeitig bearbeitetem Kontext einen Stamm und verzweigen
// anschließend für die Kandidaten. Ein vereinheitlichter KV-Cache ist erforderlich,
// damit Zweige die Zellen des Stamms gemeinsam nutzen.
class engine {
  public:
    engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs);

    result decide(const std::string & shared_text, const std::string & context_text,
                  const std::vector<field_input> & fields, const options & opt);

    // Kontexte werden gemeinsam vorverarbeitet und ihre Kandidatenzweige in Gruppen
    // innerhalb des Sequenzbudgets ausgewertet; die Ergebnisreihenfolge bleibt erhalten.
    batch_result decide_batch(const std::string & shared_text, const std::vector<std::string> & contexts,
                              const std::vector<field_input> & fields, const options & opt);

  private:
    struct prompt_part {
        const tokens_t * toks;
        llama_pos        pos0;
        llama_seq_id     seq;
    };
    struct branch {
        llama_seq_id trunk;
        llama_pos    pos0;
        tokens_t     toks;
        tokens_t     cands;
    };

    llama_context     * ctx;
    const llama_vocab * vocab;
    llama_memory_t      mem;
    llama_seq_id        seq_snap, seq_pool;
    int                 n_pool;
    bool                pad_branches; // Rekurrente/Hybrid-Modelle benötigen gleich lange Decode-Zweige
    tokens_t            cached;

    tokens_t tokenize(const std::string & text, bool add_special) const;
    void     decode_parts(const std::vector<prompt_part> & parts);
    bool     prepare_prefix(const tokens_t & shared, bool allow_cache);
    std::vector<std::vector<float>> score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free);
};

// ---- Schema-Compiler (C++-Gegenstück zu llama-mojo/tools/prepare_decisions.py)

struct field_spec {
    std::string              name;
    std::string              type;        // boolean | enum | integer | number
    std::string              description;
    std::string              aggregate;   // mode | median | mean (median/mean nur für Zahlenfelder)
    std::vector<decision_json> values;      // typisierte Werte; Index entspricht dem Kandidatenindex
    std::vector<double>      numbers;     // Zahlenfelder zusätzlich als double
    std::vector<std::string> encoded;     // JSON-Text jedes Werts
};

struct compiled_schema {
    std::string              system_text; // feste Anweisungen und Feldkatalog (cachebar)
    std::vector<field_spec>  specs;
    std::vector<field_input> inputs;
};

// Akzeptiert kompakte Felddefinitionen {"name": {"type": ..., "description": ..., ...}}
// oder ein JSON-Schema-Objekt mit "properties" (boolean, string+enum, integer min/max,
// number min/max/multipleOf).
compiled_schema compile_schema(const decision_json & schema, const std::string & instructions);

// Rendert System- und User-Nachrichten mit dem Chat-Template des Modells (Thinking aus)
// und trennt das statische Präfix vom anfragespezifischen Kontext samt JSON-Start.
std::pair<std::string, std::string> render_prompt(const common_chat_templates * tmpls, bool use_jinja,
                                                  const std::string & system_text, const std::string & context);

// Erstellt {"decision": {...}, "fields": {...}} aus den Scores und wendet für Zahlenfelder
// gegebenenfalls die angegebene Aggregation an.
decision_json assemble(const compiled_schema & cs, const result & r);

} // namespace llama_decision
