// Test: Gemma4 thinking suppression with tool-call conversations.
// Verifies that enable_thinking=false produces the empty thought channel
// marker even when the previous message is a tool_call or tool_response.
// Regression test for the template bug where the suppression marker was
// nested inside the turn-header block and got skipped for tool-call contexts.
//
// Tests exercise the full common_chat_templates_apply path (template + C++
// workaround), which is what the llama-server uses in production.

#include <string>
#include <vector>
#include <iostream>

#include <nlohmann/json.hpp>

#undef NDEBUG
#include <cassert>

#include "llama.h"
#include "common.h"
#include "chat.h"

using json = nlohmann::ordered_json;

// The fixed gemma4 template tail (suppression outside turn-header block).
static const char * GEMMA4_TAIL_FIXED =
    "{%- if add_generation_prompt -%}"
    "    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}"
    "        {{- '<|turn>model\\n' -}}"
    "    {%- endif -%}"
    "    {%- if not enable_thinking | default(false) -%}"
    "        {{- '<|channel>thought\\n<channel|>' -}}"
    "    {%- endif -%}"
    "{%- endif -%}";

// Minimal Gemma4 template wrapper — just enough to exercise the tail logic.
static std::string build_gemma4_template(const char * tail) {
    std::string tmpl;
    tmpl += "{%- set ns = namespace(prev_message_type=None) -%}";
    tmpl += "{%- set loop_messages = messages -%}";
    tmpl += "{{- bos_token -}}";
    // System block: only if tools present (simplification for test)
    tmpl += "{%- if tools -%}";
    tmpl += "    {{- '<|turn>system\\n' -}}";
    tmpl += "    {%- for tool in tools %}";
    tmpl += "        {{- '<|tool>' }}";
    tmpl += "        {{- '<tool|>' -}}";
    tmpl += "    {%- endfor %}";
    tmpl += "    {%- set ns.prev_message_type = 'tool' -%}";
    tmpl += "    {{- '<turn|>\\n' -}}";
    tmpl += "{%- endif %}";
    // Message loop
    tmpl += "{%- for message in loop_messages -%}";
    tmpl += "    {%- if message['role'] != 'tool' -%}";
    tmpl += "        {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}";
    tmpl += "        {{- '<|turn>' + role + '\\n' }}";
    tmpl += "        {%- if message['tool_calls'] -%}";
    tmpl += "            {%- for tool_call in message['tool_calls'] -%}";
    tmpl += "                {{- '<|tool_call>call:' + tool_call['function']['name'] + '{}<tool_call|>' -}}";
    tmpl += "            {%- endfor %}";
    tmpl += "            {%- set ns.prev_message_type = 'tool_call' -%}";
    tmpl += "        {%- else -%}";
    tmpl += "            {{- message['content'] | trim -}}";
    tmpl += "            {%- set ns.prev_message_type = role -%}";
    tmpl += "        {%- endif -%}";
    tmpl += "        {{- '<turn|>\\n' -}}";
    tmpl += "    {%- else -%}";
    tmpl += "        {{- '<|tool_response>response:tool{' + 'value:' + message['content'] + '}<tool_response|>' -}}";
    tmpl += "        {%- set ns.prev_message_type = 'tool_response' -%}";
    tmpl += "        {{- '<turn|>\\n' -}}";
    tmpl += "    {%- endif -%}";
    tmpl += "{%- endfor -%}";
    // Tail
    tmpl += tail;
    return tmpl;
}

static std::string apply_template(const std::string & template_str,
                                  std::vector<common_chat_msg> & messages,
                                  std::vector<common_chat_tool> & tools,
                                  bool enable_thinking) {
    auto tmpls = common_chat_templates_init(/* model= */ nullptr, template_str, "<bos>", "<end_of_turn>");
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.messages = messages;
    inputs.tools     = tools;
    inputs.add_generation_prompt = true;
    inputs.enable_thinking       = enable_thinking;
    auto result = common_chat_templates_apply(tmpls.get(), inputs);
    return result.prompt;
}

static common_chat_msg msg(const std::string & role, const std::string & content) {
    common_chat_msg m;
    m.role = role;
    m.content = content;
    return m;
}

static common_chat_msg msg_with_tool_call(const std::string & role,
                                           const std::string & tool_name) {
    common_chat_msg m;
    m.role = role;
    m.content = "";
    common_chat_tool_call tc;
    tc.name = tool_name;
    tc.arguments = "{}";
    m.tool_calls = {tc};
    return m;
}

static common_chat_tool make_tool(const std::string & name) {
    common_chat_tool tool;
    tool.name = name;
    tool.description = "Test tool";
    tool.parameters = R"({"type":"object","properties":{}})";
    return tool;
}

static const char * SUPPRESSION_MARKER = "<|channel>thought\n<channel|>";

int main() {
    std::vector<common_chat_tool> tools = { make_tool("get_weather") };
    auto tmpl = build_gemma4_template(GEMMA4_TAIL_FIXED);

    // --- Test 1: After tool_response, enable_thinking=false ---
    // The suppression marker must be present even after a tool_response.
    {
        std::vector<common_chat_msg> messages = {
            msg("user", "What is the weather?"),
            msg_with_tool_call("assistant", "get_weather"),
            msg("tool", "{\"temp\": 72}"),
        };
        auto output = apply_template(tmpl, messages, tools, false);

        std::cout << "=== Test 1: tool_response + enable_thinking=false ===\n";
        std::cout << "Output tail: ..." << output.substr(output.size() > 80 ? output.size() - 80 : 0) << "\n\n";

        assert(output.find(SUPPRESSION_MARKER) != std::string::npos
               && "suppression marker missing after tool_response");
        std::cout << "PASS: suppression marker present after tool_response\n\n";
    }

    // --- Test 2: After tool_call (no response yet), enable_thinking=false ---
    {
        std::vector<common_chat_msg> messages = {
            msg("user", "What is the weather?"),
            msg_with_tool_call("assistant", "get_weather"),
        };
        auto output = apply_template(tmpl, messages, tools, false);

        std::cout << "=== Test 2: tool_call (no response) + enable_thinking=false ===\n";
        std::cout << "Output tail: ..." << output.substr(output.size() > 80 ? output.size() - 80 : 0) << "\n\n";

        assert(output.find(SUPPRESSION_MARKER) != std::string::npos
               && "suppression marker missing after tool_call");
        std::cout << "PASS: suppression marker present after tool_call\n\n";
    }

    // --- Test 3: Normal conversation (no tools), enable_thinking=false ---
    // This already worked before the fix — regression guard.
    {
        std::vector<common_chat_msg> messages = {
            msg("user", "Hello"),
        };
        std::vector<common_chat_tool> no_tools;
        auto output = apply_template(tmpl, messages, no_tools, false);

        std::cout << "=== Test 3: normal + enable_thinking=false ===\n";
        std::cout << "Output tail: ..." << output.substr(output.size() > 80 ? output.size() - 80 : 0) << "\n\n";

        assert(output.find(SUPPRESSION_MARKER) != std::string::npos
               && "suppression marker missing in normal case");
        std::cout << "PASS: suppression marker present in normal case\n\n";
    }

    // --- Test 4: After tool_response, enable_thinking=true ---
    // With thinking enabled, no suppression marker should be present.
    {
        std::vector<common_chat_msg> messages = {
            msg("user", "What is the weather?"),
            msg_with_tool_call("assistant", "get_weather"),
            msg("tool", "{\"temp\": 72}"),
        };
        auto output = apply_template(tmpl, messages, tools, true);

        std::cout << "=== Test 4: tool_response + enable_thinking=true ===\n";
        std::cout << "Output tail: ..." << output.substr(output.size() > 80 ? output.size() - 80 : 0) << "\n\n";

        assert(output.find(SUPPRESSION_MARKER) == std::string::npos
               && "suppression marker present when thinking enabled");
        std::cout << "PASS: no suppression marker when thinking enabled\n\n";
    }

    // --- Test 5: After tool_call, enable_thinking=true ---
    {
        std::vector<common_chat_msg> messages = {
            msg("user", "What is the weather?"),
            msg_with_tool_call("assistant", "get_weather"),
        };
        auto output = apply_template(tmpl, messages, tools, true);

        std::cout << "=== Test 5: tool_call + enable_thinking=true ===\n";
        std::cout << "Output tail: ..." << output.substr(output.size() > 80 ? output.size() - 80 : 0) << "\n\n";

        assert(output.find(SUPPRESSION_MARKER) == std::string::npos
               && "suppression marker present when thinking enabled");
        std::cout << "PASS: no suppression marker when thinking enabled\n\n";
    }

    std::cout << "OK: All tests passed.\n";
    return 0;
}
