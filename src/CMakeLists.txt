llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")
# Exclude orphaned model files from sync-merge that reference APIs not present
# in this codebase version (build_attn signature, llm_graph_input_attn_kv_iswa,
# nextn_predict_layers, t_h_pre_norm, etc.). These need porting before compilation.
list(FILTER LLAMA_MODELS_SOURCES EXCLUDE REGEX ".*-iswa\\.cpp$")
list(FILTER LLAMA_MODELS_SOURCES EXCLUDE REGEX "pangu-embedded\\.cpp$")
list(FILTER LLAMA_MODELS_SOURCES EXCLUDE REGEX "qwen35.*-nextn\\.cpp$")
list(FILTER LLAMA_MODELS_SOURCES EXCLUDE REGEX "qwen3vl-moe\\.cpp$")
list(FILTER LLAMA_MODELS_SOURCES EXCLUDE REGEX "t5-dec\\.cpp$")
list(FILTER LLAMA_MODELS_SOURCES EXCLUDE REGEX "t5-enc\\.cpp$")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-kv-cache-dsa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-expected-attention.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
