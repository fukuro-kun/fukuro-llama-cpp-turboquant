# Gemeinsame Engine für llama-parallel-decision und den /decision-Endpunkt von llama-server
add_library(llama-decision STATIC decision-engine.cpp decision-engine.h)
target_include_directories(llama-decision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llama-decision PUBLIC llama-common llama)
target_compile_features(llama-decision PUBLIC cxx_std_17)
# llama-server bindet die Bibliothek in libllama-server-impl ein
set_target_properties(llama-decision PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TARGET llama-parallel-decision)
add_executable(${TARGET} parallel-decision.cpp)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries(${TARGET} PRIVATE llama-decision llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
