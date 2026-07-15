set(TARGET llama-expert-profile)
add_executable(${TARGET} expert-profile.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET2 llama-expert-overlap)
add_executable(${TARGET2} expert-overlap.cpp)
target_link_libraries(${TARGET2} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET2} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} ${TARGET2} RUNTIME)
endif()
