set(TARGET llama-diffusion-cli-tool)

add_executable(${TARGET} diffusion-cli.cpp)

# Include paths for common headers
include_directories(${CMAKE_SOURCE_DIR}/common)

target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})

# Additional include directories for llama headers
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
)

set_target_properties(${TARGET} PROPERTIES
    OUTPUT_NAME llama-diffusion-cli
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
