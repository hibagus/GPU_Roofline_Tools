set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# AMD MI300X GPU Benchmark
set(AMD_MI300X_BENCH_EXECUTABLE_NAME "amd_mi300x_bench")
add_executable(${AMD_MI300X_BENCH_EXECUTABLE_NAME} amd_mi300x_benchmark.cpp)
add_dependencies(${AMD_MI300X_BENCH_EXECUTABLE_NAME} argparse)
set_target_properties(${AMD_MI300X_BENCH_EXECUTABLE_NAME}
                        PROPERTIES 
                        HIP_SEPARABLE_COMPILATION ON
                        HIP_RESOLVE_DEVICE_SYMBOLS ON
                        HIP_ARCHITECTURES ${GPU_ARCHITECTURE_SUPPORT}
                        CXX_STANDARD 17)
  
target_link_libraries(${AMD_MI300X_BENCH_EXECUTABLE_NAME}
                      PRIVATE ${PROJECT_AMD_BENCH} external_argparse)