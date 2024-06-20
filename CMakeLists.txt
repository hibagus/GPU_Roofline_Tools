cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Declare Project Name and Project Type
set(PROJECT_NAME "GPU_Roofline_Tools")
project(${PROJECT_NAME} LANGUAGES CXX)

# Choose between CUDA or HIP
set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    set(ERROR_MESSAGE "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP or CUDA.")
    message(FATAL_ERROR ${ERROR_MESSAGE})
endif()

# Add HIP or CUDA language based on GPU_Runtime; Set standard to C++17
enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)


# prevent in source builds
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
    message(SEND_ERROR "In-source builds are not allowed.")
endif()

if("${GPU_RUNTIME}" STREQUAL "CUDA")
    find_package(CUDA REQUIRED)
elseif ("${GPU_RUNTIME}" STREQUAL "HIP")
    find_package(ROCM CONFIG REQUIRED)
    if (${ROCM_FOUND})
        if(NOT DEFINED ROCM_ROOT)
            set(ROCM_ROOT "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
        endif()
        list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
    endif()
endif()

# GPU Runtime Selector
#if("${GPU_RUNTIME}" STREQUAL "CUDA")
#   find_package(CUDA REQUIRED)
#   enable_language(CUDA)
#elseif ("${GPU_RUNTIME}" STREQUAL "ROCM")
#   find_package(ROCM CONFIG REQUIRED)
#   if (${ROCM_FOUND})
#        if(NOT DEFINED ROCM_ROOT)
#            set(ROCM_ROOT "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
#        endif()
#        include_directories(${ROCM_ROOT}/include)
#   endif()
#   enable_language(HIP)
#else()
#    message(FATAL_ERROR "Unrecognized GPU Runtime Mode")
#endif()
#
#set(GPU_RUNTIMES "ROCM" "CUDA")
#set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

# set to Release to enable optimization
#        Debug for debugging
if("${BUILD_MODE}" STREQUAL "Debug")
    set(CMAKE_BUILD_TYPE Debug)
elseif("${BUILD_MODE}" STREQUAL "Release")
    set(CMAKE_BUILD_TYPE Release)
else()
    message(FATAL_ERROR "Unrecognized build mode")
endif()

# Runtime Settings
if("${GPU_RUNTIME}" STREQUAL "CUDA")
    include(cmake/CudaSettings.cmake)
elseif ("${GPU_RUNTIME}" STREQUAL "HIP")
    include(cmake/RocmSettings.cmake)
endif()

# interface library for external library
include(cmake/ExternalLibs.cmake)

# add project include dirs for everything
include_directories(${CMAKE_SOURCE_DIR}/include)

# include ROCM directory
#if("${GPU_RUNTIME}" STREQUAL "HIP")
#    include_directories(${ROCM_ROOT}/include)
#endif()

# add project library
set(PROJECT_AMD_BENCH "amdbenchlib" )

# Add Source Code and Apps
add_subdirectory(src)
add_subdirectory(apps)