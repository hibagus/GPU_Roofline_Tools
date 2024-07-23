macro (add_sources sourcelist)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND SOURCES "${CMAKE_SOURCE_DIR}/${_relPath}/${_src}")
        else()
            list (APPEND SOURCES "${CMAKE_SOURCE_DIR}/${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SOURCES to parent directory
        set (${sourcelist} ${${sourcelist}} ${SOURCES} PARENT_SCOPE)
    endif()
endmacro()

# Iterate through source files
add_subdirectory(utils)
add_subdirectory(kernel)
add_subdirectory(launcher)


# Set library
## PROJECT_AMD_BENCH
add_library(${PROJECT_AMD_BENCH} STATIC ${PROJECT_AMD_SRC})
set_target_properties(${PROJECT_AMD_BENCH}
                        PROPERTIES
                          CXX_STANDARD 17
                          CXX_STANDARD_REQUIRED ON
                          CXX_EXTENSIONS OFF
                          HIP_STANDARD 17
                          HIP_STANDARD_REQUIRED ON
                          HIP_EXTENSIONS OFF
                          HIP_SEPARABLE_COMPILATION ON
                          HIP_ARCHITECTURES ${GPU_ARCHITECTURE_SUPPORT})
#target_link_libraries(${PROJECT_AMD_BENCH}
#                      PRIVATE hip::device)
#target_compile_definitions(${PROJECT_AMD_BENCH} PRIVATE __HIP_PLATFORM_AMD__)