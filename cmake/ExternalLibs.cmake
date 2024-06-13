include (ExternalProject)
include (GNUInstallDirs)


# argparse
add_library(external_argparse INTERFACE)
ExternalProject_Add(argparse
    PREFIX ${CMAKE_SOURCE_DIR}/build/argparse/build
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/3rdParty/argparse/
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/libs/argparse
)
target_include_directories(external_argparse
    INTERFACE ${CMAKE_SOURCE_DIR}/libs/argparse/include)