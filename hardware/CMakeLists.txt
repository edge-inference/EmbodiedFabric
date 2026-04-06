cmake_minimum_required(VERSION 3.14)
project(vla_preprocess_soc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(SystemCLanguage QUIET)
if(SystemCLanguage_FOUND)
    set(SYSTEMC_LIBS SystemC::systemc)
elseif(DEFINED ENV{SYSTEMC_HOME})
    set(SYSTEMC_HOME $ENV{SYSTEMC_HOME})
    include_directories(${SYSTEMC_HOME}/include)
    link_directories(${SYSTEMC_HOME}/lib-linux64 ${SYSTEMC_HOME}/lib)
    set(SYSTEMC_LIBS systemc)
else()
    find_library(SYSTEMC_LIB systemc)
    if(SYSTEMC_LIB)
        set(SYSTEMC_LIBS ${SYSTEMC_LIB})
    else()
        message(FATAL_ERROR "SystemC not found. Install libsystemc-dev or set SYSTEMC_HOME.")
    endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

set(PREPROCESS_SRCS
    src/image_resize.cpp
    src/state_normalize.cpp
    src/sram_interface.cpp
    src/preprocess_block.cpp
)

add_executable(tb_preprocess
    tb/tb_preprocess.cpp
    ${PREPROCESS_SRCS}
)
target_link_libraries(tb_preprocess ${SYSTEMC_LIBS})
target_compile_definitions(tb_preprocess PRIVATE
    TV_DIR="${CMAKE_SOURCE_DIR}/tb/test_vectors"
)
