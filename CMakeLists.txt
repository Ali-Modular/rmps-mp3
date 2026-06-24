cmake_minimum_required(VERSION 3.20)
project(rmps
    VERSION   0.1.0
    LANGUAGES CXX
    DESCRIPTION "Rate-Managed Perceptual Encoder — MP3 encoder with beam-search trellis quantization, perceptual SDR, ABR PID and noise shaping.")

# ----- global -----
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# Position-independent code (allows linking the static lib into shared libs later).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----- warnings / optimisations -----
add_library(rmps_warnings INTERFACE)
if(MSVC)
    target_compile_options(rmps_warnings INTERFACE /W4 /permissive- /Zc:__cplusplus)
else()
    target_compile_options(rmps_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
endif()

add_library(rmps_optflags INTERFACE)
if(MSVC)
    target_compile_options(rmps_optflags INTERFACE
        $<$<CONFIG:Release>:/O2 /Oi /Ot /Oy /GT /fp:fast>)
else()
    target_compile_options(rmps_optflags INTERFACE
        $<$<CONFIG:Release>:-O3 -ffast-math -fno-math-errno -funroll-loops -march=native>)
endif()

# ----- core library -----
add_library(rmps STATIC
    src/core/tables.cpp
    src/core/bitstream.cpp
    src/core/huffman_tables.cpp
    src/core/subband.cpp
    src/core/mdct.cpp
    src/core/reservoir.cpp
    src/core/l3_bitstream.cpp
    src/core/encoder_state.cpp
    src/quant/quantize.cpp
    src/quant/iteration_loop.cpp
    src/quant/beam_search.cpp
    src/quant/short_block.cpp
    src/quant/hf_rdo.cpp
    src/post/perceptual.cpp
    src/post/ms_stereo.cpp
    src/post/lowpass.cpp
    src/post/hf_adaptor.cpp
    src/post/masking.cpp
    src/control/abr_pid.cpp
    src/control/transient_gate.cpp
    src/encoder.cpp
)
target_include_directories(rmps
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(rmps PRIVATE rmps_warnings rmps_optflags)

# ----- CLI app -----
add_executable(rmps_cli
    apps/rmps/main.cpp
    apps/rmps/wav_io.cpp
    apps/rmps/tui.cpp
)
set_target_properties(rmps_cli PROPERTIES OUTPUT_NAME rmps)
target_link_libraries(rmps_cli PRIVATE rmps rmps_warnings rmps_optflags)
if(WIN32 AND NOT MSVC)
    target_link_options(rmps_cli PRIVATE -static)
endif()

# ----- Link-Time Optimization (LTO) / Interprocedural Optimization (IPO) -----
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported LANGUAGES CXX)
if(ipo_supported)
    set_target_properties(rmps PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(rmps_cli PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ----- install -----
include(GNUInstallDirs)
install(TARGETS rmps rmps_cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/rmps DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message(STATUS "rmps ${PROJECT_VERSION}: ${CMAKE_BUILD_TYPE} build, C++${CMAKE_CXX_STANDARD}")

# ---- README hint when invoking interactively ----
if(CMAKE_GENERATOR MATCHES "Make|Ninja")
    message(STATUS "rmps: build with 'cmake --build ${CMAKE_BINARY_DIR}'")
endif()
