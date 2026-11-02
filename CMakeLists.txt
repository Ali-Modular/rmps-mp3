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
        $<$<CONFIG:Release>:-O3 -funroll-loops -march=native>)
endif()

# ----- core library -----
add_library(rmps STATIC
    src/core/tables.cpp
    src/core/bitstream.cpp
    src/core/huffman_tables.cpp
    src/core/subband.cpp
    src/core/mdct.cpp
    src/core/resample.cpp
    src/core/zerophase_lpf.cpp
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
    src/post/hf_extender.cpp
    src/control/abr_pid.cpp
    src/control/demand.cpp
    src/control/transient_gate.cpp
    src/encoder.cpp
    src/bend/bend.cpp
    src/decode/decoder.cpp
)
target_include_directories(rmps
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(rmps PRIVATE rmps_warnings rmps_optflags)

# ----- CLI / GUI app -----
#
# Optional dependencies, found through vcpkg's installed tree when present
# (RMPS_VCPKG_ROOT, default C:/vcpkg). Without them the app builds as before:
# WAV input only, command line only.
#   libsndfile (FLAC / Vorbis / Opus enabled)  -> FLAC, Ogg Vorbis, Ogg Opus input
#   glfw3 + imgui (glfw & opengl3 bindings)    -> GUI when started without arguments
set(RMPS_VCPKG_ROOT "C:/vcpkg" CACHE PATH "vcpkg root for optional app dependencies")
# Prefer the static libraries so rmps.exe is a single self-contained file. The
# -static-md triplet keeps the dynamic C runtime, which is what the rest of the
# project builds with, so the encoder library's own build is unchanged.
if(EXISTS "${RMPS_VCPKG_ROOT}/installed/x64-windows-static-md/lib/sndfile.lib")
    set(_rmps_default_triplet "x64-windows-static-md")
else()
    set(_rmps_default_triplet "x64-windows")
endif()
set(RMPS_VCPKG_TRIPLET "${_rmps_default_triplet}" CACHE STRING "vcpkg triplet for app dependencies")
if(NOT DEFINED VCPKG_TARGET_TRIPLET AND EXISTS "${RMPS_VCPKG_ROOT}/installed/${RMPS_VCPKG_TRIPLET}")
    list(APPEND CMAKE_PREFIX_PATH "${RMPS_VCPKG_ROOT}/installed/${RMPS_VCPKG_TRIPLET}")
endif()
find_package(SndFile CONFIG QUIET)
find_package(glfw3 CONFIG QUIET)
find_package(imgui CONFIG QUIET)
find_package(OpenGL QUIET)

add_executable(rmps_cli
    apps/rmps/main.cpp
    apps/rmps/wav_io.cpp
    apps/rmps/tui.cpp
    apps/rmps/utf8.cpp
    apps/rmps/audio_in.cpp
    apps/rmps/encode_job.cpp
)
set_target_properties(rmps_cli PROPERTIES OUTPUT_NAME rmps)
target_link_libraries(rmps_cli PRIVATE rmps rmps_warnings rmps_optflags)
if(MSVC)
    # The app has Japanese UI strings; read sources as UTF-8.
    target_compile_options(rmps_cli PRIVATE /utf-8)
endif()

if(TARGET SndFile::sndfile)
    target_compile_definitions(rmps_cli PRIVATE RMPS_HAVE_SNDFILE=1)
    target_link_libraries(rmps_cli PRIVATE SndFile::sndfile)
    message(STATUS "rmps: libsndfile found - FLAC / Ogg Vorbis / Ogg Opus input enabled")
else()
    message(STATUS "rmps: libsndfile not found - WAV input only")
endif()

if(TARGET glfw AND TARGET imgui::imgui AND TARGET OpenGL::GL)
    target_sources(rmps_cli PRIVATE apps/rmps/gui.cpp apps/rmps/analysis_view.cpp)
    target_compile_definitions(rmps_cli PRIVATE RMPS_HAVE_GUI=1)
    target_link_libraries(rmps_cli PRIVATE imgui::imgui glfw OpenGL::GL)
    if(WIN32)
        target_link_libraries(rmps_cli PRIVATE comdlg32 ole32 shell32)
    endif()
    message(STATUS "rmps: glfw + imgui found - GUI enabled")
else()
    message(STATUS "rmps: glfw / imgui not found - command line only")
endif()

# Put the dependency DLLs next to rmps.exe so it runs from the build tree and
# can be copied out as a folder. $<TARGET_RUNTIME_DLLS> only lists the direct
# imports (sndfile.dll, glfw3.dll); sndfile.dll in turn imports FLAC, Vorbis,
# Opus and Ogg, found by reading its import table, so those are listed here.
if(WIN32 AND (TARGET SndFile::sndfile OR TARGET glfw) AND NOT RMPS_VCPKG_TRIPLET MATCHES "static")
    set(_rmps_bin "${RMPS_VCPKG_ROOT}/installed/${RMPS_VCPKG_TRIPLET}/bin")
    set(_rmps_dlls "")
    foreach(_d sndfile FLAC ogg vorbis vorbisenc opus glfw3)
        if(EXISTS "${_rmps_bin}/${_d}.dll")
            list(APPEND _rmps_dlls "${_rmps_bin}/${_d}.dll")
        endif()
    endforeach()
    add_custom_command(TARGET rmps_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_rmps_dlls} $<TARGET_FILE_DIR:rmps_cli>)
endif()
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
