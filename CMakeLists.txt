cmake_minimum_required(VERSION 3.5) # for add_link_options and implicit target directories.
project("whisper.cpp" C CXX)
project("whisper.cpp" VERSION 1.9.1)
include(CheckIncludeFileCXX)

set(SOVERSION 1)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(WHISPER_STANDALONE ON)

    include(git-vars)

    # configure project version
    configure_file(${CMAKE_SOURCE_DIR}/bindings/javascript/package-tmpl.json ${CMAKE_SOURCE_DIR}/bindings/javascript/package.json @ONLY)
else()
    set(WHISPER_STANDALONE OFF)
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    option(WHISPER_COMMON_FFMPEG "whisper: examples link with ffmpeg libs in order to decode more audio formats" OFF)
endif()

option(WHISPER_COREML                "whisper: enable Core ML framework"  OFF)
option(WHISPER_COREML_ALLOW_FALLBACK "whisper: allow non-CoreML fallback" OFF)
option(WHISPER_OPENVINO              "whisper: support for OpenVINO"      OFF)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)

# transition helpers
function (whisper_option_depr TYPE OLD NEW)
    if (${OLD})
        message(${TYPE} "${OLD} is deprecated and will be removed in the future.\nUse ${NEW} instead\n")
        set(${NEW} ON)
    endif()
endfunction()

if (GGML_CUDA AND NOT MSVC)
    #GGML_CUDA enabled, add the necessary compile options -Wno-deprecated-gpu-targets
    add_compile_options(-Wno-deprecated-gpu-targets)
endif()

#
# build the library
#

if (NOT TARGET ggml)
    if (WHISPER_USE_SYSTEM_GGML)
        find_package(ggml REQUIRED)
        if (NOT ggml_FOUND)
            message(FATAL_ERROR "System-installed GGML library not found.")
        endif()
        add_library(ggml ALIAS ggml::ggml)
    else()
        add_subdirectory(ggml)
        if(WIN32)
            # The following adds a _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR macro and is a workaround for
            # the Windows C++ standard library which does not support constexpr mutexes.
            # From the release notes://github.com/microsoft/STL/wiki/Changelog
            #  Disable constexpr mutex constructor on Windows
            #  Fixed mutex's constructor to be constexpr. #3824 #4000 #4339
            #  Note: Programs that aren't following the documented restrictions on binary compatibility may encounter
            #  null dereferences in mutex machinery. You must follow this rule:
            #  When you mix binaries built by different supported versions of the toolset, the Redistributable version
            #  must be at least as new as the latest toolset used by any app component.
            #  You can define _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR as an escape hatch.
            #
            # Specifically to whisper.cpp this would cause a crash when using the Java bindings.
            # resulting in a Invalid memory access error.
            target_compile_definitions(ggml-base PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
        endif()
    endif()
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()
add_subdirectory(src)

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(WHISPER_BUILD_NUMBER        ${BUILD_NUMBER})
set(WHISPER_BUILD_COMMIT        ${BUILD_COMMIT})
set(WHISPER_INSTALL_VERSION     ${CMAKE_PROJECT_VERSION})

set(WHISPER_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(WHISPER_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(WHISPER_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

get_directory_property(WHISPER_TRANSIENT_DEFINES COMPILE_DEFINITIONS)

set_target_properties(whisper PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/whisper.h)

install(TARGETS whisper LIBRARY PUBLIC_HEADER)

target_compile_definitions(whisper PRIVATE
    WHISPER_VERSION="${PROJECT_VERSION}"
)

set_target_properties(parakeet PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/parakeet.h)
install(TARGETS parakeet LIBRARY PUBLIC_HEADER)

target_compile_definitions(parakeet PRIVATE
    PARAKEET_VERSION="${PROJECT_VERSION}"
)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/whisper-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/whisper-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/whisper
    PATH_VARS
    WHISPER_INCLUDE_INSTALL_DIR
    WHISPER_LIB_INSTALL_DIR
    WHISPER_BIN_INSTALL_DIR )

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/whisper-version.cmake
    VERSION ${WHISPER_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/whisper-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/whisper-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/whisper)

configure_file(cmake/whisper.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/whisper.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/whisper.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

set(PARAKEET_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(PARAKEET_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(PARAKEET_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/parakeet-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/parakeet-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parakeet
    PATH_VARS
    PARAKEET_INCLUDE_INSTALL_DIR
    PARAKEET_LIB_INSTALL_DIR
    PARAKEET_BIN_INSTALL_DIR)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/parakeet-version.cmake
    VERSION ${WHISPER_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parakeet-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/parakeet-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parakeet)

configure_file(cmake/parakeet.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/parakeet.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/parakeet.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

add_subdirectory(examples)