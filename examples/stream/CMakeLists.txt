if (WHISPER_SDL2)
    set(TARGET whisper-stream)
    add_executable(${TARGET} stream.cpp)

    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)

    include(DefaultTargetOptions)

    target_link_libraries(${TARGET} PRIVATE pybind11::embed common common-sdl whisper ${CMAKE_THREAD_LIBS_INIT})

    install(TARGETS ${TARGET} RUNTIME)
endif ()
