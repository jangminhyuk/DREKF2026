cmake_minimum_required(VERSION 3.15)
project(dr_ekf_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Eigen (header-only, fetch if not found) ---
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

# --- pybind11 ---
find_package(pybind11 REQUIRED)

# --- MOSEK ---
set(MOSEK_DIR "" CACHE PATH "Path to MOSEK platform directory (e.g. C:/Program Files/Mosek/10.1/tools/platform/win64x86)")
if(NOT MOSEK_DIR)
    if(DEFINED ENV{MOSEK_DIR})
        set(MOSEK_DIR $ENV{MOSEK_DIR})
    endif()
endif()

find_path(MOSEK_INCLUDE_DIR mosek.h PATHS ${MOSEK_DIR}/h)
find_library(MOSEK_LIBRARY NAMES mosek64_10_1 mosek64 PATHS ${MOSEK_DIR}/bin ${MOSEK_DIR}/lib)

if(NOT MOSEK_INCLUDE_DIR OR NOT MOSEK_LIBRARY)
    message(FATAL_ERROR "MOSEK not found. Set MOSEK_DIR to the platform directory, e.g.:\n"
        "  cmake .. -DMOSEK_DIR=\"C:/Program Files/Mosek/10.1/tools/platform/win64x86\"")
endif()

message(STATUS "MOSEK include: ${MOSEK_INCLUDE_DIR}")
message(STATUS "MOSEK library: ${MOSEK_LIBRARY}")

# --- Build the Python module ---
pybind11_add_module(dr_ekf_cpp
    src/dynamics.cpp
    src/kalman_utils.cpp
    src/fw_oracle.cpp
    src/ekf.cpp
    src/dr_ekf_cdc.cpp
    src/dr_ekf_tac.cpp
    src/mosek_sdp_solver.cpp
    src/bindings.cpp
)

target_include_directories(dr_ekf_cpp PRIVATE include ${MOSEK_INCLUDE_DIR})
target_link_libraries(dr_ekf_cpp PRIVATE Eigen3::Eigen ${MOSEK_LIBRARY})

# Optimization flags and static linking for MinGW
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(dr_ekf_cpp PRIVATE -O3 -march=native -DNDEBUG)
    # Static link MinGW runtime to avoid DLL dependencies
    if(MINGW)
        target_link_options(dr_ekf_cpp PRIVATE -static-libgcc -static-libstdc++)
    endif()
elseif(MSVC)
    target_compile_options(dr_ekf_cpp PRIVATE /O2 /DNDEBUG)
endif()

# Install the module next to the Python estimator package
install(TARGETS dr_ekf_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)
