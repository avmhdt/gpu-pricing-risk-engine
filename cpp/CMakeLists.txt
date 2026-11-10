# C++20 baseline for gpre (FR-25b, D-09). M1 builds the shared Philox header and its tests;
# the multithreaded pricer (gpre_cpu_baseline) and Google Benchmark arrive in M6.
#
# Build trees live on the WSL ext4 filesystem, never in the repo:
#   cmake -S cpp -B ~/build/gpu-pricing-risk-engine/cpp -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake_minimum_required(VERSION 3.28)
project(gpre_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GPRE_NATIVE "Build with -march=native (benchmarks only; off in CI)" OFF)
set(GPRE_SANITIZE "" CACHE STRING "Sanitizers: empty, 'address,undefined' or 'thread'")

set(GPRE_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(GPRE_KERNEL_DIR ${GPRE_REPO_ROOT}/src/gpre/cuda/kernels)

add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
if(GPRE_NATIVE)
  add_compile_options(-march=native)
endif()
if(GPRE_SANITIZE)
  add_compile_options(-fsanitize=${GPRE_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${GPRE_SANITIZE})
endif()

include(cmake/get_cpm.cmake)
CPMAddPackage(NAME googletest GITHUB_REPOSITORY google/googletest GIT_TAG v1.18.0
              OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON")
CPMAddPackage(NAME nlohmann_json GITHUB_REPOSITORY nlohmann/json VERSION 3.12.0
              OPTIONS "JSON_BuildTests OFF")

# Header-only library: the Philox header shared with the CUDA kernels, plus baseline headers.
add_library(gpre_baseline INTERFACE)
target_include_directories(gpre_baseline INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                   ${GPRE_KERNEL_DIR})

enable_testing()
add_executable(test_philox tests/test_philox.cpp)
target_link_libraries(test_philox PRIVATE gpre_baseline GTest::gtest_main nlohmann_json::nlohmann_json)
target_compile_definitions(test_philox PRIVATE
  GPRE_KAT_PATH="${GPRE_REPO_ROOT}/tests/fixtures/philox_kat.json")
include(GoogleTest)
gtest_discover_tests(test_philox)
