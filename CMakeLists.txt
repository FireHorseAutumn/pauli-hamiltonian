cmake_minimum_required(VERSION 3.20)
project(pauli_hamiltonian LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen is allowed ONLY as the test_hubbard oracle. The library itself is
# dependency-free.
find_package(Eigen3 REQUIRED NO_MODULE)

enable_testing()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)

add_library(ph INTERFACE)
target_include_directories(ph INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_equality tests/test_equality.cpp)
target_link_libraries(test_equality PRIVATE ph)
target_compile_options(test_equality PRIVATE -Wall -Wextra)
add_test(NAME equality COMMAND test_equality)

add_executable(test_hubbard tests/test_hubbard.cpp)
target_link_libraries(test_hubbard PRIVATE ph Eigen3::Eigen)
target_compile_options(test_hubbard PRIVATE -Wall -Wextra)
add_test(NAME hubbard COMMAND test_hubbard)

string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)

add_executable(bench_build benchmarks/bench_build.cpp)
target_link_libraries(bench_build PRIVATE ph)
target_compile_options(bench_build PRIVATE -Wall -Wextra)
target_compile_definitions(bench_build PRIVATE
  PH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  PH_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  PH_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type_upper}}")
