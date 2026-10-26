cmake_minimum_required(VERSION 3.20)

# Select a supported CUDA host compiler before project() enables CUDA.
# With GCC 16 or newer, prefer g++-15; otherwise enable the compiler override
# and use C++20 for CUDA, since newer libstdc++ headers require it.
find_program(_cxx_compiler NAMES $ENV{CXX} c++ g++)
if(_cxx_compiler)
    execute_process(COMMAND ${_cxx_compiler} -dumpversion
                    OUTPUT_VARIABLE _gcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(_gcc_version VERSION_GREATER_EQUAL 16)
    find_program(_gcc15 NAMES g++-15)
    if(_gcc15)
        set(CMAKE_CUDA_HOST_COMPILER "${_gcc15}")
    else()
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -std=c++20")
    endif()
endif()

project(pauli_hamiltonian LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen is allowed ONLY as the test_hubbard oracle. The library itself is
# dependency-free.
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

enable_testing()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)

add_library(ph INTERFACE)
target_include_directories(ph INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(test_equality tests/test_equality.cpp)
target_link_libraries(test_equality PRIVATE ph)
target_include_directories(test_equality PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(test_equality PRIVATE -Wall -Wextra)
add_test(NAME equality COMMAND test_equality)

add_executable(test_hubbard tests/test_hubbard.cpp)
target_link_libraries(test_hubbard PRIVATE ph Eigen3::Eigen)
target_compile_options(test_hubbard PRIVATE -Wall -Wextra)
add_test(NAME hubbard COMMAND test_hubbard)

add_executable(test_gpu_equality tests/test_gpu_equality.cu)
target_link_libraries(test_gpu_equality PRIVATE ph CUDA::cudart)
target_include_directories(test_gpu_equality PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_options(test_gpu_equality PRIVATE -Wall -Wextra)
add_test(NAME gpu_equality COMMAND test_gpu_equality)

string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)

add_executable(bench_build benchmarks/bench_build.cpp)
target_link_libraries(bench_build PRIVATE ph)
target_compile_options(bench_build PRIVATE -Wall -Wextra)
target_compile_definitions(bench_build PRIVATE
  PH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  PH_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  PH_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type_upper}}")

add_executable(bench_gpu_build benchmarks/bench_gpu_build.cu)
target_link_libraries(bench_gpu_build PRIVATE ph CUDA::cudart)
target_compile_options(bench_gpu_build PRIVATE -Wall -Wextra)
target_compile_definitions(bench_gpu_build PRIVATE
  PH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  PH_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  PH_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type_upper}}")
