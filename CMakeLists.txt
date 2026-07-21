cmake_minimum_required(VERSION 3.28)
project(axiom LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(axiom STATIC
  src/core/bigint.cpp
  src/core/rational.cpp
  src/core/nt.cpp
  src/core/fft.cpp
  src/la/blas.cpp
  src/la/decomp.cpp
  src/st/rng.cpp
  src/st/sf.cpp
  src/st/dist.cpp
  src/st/desc.cpp
  src/st/htest.cpp
  src/st/reg.cpp
  src/st/mcmc.cpp
  src/st/ts.cpp
  src/num/root.cpp
  src/num/quad.cpp
  src/num/ode.cpp
  src/num/opt.cpp
  src/sym/expr.cpp
  src/sym/calc.cpp
  src/sym/poly.cpp
  src/sym/print.cpp
  src/sym/expand.cpp
  src/sym/solve.cpp
  src/sym/integrate.cpp
  src/sym/parse.cpp
  src/sym/oracle.cpp
  src/sym/jsonl.cpp
  src/sym/harness.cpp
  src/sym/print_sstr.cpp
  src/sym/count_ops.cpp
  src/sym/series.cpp
  src/sym/series_oracle.cpp
  src/search/verify.cpp
  src/search/successors.cpp
  src/search/beam.cpp
  src/search/rules.cpp
  src/search/rules2.cpp
  src/search/rules3.cpp
  src/search/engine.cpp
  src/search/emit.cpp
  src/search/firemask.cpp
  src/pyrand/sha512.cpp
  src/pyrand/pyrand.cpp
  src/mathgen/problems.cpp
  src/mathgen/ode.cpp
  src/mathgen/series_solve.cpp
)
target_include_directories(axiom PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(axiom PUBLIC cxx_std_23)
if(MSVC)
  target_compile_options(axiom PUBLIC /W4 /permissive- /EHsc)
endif()

add_executable(axiom-oracle tools/oracle_main.cpp)
target_link_libraries(axiom-oracle PRIVATE axiom)

add_executable(axiom-qual-gate tools/qual_gate_main.cpp)
target_link_libraries(axiom-qual-gate PRIVATE axiom)

# Optional Python bridge (llmopt shadow-mode consumer). Off by default so
# the STL-only core stays dependency-free; pybind11 is bridge-only, like
# GoogleTest is test-only. Point CMake at pybind11 via
# -Dpybind11_DIR=$(python -m pybind11 --cmakedir).
option(AXIOM_BUILD_PYTHON "Build the axiom_sym pybind11 module" OFF)
if(AXIOM_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(axiom_sym bindings/axiom_sym.cpp)
  target_link_libraries(axiom_sym PRIVATE axiom)
endif()

option(AXIOM_BUILD_TESTS "Build tests" ON)
if(AXIOM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

option(AXIOM_BUILD_BENCH "Build benchmarks" ON)
if(AXIOM_BUILD_BENCH)
  add_subdirectory(bench)
endif()

add_executable(axiom-chain-emit tools/chain_emit_main.cpp)
target_link_libraries(axiom-chain-emit PRIVATE axiom)

add_executable(axiom-ode-sample tools/ode_sample_main.cpp)
target_link_libraries(axiom-ode-sample PRIVATE axiom)

add_executable(axiom-series-sample tools/series_sample_main.cpp)
target_link_libraries(axiom-series-sample PRIVATE axiom)
