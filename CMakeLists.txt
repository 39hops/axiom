cmake_minimum_required(VERSION 3.28)
project(axiom LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(axiom STATIC
  src/core/bigint.cpp
  src/core/rational.cpp
  src/core/nt.cpp
)
target_include_directories(axiom PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(axiom PUBLIC cxx_std_23)
if(MSVC)
  target_compile_options(axiom PUBLIC /W4 /permissive- /EHsc)
endif()

option(AXIOM_BUILD_TESTS "Build tests" ON)
if(AXIOM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
