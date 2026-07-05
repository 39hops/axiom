cmake_minimum_required(VERSION 3.28)
project(axiom LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(axiom INTERFACE)
target_include_directories(axiom INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
  target_compile_options(axiom INTERFACE /W4 /permissive- /EHsc)
endif()

option(AXIOM_BUILD_TESTS "Build tests" ON)
if(AXIOM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
