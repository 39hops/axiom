include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(ax_tests
  smoke_test.cpp
  par/pool_test.cpp
  core/bigint_basic_test.cpp
  core/bigint_addsub_test.cpp
  core/bigint_mul_test.cpp
  core/bigint_div_test.cpp
  core/bigint_karatsuba_test.cpp
  core/bigint_gcd_pow_test.cpp
)
target_link_libraries(ax_tests PRIVATE axiom GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(ax_tests)
