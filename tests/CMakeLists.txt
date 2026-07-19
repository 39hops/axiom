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
  core/rational_test.cpp
  core/nt_test.cpp
  core/fft_test.cpp
  la/mat_test.cpp
  la/matmul_test.cpp
  la/lu_test.cpp
  la/cholesky_test.cpp
  la/qr_test.cpp
  st/rng_test.cpp
  st/sf_test.cpp
  st/dist_cont_test.cpp
  st/dist_disc_test.cpp
  st/desc_test.cpp
  st/htest_test.cpp
  st/reg_test.cpp
  st/glm_test.cpp
  st/mcmc_test.cpp
  st/ts_test.cpp
  num/root_test.cpp
  num/quad_test.cpp
  num/ode_test.cpp
  num/opt_test.cpp
  sym/expr_test.cpp
  sym/calc_test.cpp
  sym/poly_test.cpp
  sym/print_test.cpp
  sym/expand_test.cpp
  sym/solve_test.cpp
  sym/integrate_test.cpp
  sym/parse_test.cpp
  sym/oracle_test.cpp
  sym/jsonl_test.cpp
  sym/print_sstr_test.cpp
  sym/count_ops_test.cpp
  pyrand/pyrand_test.cpp
  mathgen/expression_test.cpp
)
target_compile_definitions(ax_tests PRIVATE
  AX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(ax_tests PRIVATE axiom GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(ax_tests)
