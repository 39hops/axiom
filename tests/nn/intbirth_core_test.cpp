/** @file intbirth_core_test.cpp The <Op, Acc, Round> core must be
    op-for-op the shipped i64 engine at its i64 instantiation (the
    digest drivers arbitrate the full engine; these cross-check the
    extracted primitives directly, ring-axiom style on random input). */
#include <gtest/gtest.h>

#include <ax/nn/intbirth.hpp>
#include <ax/nn/intbirth_core.hpp>

#include <random>

namespace ib = ax::nn::ib;
using ib::i64;

TEST(IntbirthCore, I64InstantiationMatchesShippedGemmForms) {
  std::mt19937_64 rng(613);
  std::uniform_int_distribution<i64> d(-1000, 1000);
  ib::Mat a(6 * 8), w(4 * 8), wn(8 * 4);
  for (auto& v : a) v = d(rng);
  for (auto& v : w) v = d(rng);
  for (auto& v : wn) v = d(rng);
  EXPECT_EQ((ib::core::gemm<i64, i64>(a, 6, 8, w, 4)),
            ib::int_gemm(a, 6, 8, w, 4));
  EXPECT_EQ((ib::core::gemm_nt<i64, i64>(a, 6, 8, wn, 4)),
            ib::int_gemm_nt(a, 6, 8, wn, 4));
  ib::Mat y(6 * 4);
  for (auto& v : y) v = d(rng);
  EXPECT_EQ((ib::core::gemm_xty<i64, i64>(a, 6, 8, y, 4)),
            ib::int_gemm_xty(a, 6, 8, y, 4));
}

TEST(IntbirthCore, RoundHalfAwayMatchesShippedRdiv) {
  ib::Mat m{7, -7, 5, -5, 512, -513, 0, 1023, -1024}, m2 = m;
  ib::rdiv_inplace(m, 2);
  ib::core::rdiv_inplace<i64, ib::core::RoundHalfAway<i64>>(m2, 2);
  EXPECT_EQ(m, m2);
  ib::Mat a{100000, -99999, 314159}, a2 = a;
  ib::rdiv_inplace(a, 512);
  ib::core::rdiv_inplace<i64, ib::core::RoundHalfAway<i64>>(a2, 512);
  EXPECT_EQ(a, a2);
}

TEST(IntbirthCore, WideAccDoesNotOverflow) {
  // K=2 dot with products near 2^62 each: i64 accumulation of the
  // intermediate pair would be UB-adjacent; i128 must return the
  // exact cancelled sum.
  std::vector<i64> a{i64{1} << 40, i64{1} << 40};
  std::vector<i64> w{(i64{1} << 22) - 1, -(i64{1} << 22)};
  auto r = ib::core::gemm<i64, __int128>(a, 1, 2, w, 1);
  EXPECT_EQ(r[0], -(i64{1} << 40));
}
