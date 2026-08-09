/** @file intbirth_test.cpp ENGINE-EXACT-1 ladder surface: the
    precision contract field. The engine's arithmetic itself is
    certified by the pinned-digest drivers in tools/int_adamw/; these
    tests cover the ladder plumbing those drivers don't see. */
#include <gtest/gtest.h>

#include <ax/nn/intbirth.hpp>

#include <stdexcept>

namespace ib = ax::nn::ib;

TEST(IntbirthPrecision, DefaultIsShippedGrain) {
  ib::contract c;
  EXPECT_EQ(c.precision, 9);
  EXPECT_EQ(ib::contract_Q(c), 512);
}

TEST(IntbirthPrecision, UnsupportedPrecisionRefusesLoudly) {
  // Refuse-if-disagree: an unwired precision must abort at
  // construction, before a single training step can run.
  ib::contract c;
  c.precision = 33;
  EXPECT_THROW(ib::block("", c), std::runtime_error);
  c.precision = 0;
  EXPECT_THROW(ib::block("", c), std::runtime_error);
}
