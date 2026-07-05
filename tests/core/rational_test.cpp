#include <gtest/gtest.h>
#include <ax/core/rational.hpp>

#include <stdexcept>

using ax::bigint;
using ax::rational;

TEST(rational, normalizes_on_construction) {
  rational r{bigint{6}, bigint{8}};
  EXPECT_EQ(r.num().to_string(), "3");
  EXPECT_EQ(r.den().to_string(), "4");
  rational s{bigint{3}, bigint{-4}};  // sign lives in numerator
  EXPECT_EQ(s.num().to_string(), "-3");
  EXPECT_EQ(s.den().to_string(), "4");
  EXPECT_THROW((rational{bigint{1}, bigint{0}}), std::domain_error);
}

TEST(rational, arithmetic) {
  rational half{bigint{1}, bigint{2}};
  rational third{bigint{1}, bigint{3}};
  EXPECT_EQ((half + third).to_string(), "5/6");
  EXPECT_EQ((half - third).to_string(), "1/6");
  EXPECT_EQ((half * third).to_string(), "1/6");
  EXPECT_EQ((half / third).to_string(), "3/2");
  EXPECT_THROW(half / rational{}, std::domain_error);
}

TEST(rational, ordering_and_integer_print) {
  EXPECT_LT((rational{bigint{1}, bigint{3}}), (rational{bigint{1}, bigint{2}}));
  EXPECT_EQ((rational{bigint{4}, bigint{2}}).to_string(), "2");
  EXPECT_EQ(rational{}.to_string(), "0");
}
