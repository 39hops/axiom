#include <gtest/gtest.h>
#include <ax/core/version.hpp>

TEST(smoke, version_is_zero_one) {
  EXPECT_EQ(ax::version_major, 0);
  EXPECT_EQ(ax::version_minor, 1);
}
