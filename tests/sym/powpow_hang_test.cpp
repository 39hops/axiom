/** Regression probes for the (x^n)^(p/q) family hang (v5 farm root
    m-l8-v5s1-330: i_sqrt_basis wedged on a nested-integral integrand
    with x/sqrt(x**2)). Each probe runs the suspect sym call under the
    work budget; a hang = budget throw = test failure surfaces where. */
#include <ax/sym/budget.hpp>
#include <ax/sym/expand.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print_sstr.hpp>

#include <gtest/gtest.h>

using namespace ax;

namespace {
const sym::expr X = sym::expr::symbol("x");

// each stage under its own budget so the failing stage names itself
template <typename F>
bool finishes(F&& f) {
  try {
    sym::work_budget_scope budget(std::chrono::milliseconds(4000));
    f();
    return true;
  } catch (const std::exception&) {
    return false;
  }
}
}  // namespace

TEST(PowPowHang, CanonicalOnNestedIntegralWithSqrtSq) {
  const sym::expr f = sym::parse(
      "x*Integral(3*x*Integral(16*sqrt(2), x), x)/sqrt(x**2)");
  const sym::expr prod = f * sym::expr::fn(
      "sqrt", std::vector<sym::expr>{sym::parse("x**2")});
  EXPECT_TRUE(finishes([&] { (void)sym::canonical(prod, X); }))
      << "canonical hangs (budget never polled) on " << sym::to_sstr(prod);
}

TEST(PowPowHang, CanonicalPowPowFamily) {
  // the general (x^n)^(p/q) family: none of these may hang or fold
  // (x^even)^(odd/2) into a plain power (that is |x|, not x)
  for (const char* s :
       {"(x**2)**(3/2)", "(x**2)**(1/2)", "(x**3)**(3/2)",
        "(x**4)**(3/4)", "(x**2)**(5/2)", "(x**6)**(1/3)",
        "(x**3)**(2/3)", "(x**5)**(7/5)"}) {
    const sym::expr e = sym::parse(s);
    EXPECT_TRUE(finishes([&] { (void)sym::canonical(e, X); })) << s;
    EXPECT_TRUE(finishes([&] { (void)sym::expand(e); })) << s;
  }
}

TEST(PowPowHang, CanonicalQuotientOfPowPow) {
  // quotient shapes that trigger the sqrt-merge path
  for (const char* s :
       {"x/sqrt(x**2)", "(x**2)**(3/2)/x", "x**2/(x**2)**(3/2)",
        "sqrt(x**2)*(x**2)**(3/2)", "(x**2)**(3/2)*(x**2)**(-1/2)",
        "(x**3)**(3/2)/(2*x)"}) {
    const sym::expr e = sym::parse(s);
    EXPECT_TRUE(finishes([&] { (void)sym::canonical(e, X); })) << s;
  }
}

#include <ax/search/search.hpp>

TEST(PowPowHang, ISqrtBasisOnWedgeNode) {
  // the exact node the v5 farm wedged on (fire-start trace): the rule
  // must return within the work budget, hang = regression
  const sym::expr node = sym::parse(
      "Integral(x*Integral(3*x*Integral(16*sqrt(2), x), x)/sqrt(x**2), x)");
  const search::rule* sqrt_basis = nullptr;
  for (const auto& r : search::default_rules().integral)
    if (r.first == "i_sqrt_basis") sqrt_basis = &r;
  ASSERT_NE(sqrt_basis, nullptr);
  EXPECT_TRUE(finishes([&] { (void)sqrt_basis->second(node); }))
      << "i_sqrt_basis hangs on the wedge node";
}

TEST(PowPowHang, VerifyEdgeOnSqrtBasisRewrite) {
  // the actual wedge (v5 farm): the hang lives in verify_edge on the
  // i_sqrt_basis rewrite, not in the rule fire itself
  const sym::expr node = sym::parse(
      "Integral(x*Integral(3*x*Integral(16*sqrt(2), x), x)/sqrt(x**2), x)");
  const search::rule* sqrt_basis = nullptr;
  for (const auto& r : search::default_rules().integral)
    if (r.first == "i_sqrt_basis") sqrt_basis = &r;
  ASSERT_NE(sqrt_basis, nullptr);
  const auto rewrites = sqrt_basis->second(node);
  const search::external_slots ext{};
  for (const auto& rw : rewrites)
    EXPECT_TRUE(finishes([&] {
      (void)search::verify_edge(node, rw, ext);
    })) << "verify_edge hangs on rewrite " << sym::to_sstr(rw);
}

TEST(PowPowHang, ISqrtBasisOnPowFormWedgeNode) {
  // to_sstr prints pow(b, 1/2) and fn-sqrt(b) identically, so the
  // farm's wedge node may be the POW form even though the trace reads
  // sqrt(x**2) — probe both spellings
  const sym::expr node = sym::parse(
      "Integral(x*Integral(3*x*Integral(16*sqrt(2), x), x)"
      "*(x**2)**(-1/2), x)");
  const search::rule* sqrt_basis = nullptr;
  for (const auto& r : search::default_rules().integral)
    if (r.first == "i_sqrt_basis") sqrt_basis = &r;
  ASSERT_NE(sqrt_basis, nullptr);
  std::vector<sym::expr> rewrites;
  EXPECT_TRUE(finishes([&] { rewrites = sqrt_basis->second(node); }))
      << "i_sqrt_basis hangs on the pow-form wedge node";
}

TEST(PowPowHang, CanonicalMixedSqrtFormsWithCarrier) {
  // inside i_sqrt_basis: h_e = canonical(f * sqrt(P)) where f holds the
  // POW spelling (x**2)**(-1/2) and an Integral carrier; the fn-sqrt
  // and pow forms share no base, and canonical must still terminate
  for (const char* s :
       {"x*(x**2)**(-1/2)*sqrt(x**2)",
        "x*Integral(3*x, x)*(x**2)**(-1/2)*sqrt(x**2)",
        "x*Integral(3*x*Integral(16*sqrt(2), x), x)*(x**2)**(-1/2)"
        "*sqrt(x**2)"}) {
    const sym::expr e = sym::parse(s);
    EXPECT_TRUE(finishes([&] { (void)sym::canonical(e, X); })) << s;
  }
}

TEST(PowPowHang, CorrectnessOfPowPowCancellation) {
  // the fix must not lose the sound cancels: dividing by the pow FACTOR
  // (not its base) still merges exponents at the final num/den step
  EXPECT_EQ(sym::to_sstr(sym::canonical(
                sym::parse("x*(x**2)**(-1/2)*sqrt(x**2)"), X)),
            "x");
  EXPECT_EQ(sym::to_sstr(sym::canonical(
                sym::parse("(x**2)**(3/2)/sqrt(x**2)"), X)),
            "x**2");
  // and must never fold (x^even)^(1/2) to a bare power (|x| != x)
  EXPECT_EQ(sym::to_sstr(sym::canonical(sym::parse("(x**2)**(1/2)"), X)),
            "sqrt(x**2)");
  EXPECT_EQ(sym::to_sstr(sym::canonical(sym::parse("(x**2)**(3/2)"), X)),
            "(x**2)**(3/2)");
}

TEST(PowPowHang, WildSpecimenFarmRootSolves) {
  // the wild artifact, not just the mechanism (llmopt discipline: keep
  // the crash artifact so the mistake can't quietly return): v5 farm
  // root m-l8-v5s1-330 hung >120s inside beam_search pre-fix. It must
  // now SOLVE under a 15s wall with every edge oracle-certified.
  const sym::expr root =
      sym::parse("Integral(16*sqrt(2)*(x**2)**(3/2), x)");
  search::beam_options opt;
  opt.width = 8;  // the width that first surfaced the wedge (bisect:
  opt.max_plies = 8;   // width 4 never reached the hung node)
  opt.max_nodes = 200;
  opt.use_macros = true;
  opt.deadline = std::chrono::steady_clock::now() +
                 std::chrono::seconds(15);
  const auto& rules = search::default_rules();
  const auto res = search::beam_search(root, rules, opt);
  ASSERT_TRUE(res.solved) << "wild specimen no longer solves";
  const auto chain = search::replay_chain(root, res.best.history, rules);
  ASSERT_TRUE(chain.has_value());
  for (std::size_t i = 0; i + 1 < chain->size(); ++i)
    EXPECT_TRUE(search::verify_edge((*chain)[i].e, (*chain)[i + 1].e,
                                    rules.external))
        << "edge " << i << " fails certification";
}

TEST(PowPowHang, ILinearBasisOnCosLogNode) {
  // second wedge family (v5 farm, m-l8-v5s1-158 hardkilled both
  // rounds): i_linear_basis on a cos(log(quadratic)) rational integrand
  const sym::expr node = sym::parse(
      "Integral(4*x*(6*x + 1)*cos(log(3*x**2 + x + 4))"
      "/(3*x**2 + x + 4), x)");
  const search::rule* lin = nullptr;
  for (const auto& r : search::default_rules().integral)
    if (r.first == "i_linear_basis") lin = &r;
  ASSERT_NE(lin, nullptr);
  EXPECT_TRUE(finishes([&] { (void)lin->second(node); }))
      << "i_linear_basis hangs on the cos(log) node";
}

TEST(PowPowHang, CanonicalCosLogCancelDifference) {
  // wedge family #2 (m-l8-v5s1-158/-202, hardkilled both v5 rounds):
  // verify_edge on a "cancel" edge hangs — the integrand difference
  // mixes atan/log opaques over gcd-sharing denominators
  const sym::expr a = sym::parse(
      "12*log(2*x)/(9*x**2 + 1) + 4*atan(3*x)/x");
  const sym::expr b = sym::parse(
      "36*x**2*atan(3*x)/(9*x**3 + x) + 12*x*log(2*x)/(9*x**3 + x)"
      " + 4*atan(3*x)/(9*x**3 + x)");
  EXPECT_TRUE(finishes([&] { (void)sym::canonical(a - b, X); }))
      << "canonical hangs on the cancel-edge integrand difference";
  EXPECT_TRUE(finishes([&] {
    (void)sym::equivalent(a, b, X);
  })) << "equivalent hangs on the cancel-edge integrands";
}

TEST(PowPowHang, VerifyEdgeCosLogCancelPair) {
  // the exact hanging verify (v5 farm trace): whole-state edge with
  // the sin(log) closed term and Integral carriers present
  const sym::expr parent = sym::parse(
      "4*x*sin(log(3*x**2 + x + 4)) + Integral(12*log(2*x)/(9*x**2 + 1)"
      " + 4*atan(3*x)/x, x)");
  const sym::expr child = sym::parse(
      "4*x*sin(log(3*x**2 + x + 4)) + Integral(36*x**2*atan(3*x)"
      "/(9*x**3 + x) + 12*x*log(2*x)/(9*x**3 + x)"
      " + 4*atan(3*x)/(9*x**3 + x), x)");
  const search::external_slots ext{};
  EXPECT_TRUE(finishes([&] {
    (void)search::verify_edge(parent, child, ext);
  })) << "verify_edge hangs on the cancel pair";
}

TEST(PowPowHang, AnnotateCosLogWildSpecimen) {
  // wedge family #2 wild specimen (m-l8-v5s1-158): annotate fired every
  // hint-probe rule BARE — no work budget — so one slow fire hung chain
  // emission after a 66s solve. The full annotate must finish under the
  // budget discipline now (hang = regression).
  const sym::expr cur = sym::parse(
      "4*x*sin(log(3*x**2 + x + 4)) + Integral(12*log(2*x)/(9*x**2 + 1)"
      " + 4*atan(3*x)/x, x)");
  const auto t0 = std::chrono::steady_clock::now();
  search::annotation ann;
  ann = search::annotate(cur, search::default_rules(), "i_unprod");
  const double dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  // generous wall: every probe is individually 8s-budgeted; the whole
  // annotate over ~20 rules must stay far under a farm hardkill window
  EXPECT_LT(dt, 60.0) << "annotate exceeded the bounded-probe budget";
}
