/** nt-callspan: Leg B pilot emitter (llmopt relay 2026-07-29-2).
    Usage:
      axiom-nt-callspan <out.jsonl> [rows] [seed_base]
    Farms the standing nt-chain families from a seed band disjoint from
    the qual delivery (default base 1000; qual used 0..39), keeps only
    rows whose cur carries a call site, and attaches the resolution
    trace: "calls": ["call: <site> -> <value>", ...] via nt_call_spans
    (exact nt_eval values, site text verbatim). Level mix 1:1:2 over
    levels 1/2/3 (relay note: harder rows are the measured direction).
    Only CERTIFIED problems contribute; an uncertified problem aborts
    the farm (this is a paired-gate diet, not a best-effort shard).
    Sidecar <out>.config.json records the ORDERED vocab_extra atom list
    per the instrument fence: resident atoms first (chars, gcd, Mod,
    **), then the two new call atoms "call:" and "->" appended LAST.

    Span modes (arms result relay, 2026-07-29): mode "end" (default)
    carries the cur site's END VALUE (gcdstep's span is the chain's
    final gcd). Mode "step" carries the IMMEDIATE step's call — for
    gcdstep the division step "call: Mod(a, b) -> r" (r the next
    remainder); every other included kind's cur IS its immediate call,
    so its span is identical in both modes. Row SELECTION is mode-
    independent, so same-seed emissions stay row-paired across arms. */
#include <ax/mathgen/ntchain.hpp>
#include <ax/sym/jsonl.hpp>

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>

namespace {

/** Step-local span for a gcdstep row: cur is exactly
    "gcd(<a>, <b>)" (positive literals by construction); the immediate
    computation is the division remainder Mod(a, b). */
std::string gcdstep_local_span(const std::string& cur) {
  if (cur.rfind("gcd(", 0) != 0 || cur.back() != ')')
    throw std::logic_error("gcdstep cur outside expected spelling: " + cur);
  const std::string site = "Mod(" + cur.substr(4, cur.size() - 5) + ")";
  return "call: " + site + " -> " + ax::mathgen::nt_eval(site).to_string();
}

}  // namespace

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 2) {
    std::cerr << "usage: axiom-nt-callspan <out.jsonl> [rows] [seed_base] "
                 "[end|step]\n";
    return 2;
  }
  const long long want = argc > 2 ? std::atoll(argv[2]) : 500;
  const long long seed_base = argc > 3 ? std::atoll(argv[3]) : 1000;
  const std::string mode = argc > 4 ? argv[4] : "end";
  if (mode != "end" && mode != "step") {
    std::cerr << "unknown span mode " << mode << "\n";
    return 2;
  }
  std::ofstream out(argv[1]);
  if (!out.good()) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  using maker = mathgen::pchain_problem (*)(int, long long);
  const maker makers[] = {
      mathgen::make_nt_gcd_chain,    mathgen::make_nt_bezout_chain,
      mathgen::make_nt_modinv_chain, mathgen::make_nt_crt_chain,
      mathgen::make_nt_modexp_chain, mathgen::make_nt_cf_chain};
  const int level_mix[] = {1, 2, 3, 3};  // 1:1:2, the hard-bias mix
  long long rows = 0, problems = 0;
  for (long long seed = seed_base; rows < want; ++seed) {
    for (const int level : level_mix) {
      for (const maker mk : makers) {
        if (rows >= want) break;
        const mathgen::pchain_problem p = mk(level, seed);
        ++problems;
        if (!p.certified) {
          std::cerr << "UNCERTIFIED problem " << p.family << "-" << level
                    << "-" << seed << ": " << p.error << "\n";
          return 1;
        }
        int n = 0;
        for (const auto& r : p.rows) {
          auto spans = mathgen::nt_call_spans(r.cur);
          const int idx = n++;
          if (spans.empty() || rows >= want) continue;
          if (mode == "step" && r.kind == "gcdstep")
            spans = {gcdstep_local_span(r.cur)};
          out << "{\"family\": \"" << p.family
              << "\", \"level\": " << p.level << ", \"seed\": " << seed
              << ", \"n\": " << idx << ", \"kind\": \"" << r.kind
              << "\", \"cur\": \"" << sym::jsonl::escape(r.cur)
              << "\", \"nxt\": \"" << sym::jsonl::escape(r.nxt)
              << "\", \"calls\": [";
          for (std::size_t i = 0; i < spans.size(); ++i)
            out << (i ? ", " : "") << "\"" << sym::jsonl::escape(spans[i])
                << "\"";
          out << "], \"source\": \"axiom-nt-callspan\", "
                 "\"verdict\": \"CERTIFIED\"}\n";
          ++rows;
        }
        out.flush();  // incremental-stream fence
      }
    }
  }
  std::ofstream cfg(std::string(argv[1]) + ".config.json");
  cfg << "{\"tool\": \"axiom-nt-callspan\", \"rows\": " << rows
      << ", \"problems_farmed\": " << problems
      << ", \"seed_base\": " << seed_base
      << ", \"span_mode\": \"" << mode << "\", \"level_mix\": [1, 2, 3, 3], "
         "\"families\": [\"nt_gcd\", \"nt_bezout\", \"nt_modinv\", "
         "\"nt_crt\", \"nt_modexp\", \"nt_cf\"], "
         "\"vocab_extra_ordered\": {\"chars\": \"0123456789+-*(), \", "
         "\"names\": [\"gcd\", \"Mod\"], \"ops\": [\"**\"], "
         "\"call_atoms\": [\"call:\", \"->\"]}, "
         "\"span_form\": \"call: <site> -> <value>\"}\n";
  std::cerr << "== " << rows << " span rows from " << problems
            << " problems (seed base " << seed_base << ")\n";
  return 0;
}
