/** pybind11 module `axiom_sym` — llmopt's native oracle bridge.

    API (per llmopt docs/superpowers/specs/2026-07-18-axiom-backend.md):
    parse_sstr, diff, canonical, equivalent, equivalent_mod_const.

    Thread safety: expr handles are immutable and the hash-cons pool is
    mutex-guarded; the module holds no state of its own. Every function
    releases the GIL for the C++ work, so llmopt worker threads verify in
    parallel. Verdicts are the protocol strings EQUIVALENT /
    NOT_EQUIVALENT / UNDECIDED; parse failures raise ValueError. */
#include <pybind11/pybind11.h>

#include <ax/search/search.hpp>
#include <ax/sym/calc.hpp>
#include <ax/sym/expr.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print.hpp>

#include <pybind11/stl.h>

#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace sym = ax::sym;

namespace {

/** axiom's printer spells pow `^`; sympy needs `**`. */
std::string to_sstr(const sym::expr& e) {
  std::string s = sym::to_string(e);
  std::string out;
  out.reserve(s.size());
  for (const char c : s) {
    if (c == '^')
      out += "**";
    else
      out += c;
  }
  return out;
}

const char* verdict_name(sym::verdict v) {
  switch (v) {
    case sym::verdict::equivalent: return "EQUIVALENT";
    case sym::verdict::not_equivalent: return "NOT_EQUIVALENT";
    case sym::verdict::undecided: return "UNDECIDED";
  }
  return "UNDECIDED";
}

sym::expr var_of(const std::string& name) { return sym::expr::symbol(name); }

}  // namespace

PYBIND11_MODULE(axiom_sym, m) {
  m.doc() =
      "axiom CAS bridge for llmopt: parse_sstr, diff, canonical, "
      "equivalent, equivalent_mod_const. Verdicts are strings; UNDECIDED "
      "means fall back to sympy, never treat as valid.";

  py::class_<sym::expr>(m, "Expr")
      .def("__str__", &to_sstr)
      .def("__repr__",
           [](const sym::expr& e) { return "Expr(" + to_sstr(e) + ")"; })
      .def(
          "same",
          [](const sym::expr& a, const sym::expr& b) { return a.same(b); },
          py::call_guard<py::gil_scoped_release>(),
          "Structural (hash-cons pointer) equality.");

  m.def(
      "parse_sstr",
      [](const std::string& src) {
        try {
          return sym::parse(src);
        } catch (const sym::parse_error& e) {
          // std::invalid_argument -> ValueError; safe to throw with the
          // GIL released (pybind11 translates after re-acquiring it).
          throw std::invalid_argument(e.what());
        }
      },
      py::arg("src"), py::call_guard<py::gil_scoped_release>(),
      "Parse sympy sstr text; raises ValueError on malformed or "
      "unrepresentable input (oo, zoo, nan, I, unknown functions).");

  m.def(
      "diff",
      [](const sym::expr& e, const std::string& var) {
        return sym::diff(e, var_of(var));
      },
      py::arg("expr"), py::arg("var"),
      py::call_guard<py::gil_scoped_release>(),
      "Symbolic derivative d expr / d var.");

  m.def(
      "canonical",
      [](const sym::expr& e, const std::string& var) {
        return sym::canonical(e, var_of(var));
      },
      py::arg("expr"), py::arg("var"),
      py::call_guard<py::gil_scoped_release>(),
      "Canonical form used for equivalence checking.");

  m.def(
      "equivalent",
      [](const sym::expr& a, const sym::expr& b, const std::string& var) {
        return verdict_name(sym::equivalent(a, b, var_of(var)));
      },
      py::arg("lhs"), py::arg("rhs"), py::arg("var"),
      py::call_guard<py::gil_scoped_release>(),
      "Three-valued equivalence: 'EQUIVALENT' | 'NOT_EQUIVALENT' | "
      "'UNDECIDED'.");

  m.def(
      "equivalent_mod_const",
      [](const sym::expr& candidate, const sym::expr& integrand,
         const std::string& var) {
        return verdict_name(
            sym::equivalent_mod_const(candidate, integrand, var_of(var)));
      },
      py::arg("candidate"), py::arg("integrand"), py::arg("var"),
      py::call_guard<py::gil_scoped_release>(),
      "Antiderivative check by differentiation: is d candidate/d var "
      "equivalent to integrand?");

  // ----------------------------------------------------------- solver
  // Hybrid-config entry: the native engine with llmopt's external slots
  // served as Python callables (llmopt/search/axiom_slots.py contract:
  // heurisch(node_sstr) -> list[str]; equivalence(lhs, rhs) -> verdict
  // string). The engine runs with the GIL released; slot invocations
  // re-acquire it. Slot failures are conservative: heurisch -> no
  // candidates, equivalence -> reject.
  m.def(
      "solve",
      [](const sym::expr& root, long long budget, int plies, int width,
         const std::string& prior_tsv, py::object heurisch,
         py::object equivalence, long long deadline_ms) {
        namespace se = ax::search;
        // Built (and destroyed) WITH the GIL held: rules captures Python
        // callables, and dropping a py::object without the GIL is UB. The
        // GIL is released only around beam_search; slot invocations
        // re-acquire it from inside.
        se::rule_set rules = se::default_rules();  // copy: slots per call
        if (!heurisch.is_none()) {
          rules.external.int_rules.emplace_back(
              "i_heurisch",
              [heurisch](const sym::expr& node) -> std::vector<sym::expr> {
                py::gil_scoped_acquire gil;
                std::vector<sym::expr> out;
                try {
                  const py::object res = heurisch(to_sstr(node));
                  for (const auto& item : res.cast<py::list>())
                    out.push_back(sym::parse(item.cast<std::string>()));
                } catch (...) {
                  out.clear();  // conservative: slot failure = no fire
                }
                return out;
              });
        }
        if (!equivalence.is_none()) {
          rules.external.equivalence =
              [equivalence](const sym::expr& a,
                            const sym::expr& b) -> bool {
            py::gil_scoped_acquire gil;
            try {
              return equivalence(to_sstr(a), to_sstr(b))
                         .cast<std::string>() == "EQUIVALENT";
            } catch (...) {
              return false;  // conservative: UNDECIDED stays rejected
            }
          };
        }
        se::beam_options opt;
        opt.width = width;
        opt.max_plies = plies;
        opt.max_nodes = budget;
        opt.use_macros = true;
        std::optional<se::markov_prior> prior;
        if (!prior_tsv.empty()) {
          prior = se::markov_prior::load_tsv(prior_tsv);
          opt.proposer = prior->proposer();
          opt.propose_k = 3;
        }
        if (deadline_ms > 0)
          opt.deadline = std::chrono::steady_clock::now() +
                         std::chrono::milliseconds(deadline_ms);
        se::search_result res{false, se::state{root}, 0};
        {
          py::gil_scoped_release run_without_gil;
          res = se::beam_search(root, rules, opt);
        }
        py::dict out;
        out["solved"] = res.solved;
        if (res.solved)
          out["answer"] = to_sstr(res.best.e);
        else
          out["answer"] = py::none();
        out["history"] = res.best.history;
        out["nodes"] = res.nodes;
        out["expired"] = res.deadline_expired;
        return out;
      },
      py::arg("root"), py::arg("budget") = 200, py::arg("plies") = 24,
      py::arg("width") = 3, py::arg("prior_tsv") = std::string(),
      py::arg("heurisch") = py::none(),
      py::arg("equivalence") = py::none(), py::arg("deadline_ms") = 20000,
      "Run the native derivation engine on Integral(f, x) with optional "
      "bridge slots (llmopt axiom_slots contract). Returns {solved, "
      "answer, history, nodes, expired}. Every emitted answer passed "
      "edge-level verification (verify_p=1).");
}
