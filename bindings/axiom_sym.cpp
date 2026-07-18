/** pybind11 module `axiom_sym` — llmopt's native oracle bridge.

    API (per llmopt docs/superpowers/specs/2026-07-18-axiom-backend.md):
    parse_sstr, diff, canonical, equivalent, equivalent_mod_const.

    Thread safety: expr handles are immutable and the hash-cons pool is
    mutex-guarded; the module holds no state of its own. Every function
    releases the GIL for the C++ work, so llmopt worker threads verify in
    parallel. Verdicts are the protocol strings EQUIVALENT /
    NOT_EQUIVALENT / UNDECIDED; parse failures raise ValueError. */
#include <pybind11/pybind11.h>

#include <ax/sym/calc.hpp>
#include <ax/sym/expr.hpp>
#include <ax/sym/oracle.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print.hpp>

#include <stdexcept>
#include <string>

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
}
