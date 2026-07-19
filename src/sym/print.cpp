#include <ax/sym/print.hpp>

#include <stdexcept>

namespace ax::sym {

namespace {

const rational kMinusOne{bigint(-1)};

/** Does this term render with a leading minus? */
bool is_negative_term(const expr& e) {
  if (e.is_num()) return e.value() < rational{};
  if (e.is_mul() && e.args()[0].is_num())
    return e.args()[0].value() < rational{};
  return false;
}

// ------------------------------------------------------------- plain text

std::string text(const expr& e);

std::string text_atom_or_paren(const expr& e) {
  // parenthesize anything that binds looser than juxtaposition
  const bool atom = e.is_sym() || e.is_fn() ||
                    (e.is_num() && !(e.value() < rational{}));
  if (atom || e.is_pow()) return text(e);
  return "(" + text(e) + ")";
}

std::string text_mul(const expr& e) {
  const auto fs = e.args();
  std::size_t start = 0;
  std::string out;
  if (fs[0].is_num() && fs[0].value() == kMinusOne) {
    out = "-";
    start = 1;
  }
  for (std::size_t i = start; i < fs.size(); ++i) {
    if (i > start) out += "*";
    out += text_atom_or_paren(fs[i]);
  }
  return out;
}

std::string text(const expr& e) {
  switch (e.k()) {
    case kind::num:
      return e.value().to_string();
    case kind::sym:
      return e.name();
    case kind::fn: {
      std::string out = e.name() + "(";
      bool first = true;
      for (const expr& a : e.args()) {
        if (!first) out += ", ";
        first = false;
        out += text(a);
      }
      return out + ")";
    }
    case kind::add: {
      std::string out;
      bool first = true;
      for (const expr& t : e.args()) {
        if (first) {
          out = text(t);
          first = false;
          continue;
        }
        if (is_negative_term(t)) {
          out += " - " + text(-t);
        } else {
          out += " + " + text(t);
        }
      }
      return out;
    }
    case kind::mul:
      return text_mul(e);
    case kind::pow: {
      const expr& b = e.args()[0];
      const expr& ex = e.args()[1];
      std::string bs = text_atom_or_paren(b);
      if (b.is_pow()) bs = "(" + text(b) + ")";  // (x^y)^z unambiguous
      std::string es;
      if (ex.is_num() && ex.value().den() == bigint(1))
        es = ex.value().to_string();  // integer exponent, incl. negative
      else if (ex.is_sym())
        es = ex.name();
      else
        es = "(" + text(ex) + ")";
      return bs + "^" + es;
    }
  }
  throw std::logic_error("to_string: unreachable");
}

// ------------------------------------------------------------------ latex

std::string latex(const expr& e);

std::string latex_num(const rational& q) {
  if (q.den() == bigint(1)) return q.to_string();
  const bool neg = q < rational{};
  const rational a = neg ? -q : q;
  const std::string body =
      "\\frac{" + a.num().to_string() + "}{" + a.den().to_string() + "}";
  return neg ? "-" + body : body;
}

std::string latex_group(const expr& e) {
  if (e.is_add() || (e.is_mul() && e.args().size() > 1))
    return "\\left(" + latex(e) + "\\right)";
  return latex(e);
}

std::string latex(const expr& e) {
  switch (e.k()) {
    case kind::num:
      return latex_num(e.value());
    case kind::sym:
      return e.name();
    case kind::fn: {
      const std::string arg = "\\left(" + latex(e.args()[0]) + "\\right)";
      if (e.name() == "sqrt") return "\\sqrt{" + latex(e.args()[0]) + "}";
      if (e.name() == "atan") return "\\arctan" + arg;
      if (e.name() == "asin") return "\\arcsin" + arg;
      if (e.name() == "acos") return "\\arccos" + arg;
      return "\\" + e.name() + arg;
    }
    case kind::add: {
      std::string out;
      bool first = true;
      for (const expr& t : e.args()) {
        if (first) {
          out = latex(t);
          first = false;
          continue;
        }
        if (is_negative_term(t)) {
          out += " - " + latex(-t);
        } else {
          out += " + " + latex(t);
        }
      }
      return out;
    }
    case kind::mul: {
      const auto fs = e.args();
      std::size_t start = 0;
      std::string out;
      if (fs[0].is_num() && fs[0].value() == kMinusOne) {
        out = "-";
        start = 1;
      }
      for (std::size_t i = start; i < fs.size(); ++i) {
        if (i > start) out += " ";
        out += fs[i].is_add() ? "\\left(" + latex(fs[i]) + "\\right)"
                              : latex_group(fs[i]);
      }
      return out;
    }
    case kind::pow: {
      const expr& b = e.args()[0];
      std::string bs = b.is_sym() || b.is_fn() ||
                               (b.is_num() && !(b.value() < rational{}) &&
                                b.value().den() == bigint(1))
                           ? latex(b)
                           : "\\left(" + latex(b) + "\\right)";
      return bs + "^{" + latex(e.args()[1]) + "}";
    }
  }
  throw std::logic_error("to_latex: unreachable");
}

}  // namespace

std::string to_string(const expr& e) { return text(e); }
std::string to_latex(const expr& e) { return latex(e); }

}  // namespace ax::sym
