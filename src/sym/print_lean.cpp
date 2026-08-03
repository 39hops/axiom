#include <ax/sym/print_lean.hpp>

#include <ax/sym/jsonl.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/print_sstr.hpp>

#include <algorithm>
#include <set>
#include <stdexcept>

namespace ax::sym {

namespace {

const bigint kOneI(1);

bool is_int(const rational& q) { return q.den() == kOneI; }

/** Shared walk state: atoms, denominators, free symbols. */
struct walk {
  std::vector<expr> atoms;              // distinct fn-subterms, in order
  std::vector<expr> dens;               // distinct symbolic denominators
  std::set<std::string> syms;           // free symbol names
  bool eligible = true;

  int atom_index(const expr& e) {
    for (std::size_t i = 0; i < atoms.size(); ++i)
      if (atoms[i].same(e)) return static_cast<int>(i);
    atoms.push_back(e);
    return static_cast<int>(atoms.size() - 1);
  }

  void add_den(const expr& base) {
    if (base.is_num()) return;  // numeric denominators need no hypothesis
    for (const expr& d : dens)
      if (d.same(base)) return;
    dens.push_back(base);
  }

  void free_syms(const expr& e) {
    if (e.is_sym()) {
      syms.insert(e.name());
      return;
    }
    for (const expr& a : e.args()) free_syms(a);
  }

  /** Ring-level eligibility + collection. fn nodes are atoms whose
      contents are frozen; sqrt and non-integer pow exponents at ring
      level are the fence. */
  void visit(const expr& e) {
    if (!eligible) return;
    switch (e.k()) {
      case kind::num:
        return;
      case kind::sym:
        syms.insert(e.name());
        return;
      case kind::fn:
        if (e.name() == "sqrt") {
          eligible = false;
          return;
        }
        atom_index(e);
        free_syms(e);  // frozen contents still bind their symbols
        return;
      case kind::pow: {
        const expr& ex = e.args()[1];
        if (!ex.is_num() || !is_int(ex.value())) {
          eligible = false;
          return;
        }
        if (ex.value() < rational{}) add_den(e.args()[0]);
        visit(e.args()[0]);
        return;
      }
      case kind::add:
      case kind::mul:
        for (const expr& a : e.args()) visit(a);
        return;
    }
  }
};

// ----------------------------------------------------------- Lean printer

/** Precedence levels for parenthesization. */
enum prec { p_add = 0, p_mul = 1, p_pow = 2, p_atom = 3 };

struct printer {
  const walk& w;
  const std::vector<std::string>& names;  // atom index -> Lean name

  std::string atom_name(const expr& e) const {
    for (std::size_t i = 0; i < w.atoms.size(); ++i)
      if (w.atoms[i].same(e)) return names[i];
    throw std::logic_error("unregistered atom");
  }

  static std::string wrap(const std::string& s, prec have, prec need) {
    return have < need ? "(" + s + ")" : s;
  }

  std::string num(const rational& q, prec need) const {
    std::string s = q.num().to_string();
    if (!is_int(q)) s += "/" + q.den().to_string();
    const bool neg = !s.empty() && s[0] == '-';
    const prec have = !is_int(q) ? p_mul : (neg ? p_add : p_atom);
    return wrap(s, have, need);
  }

  std::string print(const expr& e, prec need) const {
    switch (e.k()) {
      case kind::num:
        return num(e.value(), need);
      case kind::sym:
        return e.name();
      case kind::fn:
        return atom_name(e);
      case kind::pow: {
        const rational& q = e.args()[1].value();
        const std::string b = print(e.args()[0], p_atom);
        if (q < rational{}) {
          const std::string mag = (-q).num().to_string();
          const std::string p =
              mag == "1" ? b : b + "^" + mag;
          return wrap("1/" + p, p_mul, need);
        }
        return wrap(b + "^" + q.num().to_string(), p_pow, need);
      }
      case kind::mul: {
        // Split factors into numerator and denominator (negative pows).
        std::vector<std::string> nums, dens;
        std::string sign;
        for (const expr& f : e.args()) {
          if (f.is_num()) {
            rational q = f.value();
            std::string s;
            if (q < rational{}) {
              sign = "-";
              q = -q;
            }
            if (q == rational(kOneI) && e.args().size() > 1) continue;
            nums.push_back(num(q, p_pow));
            continue;
          }
          if (f.is_pow() && f.args()[1].value() < rational{}) {
            const rational mq = -f.args()[1].value();
            const std::string b = print(f.args()[0], p_atom);
            dens.push_back(mq.num().to_string() == "1"
                               ? b
                               : b + "^" + mq.num().to_string());
            continue;
          }
          nums.push_back(print(f, p_pow));
        }
        std::string n;
        if (nums.empty()) n = "1";
        for (std::size_t i = 0; i < nums.size(); ++i)
          n += (i ? "*" : "") + nums[i];
        std::string s = sign + n;
        if (!dens.empty()) {
          std::string d;
          for (std::size_t i = 0; i < dens.size(); ++i)
            d += (i ? "*" : "") + dens[i];
          if (dens.size() > 1) d = "(" + d + ")";
          s += "/" + d;
        }
        return wrap(s, sign.empty() ? p_mul : p_add, need);
      }
      case kind::add: {
        std::string s;
        for (std::size_t i = 0; i < e.args().size(); ++i) {
          std::string t = print(e.args()[i], p_add);
          if (i == 0) {
            s = t;
          } else if (!t.empty() && t[0] == '-') {
            s += " - " + t.substr(1);
          } else {
            s += " + " + t;
          }
        }
        return wrap(s, p_add, need);
      }
    }
    return "?";  // unreachable
  }
};

/** Atom names a1..an, switching prefix if a free symbol collides. */
std::vector<std::string> atom_names(const walk& w) {
  for (const char* prefix : {"a", "u", "w"}) {
    std::vector<std::string> names;
    bool clash = false;
    for (std::size_t i = 0; i < w.atoms.size() && !clash; ++i) {
      names.push_back(prefix + std::to_string(i + 1));
      clash = w.syms.count(names.back()) != 0;
    }
    if (!clash) return names;
  }
  throw std::runtime_error("atom name space exhausted");
}

/** Lexical fence on the raw input: expr construction merges fractional
    pows before any expr walk can see them (x**(1/2)*x**(1/2) -> x), so a
    `/` inside a `**(...)` exponent group must be rejected on the source
    text. sqrt needs no lexical fence — it survives construction as an fn
    node (only canonical() rewrites it), so the walk fences it at ring
    level and correctly freezes it inside fn arguments. Conservative: a
    false positive only shrinks coverage (counted), never emits an unsound
    certificate. */
bool raw_fenced(const std::string& s) {
  for (std::size_t i = 0; i + 2 < s.size(); ++i) {
    if (s[i] != '*' || s[i + 1] != '*' || s[i + 2] != '(') continue;
    int depth = 0;
    for (std::size_t j = i + 2; j < s.size(); ++j) {
      if (s[j] == '(') ++depth;
      if (s[j] == ')' && --depth == 0) break;
      if (s[j] == '/') return true;
    }
  }
  return false;
}

}  // namespace

lean_cert to_lean(const std::string& lhs_raw, const std::string& rhs_raw,
                  const expr& x) {
  lean_cert c;
  c.lhs = lhs_raw;
  c.rhs = rhs_raw;
  if (raw_fenced(lhs_raw) || raw_fenced(rhs_raw)) return c;
  const expr lhs = parse(lhs_raw);
  const expr rhs = parse(rhs_raw);
  walk w;
  w.visit(lhs);
  w.visit(rhs);
  if (!w.eligible) return c;

  const std::vector<std::string> names = atom_names(w);
  const printer pr{w, names};

  std::string vars;
  for (const std::string& n : names) vars += (vars.empty() ? "" : " ") + n;
  for (const std::string& s : w.syms) vars += (vars.empty() ? "" : " ") + s;
  if (vars.empty()) vars = "x";  // constant identity; bind x anyway

  std::string hyps;
  for (std::size_t i = 0; i < w.dens.size(); ++i)
    hyps += " (h" + std::to_string(i + 1) + " : " +
            pr.print(w.dens[i], p_add) + " ≠ 0)";

  c.tactic = w.dens.empty() ? "ring" : "field_simp; ring";
  c.statement = "example (" + vars + " : ℝ)" + hyps + " : " +
                pr.print(lhs, p_add) + " = " + pr.print(rhs, p_add) +
                " := by " + c.tactic;
  for (std::size_t i = 0; i < w.atoms.size(); ++i)
    c.atoms.emplace_back(names[i], to_sstr(w.atoms[i]));
  c.eligible = true;
  return c;
}

std::string sidecar_line(const std::string& id, const lean_cert& c) {
  std::string atoms;
  for (const auto& [name, sstr] : c.atoms) {
    if (!atoms.empty()) atoms += ",";
    atoms += "\"" + jsonl::escape(name) + "\":\"" + jsonl::escape(sstr) + "\"";
  }
  return "{\"id\":\"" + jsonl::escape(id) + "\",\"lhs\":\"" +
         jsonl::escape(c.lhs) + "\",\"rhs\":\"" + jsonl::escape(c.rhs) +
         "\",\"lean\":\"" + jsonl::escape(c.statement) + "\",\"tactic\":\"" +
         jsonl::escape(c.tactic) + "\",\"atoms\":{" + atoms + "}}";
}

}  // namespace ax::sym
