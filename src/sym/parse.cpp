#include <ax/sym/parse.hpp>

#include <array>
#include <cctype>
#include <string>
#include <string_view>

namespace ax::sym {

namespace {

constexpr std::array<std::string_view, 9> kKnownFns = {
    "sin", "cos", "tan", "exp", "log", "sqrt", "atan", "asin", "acos"};

constexpr std::array<std::string_view, 4> kRejectedAtoms = {"oo", "zoo",
                                                           "nan", "I"};

bool contains(const auto& arr, std::string_view name) {
  for (std::string_view v : arr)
    if (v == name) return true;
  return false;
}

class parser {
 public:
  explicit parser(std::string_view src) : s_(src) {}

  expr run() {
    expr e = sum();
    skip_ws();
    if (i_ != s_.size()) fail("trailing input");
    return e;
  }

 private:
  std::string_view s_;
  std::size_t i_ = 0;

  [[noreturn]] void fail(const std::string& msg) const {
    throw parse_error("parse: " + msg, i_);
  }

  void skip_ws() {
    while (i_ < s_.size() &&
           std::isspace(static_cast<unsigned char>(s_[i_])))
      ++i_;
  }

  bool peek(char c) const { return i_ < s_.size() && s_[i_] == c; }

  bool eat(char c) {
    if (!peek(c)) return false;
    ++i_;
    return true;
  }

  /** Consume a pow operator: `**` or `^`. */
  bool eat_pow() {
    if (i_ + 1 < s_.size() && s_[i_] == '*' && s_[i_ + 1] == '*') {
      i_ += 2;
      return true;
    }
    return eat('^');
  }

  // sum := term (('+'|'-') term)*
  expr sum() {
    expr e = term();
    for (;;) {
      skip_ws();
      if (eat('+'))
        e = e + term();
      else if (eat('-'))
        e = e - term();
      else
        return e;
    }
  }

  // term := unary (('*'|'/') unary)*   (a lone '*'; '**' belongs to power)
  expr term() {
    expr e = unary();
    for (;;) {
      skip_ws();
      if (i_ + 1 < s_.size() && s_[i_] == '*' && s_[i_ + 1] == '*')
        fail("unexpected '**'");
      if (eat('*'))
        e = e * unary();
      else if (eat('/'))
        e = e / unary();
      else
        return e;
    }
  }

  // unary := ('-'|'+') unary | power     (so -x**2 == -(x**2))
  expr unary() {
    skip_ws();
    if (eat('-')) return -unary();
    if (eat('+')) return unary();
    return power();
  }

  // power := atom ['**' unary]          (right-assoc; exponent may be unary)
  expr power() {
    expr b = atom();
    skip_ws();
    if (eat_pow()) return b.pow(unary());
    return b;
  }

  expr atom() {
    skip_ws();
    if (i_ >= s_.size()) fail("unexpected end of input");
    const char c = s_[i_];
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.')
      return number();
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_')
      return name_atom();
    if (eat('(')) {
      expr e = sum();
      skip_ws();
      if (!eat(')')) fail("expected ')'");
      return e;
    }
    fail("expected expression");
  }

  /** Integer or decimal literal, converted exactly to rational. */
  expr number() {
    const std::size_t start = i_;
    while (i_ < s_.size() &&
           std::isdigit(static_cast<unsigned char>(s_[i_])))
      ++i_;
    std::string digits(s_.substr(start, i_ - start));
    std::string frac;
    if (eat('.')) {
      const std::size_t fstart = i_;
      while (i_ < s_.size() &&
             std::isdigit(static_cast<unsigned char>(s_[i_])))
        ++i_;
      frac = std::string(s_.substr(fstart, i_ - fstart));
    }
    if (digits.empty() && frac.empty()) fail("malformed number");
    if (digits.empty()) digits = "0";
    if (frac.empty()) return expr::num(rational(bigint(digits)));
    // digits.frac == (digits*10^k + frac) / 10^k, reduced by rational.
    std::string den = "1" + std::string(frac.size(), '0');
    return expr::num(
        rational(bigint(digits + frac), bigint(den)));
  }

  expr name_atom() {
    const std::size_t start = i_;
    while (i_ < s_.size() &&
           (std::isalnum(static_cast<unsigned char>(s_[i_])) ||
            s_[i_] == '_'))
      ++i_;
    const std::string name(s_.substr(start, i_ - start));
    skip_ws();
    if (eat('(')) {
      if (!contains(kKnownFns, name)) {
        i_ = start;
        fail("unknown function '" + name + "'");
      }
      expr arg = sum();
      skip_ws();
      if (!eat(')')) fail("expected ')'");
      return expr::fn(name, arg);
    }
    if (contains(kRejectedAtoms, name)) {
      i_ = start;
      fail("unrepresentable atom '" + name + "'");
    }
    return expr::symbol(name);
  }
};

}  // namespace

expr parse(std::string_view src) { return parser(src).run(); }

}  // namespace ax::sym
