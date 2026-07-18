#include <ax/sym/jsonl.hpp>

#include <cctype>
#include <stdexcept>

namespace ax::sym::jsonl {

namespace {

[[noreturn]] void fail(const std::string& msg) {
  throw std::runtime_error("jsonl: " + msg);
}

struct cursor {
  std::string_view s;
  std::size_t i = 0;

  void ws() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
      ++i;
  }
  bool eat(char c) {
    if (i < s.size() && s[i] == c) {
      ++i;
      return true;
    }
    return false;
  }

  std::string quoted_string() {
    if (!eat('"')) fail("expected string");
    std::string out;
    while (i < s.size()) {
      const char c = s[i++];
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i >= s.size()) fail("dangling escape");
      const char e = s[i++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          if (i + 4 > s.size()) fail("bad \\u escape");
          unsigned v = 0;
          for (int k = 0; k < 4; ++k) {
            const char h = s[i++];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') v |= static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= static_cast<unsigned>(h - 'A' + 10);
            else fail("bad \\u escape");
          }
          if (v < 0x80) {
            out += static_cast<char>(v);
          } else if (v < 0x800) {
            out += static_cast<char>(0xC0 | (v >> 6));
            out += static_cast<char>(0x80 | (v & 0x3F));
          } else {
            out += static_cast<char>(0xE0 | (v >> 12));
            out += static_cast<char>(0x80 | ((v >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (v & 0x3F));
          }
          break;
        }
        default: fail("unknown escape");
      }
    }
    fail("unterminated string");
  }

  /** Bare scalar (number/true/false/null) captured as raw text. */
  std::string bare_token() {
    const std::size_t start = i;
    while (i < s.size() && s[i] != ',' && s[i] != '}' &&
           !std::isspace(static_cast<unsigned char>(s[i])))
      ++i;
    if (i == start) fail("expected value");
    const std::string_view tok = s.substr(start, i - start);
    if (tok.front() == '{' || tok.front() == '[')
      fail("nested values unsupported");
    return std::string(tok);
  }
};

}  // namespace

object parse_line(std::string_view line) {
  cursor c{line};
  c.ws();
  if (!c.eat('{')) fail("expected '{'");
  object out;
  c.ws();
  if (c.eat('}')) return out;
  for (;;) {
    c.ws();
    const std::string key = c.quoted_string();
    c.ws();
    if (!c.eat(':')) fail("expected ':'");
    c.ws();
    std::string value;
    if (c.i < c.s.size() && c.s[c.i] == '"')
      value = c.quoted_string();
    else
      value = c.bare_token();
    out[key] = std::move(value);
    c.ws();
    if (c.eat(',')) continue;
    if (c.eat('}')) break;
    fail("expected ',' or '}'");
  }
  c.ws();
  if (c.i != c.s.size()) fail("trailing input");
  return out;
}

std::string escape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (const char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          static const char* hex = "0123456789abcdef";
          out += "\\u00";
          out += hex[(c >> 4) & 0xF];
          out += hex[c & 0xF];
        } else {
          out += c;
        }
    }
  }
  return out;
}

std::string write_line(const fields& fs) {
  std::string out = "{";
  bool first = true;
  for (const auto& [k, v] : fs) {
    if (!first) out += ",";
    first = false;
    out += "\"" + escape(k) + "\":\"" + escape(v) + "\"";
  }
  out += "}";
  return out;
}

}  // namespace ax::sym::jsonl
