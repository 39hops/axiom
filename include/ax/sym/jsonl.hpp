#pragma once
/** @file jsonl.hpp Minimal STL-only JSONL io for the oracle parity harness.

    Scope is deliberately tiny: one flat JSON object per line, values are
    strings (or bare scalar tokens, captured as their raw text). This is a
    protocol codec, not a JSON library. */
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ax::sym::jsonl {

/** Flat object: key -> string value (bare tokens kept as raw text). */
using object = std::map<std::string, std::string>;

/** Ordered fields for deterministic output. */
using fields = std::vector<std::pair<std::string, std::string>>;

/** Parse one JSONL line. Throws std::runtime_error on malformed input. */
object parse_line(std::string_view line);

/** Serialize one line: {"k":"v",...} with full string escaping. */
std::string write_line(const fields& fs);

/** JSON string escaping (quotes, backslash, control chars). */
std::string escape(std::string_view s);

}  // namespace ax::sym::jsonl
