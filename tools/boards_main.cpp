/** boards: magic boards r2 emitter (relay 2026-07-27-0 ask 3).
    Usage:
      axiom-boards <states.jsonl> <out.jsonl> [expr_key]

    Reads jsonl rows carrying a state sstr under expr_key (default
    "root"; falls back to "state" / "cur"), certifies each with the
    Risch dead-state boards, and writes
      {"id": ..., "state": ..., "dead": true|false, "reason": "..."}
    rows. Only certified rows carry a non-empty reason; undecided and
    provably-elementary states emit dead=false, so the mask side is
    conservative by construction. Unparseable rows are booked and
    skipped, never silently dropped. */
#include <ax/sym/jsonl.hpp>
#include <ax/sym/parse.hpp>
#include <ax/sym/risch.hpp>

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  using namespace ax;
  if (argc < 3) {
    std::cerr << "usage: axiom-boards <states.jsonl> <out.jsonl> "
                 "[expr_key]\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  std::ofstream out(argv[2]);
  if (!in.good() || !out.good()) {
    std::cerr << "cannot open " << argv[1] << " / " << argv[2] << "\n";
    return 2;
  }
  const std::string want_key = argc > 3 ? argv[3] : "";

  long long rows = 0, dead = 0, errors = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    try {
      const auto row = sym::jsonl::parse_line(line);
      std::string key = want_key;
      if (key.empty()) {
        for (const char* k : {"root", "state", "cur"})
          if (row.count(k)) {
            key = k;
            break;
          }
      }
      const std::string& src = row.at(key);
      const auto cert = sym::dead_state(sym::parse(src));
      std::string id = row.count("id") ? row.at("id") : std::to_string(rows);
      out << "{\"id\": \"" << sym::jsonl::escape(id) << "\", \"state\": \""
          << sym::jsonl::escape(src) << "\", \"dead\": "
          << (cert.dead ? "true" : "false") << ", \"reason\": \""
          << sym::jsonl::escape(cert.reason) << "\"}\n";
      ++rows;
      if (cert.dead) ++dead;
    } catch (const std::exception& ex) {
      ++errors;
      std::cerr << "[row-error] " << ex.what() << "\n";
    }
  }
  std::cout << "rows " << rows << "  dead " << dead << "  errors "
            << errors << "\n";
  return 0;
}
