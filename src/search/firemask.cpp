/** @file firemask.cpp persistent (rule, node) no-fire memo ("magic
    math boards" rung 1). See search.hpp for the contract. Sidecar
    format: line 1 = rule-set fingerprint (hex), then one
    "<rule>\t<hash-hex>" per no-fire entry. */
#include <ax/search/search.hpp>

#include <ax/sym/print_sstr.hpp>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_set>

namespace ax::search {

namespace {

std::mutex mask_mu;
bool mask_armed = false;
std::uint64_t mask_fingerprint = 0;
std::unordered_set<std::string> mask_set;  // "<rule>\t<hash-hex>"
std::atomic<std::size_t> mask_hit_count{0};

std::uint64_t fnv1a(const std::string& s, std::uint64_t h = 1469598103934665603ull) {
  for (const unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

std::uint64_t fingerprint_of(const rule_set& rules) {
  std::uint64_t h = 1469598103934665603ull;
  for (const auto& [name, fn] : rules.core) h = fnv1a(name, h);
  for (const auto& [name, fn] : rules.macros) h = fnv1a(name, h);
  for (const auto& [name, fn] : rules.integral) h = fnv1a(name, h);
  for (const auto& [name, fn] : rules.algebra) h = fnv1a(name, h);
  return h;
}

std::string key_of(const std::string& rule_name, const sym::expr& node) {
  char hex[17];
  std::snprintf(hex, sizeof hex, "%016llx",
                static_cast<unsigned long long>(fnv1a(sym::to_sstr(node))));
  return rule_name + "\t" + hex;
}

}  // namespace

void fire_mask_enable(const rule_set& rules) {
  std::lock_guard<std::mutex> lk(mask_mu);
  mask_armed = true;
  mask_fingerprint = fingerprint_of(rules);
}

void fire_mask_reset() {
  std::lock_guard<std::mutex> lk(mask_mu);
  mask_armed = false;
  mask_fingerprint = 0;
  mask_set.clear();
  mask_hit_count = 0;
}

bool fire_mask_load(const std::string& path, const rule_set& rules) {
  std::ifstream in(path);
  if (!in.good()) return false;
  std::string line;
  if (!std::getline(in, line)) return false;
  char want[17];
  std::snprintf(want, sizeof want, "%016llx",
                static_cast<unsigned long long>(fingerprint_of(rules)));
  if (line != want) return false;  // rule set changed: sidecar is stale
  std::lock_guard<std::mutex> lk(mask_mu);
  while (std::getline(in, line))
    if (!line.empty()) mask_set.insert(line);
  return true;
}

bool fire_mask_save(const std::string& path) {
  std::lock_guard<std::mutex> lk(mask_mu);
  std::ofstream out(path);
  if (!out.good()) return false;
  char fp[17];
  std::snprintf(fp, sizeof fp, "%016llx",
                static_cast<unsigned long long>(mask_fingerprint));
  out << fp << "\n";
  for (const auto& k : mask_set) out << k << "\n";
  return out.good();
}

std::size_t fire_mask_size() {
  std::lock_guard<std::mutex> lk(mask_mu);
  return mask_set.size();
}

std::size_t fire_mask_hits() { return mask_hit_count; }

bool fire_mask_check(const std::string& rule_name, const sym::expr& node) {
  {
    std::lock_guard<std::mutex> lk(mask_mu);
    if (!mask_armed || mask_set.empty()) return false;
  }
  const std::string k = key_of(rule_name, node);
  std::lock_guard<std::mutex> lk(mask_mu);
  if (mask_set.count(k)) {
    ++mask_hit_count;
    return true;
  }
  return false;
}

void fire_mask_record(const std::string& rule_name, const sym::expr& node) {
  {
    std::lock_guard<std::mutex> lk(mask_mu);
    if (!mask_armed) return;
  }
  const std::string k = key_of(rule_name, node);
  std::lock_guard<std::mutex> lk(mask_mu);
  mask_set.insert(k);
}

}  // namespace ax::search
