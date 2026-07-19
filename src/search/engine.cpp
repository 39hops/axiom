#include <ax/search/search.hpp>

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>

/** Phase C task 6: the engine facade — Markov-prior proposer (the
    zero-NN 316/360 configuration) + solve() defaults. The prior table
    is llmopt's export (data/llmopt/markov_prior.tsv): U\tname\tcount
    unigram rows, B\tprev\tnext\tcount bigram rows. */

namespace ax::search {

markov_prior markov_prior::load_tsv(const std::string& path) {
  markov_prior p;
  std::ifstream in(path);
  if (!in.good())
    throw std::runtime_error("markov_prior: cannot open " + path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string kind, a, b;
    long long n = 0;
    ss >> kind;
    if (kind == "U") {
      ss >> a >> n;
      p.unigram[a] = n;
    } else if (kind == "B") {
      ss >> a >> b >> n;
      p.bigram[a][b] = n;
    }
  }
  return p;
}

double markov_prior::median_unigram() const {
  if (unigram.empty()) return 1.0;
  std::vector<long long> v;
  v.reserve(unigram.size());
  for (const auto& [k, n] : unigram) v.push_back(n);
  std::sort(v.begin(), v.end());
  return static_cast<double>(v[v.size() / 2]);
}

std::function<std::vector<std::pair<std::string, state>>(
    const state&, std::vector<std::pair<std::string, state>>)>
markov_prior::proposer() const {
  // Unseen-rule trial mass = 0.5*median OUTRIGHT (the measured rule:
  // 0.01*median starved new rules; re-mining regressed twice — llmopt
  // RESULTS). Rules only fire on matching structure, so the fire itself
  // is evidence.
  const double med = median_unigram();
  const auto self = *this;  // value capture: proposer outlives the prior
  return [self, med](const state& s,
                     std::vector<std::pair<std::string, state>> kids) {
    const std::string prev = s.history.empty() ? "" : s.history.back();
    const auto* table =
        prev.empty() ? nullptr : [&]() -> const std::map<std::string, long long>* {
          const auto it = self.bigram.find(prev);
          return it == self.bigram.end() ? nullptr : &it->second;
        }();
    const auto score = [&](const std::string& name) -> double {
      const auto u = self.unigram.find(name);
      if (u == self.unigram.end()) return 0.5 * med;
      double sc = 0.01 * static_cast<double>(u->second);
      if (table != nullptr) {
        const auto b = table->find(name);
        if (b != table->end()) sc += static_cast<double>(b->second);
      }
      return sc;
    };
    std::stable_sort(kids.begin(), kids.end(),
                     [&](const auto& a, const auto& b) {
                       return score(a.first) > score(b.first);
                     });
    return kids;
  };
}

search_result solve(const expr& root, const rule_set& rules,
                    const markov_prior& prior, long long budget,
                    std::optional<std::chrono::steady_clock::time_point>
                        deadline) {
  // markov3 @ width 3 — the measured-best zero-NN configuration
  // (llmopt engine.py defaults; verify_p stays 1: native verify is
  // cheap and every edge pays the oracle).
  beam_options opt;
  opt.width = 3;
  opt.max_plies = 24;
  opt.max_nodes = budget;
  opt.use_macros = true;
  opt.proposer = prior.proposer();
  opt.propose_k = 3;
  opt.deadline = deadline;
  return beam_search(root, rules, opt);
}

}  // namespace ax::search
