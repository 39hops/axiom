#pragma once
/** @file budget.hpp Cooperative work budget for sym-layer computation.

    The native RULE_WALL: expensive sym operations (expand's distribute
    loops, canonical's ratio recursion) poll a thread-local deadline and
    throw work_expired when it passes. Callers that must be bounded
    (verify_edge, rule fires) install a budget with work_budget_scope;
    the exception is caught at that boundary and treated as a
    conservative rejection. No signals, no preemption, no partial
    results — an expired computation is abandoned wholesale. */
#include <chrono>
#include <stdexcept>

namespace ax::sym {

struct work_expired : std::runtime_error {
  work_expired() : std::runtime_error("sym work budget expired") {}
};

namespace detail {
inline thread_local std::chrono::steady_clock::time_point work_deadline{};
inline thread_local bool work_budget_active = false;
inline thread_local unsigned work_poll_counter = 0;
}  // namespace detail

/** Poll the budget (cheap: real clock read only every 256 calls). */
inline void check_work_budget() {
  if (!detail::work_budget_active) return;
  if ((++detail::work_poll_counter & 0xFF) != 0) return;
  if (std::chrono::steady_clock::now() > detail::work_deadline)
    throw work_expired();
}

/** RAII budget installer (budgets do not nest; inner scopes keep the
    tighter outer deadline). */
class work_budget_scope {
 public:
  explicit work_budget_scope(std::chrono::milliseconds ms)
      : outer_active_(detail::work_budget_active),
        outer_deadline_(detail::work_deadline) {
    const auto mine = std::chrono::steady_clock::now() + ms;
    detail::work_deadline =
        outer_active_ ? std::min(outer_deadline_, mine) : mine;
    detail::work_budget_active = true;
  }
  ~work_budget_scope() {
    detail::work_budget_active = outer_active_;
    detail::work_deadline = outer_deadline_;
  }
  work_budget_scope(const work_budget_scope&) = delete;
  work_budget_scope& operator=(const work_budget_scope&) = delete;

 private:
  bool outer_active_;
  std::chrono::steady_clock::time_point outer_deadline_;
};

}  // namespace ax::sym
