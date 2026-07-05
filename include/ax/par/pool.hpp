#pragma once
/** @file pool.hpp Fixed-size thread pool and data-parallel helpers. */
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace ax {

/** Fixed-size thread pool executing submitted tasks FIFO. */
class thread_pool {
 public:
  /** @param n worker count; 0 selects hardware_concurrency(). */
  explicit thread_pool(unsigned n = 0) {
    if (n == 0) n = std::max(1u, std::thread::hardware_concurrency());
    workers_.reserve(n);
    for (unsigned i = 0; i < n; ++i)
      workers_.emplace_back([this](std::stop_token st) { run(st); });
  }

  ~thread_pool() {
    for (auto& w : workers_) w.request_stop();
    cv_.notify_all();
  }

  thread_pool(const thread_pool&) = delete;
  thread_pool& operator=(const thread_pool&) = delete;

  /** Number of worker threads. */
  unsigned size() const noexcept { return static_cast<unsigned>(workers_.size()); }

  /** Submit callable; returns future of its result. */
  template <class F>
  auto submit(F&& f) -> std::future<std::invoke_result_t<F>> {
    using R = std::invoke_result_t<F>;
    auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
    auto fut = task->get_future();
    {
      std::lock_guard lk(m_);
      q_.emplace_back([task] { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

 private:
  void run(std::stop_token st) {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock lk(m_);
        cv_.wait(lk, [&] { return st.stop_requested() || !q_.empty(); });
        if (q_.empty()) return;  // stop requested, queue drained
        job = std::move(q_.front());
        q_.pop_front();
      }
      job();
    }
  }

  std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> q_;
  std::vector<std::jthread> workers_;
};

/** Apply fn(i) for i in [first, last), chunked by grain, across pool. */
template <class F>
void parallel_for(thread_pool& p, std::size_t first, std::size_t last,
                  std::size_t grain, F&& fn) {
  if (first >= last) return;
  grain = std::max<std::size_t>(1, grain);
  std::vector<std::future<void>> futs;
  for (std::size_t b = first; b < last; b += grain) {
    std::size_t e = std::min(b + grain, last);
    futs.push_back(p.submit([b, e, &fn] {
      for (std::size_t i = b; i < e; ++i) fn(i);
    }));
  }
  for (auto& f : futs) f.get();
}

/** Chunked reduce: acc = step(acc, i) within chunk, chunks joined by join. */
template <class T, class Step, class Join>
T parallel_reduce(thread_pool& p, std::size_t first, std::size_t last,
                  std::size_t grain, T init, Step&& step, Join&& join) {
  if (first >= last) return init;
  grain = std::max<std::size_t>(1, grain);
  std::vector<std::future<T>> futs;
  for (std::size_t b = first; b < last; b += grain) {
    std::size_t e = std::min(b + grain, last);
    futs.push_back(p.submit([b, e, init, &step] {
      T acc = init;
      for (std::size_t i = b; i < e; ++i) acc = step(acc, i);
      return acc;
    }));
  }
  T total = init;
  for (auto& f : futs) total = join(total, f.get());
  return total;
}

}  // namespace ax
