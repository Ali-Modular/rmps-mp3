#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace rmps::core {

/// A small persistent worker pool with one operation: run `fn(i)` for
/// i in [0, n) and wait for all of them.
///
/// Persistent because the search calls it once per granule-channel — thousands
/// of times per song — and creating threads at that rate would cost more than
/// the work it spreads. Work is handed out by an atomic index, so the calling
/// thread takes items too and nothing is queued per item.
///
/// Results must not depend on which thread ran which item; callers write each
/// item's output into its own slot and combine them afterwards in index order.
///
/// RMPS_THREADS=<n> sets the thread count (1 = run inline on the caller);
/// default is the hardware concurrency, capped at 16.
class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    int size() const noexcept { return static_cast<int>(workers_.size()) + 1; }

    void parallel_for(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        if (workers_.empty() || n == 1) {
            for (int i = 0; i < n; ++i) fn(i);
            return;
        }
        std::lock_guard<std::mutex> job_lock(job_mu_);   // one job at a time
        {
            std::lock_guard<std::mutex> lk(mu_);
            fn_.store(&fn);
            n_.store(n);
            next_.store(0);
            remaining_ = n;
            ++generation_;
        }
        cv_.notify_all();
        run_items();
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&] { return remaining_ == 0; });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

private:
    ThreadPool() {
        int n = static_cast<int>(std::thread::hardware_concurrency());
        if (const char* e = std::getenv("RMPS_THREADS")) n = std::atoi(e);
        n = std::clamp(n, 1, 16);
        for (int i = 1; i < n; ++i)
            workers_.emplace_back([this] { worker(); });
    }

    void run_items() {
        for (;;) {
            // Index and count are read atomically, so a worker still finishing
            // the previous job's last check cannot pick up a stale item.
            const int i = next_.fetch_add(1);
            if (i >= n_.load()) return;
            (*fn_.load())(i);
            std::lock_guard<std::mutex> lk(mu_);
            if (--remaining_ == 0) done_cv_.notify_all();
        }
    }

    void worker() {
        long long seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            run_items();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex job_mu_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::atomic<const std::function<void(int)>*> fn_{nullptr};
    std::atomic<int> n_{0};
    std::atomic<int> next_{0};
    int remaining_ = 0;
    long long generation_ = 0;
    bool stop_ = false;
};

} // namespace rmps::core
