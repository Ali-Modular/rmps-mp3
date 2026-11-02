#pragma once
// The lock that guards the encoder's lazily built tables (band geometry,
// perceptual weights). A real std::mutex on hosted builds. Single-threaded
// targets define RMPS_NO_THREADS: the Daisy's toolchain is built without
// threads and has no std::mutex at all, and there is nothing to guard.

#if defined(RMPS_NO_THREADS)

namespace rmps::sync {
struct Mutex {
    void lock() noexcept {}
    void unlock() noexcept {}
};
struct Lock {
    explicit Lock(Mutex&) noexcept {}
};
} // namespace rmps::sync

#else

#include <mutex>

namespace rmps::sync {
using Mutex = std::mutex;
using Lock  = std::lock_guard<std::mutex>;
} // namespace rmps::sync

#endif
