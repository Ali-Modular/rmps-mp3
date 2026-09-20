#include "utf8.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#endif

namespace rmps::app {

#ifdef _WIN32
std::wstring widen(const std::string& utf8) {
    if (utf8.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.data(),
                                      static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()),
                        w.data(), n);
    return w;
}

std::string narrow(const std::wstring& wide) {
    if (wide.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, wide.data(),
                                      static_cast<int>(wide.size()), nullptr, 0,
                                      nullptr, nullptr);
    std::string s(static_cast<std::size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
                        s.data(), n, nullptr, nullptr);
    return s;
}

std::filesystem::path to_path(const std::string& utf8) {
    return std::filesystem::path(widen(utf8));
}
std::string from_path(const std::filesystem::path& p) {
    return narrow(p.wstring());
}

std::vector<std::string> utf8_args(int, char**) {
    std::vector<std::string> out;
    int n = 0;
    LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n);
    if (!w) return out;
    for (int i = 0; i < n; ++i) out.push_back(narrow(w[i]));
    LocalFree(w);
    return out;
}
#else
std::filesystem::path to_path(const std::string& utf8) { return std::filesystem::path(utf8); }
std::string from_path(const std::filesystem::path& p) { return p.string(); }
std::vector<std::string> utf8_args(int argc, char** argv) {
    return std::vector<std::string>(argv, argv + argc);
}
#endif

} // namespace rmps::app
