#pragma once
#include <filesystem>
#include <string>
#include <vector>

namespace rmps::app {

/// Paths are carried through the app as UTF-8 `std::string` and turned into
/// `std::filesystem::path` only at the point a file is opened.
///
/// Narrow strings are not safe on Windows: `argv` and `fopen` go through the
/// ANSI code page, which on this machine is 1252, so a path such as
/// `I:\マイドライブ\...` arrived as `I:\??????\...` and the output could not be
/// opened. Converting through UTF-8 and opening via the wide-character APIs
/// that `std::filesystem::path` uses under MSVC handles any script.
std::filesystem::path to_path(const std::string& utf8);
std::string from_path(const std::filesystem::path& p);

#ifdef _WIN32
std::wstring widen(const std::string& utf8);
std::string narrow(const std::wstring& wide);
#endif

/// The process's command-line arguments as UTF-8, including argv[0].
std::vector<std::string> utf8_args(int argc, char** argv);

} // namespace rmps::app
