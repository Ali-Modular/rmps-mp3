#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rmps::app {

enum class Lang { Ja, En };

/// UI language of the GUI. Starts from the Windows display language and can
/// be switched at runtime.
inline Lang& ui_lang() {
    static Lang lang = [] {
#ifdef _WIN32
        return PRIMARYLANGID(GetUserDefaultUILanguage()) == LANG_JAPANESE ? Lang::Ja : Lang::En;
#else
        return Lang::En;
#endif
    }();
    return lang;
}

/// Pick the string for the current UI language.
inline const char* tr(const char* ja, const char* en) { return ui_lang() == Lang::Ja ? ja : en; }
inline const wchar_t* tr(const wchar_t* ja, const wchar_t* en) { return ui_lang() == Lang::Ja ? ja : en; }

} // namespace rmps::app
