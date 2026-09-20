#include "gui.hpp"
#include "i18n.hpp"
#include "analysis.hpp"
#include "analysis_view.hpp"
#include "audio_in.hpp"
#include "encode_job.hpp"
#include "utf8.hpp"

#include "rmps/version.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <commdlg.h>
#include <shobjidl.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#endif

#include <GLFW/glfw3.h>
#ifdef _WIN32
#include <GLFW/glfw3native.h>
#endif
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rmps::app {

namespace {

// ---------------------------------------------------------------------------
// Queue state shared between the UI thread and the encode worker.
// ---------------------------------------------------------------------------

enum class ItemState { Waiting, Running, Done, Failed, Cancelled };

struct QueueItem {
    std::string input;      // UTF-8
    std::string output;     // UTF-8, decided when the run starts
    ItemState state = ItemState::Waiting;
    std::string message;
    double progress = 0.0;  // 0..1
    double elapsed_s = 0.0;
    double duration_s = 0.0;
    std::size_t bytes = 0;
    std::shared_ptr<Analysis> analysis;   // created when the run starts
};

struct Shared {
    std::mutex mu;
    std::vector<QueueItem> items;
    bool running = false;
    std::atomic<bool> cancel{false};
    std::string current_detail;
};

struct Settings {
    int mode = 1;            // 0 CBR, 1 ABR, 2 quality
    int bitrate_idx = 8;     // into BITRATES (128)
    int quality = 50;
    int effort = 2;
    bool same_folder = true;
    std::string out_dir;     // UTF-8
};

constexpr int BITRATES[] = {32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320};
constexpr int NUM_BITRATES = static_cast<int>(sizeof BITRATES / sizeof BITRATES[0]);

std::string output_for(const std::string& input, const Settings& st) {
    std::filesystem::path in = to_path(input);
    std::filesystem::path dir = (st.same_folder || st.out_dir.empty())
                                    ? in.parent_path()
                                    : to_path(st.out_dir);
    std::filesystem::path out = dir / in.stem();
    out += L".mp3";
    return from_path(out);
}

void add_files(Shared& sh, const std::vector<std::string>& paths) {
    std::lock_guard<std::mutex> lk(sh.mu);
    for (const auto& p : paths) {
        std::error_code ec;
        if (std::filesystem::is_directory(to_path(p), ec)) continue;
        bool dup = std::any_of(sh.items.begin(), sh.items.end(),
                               [&](const QueueItem& q) { return q.input == p; });
        if (!dup) sh.items.push_back(QueueItem{p});
    }
}

// ---------------------------------------------------------------------------
// Native dialogs
// ---------------------------------------------------------------------------

#ifdef _WIN32
std::vector<std::string> pick_input_files(HWND owner) {
    std::wstring buf(64 * 1024, L'\0');
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof ofn;
    ofn.hwndOwner = owner;
    ofn.lpstrFilter = tr(
        L"音声ファイル (*.wav;*.flac;*.ogg;*.opus;*.oga;*.aif;*.aiff)\0"
        L"*.wav;*.flac;*.ogg;*.opus;*.oga;*.aif;*.aiff\0"
        L"すべてのファイル (*.*)\0*.*\0",
        L"Audio files (*.wav;*.flac;*.ogg;*.opus;*.oga;*.aif;*.aiff)\0"
        L"*.wav;*.flac;*.ogg;*.opus;*.oga;*.aif;*.aiff\0"
        L"All files (*.*)\0*.*\0");
    ofn.lpstrFile = buf.data();
    ofn.nMaxFile = static_cast<DWORD>(buf.size());
    ofn.Flags = OFN_ALLOWMULTISELECT | OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    std::vector<std::string> out;
    if (!GetOpenFileNameW(&ofn)) return out;
    // Multi-select returns "dir\0file1\0file2\0\0"; single select returns the
    // full path followed by "\0\0".
    const wchar_t* p = buf.c_str();
    std::wstring first = p;
    p += first.size() + 1;
    if (*p == L'\0') {
        out.push_back(narrow(first));
    } else {
        while (*p) {
            std::wstring name = p;
            out.push_back(narrow((std::filesystem::path(first) / name).wstring()));
            p += name.size() + 1;
        }
    }
    return out;
}

std::string pick_folder(HWND owner) {
    std::string result;
    if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)))
        return result;
    IFileOpenDialog* dlg = nullptr;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                   IID_IFileOpenDialog, reinterpret_cast<void**>(&dlg)))) {
        DWORD opts = 0;
        dlg->GetOptions(&opts);
        dlg->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
        if (SUCCEEDED(dlg->Show(owner))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dlg->GetResult(&item))) {
                PWSTR path = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
                    result = narrow(path);
                    CoTaskMemFree(path);
                }
                item->Release();
            }
        }
        dlg->Release();
    }
    CoUninitialize();
    return result;
}

void open_folder_of(const std::string& utf8_file) {
    std::wstring arg = L"/select,\"" + widen(utf8_file) + L"\"";
    ShellExecuteW(nullptr, L"open", L"explorer.exe", arg.c_str(), nullptr, SW_SHOWNORMAL);
}
#endif

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

void worker(Shared* sh, Settings st) {
    for (;;) {
        std::size_t idx = 0;
        EncodeJob job;
        std::shared_ptr<Analysis> analysis;
        {
            std::lock_guard<std::mutex> lk(sh->mu);
            auto it = std::find_if(sh->items.begin(), sh->items.end(), [](const QueueItem& q) {
                return q.state == ItemState::Waiting;
            });
            if (it == sh->items.end() || sh->cancel.load()) break;
            idx = static_cast<std::size_t>(it - sh->items.begin());
            it->state = ItemState::Running;
            it->progress = 0.0;
            it->output = output_for(it->input, st);
            job.input = it->input;
            job.output = it->output;
            it->analysis = std::make_shared<Analysis>();
            analysis = it->analysis;
        }
        job.effort = st.effort;
        job.bitrate = BITRATES[st.bitrate_idx];
        if (st.mode == 1) job.abr_kbps = job.bitrate;
        if (st.mode == 2) job.quality_sdr_db = quality_to_score_db(st.quality);
        {
            char buf[128];
            if (st.mode == 2)
                std::snprintf(buf, sizeof buf, tr("品質 %d (%.1f dB)  effort %d", "Quality %d (%.1f dB)  effort %d"), st.quality,
                              quality_to_score_db(st.quality), st.effort);
            else
                std::snprintf(buf, sizeof buf, "%s %d kbps  effort %d", st.mode == 0 ? "CBR" : "ABR",
                              job.bitrate, st.effort);
            std::lock_guard<std::mutex> lk(analysis->mu);
            analysis->title = from_path(to_path(job.input).filename());
            analysis->settings = buf;
            analysis->target_kbps = st.mode == 2 ? 0 : job.bitrate;
        }

        EncodeCallbacks cb;
        cb.cancel = &sh->cancel;
        cb.on_start = [&](const EncodeProgress& p) {
            std::lock_guard<std::mutex> lk(sh->mu);
            sh->items[idx].duration_s = p.duration_s;
            char buf[256];
            std::snprintf(buf, sizeof buf, "%s  %d Hz  %d ch%s", p.source_format.c_str(),
                          p.input_samplerate, p.source_channels,
                          p.coded_samplerate != p.input_samplerate
                              ? ("  → " + std::to_string(p.coded_samplerate) + tr(" Hz で符号化", " Hz coded")).c_str()
                              : "");
            sh->current_detail = buf;
        };
        cb.on_frame = [&](const EncodeProgress& p, const FrameStats& fs, const AbrState*) {
            analysis->append(fs);
            if (p.mp3_frames_done % 16 != 0) return;   // keep lock traffic low
            std::lock_guard<std::mutex> lk(sh->mu);
            auto& q = sh->items[idx];
            q.progress = p.input_frames_total
                             ? static_cast<double>(p.input_frames_done) / p.input_frames_total
                             : 0.0;
            q.elapsed_s = p.elapsed_s;
            q.bytes = p.bytes_out;
        };

        const EncodeResult r = run_encode(job, cb);
        {
            std::lock_guard<std::mutex> alk(analysis->mu);
            analysis->finished = true;
        }
        std::lock_guard<std::mutex> lk(sh->mu);
        auto& q = sh->items[idx];
        q.elapsed_s = r.elapsed_s;
        q.bytes = r.bytes_out;
        if (r.cancelled) {
            q.state = ItemState::Cancelled;
            q.message = tr("中止しました", "Cancelled");
        } else if (!r.ok) {
            q.state = ItemState::Failed;
            q.message = r.error;
        } else {
            q.state = ItemState::Done;
            q.progress = 1.0;
            char buf[160];
            const double kbps = r.duration_s > 0 ? r.bytes_out * 8.0 / r.duration_s / 1000.0 : 0.0;
            std::snprintf(buf, sizeof buf, tr("%.2f MB  %.1f kbps  %.1f 秒 (x%.1f)", "%.2f MB  %.1f kbps  %.1f s (x%.1f)"),
                          r.bytes_out / 1e6, kbps, r.elapsed_s,
                          r.elapsed_s > 0 ? r.duration_s / r.elapsed_s : 0.0);
            q.message = buf;
        }
    }
    {
        std::lock_guard<std::mutex> lk(sh->mu);
        sh->running = false;
        sh->current_detail.clear();
    }
    // The UI thread sleeps in glfwWaitEvents when idle; wake it so the final
    // state is drawn.
    glfwPostEmptyEvent();
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

Shared* g_shared = nullptr;

void drop_callback(GLFWwindow*, int count, const char** paths) {
    // GLFW hands dropped paths over as UTF-8. Adding while an encode runs is
    // fine: the worker picks up Waiting items as it goes.
    add_files(*g_shared, std::vector<std::string>(paths, paths + count));
}

void load_japanese_font(ImGuiIO& io) {
    const char* candidates[] = {
        "C:\\Windows\\Fonts\\meiryo.ttc",
        "C:\\Windows\\Fonts\\YuGothM.ttc",
        "C:\\Windows\\Fonts\\msgothic.ttc",
        "C:\\Windows\\Fonts\\NotoSansJP-VF.ttf",
    };
    for (const char* f : candidates) {
        std::error_code ec;
        if (!std::filesystem::exists(f, ec)) continue;
        ImFontConfig cfg;
        cfg.OversampleH = 2;
        if (io.Fonts->AddFontFromFileTTF(f, 18.0f, &cfg, io.Fonts->GetGlyphRangesJapanese()))
            return;
    }
    io.Fonts->AddFontDefault();
}

const char* state_label(ItemState s) {
    switch (s) {
    case ItemState::Waiting: return tr("待機", "Waiting");
    case ItemState::Running: return tr("処理中", "Encoding");
    case ItemState::Done: return tr("完了", "Done");
    case ItemState::Failed: return tr("エラー", "Error");
    case ItemState::Cancelled: return tr("中止", "Cancelled");
    }
    return "";
}

struct AnalysisUi {
    AnalysisView view;
    std::shared_ptr<Analysis> selected;
    bool active = false;        // analysis tab visible
    bool auto_follow = true;    // show whatever is encoding now
    bool open_request = false;  // switch to the analysis tab next frame
};

void draw_analysis(Shared& sh, AnalysisUi& ui) {
    std::vector<std::pair<std::string, std::shared_ptr<Analysis>>> choices;
    std::shared_ptr<Analysis> running_one;
    {
        std::lock_guard<std::mutex> lk(sh.mu);
        for (const auto& q : sh.items)
            if (q.analysis) {
                choices.emplace_back(from_path(to_path(q.input).filename()) + "  [" + state_label(q.state) + "]",
                                     q.analysis);
                if (q.state == ItemState::Running) running_one = q.analysis;
            }
    }
    if (ui.auto_follow && running_one) ui.selected = running_one;
    // Default to the most recent run (the one encoding now, or the last one).
    if (ui.selected && std::none_of(choices.begin(), choices.end(),
                                    [&](const auto& c) { return c.second == ui.selected; }))
        ui.selected.reset();
    if (!ui.selected && !choices.empty()) ui.selected = choices.back().second;
    std::string preview = tr("（なし）", "(none)");
    for (const auto& c : choices)
        if (c.second == ui.selected) preview = c.first;
    ImGui::SetNextItemWidth(480);
    if (ImGui::BeginCombo(tr("ファイル", "File"), preview.c_str())) {
        for (std::size_t i = 0; i < choices.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::Selectable(choices[i].first.c_str(), choices[i].second == ui.selected)) {
                ui.selected = choices[i].second;
                ui.auto_follow = choices[i].second == running_one;
            }
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::Checkbox(tr("エンコード中のファイルを自動表示", "Follow the file being encoded"), &ui.auto_follow);
    ui.view.draw(ui.selected.get());
}

void draw(GLFWwindow* window, Shared& sh, Settings& st, std::thread& th, AnalysisUi& aui) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("main", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings);

    aui.active = false;
    if (!ImGui::BeginTabBar("top")) { ImGui::End(); return; }
    if (ImGui::TabItemButton(ui_lang() == Lang::Ja ? "English" : "日本語", ImGuiTabItemFlags_Trailing)) {
        ui_lang() = ui_lang() == Lang::Ja ? Lang::En : Lang::Ja;
        glfwSetWindowTitle(window, (std::string(tr("rmps MP3 エンコーダ ", "rmps MP3 encoder ")) + VERSION_STRING).c_str());
        glfwPostEmptyEvent();
    }
    if (!ImGui::BeginTabItem(tr("エンコード###enc", "Encode###enc"))) {
        if (ImGui::BeginTabItem(tr("解析###ana", "Analysis###ana"))) {
            aui.active = true;
            draw_analysis(sh, aui);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
        ImGui::End();
        return;
    }

    bool running;
    {
        std::lock_guard<std::mutex> lk(sh.mu);
        running = sh.running;
    }
#ifdef _WIN32
    HWND hwnd = glfwGetWin32Window(window);
#else
    (void)window;
#endif

    // ---- Input ----
    ImGui::SeparatorText(tr("入力ファイル", "Input files"));
    ImGui::TextDisabled(tr("ここにファイルをドラッグ＆ドロップ、または追加ボタン（対応: %s）", "Drag & drop files here, or use Add files (supported: %s)"),
                        supported_formats());
#ifdef _WIN32
    if (ImGui::Button(tr("ファイルを追加…", "Add files..."))) add_files(sh, pick_input_files(hwnd));
    ImGui::SameLine();
#endif
    ImGui::BeginDisabled(running);
    if (ImGui::Button(tr("完了分を消去", "Clear finished"))) {
        std::lock_guard<std::mutex> lk(sh.mu);
        sh.items.erase(std::remove_if(sh.items.begin(), sh.items.end(),
                                      [](const QueueItem& q) { return q.state != ItemState::Waiting; }),
                       sh.items.end());
    }
    ImGui::SameLine();
    if (ImGui::Button(tr("すべて消去", "Clear all"))) {
        std::lock_guard<std::mutex> lk(sh.mu);
        sh.items.clear();
    }
    ImGui::EndDisabled();

    const float list_h = std::max(140.0f, ImGui::GetContentRegionAvail().y - 240.0f);
    if (ImGui::BeginTable("queue", 4,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable,
                          ImVec2(0, list_h))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn(tr("ファイル", "File"), ImGuiTableColumnFlags_WidthStretch, 3.0f);
        ImGui::TableSetupColumn(tr("状態", "Status"), ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableSetupColumn(tr("進捗", "Progress"), ImGuiTableColumnFlags_WidthStretch, 1.5f);
        ImGui::TableSetupColumn(tr("結果", "Result"), ImGuiTableColumnFlags_WidthStretch, 2.5f);
        ImGui::TableHeadersRow();
        std::lock_guard<std::mutex> lk(sh.mu);
        int remove_at = -1;
        for (int i = 0; i < static_cast<int>(sh.items.size()); ++i) {
            auto& q = sh.items[static_cast<std::size_t>(i)];
            ImGui::PushID(i);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(from_path(to_path(q.input).filename()).c_str());
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", q.input.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(state_label(q.state));
            ImGui::TableNextColumn();
            if (q.analysis && (q.state == ItemState::Running || q.state == ItemState::Done)) {
                if (ImGui::SmallButton(tr("解析", "Analyze"))) {
                    aui.selected = q.analysis;
                    aui.auto_follow = q.state == ItemState::Running;
                    aui.open_request = true;
                }
                ImGui::SameLine();
            }
            ImGui::ProgressBar(static_cast<float>(q.progress), ImVec2(-1, 0));
            ImGui::TableNextColumn();
            if (q.state == ItemState::Done) {
#ifdef _WIN32
                if (ImGui::SmallButton(tr("場所を開く", "Show in folder"))) open_folder_of(q.output);
                ImGui::SameLine();
#endif
                ImGui::TextUnformatted(q.message.c_str());
            } else if (q.state == ItemState::Failed) {
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "%s", q.message.c_str());
            } else if (q.state == ItemState::Waiting && !running) {
                if (ImGui::SmallButton(tr("削除", "Remove"))) remove_at = i;
            } else {
                ImGui::TextUnformatted(q.message.c_str());
            }
            ImGui::PopID();
        }
        if (remove_at >= 0) sh.items.erase(sh.items.begin() + remove_at);
        ImGui::EndTable();
    }

    // ---- Settings ----
    ImGui::BeginDisabled(running);
    ImGui::SeparatorText(tr("設定", "Settings"));
    ImGui::TextUnformatted(tr("モード:", "Mode:"));
    ImGui::SameLine();
    ImGui::RadioButton(tr("CBR（固定）", "CBR (constant)"), &st.mode, 0);
    ImGui::SameLine();
    ImGui::RadioButton(tr("ABR（平均）", "ABR (average)"), &st.mode, 1);
    ImGui::SameLine();
    ImGui::RadioButton(tr("品質指定 VBR", "Quality VBR"), &st.mode, 2);

    ImGui::SetNextItemWidth(260);
    if (st.mode == 2) {
        ImGui::SliderInt(tr("品質 (1-100)", "Quality (1-100)"), &st.quality, 1, 100);
        ImGui::SameLine();
        ImGui::TextDisabled(tr("目標 %.1f dB", "target %.1f dB"), quality_to_score_db(st.quality));
    } else {
        const std::string preview = std::to_string(BITRATES[st.bitrate_idx]) + " kbps";
        if (ImGui::BeginCombo(st.mode == 0 ? tr("ビットレート", "Bitrate") : tr("目標ビットレート", "Target bitrate"), preview.c_str())) {
            for (int i = 0; i < NUM_BITRATES; ++i) {
                const std::string label = std::to_string(BITRATES[i]) + " kbps";
                if (ImGui::Selectable(label.c_str(), st.bitrate_idx == i)) st.bitrate_idx = i;
            }
            ImGui::EndCombo();
        }
    }

    ImGui::SetNextItemWidth(260);
    ImGui::SliderInt("Effort (0-10)", &st.effort, 0, 10);
    ImGui::SameLine();
    ImGui::TextDisabled(st.effort <= 2 ? tr("標準", "standard") : st.effort <= 4 ? tr("高品質・やや遅い", "higher quality, slower") : tr("最高品質・遅い", "best quality, slow"));

    ImGui::TextUnformatted(tr("出力先:", "Output:"));
    ImGui::SameLine();
    int dest = st.same_folder ? 0 : 1;
    ImGui::RadioButton(tr("入力と同じフォルダ", "Same folder as input"), &dest, 0);
    ImGui::SameLine();
    ImGui::RadioButton(tr("フォルダを指定", "Choose folder"), &dest, 1);
    st.same_folder = dest == 0;
    if (!st.same_folder) {
#ifdef _WIN32
        if (ImGui::Button(tr("参照…", "Browse..."))) {
            const std::string d = pick_folder(hwnd);
            if (!d.empty()) st.out_dir = d;
        }
        ImGui::SameLine();
#endif
        ImGui::TextUnformatted(st.out_dir.empty() ? tr("（未選択）", "(not selected)") : st.out_dir.c_str());
    }
    ImGui::EndDisabled();

    // ---- Run ----
    ImGui::SeparatorText(tr("実行", "Run"));
    std::size_t waiting = 0, total = 0;
    double overall = 0.0;
    std::string detail;
    {
        std::lock_guard<std::mutex> lk(sh.mu);
        total = sh.items.size();
        for (const auto& q : sh.items) {
            if (q.state == ItemState::Waiting) ++waiting;
            overall += (q.state == ItemState::Running) ? q.progress
                       : (q.state == ItemState::Waiting) ? 0.0 : 1.0;
        }
        detail = sh.current_detail;
    }
    const bool can_start = !running && waiting > 0 && (st.same_folder || !st.out_dir.empty());
    if (!running) {
        ImGui::BeginDisabled(!can_start);
        if (ImGui::Button(tr("エンコード開始", "Start encoding"), ImVec2(180, 36))) {
            if (th.joinable()) th.join();
            sh.cancel = false;
            {
                std::lock_guard<std::mutex> lk(sh.mu);
                sh.running = true;
            }
            th = std::thread(worker, &sh, st);
        }
        ImGui::EndDisabled();
    } else {
        if (ImGui::Button(tr("中止", "Cancel"), ImVec2(180, 36))) sh.cancel = true;
    }
    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::ProgressBar(total ? static_cast<float>(overall / total) : 0.0f, ImVec2(-1, 0));
    if (running) ImGui::TextDisabled("%s", detail.c_str());
    else if (total == 0) ImGui::TextDisabled(tr("ファイルがありません", "No files"));
    else ImGui::TextDisabled(tr("待機 %zu 件 / 全 %zu 件", "%zu waiting / %zu total"), waiting, total);
    ImGui::EndGroup();

    ImGui::EndTabItem();
    if (ImGui::BeginTabItem(tr("解析###ana", "Analysis###ana"), nullptr, aui.open_request ? ImGuiTabItemFlags_SetSelected : 0))
        ImGui::EndTabItem();   // not selected this frame: keep it in the bar
    if (aui.open_request) {
        aui.open_request = false;
        glfwPostEmptyEvent();   // the switch lands on the next frame
    }
    ImGui::EndTabBar();
    ImGui::End();
}

} // namespace

int run_gui() {
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    const std::string title = std::string(tr("rmps MP3 エンコーダ ", "rmps MP3 encoder ")) + VERSION_STRING;
    GLFWwindow* window = glfwCreateWindow(1100, 760, title.c_str(), nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    load_japanese_font(io);
    ImGui::StyleColorsDark();
    ImGui::GetStyle().FrameRounding = 4.0f;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    Shared shared;
    g_shared = &shared;
    Settings settings;
    std::thread th;
    AnalysisUi analysis_ui;
    glfwSetDropCallback(window, drop_callback);

    while (!glfwWindowShouldClose(window)) {
        // Idle politely when nothing is happening; redraw steadily while
        // encoding so progress moves.
        bool busy;
        {
            std::lock_guard<std::mutex> lk(shared.mu);
            busy = shared.running;
        }
        if (busy) glfwWaitEventsTimeout(0.1); else glfwWaitEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        draw(window, shared, settings, th, analysis_ui);
        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.10f, 0.10f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    shared.cancel = true;
    if (th.joinable()) th.join();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

} // namespace rmps::app
