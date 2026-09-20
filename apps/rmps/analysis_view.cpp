#include "analysis_view.hpp"
#include "i18n.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace rmps::app {

// ===========================================================================
// Data capture
// ===========================================================================

void Analysis::append(const FrameStats& fs) {
    FrameRec r;
    r.kbps = static_cast<std::uint16_t>(fs.bitrate_kbps);
    r.ms = fs.ms_stereo;
    r.resv = fs.reservoir_bits;
    r.resv_max = fs.reservoir_max_bits;
    r.overflow = fs.overflow_bits;
    r.lpf_hz = static_cast<float>(fs.lowpass_mid_hz);
    int max_gr = 0, max_ch = 0;
    for (const auto& s : fs.side) {
        if (s.gr > 1 || s.ch > 1) continue;
        max_gr = std::max(max_gr, s.gr + 1);
        max_ch = std::max(max_ch, s.ch + 1);
        GranRec& g = r.g[static_cast<std::size_t>(s.gr)][static_cast<std::size_t>(s.ch)];
        g.block = static_cast<std::uint8_t>(s.block_type);
        g.scale = static_cast<std::uint8_t>(s.scalefac_scale);
        g.preflag = static_cast<std::uint8_t>(s.preflag);
        g.compress = static_cast<std::uint8_t>(s.scalefac_compress);
        g.c1tab = static_cast<std::uint8_t>(s.count1_table);
        g.r0 = static_cast<std::uint8_t>(s.region0_count);
        g.r1 = static_cast<std::uint8_t>(s.region1_count);
        for (int i = 0; i < 3; ++i) {
            g.tbl[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(s.table_select[static_cast<std::size_t>(i)]);
            g.sbg[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(s.subblock_gain[static_cast<std::size_t>(i)]);
        }
        g.gg = static_cast<std::int16_t>(s.global_gain);
        g.part2 = static_cast<std::uint16_t>(s.part2_bits);
        g.part23 = static_cast<std::uint16_t>(s.part2_3_bits);
        g.big_values = static_cast<std::uint16_t>(s.big_values);
        g.count1 = static_cast<std::uint16_t>(s.count1);
        g.sf_l = s.sf_long;
        g.sf_s = s.sf_short;
    }
    if (max_gr == 0) return;   // no audio frame was produced by this call
    r.n_gr = static_cast<std::uint8_t>(std::max(max_gr, 1));
    r.n_ch = static_cast<std::uint8_t>(std::max(max_ch, 1));
    std::lock_guard<std::mutex> lk(mu);
    if (samplerate == 0) {
        samplerate = fs.samplerate;
        layout = band_layout(fs.samplerate);
    }
    frames.push_back(r);
}

namespace {

// ===========================================================================
// Helpers
// ===========================================================================

constexpr std::array<int, 22> PRETAB = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        1, 1, 1, 1, 2, 2, 3, 3, 3, 2, 0};
constexpr std::array<int, 16> SLEN1 = {0, 0, 0, 0, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4};
constexpr std::array<int, 16> SLEN2 = {0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3};
constexpr float LEFT = 64.0f;       // axis label margin

constexpr int CAT_C1A = 32, CAT_C1B = 33, CAT_ZERO = 34;

ImU32 rgba(int r, int g, int b, int a = 255) { return IM_COL32(r, g, b, a); }

/// Viridis-like colormap, t in [0, 1].
void colormap(double t, std::uint8_t* out) {
    static constexpr double stops[][3] = {
        {0.267, 0.005, 0.329}, {0.283, 0.141, 0.458}, {0.254, 0.265, 0.530},
        {0.207, 0.372, 0.553}, {0.164, 0.471, 0.558}, {0.128, 0.567, 0.551},
        {0.135, 0.659, 0.518}, {0.267, 0.749, 0.441}, {0.478, 0.821, 0.318},
        {0.741, 0.873, 0.150}, {0.993, 0.906, 0.144}};
    t = std::clamp(t, 0.0, 1.0) * 10.0;
    const int i = std::min(static_cast<int>(t), 9);
    const double f = t - i;
    for (int c = 0; c < 3; ++c)
        out[c] = static_cast<std::uint8_t>(255.0 * (stops[i][c] + f * (stops[i + 1][c] - stops[i][c])));
    out[3] = 255;
}

/// Colour for a Huffman category: tables 0-31, count1 A/B, zero region.
std::array<std::uint8_t, 4> table_color(int cat) {
    if (cat == CAT_ZERO) return {18, 18, 22, 255};
    if (cat == CAT_C1A) return {150, 150, 160, 255};
    if (cat == CAT_C1B) return {215, 215, 225, 255};
    if (cat == 0) return {55, 55, 65, 255};
    auto hsv = [](double h, double s, double v) -> std::array<std::uint8_t, 4> {
        double r = 0, g = 0, b = 0;
        const int i = static_cast<int>(h * 6.0) % 6;
        const double f = h * 6.0 - std::floor(h * 6.0);
        const double p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
        switch (i) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
        }
        return {static_cast<std::uint8_t>(r * 255), static_cast<std::uint8_t>(g * 255),
                static_cast<std::uint8_t>(b * 255), 255};
    };
    if (cat <= 15) return hsv((cat - 1) / 15.0 * 0.85, 0.70, 0.95);      // small-value tables
    if (cat <= 23) return hsv(0.83, 0.25 + 0.08 * (cat - 16), 0.95);     // ESC family 16
    return hsv(0.50, 0.25 + 0.08 * (cat - 24), 0.95);                    // ESC family 24
}

const char* category_name(int cat, char* buf, std::size_t n) {
    if (cat == CAT_ZERO) return tr("ゼロ領域", "zero region");
    if (cat == CAT_C1A) return "count1 A";
    if (cat == CAT_C1B) return "count1 B";
    std::snprintf(buf, n, tr("テーブル %d", "table %d"), cat);
    return buf;
}

const char* block_name(int b) {
    switch (b) {
    case 1: return "START";
    case 2: return "SHORT";
    case 3: return "STOP";
    default: return "LONG";
    }
}

ImU32 block_color(int b) {
    switch (b) {
    case 1: return rgba(230, 180, 60);
    case 2: return rgba(235, 80, 70);
    case 3: return rgba(230, 180, 60);
    default: return rgba(70, 90, 110);
    }
}

std::string hz_label(double hz) {
    char b[32];
    if (hz >= 1000.0) std::snprintf(b, sizeof b, "%.1fk", hz / 1000.0);
    else std::snprintf(b, sizeof b, "%.0f", hz);
    return b;
}

int side_info_bits(int channels) { return channels == 1 ? (4 + 17) * 8 : (4 + 32) * 8; }

double frame_capacity_bits(const FrameRec& f, int sr) {
    return sr > 0 ? f.kbps * 1000.0 * 1152.0 / sr : 0.0;
}

} // namespace

// ===========================================================================
// GL texture for heatmaps
// ===========================================================================

struct AnalysisView::Tex {
    GLuint id = 0;
    int w = 0, h = 0;
    std::string key;   ///< what the pixels were built for
    void upload(int nw, int nh, const std::vector<std::uint8_t>& px) {
        if (!id) glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F);   // CLAMP_TO_EDGE
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, nw, nh, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        w = nw;
        h = nh;
    }
    ~Tex() { if (id) glDeleteTextures(1, &id); }
};

AnalysisView::AnalysisView() : sf_tex_(std::make_unique<Tex>()), huff_tex_(std::make_unique<Tex>()) {}
AnalysisView::~AnalysisView() = default;

// ===========================================================================
// Sync
// ===========================================================================

void AnalysisView::sync(Analysis* a) {
    if (a != src_) {
        src_ = a;
        fr_.clear();
        huff_lines_ = {};
        short_tx_of_freq_.clear();
        v0_ = 0.0;
        v1_ = 1.0;
        follow_ = true;
        sel_ = 0;
        sf_tex_->key.clear();
        huff_tex_->key.clear();
    }
    if (!a) return;
    std::size_t start;
    {
        std::lock_guard<std::mutex> lk(a->mu);
        title_ = a->title;
        settings_ = a->settings;
        target_kbps_ = a->target_kbps;
        if (a->samplerate) samplerate_ = a->samplerate;
        layout_ = a->layout;
        finished_ = a->finished;
        start = fr_.size();
        if (a->frames.size() > start)
            fr_.insert(fr_.end(), a->frames.begin() + static_cast<std::ptrdiff_t>(start), a->frames.end());
    }
    if (layout_.long_edges.size() < 23 || layout_.short_edges.size() < 14) return;

    // Long-equivalent frequency line -> short transmission index (middle
    // window), for drawing short-block Huffman regions on a frequency axis.
    if (short_tx_of_freq_.empty()) {
        short_tx_of_freq_.assign(576, 0);
        const auto& se = layout_.short_edges;
        for (int L = 0; L < 576; ++L) {
            const int sl = std::min(L / 3, 191);
            int sb = 0;
            while (sb < 12 && sl >= se[static_cast<std::size_t>(sb + 1)]) ++sb;
            int tx = 0;
            for (int s = 0; s < sb; ++s) tx += 3 * (se[static_cast<std::size_t>(s + 1)] - se[static_cast<std::size_t>(s)]);
            const int width = se[static_cast<std::size_t>(sb + 1)] - se[static_cast<std::size_t>(sb)];
            tx += width + (sl - se[static_cast<std::size_t>(sb)]);
            short_tx_of_freq_[static_cast<std::size_t>(L)] = std::min(tx, 575);
        }
    }

    // Accumulate Huffman usage for the new frames.
    const auto& le = layout_.long_edges;
    for (std::size_t f = start; f < fr_.size(); ++f) {
        const FrameRec& fr = fr_[f];
        for (int gr = 0; gr < fr.n_gr; ++gr)
            for (int ch = 0; ch < fr.n_ch; ++ch) {
                const GranRec& g = fr.g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(ch)];
                const int bv = 2 * g.big_values;
                const int c1 = bv + 4 * g.count1;
                int a1, a2;
                if (g.block == 2) { a1 = std::min(36, bv); a2 = bv; }
                else {
                    a1 = std::min(le[static_cast<std::size_t>(std::min(g.r0 + 1, 22))], bv);
                    a2 = std::min(le[static_cast<std::size_t>(std::min(g.r0 + g.r1 + 2, 22))], bv);
                }
                auto& h = huff_lines_[static_cast<std::size_t>(ch)];
                h[g.tbl[0]] += a1;
                if (g.block == 2) h[g.tbl[1]] += bv - a1;
                else { h[g.tbl[1]] += a2 - a1; h[g.tbl[2]] += bv - a2; }
                h[static_cast<std::size_t>(g.c1tab ? CAT_C1B : CAT_C1A)] += std::min(c1, 576) - bv;
                h[CAT_ZERO] += 576 - std::min(c1, 576);
            }
    }

    const double n = static_cast<double>(std::max<std::size_t>(fr_.size(), 1));
    if (follow_) { v0_ = 0.0; v1_ = n; }
    sel_ = std::clamp(sel_, 0, static_cast<int>(fr_.size()) - 1);
    if (sel_ < 0) sel_ = 0;
}

// ===========================================================================
// Timeline plumbing
// ===========================================================================

namespace {

struct Plot {
    ImVec2 p0, p1;     // data area
    bool hovered = false;
    double v0 = 0, v1 = 1;
    float x_of(double f) const {
        return p0.x + static_cast<float>((f - v0) / (v1 - v0)) * (p1.x - p0.x);
    }
    double frame_at(float x) const {
        return v0 + (x - p0.x) / std::max(p1.x - p0.x, 1.0f) * (v1 - v0);
    }
    int width() const { return std::max(1, static_cast<int>(p1.x - p0.x)); }
    /// Frame range [f0, f1) covered by pixel column `px`.
    void column(int px, int nframes, int& f0, int& f1) const {
        const double span = (v1 - v0) / width();
        f0 = static_cast<int>(std::floor(v0 + px * span));
        f1 = std::max(f0 + 1, static_cast<int>(std::floor(v0 + (px + 1) * span)));
        f0 = std::clamp(f0, 0, nframes);
        f1 = std::clamp(f1, 0, nframes);
    }
};

/// Reserve a plot area and handle zoom (wheel), pan (drag), reset (double
/// click) and selection (click). The view range is shared by every plot.
Plot plot_area(const char* id, float height, double& v0, double& v1, bool& follow,
               int nframes, int& sel) {
    const ImVec2 cur = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    ImGui::InvisibleButton(id, ImVec2(std::max(w, 50.0f), height));
    Plot p;
    p.p0 = ImVec2(cur.x + LEFT, cur.y + 2);
    p.p1 = ImVec2(cur.x + std::max(w, 50.0f) - 4, cur.y + height - 2);
    p.hovered = ImGui::IsItemHovered();
    const double total = std::max(nframes, 1);
    if (p.hovered) {
        ImGuiIO& io = ImGui::GetIO();
        p.v0 = v0; p.v1 = v1;
        if (io.MouseWheel != 0.0f) {
            const double at = p.frame_at(io.MousePos.x);
            const double scale = io.MouseWheel > 0 ? 0.8 : 1.25;
            double n0 = at - (at - v0) * scale, n1 = at + (v1 - at) * scale;
            if (n1 - n0 < 8) { const double c = (n0 + n1) / 2; n0 = c - 4; n1 = c + 4; }
            if (n1 - n0 > total) { n0 = 0; n1 = total; }
            if (n0 < 0) { n1 -= n0; n0 = 0; }
            if (n1 > total) { n0 -= n1 - total; n1 = total; n0 = std::max(n0, 0.0); }
            v0 = n0; v1 = n1;
            follow = (v0 <= 0.0 && v1 >= total);
        }
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            v0 = 0; v1 = total; follow = true;
        } else if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 3.0f)) {
            const double df = -io.MouseDelta.x / std::max(p.p1.x - p.p0.x, 1.0f) * (v1 - v0);
            double n0 = std::clamp(v0 + df, 0.0, std::max(total - (v1 - v0), 0.0));
            v1 = n0 + (v1 - v0); v0 = n0;
            follow = false;
        } else if (ImGui::IsItemDeactivated() && io.MouseDragMaxDistanceSqr[0] < 9.0f) {
            sel = std::clamp(static_cast<int>(p.frame_at(io.MousePos.x)), 0, std::max(nframes - 1, 0));
        }
    }
    p.v0 = v0; p.v1 = v1;
    auto* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p.p0, p.p1, rgba(24, 24, 30));
    return p;
}

void legend_text(float x, float y, ImU32 col, const char* t) {
    auto* dl = ImGui::GetWindowDrawList();
    const ImVec2 sz = ImGui::CalcTextSize(t);
    dl->AddRectFilled(ImVec2(x - 3, y - 1), ImVec2(x + sz.x + 3, y + sz.y + 1), rgba(24, 24, 30, 220));
    dl->AddText(ImVec2(x, y), col, t);
}

void plot_frame_marker(const Plot& p, int sel) {
    auto* dl = ImGui::GetWindowDrawList();
    const float x = p.x_of(sel + 0.5);
    if (x >= p.p0.x && x <= p.p1.x)
        dl->AddLine(ImVec2(x, p.p0.y), ImVec2(x, p.p1.y), rgba(255, 255, 255, 110));
}

void plot_title(const Plot& p, const char* t) {
    auto* dl = ImGui::GetWindowDrawList();
    const ImVec2 sz = ImGui::CalcTextSize(t);
    const ImVec2 at(p.p0.x + 6, p.p0.y + 2);
    dl->AddRectFilled(ImVec2(at.x - 3, at.y - 1), ImVec2(at.x + sz.x + 3, at.y + sz.y + 1), rgba(24, 24, 30, 220));
    dl->AddText(at, rgba(225, 225, 235), t);
}

double seconds_of(int frame, int sr) { return sr > 0 ? frame * 1152.0 / sr : 0.0; }

void time_axis(const Plot& p, int sr) {
    auto* dl = ImGui::GetWindowDrawList();
    const double s0 = seconds_of(static_cast<int>(p.v0), sr), s1 = seconds_of(static_cast<int>(p.v1), sr);
    const double span = std::max(s1 - s0, 1e-3);
    const double steps[] = {0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600};
    double step = steps[0];
    for (double st : steps) { step = st; if (span / st <= 10) break; }
    for (double t = std::ceil(s0 / step) * step; t <= s1; t += step) {
        const double f = t * sr / 1152.0;
        const float x = p.x_of(f);
        dl->AddLine(ImVec2(x, p.p1.y), ImVec2(x, p.p1.y + 4), rgba(160, 160, 170));
        char b[32];
        if (t >= 60) std::snprintf(b, sizeof b, "%d:%04.1f", static_cast<int>(t) / 60, std::fmod(t, 60.0));
        else std::snprintf(b, sizeof b, "%.1fs", t);
        dl->AddText(ImVec2(x + 2, p.p1.y + 1), rgba(160, 160, 170), b);
    }
}

} // namespace

// ===========================================================================
// Top level
// ===========================================================================

void AnalysisView::draw(Analysis* a) {
    sync(a);
    if (!a) {
        ImGui::TextDisabled(tr("解析するファイルを選んでください（エンコード済み、またはエンコード中のもの）。", "Pick a file to analyze (encoded, or being encoded)."));
        return;
    }
    if (fr_.empty()) {
        ImGui::TextDisabled(tr("まだフレームがありません。", "No frames yet."));
        return;
    }
    const double dur = seconds_of(static_cast<int>(fr_.size()), samplerate_);
    ImGui::Text("%s", title_.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled(tr("  %s   %d Hz   %zu フレーム   %.1f 秒%s", "  %s   %d Hz   %zu frames   %.1f s%s"), settings_.c_str(), samplerate_,
                        fr_.size(), dur, finished_ ? "" : tr("   （エンコード中）", "   (encoding)"));
    ImGui::TextDisabled(tr("ホイール: ズーム   ドラッグ: 移動   ダブルクリック: 全体表示   クリック: フレーム選択", "Wheel: zoom   Drag: pan   Double-click: show all   Click: select frame"));

    if (ImGui::BeginTabBar("analysis_tabs")) {
        if (ImGui::BeginTabItem(tr("概要###ov", "Overview###ov"))) { tab_overview(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem(tr("スケールファクタ###sf", "Scalefactors###sf"))) { tab_scalefactors(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem(tr("ハフマンテーブル###hf", "Huffman tables###hf"))) { tab_huffman(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem(tr("フレーム詳細###fd", "Frame detail###fd"))) { tab_frame(); ImGui::EndTabItem(); }
        if (ImGui::BeginTabItem(tr("統計###st", "Statistics###st"))) { tab_stats(); ImGui::EndTabItem(); }
        ImGui::EndTabBar();
    }
}

// ===========================================================================
// Overview
// ===========================================================================

void AnalysisView::tab_overview() {
    ImGui::BeginChild("ov", ImVec2(0, 0), 0, ImGuiWindowFlags_NoScrollWithMouse);
    const int n = static_cast<int>(fr_.size());
    auto* dl = ImGui::GetWindowDrawList();
    int hover_frame = -1;

    // ---- Bitrate ----
    {
        Plot p = plot_area("br", 170, v0_, v1_, follow_, n, sel_);
        const double ymax = 320.0;
        auto y_of = [&](double k) { return p.p1.y - static_cast<float>(k / ymax) * (p.p1.y - p.p0.y); };
        for (int k : {32, 64, 96, 128, 160, 192, 256, 320}) {
            dl->AddLine(ImVec2(p.p0.x, y_of(k)), ImVec2(p.p1.x, y_of(k)), rgba(60, 60, 70));
            char b[16]; std::snprintf(b, sizeof b, "%d", k);
            dl->AddText(ImVec2(p.p0.x - 26, y_of(k) - 7), rgba(140, 140, 150), b);
        }
        float prev_x = 0, prev_y = 0;
        for (int px = 0; px < p.width(); ++px) {
            int f0, f1; p.column(px, n, f0, f1);
            if (f0 >= f1) continue;
            int lo = 999, hi = 0; double sum = 0;
            for (int f = f0; f < f1; ++f) { lo = std::min<int>(lo, fr_[static_cast<std::size_t>(f)].kbps); hi = std::max<int>(hi, fr_[static_cast<std::size_t>(f)].kbps); sum += fr_[static_cast<std::size_t>(f)].kbps; }
            const float x = p.p0.x + px + 0.5f;
            dl->AddLine(ImVec2(x, y_of(lo)), ImVec2(x, y_of(hi) - 1), rgba(70, 130, 220, 150));
            const float y = y_of(sum / (f1 - f0));
            if (px > 0) dl->AddLine(ImVec2(prev_x, prev_y), ImVec2(x, y), rgba(120, 180, 255), 1.5f);
            prev_x = x; prev_y = y;
        }
        if (target_kbps_ > 0) {
            const float y = y_of(target_kbps_);
            for (float x = p.p0.x; x < p.p1.x; x += 10) dl->AddLine(ImVec2(x, y), ImVec2(std::min(x + 5, p.p1.x), y), rgba(255, 170, 60), 1.5f);
            legend_text(p.p1.x - 40, y - 18, rgba(255, 170, 60), tr("目標", "target"));
        }
        plot_frame_marker(p, sel_);
        plot_title(p, tr("ビットレート", "Bitrate"));
        if (p.hovered) hover_frame = static_cast<int>(p.frame_at(ImGui::GetIO().MousePos.x));
    }

    // ---- Frame usage ----
    {
        Plot p = plot_area("use", 110, v0_, v1_, follow_, n, sel_);
        auto y_of = [&](double r) { return p.p1.y - static_cast<float>(std::min(r, 1.2) / 1.2) * (p.p1.y - p.p0.y); };
        dl->AddLine(ImVec2(p.p0.x, y_of(1.0)), ImVec2(p.p1.x, y_of(1.0)), rgba(200, 200, 200, 90));
        dl->AddText(ImVec2(p.p0.x - 38, y_of(1.0) - 7), rgba(140, 140, 150), "100%");
        const int side = side_info_bits(fr_.empty() ? 2 : fr_[0].n_ch);
        for (int px = 0; px < p.width(); ++px) {
            int f0, f1; p.column(px, n, f0, f1);
            if (f0 >= f1) continue;
            double p2 = 0, p3 = 0, cap = 0;
            for (int f = f0; f < f1; ++f) {
                const auto& fr = fr_[static_cast<std::size_t>(f)];
                cap += frame_capacity_bits(fr, samplerate_);
                for (int gr = 0; gr < fr.n_gr; ++gr)
                    for (int ch = 0; ch < fr.n_ch; ++ch) {
                        const auto& g = fr.g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(ch)];
                        p2 += g.part2; p3 += g.part23 - g.part2;
                    }
                p3 += side;
            }
            if (cap <= 0) continue;
            const float x = p.p0.x + px;
            dl->AddRectFilled(ImVec2(x, y_of(p2 / cap)), ImVec2(x + 1, p.p1.y), rgba(170, 110, 220));
            dl->AddRectFilled(ImVec2(x, y_of((p2 + p3) / cap)), ImVec2(x + 1, y_of(p2 / cap)), rgba(80, 180, 120));
        }
        legend_text(p.p1.x - 250, p.p0.y + 2, rgba(190, 130, 240), tr("■スケールファクタ", "■scalefactors"));
        legend_text(p.p1.x - 120, p.p0.y + 2, rgba(80, 180, 120), tr("■ハフマン+他", "■Huffman+other"));
        plot_frame_marker(p, sel_);
        plot_title(p, tr("フレーム使用率", "Frame usage"));
        if (p.hovered) hover_frame = static_cast<int>(p.frame_at(ImGui::GetIO().MousePos.x));
    }

    // ---- Reservoir ----
    {
        Plot p = plot_area("resv", 80, v0_, v1_, follow_, n, sel_);
        float prev_x = 0, prev_y = 0;
        for (int px = 0; px < p.width(); ++px) {
            int f0, f1; p.column(px, n, f0, f1);
            if (f0 >= f1) continue;
            double fill = 0; bool ovf = false;
            for (int f = f0; f < f1; ++f) {
                const auto& fr = fr_[static_cast<std::size_t>(f)];
                fill += fr.resv_max > 0 ? static_cast<double>(fr.resv) / fr.resv_max : 0.0;
                ovf |= fr.overflow > 0;
            }
            fill /= (f1 - f0);
            const float x = p.p0.x + px + 0.5f;
            const float y = p.p1.y - static_cast<float>(std::clamp(fill, 0.0, 1.0)) * (p.p1.y - p.p0.y);
            if (px > 0) dl->AddLine(ImVec2(prev_x, prev_y), ImVec2(x, y), rgba(230, 200, 90), 1.5f);
            if (ovf) dl->AddLine(ImVec2(x, p.p0.y), ImVec2(x, p.p0.y + 6), rgba(255, 70, 70));
            prev_x = x; prev_y = y;
        }
        legend_text(p.p1.x - 190, p.p0.y + 8, rgba(255, 90, 90), tr("▮溢れ（詰め物で廃棄）", "▮overflow (stuffed away)"));
        plot_frame_marker(p, sel_);
        plot_title(p, tr("リザーバ", "Reservoir"));
        if (p.hovered) hover_frame = static_cast<int>(p.frame_at(ImGui::GetIO().MousePos.x));
    }

    // ---- Block types & stereo ----
    {
        const int nch = fr_[0].n_ch;
        Plot p = plot_area("blk", 20.0f * (nch + 1) + 8, v0_, v1_, follow_, n, sel_);
        const float row = (p.p1.y - p.p0.y) / (nch + 1);
        for (int px = 0; px < p.width(); ++px) {
            int f0, f1; p.column(px, n, f0, f1);
            if (f0 >= f1) continue;
            const float x = p.p0.x + px;
            for (int ch = 0; ch < nch; ++ch) {
                // Most notable block type in the column: SHORT > START/STOP > LONG.
                auto rank = [](int b) { return b == 2 ? 2 : b != 0 ? 1 : 0; };
                int worst = 0;
                for (int f = f0; f < f1; ++f)
                    for (int gr = 0; gr < fr_[static_cast<std::size_t>(f)].n_gr; ++gr) {
                        const int b = fr_[static_cast<std::size_t>(f)].g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(ch)].block;
                        if (rank(b) > rank(worst)) worst = b;
                    }
                dl->AddRectFilled(ImVec2(x, p.p0.y + row * ch + 1), ImVec2(x + 1, p.p0.y + row * (ch + 1) - 1), block_color(worst));
            }
            int ms = 0;
            for (int f = f0; f < f1; ++f) ms += fr_[static_cast<std::size_t>(f)].ms;
            const double frac = static_cast<double>(ms) / (f1 - f0);
            dl->AddRectFilled(ImVec2(x, p.p0.y + row * nch + 1), ImVec2(x + 1, p.p1.y - 1),
                              frac > 0.5 ? rgba(60, 120, 200) : rgba(90, 90, 90));
        }
        for (int ch = 0; ch < nch; ++ch) {
            char b[16]; std::snprintf(b, sizeof b, "ch%d", ch);
            dl->AddText(ImVec2(p.p0.x - 30, p.p0.y + row * ch + 2), rgba(140, 140, 150), b);
        }
        dl->AddText(ImVec2(p.p0.x - 30, p.p0.y + row * nch + 2), rgba(140, 140, 150), "M/S");
        plot_frame_marker(p, sel_);
        if (p.hovered) hover_frame = static_cast<int>(p.frame_at(ImGui::GetIO().MousePos.x));
        ImGui::TextDisabled("  ");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.27f, 0.35f, 0.43f, 1), "■LONG");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.24f, 1), "■START/STOP");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.92f, 0.31f, 0.27f, 1), "■SHORT");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.24f, 0.47f, 0.78f, 1), "■M/S");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.35f, 0.35f, 0.35f, 1), "■L/R");
    }

    // ---- Global gain ----
    {
        Plot p = plot_area("gg", 110, v0_, v1_, follow_, n, sel_);
        int gmin = 255, gmax = 0;
        for (const auto& fr : fr_)
            for (int gr = 0; gr < fr.n_gr; ++gr)
                for (int ch = 0; ch < fr.n_ch; ++ch) {
                    const int g = fr.g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(ch)].gg;
                    gmin = std::min(gmin, g); gmax = std::max(gmax, g);
                }
        if (gmax <= gmin) gmax = gmin + 1;
        auto y_of = [&](double g) { return p.p1.y - static_cast<float>((g - gmin) / (gmax - gmin)) * (p.p1.y - p.p0.y); };
        char lb[16];
        std::snprintf(lb, sizeof lb, "%d", gmax); dl->AddText(ImVec2(p.p0.x - 26, p.p0.y), rgba(140, 140, 150), lb);
        std::snprintf(lb, sizeof lb, "%d", gmin); dl->AddText(ImVec2(p.p0.x - 26, p.p1.y - 14), rgba(140, 140, 150), lb);
        const ImU32 cols[2] = {rgba(110, 190, 255), rgba(255, 150, 90)};
        for (int ch = 0; ch < fr_[0].n_ch; ++ch) {
            float prev_x = 0, prev_y = 0;
            for (int px = 0; px < p.width(); ++px) {
                int f0, f1; p.column(px, n, f0, f1);
                if (f0 >= f1) continue;
                double s = 0; int c = 0;
                for (int f = f0; f < f1; ++f)
                    for (int gr = 0; gr < fr_[static_cast<std::size_t>(f)].n_gr; ++gr) {
                        s += fr_[static_cast<std::size_t>(f)].g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(ch)].gg; ++c;
                    }
                const float x = p.p0.x + px + 0.5f, y = y_of(s / std::max(c, 1));
                if (px > 0) dl->AddLine(ImVec2(prev_x, prev_y), ImVec2(x, y), cols[ch], 1.2f);
                prev_x = x; prev_y = y;
            }
        }
        if (fr_[0].n_ch > 1) {
            legend_text(p.p1.x - 110, p.p0.y + 2, cols[0], "■ch0");
            legend_text(p.p1.x - 55, p.p0.y + 2, cols[1], "■ch1");
        }
        time_axis(p, samplerate_);
        plot_frame_marker(p, sel_);
        plot_title(p, tr("グローバルゲイン", "Global gain"));
        if (p.hovered) hover_frame = static_cast<int>(p.frame_at(ImGui::GetIO().MousePos.x));
        ImGui::Dummy(ImVec2(0, 16));
    }

    if (hover_frame >= 0 && hover_frame < n) {
        const auto& fr = fr_[static_cast<std::size_t>(hover_frame)];
        ImGui::BeginTooltip();
        ImGui::Text(tr("フレーム %d  (%.2f 秒)", "Frame %d  (%.2f s)"), hover_frame, seconds_of(hover_frame, samplerate_));
        ImGui::Text(tr("ビットレート %d kbps   %s", "Bitrate %d kbps   %s"), fr.kbps, fr.ms ? "M/S" : "L/R");
        ImGui::Text(tr("リザーバ %d / %d bit%s", "Reservoir %d / %d bit%s"), fr.resv, fr.resv_max, fr.overflow ? tr("  （溢れ）", "  (overflow)") : "");
        if (fr.lpf_hz > 0) ImGui::Text(tr("ローパス %.0f Hz", "Lowpass %.0f Hz"), fr.lpf_hz);
        for (int gr = 0; gr < fr.n_gr; ++gr)
            for (int ch = 0; ch < fr.n_ch; ++ch) {
                const auto& g = fr.g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(ch)];
                ImGui::Text("gr%d ch%d  %-5s gain %d  %d bit (sf %d)", gr, ch, block_name(g.block), g.gg, g.part23, g.part2);
            }
        ImGui::EndTooltip();
    }
    ImGui::EndChild();
}

// ===========================================================================
// Scalefactors
// ===========================================================================

void AnalysisView::tab_scalefactors() {
    const int n = static_cast<int>(fr_.size());
    const int nch = fr_[0].n_ch;
    ImGui::SetNextItemWidth(160);
    if (nch > 1) {
        ImGui::RadioButton("ch0 (L / Mid)", &channel_, 0); ImGui::SameLine();
        ImGui::RadioButton("ch1 (R / Side)", &channel_, 1); ImGui::SameLine();
        ImGui::TextDisabled("|"); ImGui::SameLine();
    } else channel_ = 0;
    ImGui::RadioButton(tr("スケールファクタ値", "Scalefactor value"), &sf_mode_, 0); ImGui::SameLine();
    ImGui::RadioButton(tr("実効量子化ステップ (dB)", "Effective quantizer step (dB)"), &sf_mode_, 1);
    ImGui::TextDisabled(sf_mode_ == 0
        ? tr("帯域ごとの sf（preflag の pretab 込み）。大きいほど、その帯域だけグローバルより細かく量子化。", "Per-band sf (including pretab when preflag is set). Larger = that band is quantized finer than global_gain alone.")
        : tr("global_gain とスケールファクタ（short は subblock_gain も）から求めた実際の量子化の細かさ。明るいほど細かい（ビットを使っている）。", "Actual quantizer resolution from global_gain and scalefactors (plus subblock_gain for short blocks). Brighter = finer (more bits spent)."));

    ImGui::BeginChild("sf", ImVec2(0, 0), 0, ImGuiWindowFlags_NoScrollWithMouse);
    auto* dl = ImGui::GetWindowDrawList();
    const float h = std::max(220.0f, ImGui::GetContentRegionAvail().y - 90.0f);
    Plot p = plot_area("sfmap", h, v0_, v1_, follow_, n, sel_);
    const int W = p.width();
    constexpr int ROWS = 21;
    const auto& le = layout_.long_edges;
    const auto& se = layout_.short_edges;

    auto band_value = [&](const FrameRec& fr, int gr, int ch, int b, bool& is_short) -> double {
        const GranRec& g = fr.g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(ch)];
        is_short = g.block == 2;
        double sfv, sbg = 0;
        if (!is_short) {
            sfv = g.sf_l[static_cast<std::size_t>(b)] + (g.preflag ? PRETAB[static_cast<std::size_t>(b)] : 0);
        } else {
            const int center = (le[static_cast<std::size_t>(b)] + le[static_cast<std::size_t>(b + 1)]) / 2;
            const int sl = std::min(center / 3, 191);
            int sb = 0;
            while (sb < 12 && sl >= se[static_cast<std::size_t>(sb + 1)]) ++sb;
            double acc = 0;
            for (int w = 0; w < 3; ++w) {
                acc += sb < 12 ? g.sf_s[static_cast<std::size_t>(w)][static_cast<std::size_t>(sb)] : 0;
                sbg += g.sbg[static_cast<std::size_t>(w)];
            }
            sfv = acc / 3.0;
            sbg /= 3.0;
        }
        if (sf_mode_ == 0) return sfv;
        // Step in dB relative to gain 210: larger = coarser. Negate so larger
        // means finer, matching the brightness.
        const double step_db = 1.505 * (g.gg - 210) - 3.0103 * (1 + g.scale) * sfv - 12.04 * sbg;
        return -step_db;
    };

    // Value range over the visible frames.
    double vmin = 1e9, vmax = -1e9;
    {
        const int a = std::clamp(static_cast<int>(v0_), 0, n), b = std::clamp(static_cast<int>(std::ceil(v1_)), 0, n);
        const int stride = std::max(1, (b - a) / 4000);
        for (int f = a; f < b; f += stride)
            for (int gr = 0; gr < fr_[static_cast<std::size_t>(f)].n_gr; ++gr)
                for (int bb = 0; bb < ROWS; ++bb) {
                    bool sh;
                    const double v = band_value(fr_[static_cast<std::size_t>(f)], gr, channel_, bb, sh);
                    vmin = std::min(vmin, v); vmax = std::max(vmax, v);
                }
        if (sf_mode_ == 0) { vmin = 0; vmax = std::max(vmax, 1.0); }
        if (vmax <= vmin) vmax = vmin + 1;
    }

    char key[160];
    std::snprintf(key, sizeof key, "%d|%.3f|%.3f|%d|%d|%d|%.2f|%.2f", n, v0_, v1_, W, channel_, sf_mode_, vmin, vmax);
    if (sf_tex_->key != key) {
        std::vector<std::uint8_t> px(static_cast<std::size_t>(W) * ROWS * 4, 0);
        for (int x = 0; x < W; ++x) {
            int f0, f1; p.column(x, n, f0, f1);
            if (f0 >= f1) continue;
            for (int b = 0; b < ROWS; ++b) {
                double s = 0; int c = 0;
                for (int f = f0; f < f1; ++f)
                    for (int gr = 0; gr < fr_[static_cast<std::size_t>(f)].n_gr; ++gr) {
                        bool sh; s += band_value(fr_[static_cast<std::size_t>(f)], gr, channel_, b, sh); ++c;
                    }
                const std::size_t o = (static_cast<std::size_t>(ROWS - 1 - b) * W + x) * 4;
                colormap((s / c - vmin) / (vmax - vmin), &px[o]);
            }
        }
        sf_tex_->upload(W, ROWS, px);
        sf_tex_->key = key;
    }
    dl->AddImage(static_cast<ImTextureID>(sf_tex_->id), p.p0, p.p1);

    // Band labels.
    for (int b = 0; b <= ROWS; b += 3) {
        const double hz = le[static_cast<std::size_t>(b)] * samplerate_ / 1152.0;
        const float y = p.p1.y - (p.p1.y - p.p0.y) * b / ROWS;
        dl->AddLine(ImVec2(p.p0.x - 4, y), ImVec2(p.p0.x, y), rgba(150, 150, 160));
        dl->AddText(ImVec2(p.p0.x - 50, std::max(y - 14, p.p0.y)), rgba(150, 150, 160), hz_label(hz).c_str());
    }
    // Short-block ticks.
    for (int x = 0; x < W; ++x) {
        int f0, f1; p.column(x, n, f0, f1);
        bool sh = false;
        for (int f = f0; f < f1 && !sh; ++f)
            for (int gr = 0; gr < fr_[static_cast<std::size_t>(f)].n_gr; ++gr)
                sh |= fr_[static_cast<std::size_t>(f)].g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(channel_)].block == 2;
        if (sh) dl->AddLine(ImVec2(p.p0.x + x, p.p0.y - 4), ImVec2(p.p0.x + x, p.p0.y), rgba(235, 80, 70));
    }
    time_axis(p, samplerate_);
    plot_frame_marker(p, sel_);

    if (p.hovered) {
        const ImVec2 m = ImGui::GetIO().MousePos;
        const int f = std::clamp(static_cast<int>(p.frame_at(m.x)), 0, n - 1);
        const int b = std::clamp(static_cast<int>((p.p1.y - m.y) / (p.p1.y - p.p0.y) * ROWS), 0, ROWS - 1);
        const auto& fr = fr_[static_cast<std::size_t>(f)];
        ImGui::BeginTooltip();
        ImGui::Text(tr("フレーム %d  sfb %d  (%.0f - %.0f Hz)", "Frame %d  sfb %d  (%.0f - %.0f Hz)"), f, b,
                    le[static_cast<std::size_t>(b)] * samplerate_ / 1152.0, le[static_cast<std::size_t>(b + 1)] * samplerate_ / 1152.0);
        for (int gr = 0; gr < fr.n_gr; ++gr) {
            const auto& g = fr.g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(channel_)];
            bool sh; const double v = band_value(fr, gr, channel_, b, sh);
            if (g.block == 2)
                ImGui::Text(tr("gr%d SHORT  sf(窓平均) → %s %.1f   gain %d  sbg %d/%d/%d", "gr%d SHORT  sf(window mean) → %s %.1f   gain %d  sbg %d/%d/%d"), gr, sf_mode_ ? tr("細かさ", "fineness") : tr("値", "value"), v, g.gg, g.sbg[0], g.sbg[1], g.sbg[2]);
            else
                ImGui::Text("gr%d %-5s sf %d%s → %s %.1f   gain %d  scale %d", gr, block_name(g.block), g.sf_l[static_cast<std::size_t>(b)],
                            g.preflag ? "+pretab" : "", sf_mode_ ? tr("細かさ", "fineness") : tr("値", "value"), v, g.gg, g.scale);
        }
        ImGui::EndTooltip();
    }

    // Colour bar.
    ImGui::Dummy(ImVec2(0, 20));
    const ImVec2 c0 = ImGui::GetCursorScreenPos();
    const float cw = 260.0f;
    for (int i = 0; i < 64; ++i) {
        std::uint8_t c[4]; colormap(i / 63.0, c);
        dl->AddRectFilled(ImVec2(c0.x + LEFT + cw * i / 64, c0.y), ImVec2(c0.x + LEFT + cw * (i + 1) / 64 + 1, c0.y + 14), IM_COL32(c[0], c[1], c[2], 255));
    }
    char lo[32], hi[32];
    if (sf_mode_ == 0) { std::snprintf(lo, sizeof lo, "%.0f", vmin); std::snprintf(hi, sizeof hi, "%.0f", vmax); }
    else { std::snprintf(lo, sizeof lo, tr("粗い", "coarse")); std::snprintf(hi, sizeof hi, tr("細かい (幅 %.0f dB)", "fine (range %.0f dB)"), vmax - vmin); }
    dl->AddText(ImVec2(c0.x + 4, c0.y), rgba(170, 170, 180), lo);
    dl->AddText(ImVec2(c0.x + LEFT + cw + 8, c0.y), rgba(170, 170, 180), hi);
    ImGui::Dummy(ImVec2(0, 18));
    ImGui::TextDisabled(tr("上端の赤い印 = SHORT ブロック", "Red ticks on top = SHORT blocks"));
    ImGui::EndChild();
}

// ===========================================================================
// Huffman
// ===========================================================================

namespace {
int category_at(const GranRec& g, int L, const BandLayout& lay, const std::vector<int>& short_tx) {
    const int bv = 2 * g.big_values;
    const int c1 = bv + 4 * g.count1;
    if (g.block == 2) {
        const int i = short_tx[static_cast<std::size_t>(L)];
        if (i < std::min(36, bv)) return g.tbl[0];
        if (i < bv) return g.tbl[1];
        if (i < c1) return g.c1tab ? CAT_C1B : CAT_C1A;
        return CAT_ZERO;
    }
    const auto& le = lay.long_edges;
    const int a1 = std::min(le[static_cast<std::size_t>(std::min(g.r0 + 1, 22))], bv);
    const int a2 = std::min(le[static_cast<std::size_t>(std::min(g.r0 + g.r1 + 2, 22))], bv);
    if (L < a1) return g.tbl[0];
    if (L < a2) return g.tbl[1];
    if (L < bv) return g.tbl[2];
    if (L < c1) return g.c1tab ? CAT_C1B : CAT_C1A;
    return CAT_ZERO;
}
} // namespace

void AnalysisView::tab_huffman() {
    const int n = static_cast<int>(fr_.size());
    const int nch = fr_[0].n_ch;
    if (nch > 1) {
        ImGui::RadioButton("ch0 (L / Mid)", &channel_, 0); ImGui::SameLine();
        ImGui::RadioButton("ch1 (R / Side)", &channel_, 1); ImGui::SameLine();
    } else channel_ = 0;
    ImGui::TextDisabled(tr("  縦軸 = 周波数。各スペクトル線がどのテーブル／領域で符号化されたか（SHORT は中央窓の位置で表示）。", "  Vertical = frequency. Which table / region codes each spectral line (SHORT shown at the middle window's position)."));

    ImGui::BeginChild("hf", ImVec2(0, 0), 0, ImGuiWindowFlags_NoScrollWithMouse);
    auto* dl = ImGui::GetWindowDrawList();
    const float h = std::max(220.0f, ImGui::GetContentRegionAvail().y - 120.0f);
    Plot p = plot_area("hfmap", h, v0_, v1_, follow_, n, sel_);
    const int W = p.width();
    constexpr int ROWS = 144;   // 4 lines per row

    char key[128];
    std::snprintf(key, sizeof key, "%d|%.3f|%.3f|%d|%d", n, v0_, v1_, W, channel_);
    if (huff_tex_->key != key) {
        std::vector<std::uint8_t> px(static_cast<std::size_t>(W) * ROWS * 4, 0);
        for (int x = 0; x < W; ++x) {
            int f0, f1; p.column(x, n, f0, f1);
            if (f0 >= f1) continue;
            const int f = (f0 + f1 - 1) / 2;   // representative frame
            const auto& fr = fr_[static_cast<std::size_t>(f)];
            const int gr = (x & 1) ? std::min(1, fr.n_gr - 1) : 0;
            const auto& g = fr.g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(channel_)];
            for (int r = 0; r < ROWS; ++r) {
                const auto c = table_color(category_at(g, r * 4 + 2, layout_, short_tx_of_freq_));
                const std::size_t o = (static_cast<std::size_t>(ROWS - 1 - r) * W + x) * 4;
                std::copy(c.begin(), c.end(), px.begin() + static_cast<std::ptrdiff_t>(o));
            }
        }
        huff_tex_->upload(W, ROWS, px);
        huff_tex_->key = key;
    }
    dl->AddImage(static_cast<ImTextureID>(huff_tex_->id), p.p0, p.p1);
    const double nyq = samplerate_ / 2.0;
    for (int k = 0; k <= 4; ++k) {
        const float y = p.p1.y - (p.p1.y - p.p0.y) * k / 4;
        dl->AddText(ImVec2(p.p0.x - 50, std::max(y - 14, p.p0.y)), rgba(150, 150, 160), hz_label(nyq * k / 4).c_str());
    }
    time_axis(p, samplerate_);
    plot_frame_marker(p, sel_);

    if (p.hovered) {
        const ImVec2 m = ImGui::GetIO().MousePos;
        const int f = std::clamp(static_cast<int>(p.frame_at(m.x)), 0, n - 1);
        const int L = std::clamp(static_cast<int>((p.p1.y - m.y) / (p.p1.y - p.p0.y) * 576), 0, 575);
        const auto& fr = fr_[static_cast<std::size_t>(f)];
        ImGui::BeginTooltip();
        ImGui::Text(tr("フレーム %d   %.0f Hz (線 %d)", "Frame %d   %.0f Hz (line %d)"), f, L * samplerate_ / 1152.0, L);
        for (int gr = 0; gr < fr.n_gr; ++gr) {
            const auto& g = fr.g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(channel_)];
            char b[32];
            ImGui::Text(tr("gr%d %-5s → %s   big_values %d  count1 %d  テーブル %d/%d/%d", "gr%d %-5s → %s   big_values %d  count1 %d  tables %d/%d/%d"), gr, block_name(g.block),
                        category_name(category_at(g, L, layout_, short_tx_of_freq_), b, sizeof b),
                        g.big_values, g.count1, g.tbl[0], g.tbl[1], g.tbl[2]);
        }
        ImGui::EndTooltip();
    }

    // Legend with usage share.
    ImGui::Dummy(ImVec2(0, 20));
    const auto& use = huff_lines_[static_cast<std::size_t>(channel_)];
    double total = 0;
    for (double v : use) total += v;
    ImGui::TextUnformatted(tr("使用率（スペクトル線数の割合、ファイル全体）:", "Usage (share of spectral lines, whole file):"));
    int shown = 0;
    for (int cat = 0; cat < 35; ++cat) {
        if (use[static_cast<std::size_t>(cat)] <= 0) continue;
        const auto c = table_color(cat);
        char b[32];
        if (shown++ % 6) ImGui::SameLine();
        ImGui::TextColored(ImVec4(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, 1), "■");
        ImGui::SameLine(0, 2);
        ImGui::Text("%s %.1f%%", category_name(cat, b, sizeof b), 100.0 * use[static_cast<std::size_t>(cat)] / std::max(total, 1.0));
    }
    ImGui::EndChild();
}

// ===========================================================================
// Frame detail
// ===========================================================================

void AnalysisView::tab_frame() {
    const int n = static_cast<int>(fr_.size());
    if (ImGui::ArrowButton("prev", ImGuiDir_Left)) sel_ = std::max(0, sel_ - 1);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(360);
    ImGui::SliderInt("##frame", &sel_, 0, n - 1, tr("フレーム %d", "Frame %d"));
    ImGui::SameLine();
    if (ImGui::ArrowButton("next", ImGuiDir_Right)) sel_ = std::min(n - 1, sel_ + 1);
    const auto& fr = fr_[static_cast<std::size_t>(sel_)];
    ImGui::SameLine();
    ImGui::Text(tr("  %.2f 秒   %d kbps   %s   リザーバ %d/%d bit%s", "  %.2f s   %d kbps   %s   reservoir %d/%d bit%s"), seconds_of(sel_, samplerate_), fr.kbps,
                fr.ms ? "M/S" : "L/R", fr.resv, fr.resv_max, fr.overflow ? tr("  溢れ", "  overflow") : "");

    const auto& le = layout_.long_edges;
    const auto& se = layout_.short_edges;
    const double cap = frame_capacity_bits(fr, samplerate_) / std::max(1, fr.n_gr * fr.n_ch);

    ImGui::BeginChild("frame_detail", ImVec2(0, 0));
    for (int gr = 0; gr < fr.n_gr; ++gr) {
        for (int ch = 0; ch < fr.n_ch; ++ch) {
            const GranRec& g = fr.g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(ch)];
            ImGui::PushID(gr * 4 + ch);
            ImGui::SeparatorText((std::string("gr") + std::to_string(gr) + "  ch" + std::to_string(ch) +
                                  (fr.ms ? (ch ? " (Side)" : " (Mid)") : (ch ? " (R)" : " (L)"))).c_str());
            ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.95f, 1), "%s", block_name(g.block));
            ImGui::SameLine();
            ImGui::Text("  global_gain %d   scalefac_scale %d   preflag %d   compress %d (slen %d/%d)",
                        g.gg, g.scale, g.preflag, g.compress, SLEN1[g.compress & 15], SLEN2[g.compress & 15]);
            if (g.block == 2) {
                ImGui::SameLine();
                ImGui::Text("   subblock_gain %d/%d/%d", g.sbg[0], g.sbg[1], g.sbg[2]);
            }

            auto* dl = ImGui::GetWindowDrawList();
            const float w = ImGui::GetContentRegionAvail().x;

            // ---- Bits bar ----
            {
                const ImVec2 c = ImGui::GetCursorScreenPos();
                ImGui::Dummy(ImVec2(w, 20));
                const float bw = w - LEFT - 8;
                const double scale = bw / std::max(cap * 1.4, static_cast<double>(g.part23) * 1.05);
                dl->AddText(ImVec2(c.x, c.y + 2), rgba(150, 150, 160), tr("ビット", "Bits"));
                dl->AddRectFilled(ImVec2(c.x + LEFT, c.y + 3), ImVec2(c.x + LEFT + static_cast<float>(g.part2 * scale), c.y + 17), rgba(170, 110, 220));
                dl->AddRectFilled(ImVec2(c.x + LEFT + static_cast<float>(g.part2 * scale), c.y + 3),
                                  ImVec2(c.x + LEFT + static_cast<float>(g.part23 * scale), c.y + 17), rgba(80, 180, 120));
                const float cx = c.x + LEFT + static_cast<float>(cap * scale);
                dl->AddLine(ImVec2(cx, c.y), ImVec2(cx, c.y + 20), rgba(255, 170, 60), 2);
                char b[96];
                std::snprintf(b, sizeof b, tr("sf %d + ハフマン %d = %d bit  （平均割当 %.0f）", "sf %d + Huffman %d = %d bit  (average budget %.0f)"), g.part2, g.part23 - g.part2, g.part23, cap);
                dl->AddText(ImVec2(c.x + LEFT + 6, c.y + 3), rgba(255, 255, 255), b);
            }

            // ---- Region lane ----
            {
                const ImVec2 c = ImGui::GetCursorScreenPos();
                ImGui::Dummy(ImVec2(w, 34));
                const float bw = w - LEFT - 8;
                dl->AddText(ImVec2(c.x, c.y + 8), rgba(150, 150, 160), g.block == 2 ? tr("領域(送信順)", "Regions (tx order)") : tr("領域", "Regions"));
                const int bv = 2 * g.big_values;
                const int c1 = std::min(bv + 4 * g.count1, 576);
                struct Seg { int a, b, cat; };
                std::vector<Seg> segs;
                if (g.block == 2) {
                    const int a1 = std::min(36, bv);
                    segs = {{0, a1, g.tbl[0]}, {a1, bv, g.tbl[1]}, {bv, c1, g.c1tab ? CAT_C1B : CAT_C1A}, {c1, 576, CAT_ZERO}};
                } else {
                    const int a1 = std::min(le[static_cast<std::size_t>(std::min(g.r0 + 1, 22))], bv);
                    const int a2 = std::min(le[static_cast<std::size_t>(std::min(g.r0 + g.r1 + 2, 22))], bv);
                    segs = {{0, a1, g.tbl[0]}, {a1, a2, g.tbl[1]}, {a2, bv, g.tbl[2]}, {bv, c1, g.c1tab ? CAT_C1B : CAT_C1A}, {c1, 576, CAT_ZERO}};
                }
                for (const auto& s : segs) {
                    if (s.b <= s.a) continue;
                    const float x0 = c.x + LEFT + bw * s.a / 576.0f, x1 = c.x + LEFT + bw * s.b / 576.0f;
                    const auto col = table_color(s.cat);
                    dl->AddRectFilled(ImVec2(x0, c.y + 4), ImVec2(x1, c.y + 26), IM_COL32(col[0], col[1], col[2], 255));
                    dl->AddRect(ImVec2(x0, c.y + 4), ImVec2(x1, c.y + 26), rgba(0, 0, 0, 160));
                    char b[32];
                    const char* label = s.cat == CAT_ZERO ? "0" : s.cat == CAT_C1A ? "Q-A" : s.cat == CAT_C1B ? "Q-B" : (std::snprintf(b, sizeof b, "T%d", s.cat), b);
                    const ImVec2 ts = ImGui::CalcTextSize(label);
                    if (x1 - x0 > ts.x + 4) {
                        const bool light = col[0] + col[1] + col[2] > 380;
                        dl->AddText(ImVec2((x0 + x1 - ts.x) / 2, c.y + 6), light ? rgba(0, 0, 0) : rgba(255, 255, 255), label);
                    }
                }
                if (g.block != 2) {
                    for (int k : {0, 4000, 8000, 12000, 16000, 20000}) {
                        const double line = k * 1152.0 / samplerate_;
                        if (line > 576) break;
                        const float x = c.x + LEFT + bw * static_cast<float>(line / 576.0);
                        dl->AddText(ImVec2(x, c.y + 26), rgba(120, 120, 130), hz_label(k).c_str());
                    }
                }
                ImGui::Dummy(ImVec2(0, 6));
            }

            // ---- Scalefactor bars ----
            {
                const ImVec2 c = ImGui::GetCursorScreenPos();
                const float hgt = 90.0f;
                ImGui::Dummy(ImVec2(w, hgt + 18));
                const float bw = w - LEFT - 8;
                dl->AddText(ImVec2(c.x, c.y + hgt / 2 - 8), rgba(150, 150, 160), "sf");
                dl->AddRectFilled(ImVec2(c.x + LEFT, c.y), ImVec2(c.x + LEFT + bw, c.y + hgt), rgba(24, 24, 30));
                constexpr int SFMAX = 15;
                auto yh = [&](double v) { return static_cast<float>(std::min(v, static_cast<double>(SFMAX)) / SFMAX) * hgt; };
                if (g.block != 2) {
                    const float cwid = bw / 21.0f;
                    for (int b = 0; b < 21; ++b) {
                        const float x0 = c.x + LEFT + cwid * b + 1, x1 = x0 + cwid - 2;
                        const int sf = g.sf_l[static_cast<std::size_t>(b)];
                        const int pre = g.preflag ? PRETAB[static_cast<std::size_t>(b)] : 0;
                        dl->AddRectFilled(ImVec2(x0, c.y + hgt - yh(sf)), ImVec2(x1, c.y + hgt), rgba(90, 160, 240));
                        if (pre) dl->AddRectFilled(ImVec2(x0, c.y + hgt - yh(sf + pre)), ImVec2(x1, c.y + hgt - yh(sf)), rgba(240, 160, 90));
                        char v[8]; std::snprintf(v, sizeof v, "%d", sf + pre);
                        if (sf + pre) dl->AddText(ImVec2(x0 + 2, c.y + hgt - yh(sf + pre) - 14), rgba(220, 220, 230), v);
                        if (b % 3 == 0)
                            dl->AddText(ImVec2(x0, c.y + hgt + 1), rgba(120, 120, 130),
                                        hz_label(le[static_cast<std::size_t>(b)] * samplerate_ / 1152.0).c_str());
                    }
                    if (g.preflag) dl->AddText(ImVec2(c.x + LEFT + bw - 120, c.y + 2), rgba(240, 160, 90), "■ pretab (preflag)");
                } else {
                    const float cwid = bw / 12.0f;
                    const ImU32 wc[3] = {rgba(90, 160, 240), rgba(120, 200, 150), rgba(230, 190, 90)};
                    for (int b = 0; b < 12; ++b)
                        for (int wdw = 0; wdw < 3; ++wdw) {
                            const float x0 = c.x + LEFT + cwid * b + 1 + wdw * (cwid - 2) / 3, x1 = x0 + (cwid - 2) / 3 - 1;
                            const int sf = g.sf_s[static_cast<std::size_t>(wdw)][static_cast<std::size_t>(b)];
                            dl->AddRectFilled(ImVec2(x0, c.y + hgt - yh(sf)), ImVec2(x1, c.y + hgt), wc[wdw]);
                            if (wdw == 0 && b % 2 == 0)
                                dl->AddText(ImVec2(x0, c.y + hgt + 1), rgba(120, 120, 130),
                                            hz_label(se[static_cast<std::size_t>(b)] * samplerate_ / 384.0).c_str());
                        }
                    dl->AddText(ImVec2(c.x + LEFT + bw - 150, c.y + 2), rgba(200, 200, 210), tr("窓 ■1 ■2 ■3", "window ■1 ■2 ■3"));
                }
            }
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
}

// ===========================================================================
// Statistics
// ===========================================================================

void AnalysisView::tab_stats() {
    const int n = static_cast<int>(fr_.size());
    ImGui::BeginChild("stats", ImVec2(0, 0));
    auto* dl = ImGui::GetWindowDrawList();

    // ---- Bitrate distribution ----
    {
        constexpr int RATES[] = {32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320};
        std::array<int, 14> cnt{};
        double kb = 0;
        for (const auto& f : fr_) {
            kb += f.kbps;
            for (int i = 0; i < 14; ++i) if (RATES[i] == f.kbps) ++cnt[static_cast<std::size_t>(i)];
        }
        ImGui::SeparatorText(tr("ビットレートの選択分布", "Bitrate selection"));
        ImGui::Text(tr("平均 %.1f kbps（フレームのビットレート指定の平均）", "Mean %.1f kbps (average of per-frame bitrate indices)"), kb / std::max(n, 1));
        const ImVec2 c = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x, h = 130;
        ImGui::Dummy(ImVec2(w, h + 20));
        int mx = 1;
        for (int v : cnt) mx = std::max(mx, v);
        const float bw = (w - LEFT) / 14.0f;
        for (int i = 0; i < 14; ++i) {
            const float x0 = c.x + LEFT + bw * i + 3, x1 = x0 + bw - 6;
            const float bh = h * cnt[static_cast<std::size_t>(i)] / mx;
            dl->AddRectFilled(ImVec2(x0, c.y + h - bh), ImVec2(x1, c.y + h), rgba(90, 150, 230));
            char b[32];
            std::snprintf(b, sizeof b, "%d", RATES[i]);
            dl->AddText(ImVec2(x0, c.y + h + 2), rgba(160, 160, 170), b);
            if (cnt[static_cast<std::size_t>(i)]) {
                std::snprintf(b, sizeof b, "%.0f%%", 100.0 * cnt[static_cast<std::size_t>(i)] / n);
                dl->AddText(ImVec2(x0, c.y + h - bh - 15), rgba(220, 220, 230), b);
            }
        }
    }

    // ---- Blocks, stereo ----
    {
        ImGui::SeparatorText(tr("ブロック種別とステレオ", "Block types and stereo"));
        int ms = 0;
        std::array<std::array<long, 4>, 2> blk{};
        for (const auto& f : fr_) {
            ms += f.ms;
            for (int gr = 0; gr < f.n_gr; ++gr)
                for (int ch = 0; ch < f.n_ch; ++ch)
                    ++blk[static_cast<std::size_t>(ch)][f.g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(ch)].block & 3];
        }
        ImGui::Text(tr("M/S フレーム: %.1f%%", "M/S frames: %.1f%%"), 100.0 * ms / std::max(n, 1));
        for (int ch = 0; ch < fr_[0].n_ch; ++ch) {
            long tot = 0;
            for (long v : blk[static_cast<std::size_t>(ch)]) tot += v;
            ImGui::Text("ch%d:  LONG %.1f%%   START %.1f%%   SHORT %.1f%%   STOP %.1f%%", ch,
                        100.0 * blk[static_cast<std::size_t>(ch)][0] / std::max(tot, 1L), 100.0 * blk[static_cast<std::size_t>(ch)][1] / std::max(tot, 1L),
                        100.0 * blk[static_cast<std::size_t>(ch)][2] / std::max(tot, 1L), 100.0 * blk[static_cast<std::size_t>(ch)][3] / std::max(tot, 1L));
        }
    }

    // ---- Mean scalefactor per band ----
    {
        ImGui::SeparatorText(tr("帯域ごとの平均スケールファクタ（LONG ブロック、pretab 込み）", "Mean scalefactor per band (LONG blocks, including pretab)"));
        const ImVec2 c = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x, h = 110;
        ImGui::Dummy(ImVec2(w, h + 20));
        const auto& le = layout_.long_edges;
        const ImU32 cols[2] = {rgba(110, 190, 255), rgba(255, 150, 90)};
        const float bw = (w - LEFT) / 21.0f;
        double mx = 1;
        std::array<std::array<double, 21>, 2> mean{};
        for (int ch = 0; ch < fr_[0].n_ch; ++ch) {
            std::array<long, 21> cnt{};
            for (const auto& f : fr_)
                for (int gr = 0; gr < f.n_gr; ++gr) {
                    const auto& g = f.g[static_cast<std::size_t>(gr)][static_cast<std::size_t>(ch)];
                    if (g.block == 2) continue;
                    for (int b = 0; b < 21; ++b) {
                        mean[static_cast<std::size_t>(ch)][static_cast<std::size_t>(b)] += g.sf_l[static_cast<std::size_t>(b)] + (g.preflag ? PRETAB[static_cast<std::size_t>(b)] : 0);
                        ++cnt[static_cast<std::size_t>(b)];
                    }
                }
            for (int b = 0; b < 21; ++b) {
                mean[static_cast<std::size_t>(ch)][static_cast<std::size_t>(b)] /= std::max(cnt[static_cast<std::size_t>(b)], 1L);
                mx = std::max(mx, mean[static_cast<std::size_t>(ch)][static_cast<std::size_t>(b)]);
            }
        }
        for (int b = 0; b < 21; ++b) {
            for (int ch = 0; ch < fr_[0].n_ch; ++ch) {
                const float x0 = c.x + LEFT + bw * b + 2 + ch * (bw - 4) / 2, x1 = x0 + (bw - 4) / 2 - 1;
                const float bh = static_cast<float>(h * mean[static_cast<std::size_t>(ch)][static_cast<std::size_t>(b)] / mx);
                dl->AddRectFilled(ImVec2(x0, c.y + h - bh), ImVec2(x1, c.y + h), cols[ch]);
            }
            if (b % 3 == 0)
                dl->AddText(ImVec2(c.x + LEFT + bw * b, c.y + h + 2), rgba(160, 160, 170),
                            hz_label(le[static_cast<std::size_t>(b)] * samplerate_ / 1152.0).c_str());
        }
        char b[32]; std::snprintf(b, sizeof b, tr("最大 %.2f", "max %.2f"), mx);
        dl->AddText(ImVec2(c.x, c.y), rgba(160, 160, 170), b);
        ImGui::TextColored(ImVec4(0.43f, 0.75f, 1, 1), "■ch0");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1, 0.59f, 0.35f, 1), "■ch1");
    }

    // ---- Huffman usage, both channels ----
    {
        ImGui::SeparatorText(tr("ハフマンテーブル使用率（全チャンネル、スペクトル線の割合）", "Huffman table usage (all channels, share of spectral lines)"));
        std::array<double, 35> use{};
        double total = 0;
        for (int ch = 0; ch < 2; ++ch)
            for (int k = 0; k < 35; ++k) { use[static_cast<std::size_t>(k)] += huff_lines_[static_cast<std::size_t>(ch)][static_cast<std::size_t>(k)]; total += huff_lines_[static_cast<std::size_t>(ch)][static_cast<std::size_t>(k)]; }
        const ImVec2 c = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x;
        float x = c.x;
        const float bh = 26;
        ImGui::Dummy(ImVec2(w, bh + 4));
        for (int k = 0; k < 35; ++k) {
            const double share = use[static_cast<std::size_t>(k)] / std::max(total, 1.0);
            if (share <= 0) continue;
            const float seg = static_cast<float>(share * w);
            const auto col = table_color(k);
            dl->AddRectFilled(ImVec2(x, c.y), ImVec2(x + seg, c.y + bh), IM_COL32(col[0], col[1], col[2], 255));
            char b[32];
            const char* label = k == CAT_ZERO ? "0" : k == CAT_C1A ? "Q-A" : k == CAT_C1B ? "Q-B" : (std::snprintf(b, sizeof b, "T%d", k), b);
            if (seg > ImGui::CalcTextSize(label).x + 4) {
                const bool light = col[0] + col[1] + col[2] > 380;
                dl->AddText(ImVec2(x + 2, c.y + 5), light ? rgba(0, 0, 0) : rgba(255, 255, 255), label);
            }
            x += seg;
        }
    }
    ImGui::EndChild();
}

} // namespace rmps::app
