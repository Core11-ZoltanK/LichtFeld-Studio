/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sequencer_panel.hpp"
#include "core/events.hpp"
#include "theme/theme.hpp"
#include <algorithm>
#include <cmath>
#include <format>

namespace lfs::vis {

namespace {
constexpr float DEFAULT_TIMELINE_DURATION = 10.0f;
constexpr float TIMELINE_END_PADDING = 1.0f;
constexpr float MIN_KEYFRAME_SPACING = 0.1f;
constexpr float ICON_SIZE = 7.0f;
constexpr float PLAY_ICON_SIZE = 8.0f;
constexpr float PAUSE_BAR_W = 2.5f;
constexpr float PAUSE_BAR_H = 9.0f;
constexpr float PAUSE_GAP = 3.0f;
constexpr float PLAYHEAD_HANDLE_SIZE = 7.0f;
constexpr float TIMELINE_ROUNDING = 4.0f;
constexpr float SKIP_ICON_SIZE = 5.0f;
constexpr float MAJOR_TICK_HEIGHT = 8.0f;
constexpr float MINOR_TICK_HEIGHT = 4.0f;

constexpr ImGuiWindowFlags PANEL_FLAGS =
    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
    ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
    ImGuiWindowFlags_NoFocusOnAppearing;

[[nodiscard]] std::string formatTime(const float seconds) {
    const int mins = static_cast<int>(seconds) / 60;
    const float secs = seconds - static_cast<float>(mins * 60);
    return std::format("{}:{:05.2f}", mins, secs);
}

[[nodiscard]] std::string formatTimeShort(const float seconds) {
    const int mins = static_cast<int>(seconds) / 60;
    const int secs = static_cast<int>(seconds) % 60;
    if (mins > 0) {
        return std::format("{}:{:02d}", mins, secs);
    }
    return std::format("{}s", secs);
}
} // namespace

using namespace panel_config;

SequencerPanel::SequencerPanel(SequencerController& controller)
    : controller_(controller) {}

void SequencerPanel::render(const float viewport_x, const float viewport_width, const float viewport_y_bottom) {
    const auto& t = theme();

    const float panel_x = viewport_x + PADDING_H;
    const float panel_width = viewport_width - 2.0f * PADDING_H;
    const ImVec2 panel_pos = {panel_x, viewport_y_bottom - HEIGHT - PADDING_BOTTOM};
    const ImVec2 panel_size = {panel_width, HEIGHT};

    ImGui::SetNextWindowPos(panel_pos);
    ImGui::SetNextWindowSize(panel_size);
    ImGui::SetNextWindowBgAlpha(0.95f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, toU32(t.palette.surface));
    ImGui::PushStyleColor(ImGuiCol_Border, toU32WithAlpha(t.palette.border, 0.4f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.window_rounding);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);

    if (!ImGui::Begin("##SequencerPanel", nullptr, PANEL_FLAGS)) {
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(2);
        ImGui::End();
        return;
    }

    const float content_width = panel_size.x - 2.0f * INNER_PADDING;
    const float timeline_width = content_width - TRANSPORT_WIDTH - TIME_DISPLAY_WIDTH;
    const float content_height = HEIGHT - 2.0f * INNER_PADDING;

    const ImVec2 transport_pos = {panel_pos.x + INNER_PADDING, panel_pos.y + INNER_PADDING};
    const ImVec2 timeline_pos = {transport_pos.x + TRANSPORT_WIDTH, panel_pos.y + INNER_PADDING};
    const ImVec2 time_display_pos = {timeline_pos.x + timeline_width, panel_pos.y + INNER_PADDING};

    renderTransportControls(transport_pos, content_height);
    renderTimeline(timeline_pos, timeline_width, content_height);
    renderTimeDisplay(time_display_pos, content_height);

    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

void SequencerPanel::renderTransportControls(const ImVec2& pos, const float height) {
    const auto& t = theme();
    const float y_center = pos.y + height / 2.0f;
    const float btn_half = BUTTON_SIZE / 2.0f;
    float x_offset = 0.0f;

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, btn_half);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 0});
    ImGui::PushStyleColor(ImGuiCol_Button, t.button_normal());
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.button_hovered());
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.button_active());

    // |◀ First keyframe
    ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
    if (ImGui::Button("##first", {BUTTON_SIZE, BUTTON_SIZE})) {
        controller_.seekToFirstKeyframe();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Go to first keyframe");
    }
    {
        ImDrawList* const dl = ImGui::GetWindowDrawList();
        const ImVec2 center = {pos.x + x_offset + btn_half, y_center};
        dl->AddRectFilled(
            {center.x - SKIP_ICON_SIZE - 1, center.y - SKIP_ICON_SIZE},
            {center.x - SKIP_ICON_SIZE + 1, center.y + SKIP_ICON_SIZE},
            t.text_u32());
        dl->AddTriangleFilled(
            {center.x + SKIP_ICON_SIZE, center.y - SKIP_ICON_SIZE},
            {center.x + SKIP_ICON_SIZE, center.y + SKIP_ICON_SIZE},
            {center.x - SKIP_ICON_SIZE + 2, center.y},
            t.text_u32());
    }
    x_offset += BUTTON_SIZE + BUTTON_SPACING;

    // ■ Stop
    ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
    if (ImGui::Button("##stop", {BUTTON_SIZE, BUTTON_SIZE})) {
        controller_.stop();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Stop");
    }
    {
        ImDrawList* const dl = ImGui::GetWindowDrawList();
        const ImVec2 center = {pos.x + x_offset + btn_half, y_center};
        dl->AddRectFilled(
            {center.x - ICON_SIZE / 2, center.y - ICON_SIZE / 2},
            {center.x + ICON_SIZE / 2, center.y + ICON_SIZE / 2},
            t.text_u32());
    }
    x_offset += BUTTON_SIZE + BUTTON_SPACING;

    // ▶/❚❚ Play/Pause
    ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
    if (ImGui::Button("##playpause", {BUTTON_SIZE, BUTTON_SIZE})) {
        controller_.togglePlayPause();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(controller_.isPlaying() ? "Pause (Space)" : "Play (Space)");
    }
    {
        ImDrawList* const dl = ImGui::GetWindowDrawList();
        const ImVec2 center = {pos.x + x_offset + btn_half, y_center};

        if (controller_.isPlaying()) {
            dl->AddRectFilled(
                {center.x - PAUSE_GAP - PAUSE_BAR_W, center.y - PAUSE_BAR_H / 2},
                {center.x - PAUSE_GAP, center.y + PAUSE_BAR_H / 2},
                t.text_u32());
            dl->AddRectFilled(
                {center.x + PAUSE_GAP - PAUSE_BAR_W, center.y - PAUSE_BAR_H / 2},
                {center.x + PAUSE_GAP, center.y + PAUSE_BAR_H / 2},
                t.text_u32());
        } else {
            dl->AddTriangleFilled(
                {center.x - PLAY_ICON_SIZE * 0.4f, center.y - PLAY_ICON_SIZE},
                {center.x - PLAY_ICON_SIZE * 0.4f, center.y + PLAY_ICON_SIZE},
                {center.x + PLAY_ICON_SIZE * 0.8f, center.y},
                t.text_u32());
        }
    }
    x_offset += BUTTON_SIZE + BUTTON_SPACING;

    // ▶| Last keyframe
    ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
    if (ImGui::Button("##last", {BUTTON_SIZE, BUTTON_SIZE})) {
        controller_.seekToLastKeyframe();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Go to last keyframe");
    }
    {
        ImDrawList* const dl = ImGui::GetWindowDrawList();
        const ImVec2 center = {pos.x + x_offset + btn_half, y_center};
        dl->AddTriangleFilled(
            {center.x - SKIP_ICON_SIZE, center.y - SKIP_ICON_SIZE},
            {center.x - SKIP_ICON_SIZE, center.y + SKIP_ICON_SIZE},
            {center.x + SKIP_ICON_SIZE - 2, center.y},
            t.text_u32());
        dl->AddRectFilled(
            {center.x + SKIP_ICON_SIZE - 1, center.y - SKIP_ICON_SIZE},
            {center.x + SKIP_ICON_SIZE + 1, center.y + SKIP_ICON_SIZE},
            t.text_u32());
    }
    x_offset += BUTTON_SIZE + BUTTON_SPACING + 4.0f;

    // ↻ Loop toggle
    const bool is_looping = controller_.loopMode() != LoopMode::ONCE;
    if (is_looping) {
        ImGui::PushStyleColor(ImGuiCol_Button, t.primary_u32());
    }
    ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
    if (ImGui::Button("##loop", {BUTTON_SIZE, BUTTON_SIZE})) {
        controller_.toggleLoop();
    }
    if (is_looping) {
        ImGui::PopStyleColor();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(is_looping ? "Loop: ON" : "Loop: OFF");
    }
    {
        ImDrawList* const dl = ImGui::GetWindowDrawList();
        const ImVec2 center = {pos.x + x_offset + btn_half, y_center};
        const float r = ICON_SIZE * 0.8f;
        const ImU32 col = is_looping ? toU32(t.palette.text) : t.text_dim_u32();
        dl->PathArcTo(center, r, 0.5f, 2.5f, 8);
        dl->PathStroke(col, 0, 1.5f);
        dl->PathArcTo(center, r, 3.64f, 5.64f, 8);
        dl->PathStroke(col, 0, 1.5f);
        const float ah = 3.0f;
        dl->AddTriangleFilled(
            {center.x + r - ah, center.y - ah},
            {center.x + r + ah, center.y},
            {center.x + r - ah, center.y + ah},
            col);
        dl->AddTriangleFilled(
            {center.x - r + ah, center.y + ah},
            {center.x - r - ah, center.y},
            {center.x - r + ah, center.y - ah},
            col);
    }
    x_offset += BUTTON_SIZE + BUTTON_SPACING;

    // + Add keyframe
    ImGui::SetCursorScreenPos({pos.x + x_offset, y_center - btn_half});
    if (ImGui::Button("+##addkf", {BUTTON_SIZE, BUTTON_SIZE})) {
        lfs::core::events::cmd::SequencerAddKeyframe{}.emit();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Add keyframe (K)");
    }

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
}

void SequencerPanel::renderTimeline(const ImVec2& pos, const float width, const float height) {
    const auto& t = theme();
    ImDrawList* const dl = ImGui::GetWindowDrawList();

    const float ruler_y = pos.y;
    const float timeline_y = pos.y + RULER_HEIGHT + 4.0f;
    const float timeline_height = height - RULER_HEIGHT - 4.0f;
    const float bar_half = std::min(timeline_height, TIMELINE_HEIGHT) / 2.0f;
    const float y_center = timeline_y + timeline_height / 2.0f;

    const ImVec2 bar_min = {pos.x, y_center - bar_half};
    const ImVec2 bar_max = {pos.x + width, y_center + bar_half};

    dl->AddRectFilled(bar_min, bar_max, toU32WithAlpha(t.palette.background, 0.8f), TIMELINE_ROUNDING);
    dl->AddRect(bar_min, bar_max, toU32WithAlpha(t.palette.border, 0.3f), TIMELINE_ROUNDING, 0, 1.0f);

    const auto& timeline = controller_.timeline();

    renderTimeRuler(dl, {pos.x, ruler_y}, width);

    if (timeline.empty()) {
        constexpr const char* HINT = "Position camera and press K to add keyframes";
        const ImVec2 text_size = ImGui::CalcTextSize(HINT);
        dl->AddText({pos.x + (width - text_size.x) / 2, y_center - text_size.y / 2},
                    toU32WithAlpha(t.palette.text_dim, 0.5f), HINT);
        return;
    }

    const ImVec2 mouse = ImGui::GetMousePos();
    const bool mouse_in_timeline = mouse.x >= bar_min.x && mouse.x <= bar_max.x &&
                                    mouse.y >= bar_min.y - RULER_HEIGHT && mouse.y <= bar_max.y;

    // Playhead dragging
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mouse_in_timeline && !dragging_keyframe_) {
        dragging_playhead_ = true;
        controller_.beginScrub();
    }
    if (dragging_playhead_) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            const float time = std::clamp(xToTime(mouse.x, pos.x, width), 0.0f, timeline.endTime());
            controller_.scrub(time);
        } else {
            dragging_playhead_ = false;
            controller_.endScrub();
        }
    }

    // Keyframes
    hovered_keyframe_ = std::nullopt;
    const auto& keyframes = timeline.keyframes();
    for (size_t i = 0; i < keyframes.size(); ++i) {
        const float x = timeToX(keyframes[i].time, pos.x, width);
        const ImVec2 kf_pos = {x, y_center};

        const float dist = std::abs(mouse.x - x);
        const bool hovered = mouse_in_timeline && dist < KEYFRAME_RADIUS * 2;
        if (hovered) {
            hovered_keyframe_ = i;
        }

        const bool selected = controller_.selectedKeyframe() == i;
        const bool is_first = (i == 0);
        drawKeyframeMarker(dl, kf_pos, selected, hovered, keyframes[i].time);

        if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            controller_.selectKeyframe(i);
            if (!is_first) {
                dragging_keyframe_ = true;
                dragged_keyframe_index_ = i;
            }
        }
    }

    // Keyframe dragging
    if (dragging_keyframe_) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            const float new_time = std::max(xToTime(mouse.x, pos.x, width), MIN_KEYFRAME_SPACING);
            controller_.timeline().setKeyframeTime(dragged_keyframe_index_, new_time, false);
        } else {
            controller_.timeline().sortKeyframes();
            dragging_keyframe_ = false;
        }
    }

    // Delete keyframe
    if (controller_.hasSelection() && ImGui::IsKeyPressed(ImGuiKey_Delete)) {
        controller_.removeSelectedKeyframe();
    }

    // Context menu
    if (mouse_in_timeline && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        context_menu_time_ = xToTime(mouse.x, pos.x, width);
        context_menu_keyframe_ = hovered_keyframe_;
        context_menu_open_ = true;
        context_menu_pos_ = mouse;
        ImGui::OpenPopup("TimelineContextMenu");
    }

    ImGui::SetNextWindowPos(context_menu_pos_, ImGuiCond_Always, {0.0f, 1.0f});
    if (ImGui::BeginPopup("TimelineContextMenu")) {
        if (context_menu_keyframe_.has_value()) {
            const size_t idx = *context_menu_keyframe_;
            const bool is_first = (idx == 0);

            if (ImGui::MenuItem("Update to Current View", "U")) {
                controller_.selectKeyframe(idx);
                lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
            }
            if (ImGui::MenuItem("Go to Keyframe")) {
                controller_.selectKeyframe(idx);
                controller_.seek(keyframes[idx].time);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Delete Keyframe", "Del", false, !is_first)) {
                controller_.selectKeyframe(idx);
                controller_.removeSelectedKeyframe();
            }
        } else {
            if (ImGui::MenuItem("Add Keyframe Here", "K")) {
                lfs::core::events::cmd::SequencerAddKeyframe{}.emit();
            }
        }
        ImGui::EndPopup();
    } else {
        context_menu_open_ = false;
    }

    // Playhead
    const float playhead_x = timeToX(controller_.playhead(), pos.x, width);
    drawPlayhead(dl, {playhead_x, ruler_y}, {playhead_x, bar_max.y + 4});
}

void SequencerPanel::renderTimeRuler(ImDrawList* const dl, const ImVec2& pos, const float width) {
    const auto& t = theme();
    const float end_time = getDisplayEndTime();

    float major_interval = 1.0f;
    if (end_time > 60.0f) {
        major_interval = 10.0f;
    } else if (end_time > 30.0f) {
        major_interval = 5.0f;
    } else if (end_time > 10.0f) {
        major_interval = 2.0f;
    } else if (end_time <= 2.0f) {
        major_interval = 0.5f;
    }

    const float minor_interval = major_interval / 4.0f;

    for (float t_val = 0.0f; t_val <= end_time; t_val += minor_interval) {
        const float x = pos.x + (t_val / end_time) * width;
        if (x < pos.x || x > pos.x + width) continue;

        const bool is_major = std::fmod(t_val + 0.001f, major_interval) < 0.01f;

        if (is_major) {
            dl->AddLine({x, pos.y + RULER_HEIGHT - MAJOR_TICK_HEIGHT},
                       {x, pos.y + RULER_HEIGHT},
                       t.text_dim_u32(), 1.0f);

            const std::string label = formatTimeShort(t_val);
            const ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
            dl->AddText({x - text_size.x / 2, pos.y}, t.text_dim_u32(), label.c_str());
        } else {
            dl->AddLine({x, pos.y + RULER_HEIGHT - MINOR_TICK_HEIGHT},
                       {x, pos.y + RULER_HEIGHT},
                       toU32WithAlpha(t.palette.text_dim, 0.5f), 1.0f);
        }
    }

    dl->AddLine({pos.x, pos.y + RULER_HEIGHT},
               {pos.x + width, pos.y + RULER_HEIGHT},
               toU32WithAlpha(t.palette.border, 0.5f), 1.0f);
}

void SequencerPanel::renderTimeDisplay(const ImVec2& pos, const float height) {
    const auto& t = theme();
    const float y_center = pos.y + height / 2.0f;

    const std::string time_str = formatTime(controller_.playhead());
    const ImVec2 text_size = ImGui::CalcTextSize(time_str.c_str());

    ImDrawList* const dl = ImGui::GetWindowDrawList();
    dl->AddText({pos.x + 8.0f, y_center - text_size.y / 2}, t.text_u32(), time_str.c_str());

    if (!controller_.timeline().empty()) {
        const std::string dur_str = " / " + formatTime(controller_.timeline().endTime());
        dl->AddText({pos.x + 8.0f + text_size.x, y_center - text_size.y / 2}, t.text_dim_u32(), dur_str.c_str());
    }
}

void SequencerPanel::drawKeyframeMarker(ImDrawList* const dl, const ImVec2& pos,
                                         const bool selected, const bool hovered, const float time) const {
    const auto& t = theme();

    ImU32 fill = t.primary_u32();
    if (selected) {
        fill = toU32(lighten(t.palette.primary, 0.2f));
    } else if (hovered) {
        fill = toU32(lighten(t.palette.primary, 0.1f));
    }

    dl->AddQuadFilled(
        {pos.x, pos.y - KEYFRAME_RADIUS},
        {pos.x + KEYFRAME_RADIUS, pos.y},
        {pos.x, pos.y + KEYFRAME_RADIUS},
        {pos.x - KEYFRAME_RADIUS, pos.y},
        fill);

    if (selected) {
        dl->AddQuad(
            {pos.x, pos.y - KEYFRAME_RADIUS - 1},
            {pos.x + KEYFRAME_RADIUS + 1, pos.y},
            {pos.x, pos.y + KEYFRAME_RADIUS + 1},
            {pos.x - KEYFRAME_RADIUS - 1, pos.y},
            toU32(t.palette.text), 1.5f);
    }

    if (hovered) {
        ImGui::SetTooltip("Keyframe @ %s", formatTime(time).c_str());
    }
}

void SequencerPanel::drawPlayhead(ImDrawList* const dl, const ImVec2& top, const ImVec2& bottom) const {
    const auto& t = theme();
    dl->AddLine(top, bottom, t.error_u32(), PLAYHEAD_WIDTH);
    dl->AddTriangleFilled(
        {top.x - PLAYHEAD_HANDLE_SIZE, top.y},
        {top.x + PLAYHEAD_HANDLE_SIZE, top.y},
        {top.x, top.y + PLAYHEAD_HANDLE_SIZE},
        t.error_u32());
}

float SequencerPanel::getDisplayEndTime() const {
    const auto& timeline = controller_.timeline();
    if (timeline.size() < 2) {
        return DEFAULT_TIMELINE_DURATION;
    }
    return std::max(timeline.endTime() + TIMELINE_END_PADDING, DEFAULT_TIMELINE_DURATION);
}

float SequencerPanel::timeToX(const float time, const float timeline_x, const float timeline_width) const {
    const float end = getDisplayEndTime();
    return timeline_x + (time / end) * timeline_width;
}

float SequencerPanel::xToTime(const float x, const float timeline_x, const float timeline_width) const {
    const float end = getDisplayEndTime();
    return ((x - timeline_x) / timeline_width) * end;
}

} // namespace lfs::vis
