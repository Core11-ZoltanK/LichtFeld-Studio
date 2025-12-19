/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sequencer_settings_panel.hpp"
#include "core/events.hpp"
#include "gui/ui_widgets.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::gui::panels {

using namespace lfs::io::video;
using namespace lfs::core::events;

namespace {
constexpr int MIN_WIDTH = 320;
constexpr int MAX_WIDTH = 7680;
constexpr int MIN_HEIGHT = 240;
constexpr int MAX_HEIGHT = 4320;
constexpr const char* FPS_ITEMS[] = {"24 fps", "30 fps", "60 fps"};
constexpr int FPS_VALUES[] = {24, 30, 60};
} // namespace

void DrawSequencerSection(const UIContext& ctx, SequencerUIState& state) {
    widgets::SectionHeader("SEQUENCER", ctx.fonts);

    ImGui::Checkbox("Show Camera Path", &state.show_camera_path);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Display camera path in viewport");
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Video Export");
    ImGui::Spacing();

    const auto current_info = getPresetInfo(state.preset);
    if (ImGui::BeginCombo("Format", current_info.name)) {
        for (int i = 0; i < getPresetCount(); ++i) {
            const auto p = static_cast<VideoPreset>(i);
            const auto info = getPresetInfo(p);
            const bool selected = (state.preset == p);

            if (ImGui::Selectable(info.name, selected)) {
                state.preset = p;
                if (p != VideoPreset::CUSTOM) {
                    state.framerate = info.framerate;
                    state.quality = info.crf;
                }
            }

            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", info.description);
            }

            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    if (state.preset == VideoPreset::CUSTOM) {
        ImGui::InputInt("Width", &state.custom_width, 16, 64);
        ImGui::InputInt("Height", &state.custom_height, 16, 64);
        state.custom_width = std::clamp(state.custom_width, MIN_WIDTH, MAX_WIDTH);
        state.custom_height = std::clamp(state.custom_height, MIN_HEIGHT, MAX_HEIGHT);

        int fps_idx = (state.framerate == 24) ? 0 : (state.framerate == 60) ? 2 : 1;
        if (ImGui::Combo("Framerate", &fps_idx, FPS_ITEMS, 3)) {
            state.framerate = FPS_VALUES[fps_idx];
        }
    } else {
        ImGui::TextDisabled("%s", current_info.description);
    }

    ImGui::SliderInt("Quality", &state.quality, 15, 28, "CRF %d");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Lower = higher quality, larger file");
    }

    ImGui::Spacing();

    const bool has_keyframes = ctx.sequencer_controller &&
                                !ctx.sequencer_controller->timeline().empty();

    if (!has_keyframes) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button("Export Video...", ImVec2(-1, 0))) {
        const auto info = getPresetInfo(state.preset);
        const int width = (state.preset == VideoPreset::CUSTOM) ? state.custom_width : info.width;
        const int height = (state.preset == VideoPreset::CUSTOM) ? state.custom_height : info.height;

        cmd::SequencerExportVideo{
            .width = width,
            .height = height,
            .framerate = state.framerate,
            .crf = state.quality
        }.emit();
    }

    if (!has_keyframes) {
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Add keyframes first (press K)");
        }
    }
}

} // namespace lfs::vis::gui::panels
