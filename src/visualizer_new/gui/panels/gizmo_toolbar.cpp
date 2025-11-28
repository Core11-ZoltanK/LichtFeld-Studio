/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <glad/glad.h>

#include "gui/panels/gizmo_toolbar.hpp"
#include "core_new/image_io.hpp"
#include "core_new/logger.hpp"
#include "internal/resource_paths.hpp"
#include <imgui.h>

namespace lfs::vis::gui::panels {

    static unsigned int LoadIconTexture(const std::string& icon_name) {
        try {
            const auto path = lfs::vis::getAssetPath("icon/" + icon_name);
            const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);

            unsigned int texture_id;
            glGenTextures(1, &texture_id);
            glBindTexture(GL_TEXTURE_2D, texture_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            const GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

            lfs::core::free_image(data);
            glBindTexture(GL_TEXTURE_2D, 0);
            return texture_id;
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load toolbar icon {}: {}", icon_name, e.what());
            return 0;
        }
    }

    void InitGizmoToolbar(GizmoToolbarState& state) {
        if (state.initialized) return;

        state.translation_texture = LoadIconTexture("translation.png");
        state.rotation_texture = LoadIconTexture("rotation.png");
        state.scaling_texture = LoadIconTexture("scaling.png");
        state.brush_texture = LoadIconTexture("brush.png");
        state.align_texture = LoadIconTexture("align.png");
        state.cropbox_texture = LoadIconTexture("cropbox.png");
        state.bounds_texture = LoadIconTexture("bounds.png");
        state.reset_texture = LoadIconTexture("reset.png");
        state.initialized = true;
    }

    void ShutdownGizmoToolbar(GizmoToolbarState& state) {
        if (!state.initialized) return;

        if (state.translation_texture) glDeleteTextures(1, &state.translation_texture);
        if (state.rotation_texture) glDeleteTextures(1, &state.rotation_texture);
        if (state.scaling_texture) glDeleteTextures(1, &state.scaling_texture);
        if (state.brush_texture) glDeleteTextures(1, &state.brush_texture);
        if (state.align_texture) glDeleteTextures(1, &state.align_texture);
        if (state.cropbox_texture) glDeleteTextures(1, &state.cropbox_texture);
        if (state.bounds_texture) glDeleteTextures(1, &state.bounds_texture);
        if (state.reset_texture) glDeleteTextures(1, &state.reset_texture);

        state.translation_texture = 0;
        state.rotation_texture = 0;
        state.scaling_texture = 0;
        state.brush_texture = 0;
        state.align_texture = 0;
        state.cropbox_texture = 0;
        state.bounds_texture = 0;
        state.reset_texture = 0;
        state.initialized = false;
    }

    void DrawGizmoToolbar([[maybe_unused]] const UIContext& ctx, GizmoToolbarState& state,
                          const ImVec2& viewport_pos, const ImVec2& viewport_size) {
        if (!state.initialized) {
            InitGizmoToolbar(state);
        }

        const ImGuiViewport* viewport = ImGui::GetMainViewport();

        constexpr float toolbar_width = 260.0f;
        constexpr float toolbar_height = 36.0f;

        const float pos_x = viewport->WorkPos.x + viewport_pos.x + (viewport_size.x - toolbar_width) * 0.5f;
        const float pos_y = viewport->WorkPos.y + viewport_pos.y + 5.0f;

        ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(toolbar_width, toolbar_height), ImGuiCond_Always);

        constexpr ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoSavedSettings;

        constexpr float button_size = 24.0f;
        const float vertical_padding = (toolbar_height - button_size) * 0.5f - 3.0f;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, vertical_padding));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.15f, 0.9f));

        if (ImGui::Begin("##GizmoToolbar", nullptr, flags)) {
            const ImVec2 btn_size(button_size, button_size);

            const auto IconButton = [&](const char* id, const unsigned int texture, const ToolMode tool,
                                        const ImGuizmo::OPERATION op, const char* fallback) {
                const bool is_selected = (state.current_tool == tool);
                ImGui::PushStyleColor(ImGuiCol_Button, is_selected ?
                    ImVec4(0.3f, 0.5f, 0.8f, 1.0f) : ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, is_selected ?
                    ImVec4(0.4f, 0.6f, 0.9f, 1.0f) : ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

                const bool clicked = texture ?
                    ImGui::ImageButton(id, (ImTextureID)(intptr_t)texture, btn_size,
                                       ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0)) :
                    ImGui::Button(fallback, btn_size);

                ImGui::PopStyleColor(2);
                if (clicked) {
                    if (is_selected) {
                        state.current_tool = ToolMode::None;
                    } else {
                        state.current_tool = tool;
                        state.current_operation = op;
                    }
                }
                return clicked;
            };

            IconButton("##translate", state.translation_texture, ToolMode::Translate, ImGuizmo::TRANSLATE, "T");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Translate");
            ImGui::SameLine();

            IconButton("##rotate", state.rotation_texture, ToolMode::Rotate, ImGuizmo::ROTATE, "R");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rotate");
            ImGui::SameLine();

            IconButton("##scale", state.scaling_texture, ToolMode::Scale, ImGuizmo::SCALE, "S");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Scale");
            ImGui::SameLine();

            // Separator
            ImGui::SameLine(0.0f, 8.0f);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddLine(
                ImVec2(p.x, p.y + 2.0f), ImVec2(p.x, p.y + button_size - 2.0f),
                IM_COL32(128, 128, 128, 128), 1.0f);
            ImGui::Dummy(ImVec2(2.0f, 0.0f));
            ImGui::SameLine(0.0f, 8.0f);

            IconButton("##brush", state.brush_texture, ToolMode::Brush, ImGuizmo::TRANSLATE, "B");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Brush Selection");
            ImGui::SameLine();

            IconButton("##align", state.align_texture, ToolMode::Align, ImGuizmo::TRANSLATE, "A");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("3-Point Align");
            ImGui::SameLine();

            IconButton("##cropbox", state.cropbox_texture, ToolMode::CropBox, ImGuizmo::BOUNDS, "C");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Crop Box");
        }
        ImGui::End();

        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);

        // Secondary toolbar for cropbox operations
        if (state.current_tool == ToolMode::CropBox) {
            constexpr float sub_toolbar_width = 205.0f;
            constexpr float sub_toolbar_height = 36.0f;

            const float sub_pos_x = viewport->WorkPos.x + viewport_pos.x + (viewport_size.x - sub_toolbar_width) * 0.5f;
            const float sub_pos_y = viewport->WorkPos.y + viewport_pos.y + toolbar_height + 8.0f;

            ImGui::SetNextWindowPos(ImVec2(sub_pos_x, sub_pos_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(sub_toolbar_width, sub_toolbar_height), ImGuiCond_Always);

            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, vertical_padding));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.12f, 0.12f, 0.12f, 0.95f));

            if (ImGui::Begin("##CropBoxToolbar", nullptr, flags)) {
                const ImVec2 btn_size(button_size, button_size);

                const auto CropOpButton = [&](const char* id, const unsigned int texture,
                                              const CropBoxOperation op, const char* fallback, const char* tooltip) {
                    const bool is_selected = (state.cropbox_operation == op);
                    ImGui::PushStyleColor(ImGuiCol_Button, is_selected ?
                        ImVec4(0.3f, 0.5f, 0.8f, 1.0f) : ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, is_selected ?
                        ImVec4(0.4f, 0.6f, 0.9f, 1.0f) : ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

                    const bool clicked = texture ?
                        ImGui::ImageButton(id, (ImTextureID)(intptr_t)texture, btn_size,
                                           ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0)) :
                        ImGui::Button(fallback, btn_size);

                    ImGui::PopStyleColor(2);
                    if (clicked) {
                        state.cropbox_operation = op;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
                };

                CropOpButton("##crop_bounds", state.bounds_texture, CropBoxOperation::Bounds, "B", "Resize Bounds");
                ImGui::SameLine();
                CropOpButton("##crop_translate", state.translation_texture, CropBoxOperation::Translate, "T", "Translate");
                ImGui::SameLine();
                CropOpButton("##crop_rotate", state.rotation_texture, CropBoxOperation::Rotate, "R", "Rotate");
                ImGui::SameLine();
                CropOpButton("##crop_scale", state.scaling_texture, CropBoxOperation::Scale, "S", "Scale");
                ImGui::SameLine();

                // Reset button
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));

                const bool reset_clicked = state.reset_texture ?
                    ImGui::ImageButton("##crop_reset", (ImTextureID)(intptr_t)state.reset_texture, btn_size,
                                       ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0)) :
                    ImGui::Button("X", btn_size);

                ImGui::PopStyleColor(2);
                if (reset_clicked) {
                    state.reset_cropbox_requested = true;
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reset to Default");
            }
            ImGui::End();

            ImGui::PopStyleColor();
            ImGui::PopStyleVar(2);
        }
    }

} // namespace lfs::vis::gui::panels
