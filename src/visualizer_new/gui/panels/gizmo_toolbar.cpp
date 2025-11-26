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
            auto path = lfs::vis::getAssetPath("icon/" + icon_name);
            auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);

            unsigned int texture_id;
            glGenTextures(1, &texture_id);
            glBindTexture(GL_TEXTURE_2D, texture_id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

            lfs::core::free_image(data);
            glBindTexture(GL_TEXTURE_2D, 0);

            LOG_DEBUG("Loaded toolbar icon: {} ({}x{}, {} channels)", icon_name, width, height, channels);
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
        state.initialized = true;
    }

    void ShutdownGizmoToolbar(GizmoToolbarState& state) {
        if (!state.initialized) return;

        if (state.translation_texture) glDeleteTextures(1, &state.translation_texture);
        if (state.rotation_texture) glDeleteTextures(1, &state.rotation_texture);
        if (state.scaling_texture) glDeleteTextures(1, &state.scaling_texture);
        if (state.brush_texture) glDeleteTextures(1, &state.brush_texture);
        if (state.align_texture) glDeleteTextures(1, &state.align_texture);

        state.translation_texture = 0;
        state.rotation_texture = 0;
        state.scaling_texture = 0;
        state.brush_texture = 0;
        state.align_texture = 0;
        state.initialized = false;
    }

    void DrawGizmoToolbar([[maybe_unused]] const UIContext& ctx, GizmoToolbarState& state,
                          const ImVec2& viewport_pos, const ImVec2& viewport_size) {
        if (!state.initialized) {
            InitGizmoToolbar(state);
        }

        const ImGuiViewport* viewport = ImGui::GetMainViewport();

        // Position at top center of the viewport area
        float toolbar_width = 220.0f;  // Wide enough for 5 buttons + separator
        float toolbar_height = 36.0f;

        float pos_x = viewport->WorkPos.x + viewport_pos.x + (viewport_size.x - toolbar_width) * 0.5f;
        float pos_y = viewport->WorkPos.y + viewport_pos.y + 5.0f;

        ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(toolbar_width, toolbar_height), ImGuiCond_Always);

        ImGuiWindowFlags flags =
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoSavedSettings;

        float button_size = 24.0f;
        float vertical_padding = (toolbar_height - button_size) * 0.5f - 3.0f;  // Shift up slightly

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6.0f, vertical_padding));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.15f, 0.9f));

        if (ImGui::Begin("##GizmoToolbar", nullptr, flags)) {
            ImVec2 btn_size(button_size, button_size);

            auto IconButton = [&](const char* id, unsigned int texture, ToolMode tool,
                                  ImGuizmo::OPERATION op, const char* fallback) {
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
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Translate (W)");
            ImGui::SameLine();

            IconButton("##rotate", state.rotation_texture, ToolMode::Rotate, ImGuizmo::ROTATE, "R");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rotate (E)");
            ImGui::SameLine();

            IconButton("##scale", state.scaling_texture, ToolMode::Scale, ImGuizmo::SCALE, "S");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Scale (R)");
            ImGui::SameLine();

            // Separator
            ImGui::SameLine(0.0f, 8.0f);
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 p = ImGui::GetCursorScreenPos();
            draw_list->AddLine(ImVec2(p.x, p.y + 2.0f), ImVec2(p.x, p.y + button_size - 2.0f),
                              IM_COL32(128, 128, 128, 128), 1.0f);
            ImGui::Dummy(ImVec2(2.0f, 0.0f));
            ImGui::SameLine(0.0f, 8.0f);

            IconButton("##brush", state.brush_texture, ToolMode::Brush, ImGuizmo::TRANSLATE, "B");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Brush Selection (B)");
            ImGui::SameLine();

            IconButton("##align", state.align_texture, ToolMode::Align, ImGuizmo::TRANSLATE, "A");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("3-Point Align (A)");
        }
        ImGui::End();

        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
    }

} // namespace lfs::vis::gui::panels
