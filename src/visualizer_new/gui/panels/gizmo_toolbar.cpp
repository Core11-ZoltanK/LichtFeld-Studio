/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <glad/glad.h>

#include "gui/panels/gizmo_toolbar.hpp"
#include "core_new/image_io.hpp"
#include "core_new/logger.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"
#include <imgui.h>

namespace lfs::vis::gui::panels {

    // Layout constant (not theme-dependent)
    constexpr float SUBTOOLBAR_OFFSET_Y = 8.0f;

    // Computes toolbar dimensions for N buttons
    static ImVec2 ComputeToolbarSize(const int num_buttons) {
        const auto& t = theme();
        const float width = num_buttons * t.sizes.toolbar_button_size +
                            (num_buttons - 1) * t.sizes.toolbar_spacing +
                            2.0f * t.sizes.toolbar_padding;
        const float height = t.sizes.toolbar_button_size + 2.0f * t.sizes.toolbar_padding;
        return ImVec2(width, height);
    }

    // RAII helper for toolbar style setup
    struct ToolbarStyle {
        ToolbarStyle() {
            const auto& t = theme();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.window_rounding);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(t.sizes.toolbar_padding, t.sizes.toolbar_padding));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(t.sizes.toolbar_spacing, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, t.toolbar_background());
        }
        ~ToolbarStyle() {
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(4);
        }
    };

    // Secondary toolbar style (slightly darker)
    struct SubToolbarStyle {
        SubToolbarStyle() {
            const auto& t = theme();
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, t.sizes.window_rounding);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(t.sizes.toolbar_padding, t.sizes.toolbar_padding));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(t.sizes.toolbar_spacing, 0.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_WindowBg, t.subtoolbar_background());
        }
        ~SubToolbarStyle() {
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(4);
        }
    };

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

        state.selection_texture = LoadIconTexture("selection.png");
        state.rectangle_texture = LoadIconTexture("rectangle.png");
        state.polygon_texture = LoadIconTexture("polygon.png");
        state.lasso_texture = LoadIconTexture("lasso.png");
        state.ring_texture = LoadIconTexture("ring.png");
        state.translation_texture = LoadIconTexture("translation.png");
        state.rotation_texture = LoadIconTexture("rotation.png");
        state.scaling_texture = LoadIconTexture("scaling.png");
        state.brush_texture = LoadIconTexture("brush.png");
        state.align_texture = LoadIconTexture("align.png");
        state.cropbox_texture = LoadIconTexture("cropbox.png");
        state.bounds_texture = LoadIconTexture("bounds.png");
        state.reset_texture = LoadIconTexture("reset.png");
        state.local_texture = LoadIconTexture("local.png");
        state.world_texture = LoadIconTexture("world.png");
        state.initialized = true;
    }

    void ShutdownGizmoToolbar(GizmoToolbarState& state) {
        if (!state.initialized) return;

        if (state.selection_texture) glDeleteTextures(1, &state.selection_texture);
        if (state.rectangle_texture) glDeleteTextures(1, &state.rectangle_texture);
        if (state.polygon_texture) glDeleteTextures(1, &state.polygon_texture);
        if (state.lasso_texture) glDeleteTextures(1, &state.lasso_texture);
        if (state.ring_texture) glDeleteTextures(1, &state.ring_texture);
        if (state.translation_texture) glDeleteTextures(1, &state.translation_texture);
        if (state.rotation_texture) glDeleteTextures(1, &state.rotation_texture);
        if (state.scaling_texture) glDeleteTextures(1, &state.scaling_texture);
        if (state.brush_texture) glDeleteTextures(1, &state.brush_texture);
        if (state.align_texture) glDeleteTextures(1, &state.align_texture);
        if (state.cropbox_texture) glDeleteTextures(1, &state.cropbox_texture);
        if (state.bounds_texture) glDeleteTextures(1, &state.bounds_texture);
        if (state.reset_texture) glDeleteTextures(1, &state.reset_texture);
        if (state.local_texture) glDeleteTextures(1, &state.local_texture);
        if (state.world_texture) glDeleteTextures(1, &state.world_texture);

        state.selection_texture = 0;
        state.rectangle_texture = 0;
        state.polygon_texture = 0;
        state.lasso_texture = 0;
        state.ring_texture = 0;
        state.translation_texture = 0;
        state.rotation_texture = 0;
        state.scaling_texture = 0;
        state.brush_texture = 0;
        state.align_texture = 0;
        state.cropbox_texture = 0;
        state.bounds_texture = 0;
        state.reset_texture = 0;
        state.local_texture = 0;
        state.world_texture = 0;
        state.initialized = false;
    }

    void DrawGizmoToolbar(const UIContext& ctx, GizmoToolbarState& state,
                          const ImVec2& viewport_pos, const ImVec2& viewport_size) {
        if (!state.initialized) {
            InitGizmoToolbar(state);
        }

        // EditorContext is the single source of truth
        auto* editor = ctx.editor;
        if (!editor) return;

        // During training, don't show toolbar
        if (editor->isTrainingOrPaused()) {
            return;
        }

        // Validate and auto-deselect unavailable tools
        editor->validateActiveTool();

        const ImGuiViewport* viewport = ImGui::GetMainViewport();

        constexpr ImGuiWindowFlags FLAGS =
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoSavedSettings;

        // Main toolbar: 7 buttons
        constexpr int NUM_MAIN_BUTTONS = 7;
        const ImVec2 toolbar_size = ComputeToolbarSize(NUM_MAIN_BUTTONS);

        const float pos_x = viewport->WorkPos.x + viewport_pos.x + (viewport_size.x - toolbar_size.x) * 0.5f;
        const float pos_y = viewport->WorkPos.y + viewport_pos.y + 5.0f;

        ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y), ImGuiCond_Always);
        ImGui::SetNextWindowSize(toolbar_size, ImGuiCond_Always);

        {
            const ToolbarStyle style;
            if (ImGui::Begin("##GizmoToolbar", nullptr, FLAGS)) {
                const auto& t = theme();
                const ImVec2 btn_size(t.sizes.toolbar_button_size, t.sizes.toolbar_button_size);

                // Tool button helper
                const auto ToolButton = [&](const char* id, unsigned int texture,
                                            ToolType tool, ImGuizmo::OPERATION op,
                                            const char* fallback, const char* tooltip) {
                    const bool is_selected = (editor->getActiveTool() == tool);
                    const bool enabled = editor->isToolAvailable(tool);
                    const char* disabled_reason = editor->getToolUnavailableReason(tool);

                    if (!enabled) {
                        ImGui::BeginDisabled();
                    }

                    ImGui::PushStyleColor(ImGuiCol_Button, is_selected ? t.button_selected() : t.button_normal());
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, is_selected ? t.button_selected_hovered() : t.button_hovered());

                    const bool clicked = texture
                        ? ImGui::ImageButton(id, static_cast<ImTextureID>(texture),
                                             btn_size, ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0))
                        : ImGui::Button(fallback, btn_size);

                    ImGui::PopStyleColor(2);

                    if (!enabled) {
                        ImGui::EndDisabled();
                    }

                    if (clicked && enabled) {
                        editor->setActiveTool(is_selected ? ToolType::None : tool);
                        if (!is_selected) state.current_operation = op;
                    }

                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                        if (enabled) {
                            ImGui::SetTooltip("%s", tooltip);
                        } else if (disabled_reason) {
                            ImGui::SetTooltip("%s (%s)", tooltip, disabled_reason);
                        }
                    }
                };

                ToolButton("##selection", state.selection_texture, ToolType::Selection, ImGuizmo::TRANSLATE, "S", "Selection");
                ImGui::SameLine();
                ToolButton("##translate", state.translation_texture, ToolType::Translate, ImGuizmo::TRANSLATE, "T", "Translate");
                ImGui::SameLine();
                ToolButton("##rotate", state.rotation_texture, ToolType::Rotate, ImGuizmo::ROTATE, "R", "Rotate");
                ImGui::SameLine();
                ToolButton("##scale", state.scaling_texture, ToolType::Scale, ImGuizmo::SCALE, "S", "Scale");
                ImGui::SameLine();
                ToolButton("##brush", state.brush_texture, ToolType::Brush, ImGuizmo::TRANSLATE, "B", "Brush Selection");
                ImGui::SameLine();
                ToolButton("##align", state.align_texture, ToolType::Align, ImGuizmo::TRANSLATE, "A", "3-Point Align");
                ImGui::SameLine();
                ToolButton("##cropbox", state.cropbox_texture, ToolType::CropBox, ImGuizmo::BOUNDS, "C", "Crop Box");
            }
            ImGui::End();
        }

        const ToolType active_tool = editor->getActiveTool();

        // Secondary toolbar for selection mode
        if (active_tool == ToolType::Selection) {
            constexpr int NUM_SEL_BUTTONS = 5;
            const ImVec2 sub_size = ComputeToolbarSize(NUM_SEL_BUTTONS);

            const float sub_pos_x = viewport->WorkPos.x + viewport_pos.x + (viewport_size.x - sub_size.x) * 0.5f;
            const float sub_pos_y = viewport->WorkPos.y + viewport_pos.y + toolbar_size.y + SUBTOOLBAR_OFFSET_Y;

            ImGui::SetNextWindowPos(ImVec2(sub_pos_x, sub_pos_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(sub_size, ImGuiCond_Always);

            {
                const SubToolbarStyle style;
                if (ImGui::Begin("##SelectionModeToolbar", nullptr, FLAGS)) {
                    const auto& t = theme();
                    const ImVec2 btn_size(t.sizes.toolbar_button_size, t.sizes.toolbar_button_size);

                    const auto SelectionModeButton = [&](const char* id, unsigned int texture,
                                                         SelectionSubMode mode, const char* fallback,
                                                         const char* tooltip) {
                        const bool is_selected = (state.selection_mode == mode);
                        ImGui::PushStyleColor(ImGuiCol_Button, is_selected ? t.button_selected() : t.button_normal());
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, is_selected ? t.button_selected_hovered() : t.button_hovered());

                        const bool clicked = texture
                            ? ImGui::ImageButton(id, static_cast<ImTextureID>(texture),
                                                 btn_size, ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0))
                            : ImGui::Button(fallback, btn_size);

                        ImGui::PopStyleColor(2);
                        if (clicked) state.selection_mode = mode;
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
                    };

                    SelectionModeButton("##centers", state.selection_texture, SelectionSubMode::Centers,
                                        "C", "Brush selection (Ctrl+1)");
                    ImGui::SameLine();
                    SelectionModeButton("##rect", state.rectangle_texture, SelectionSubMode::Rectangle,
                                        "R", "Rectangle selection (Ctrl+2)");
                    ImGui::SameLine();
                    SelectionModeButton("##polygon", state.polygon_texture, SelectionSubMode::Polygon,
                                        "P", "Polygon selection (Ctrl+3)");
                    ImGui::SameLine();
                    SelectionModeButton("##lasso", state.lasso_texture, SelectionSubMode::Lasso,
                                        "L", "Lasso selection (Ctrl+4)");
                    ImGui::SameLine();
                    SelectionModeButton("##rings", state.ring_texture, SelectionSubMode::Rings,
                                        "O", "Ring selection (Ctrl+5)");
                }
                ImGui::End();
            }
        }

        // Transform space toolbar (Local/World toggle)
        const bool is_transform_tool = (active_tool == ToolType::Translate ||
                                        active_tool == ToolType::Rotate ||
                                        active_tool == ToolType::Scale);
        if (is_transform_tool) {
            constexpr int NUM_BUTTONS = 2;
            const ImVec2 sub_size = ComputeToolbarSize(NUM_BUTTONS);
            const float sub_x = viewport->WorkPos.x + viewport_pos.x + (viewport_size.x - sub_size.x) * 0.5f;
            const float sub_y = viewport->WorkPos.y + viewport_pos.y + toolbar_size.y + SUBTOOLBAR_OFFSET_Y;

            ImGui::SetNextWindowPos(ImVec2(sub_x, sub_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(sub_size, ImGuiCond_Always);

            const SubToolbarStyle style;
            if (ImGui::Begin("##TransformSpaceToolbar", nullptr, FLAGS)) {
                const auto& t = theme();
                const ImVec2 btn_size(t.sizes.toolbar_button_size, t.sizes.toolbar_button_size);

                const auto SpaceButton = [&](const char* id, unsigned int tex,
                                             TransformSpace space, const char* fallback,
                                             const char* tooltip) {
                    const bool selected = (state.transform_space == space);
                    ImGui::PushStyleColor(ImGuiCol_Button, selected ? t.button_selected() : t.button_normal());
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, selected ? t.button_selected_hovered() : t.button_hovered());

                    const bool clicked = tex
                        ? ImGui::ImageButton(id, static_cast<ImTextureID>(tex),
                                             btn_size, ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0))
                        : ImGui::Button(fallback, btn_size);

                    ImGui::PopStyleColor(2);
                    if (clicked) { state.transform_space = space; }
                    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("%s", tooltip); }
                };

                SpaceButton("##local", state.local_texture, TransformSpace::Local, "L", "Local Space");
                ImGui::SameLine();
                SpaceButton("##world", state.world_texture, TransformSpace::World, "W", "World Space");
            }
            ImGui::End();
        }

        // Cropbox operations toolbar
        if (active_tool == ToolType::CropBox) {
            constexpr int NUM_CROP_BUTTONS = 5;
            const ImVec2 sub_size = ComputeToolbarSize(NUM_CROP_BUTTONS);
            const float sub_x = viewport->WorkPos.x + viewport_pos.x + (viewport_size.x - sub_size.x) * 0.5f;
            const float sub_y = viewport->WorkPos.y + viewport_pos.y + toolbar_size.y + SUBTOOLBAR_OFFSET_Y;

            ImGui::SetNextWindowPos(ImVec2(sub_x, sub_y), ImGuiCond_Always);
            ImGui::SetNextWindowSize(sub_size, ImGuiCond_Always);

            const SubToolbarStyle style;
            if (ImGui::Begin("##CropBoxToolbar", nullptr, FLAGS)) {
                const auto& t = theme();
                const ImVec2 btn_size(t.sizes.toolbar_button_size, t.sizes.toolbar_button_size);

                const auto CropOpButton = [&](const char* id, unsigned int tex,
                                              CropBoxOperation op, const char* fallback,
                                              const char* tooltip) {
                    const bool selected = (state.cropbox_operation == op);
                    ImGui::PushStyleColor(ImGuiCol_Button, selected ? t.button_selected() : t.button_normal());
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, selected ? t.button_selected_hovered() : t.button_hovered());

                    const bool clicked = tex
                        ? ImGui::ImageButton(id, static_cast<ImTextureID>(tex),
                                             btn_size, ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0))
                        : ImGui::Button(fallback, btn_size);

                    ImGui::PopStyleColor(2);
                    if (clicked) { state.cropbox_operation = op; }
                    if (ImGui::IsItemHovered()) { ImGui::SetTooltip("%s", tooltip); }
                };

                CropOpButton("##crop_bounds", state.bounds_texture, CropBoxOperation::Bounds, "B", "Resize Bounds");
                ImGui::SameLine();
                CropOpButton("##crop_translate", state.translation_texture, CropBoxOperation::Translate, "T", "Translate");
                ImGui::SameLine();
                CropOpButton("##crop_rotate", state.rotation_texture, CropBoxOperation::Rotate, "R", "Rotate");
                ImGui::SameLine();
                CropOpButton("##crop_scale", state.scaling_texture, CropBoxOperation::Scale, "S", "Scale");
                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Button, t.button_normal());
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.button_hovered());
                const bool reset_clicked = state.reset_texture
                    ? ImGui::ImageButton("##crop_reset", static_cast<ImTextureID>(state.reset_texture),
                                         btn_size, ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0))
                    : ImGui::Button("X", btn_size);
                ImGui::PopStyleColor(2);
                if (reset_clicked) { state.reset_cropbox_requested = true; }
                if (ImGui::IsItemHovered()) { ImGui::SetTooltip("Reset to Default"); }
            }
            ImGui::End();
        }
    }

} // namespace lfs::vis::gui::panels
