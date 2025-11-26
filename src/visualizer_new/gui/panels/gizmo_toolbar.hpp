/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include <imgui.h>
#include <ImGuizmo.h>

namespace lfs::vis::gui::panels {

    enum class ToolMode {
        None,       // No tool active (splat selected but no gizmo)
        Translate,
        Rotate,
        Scale,
        Brush,
        Align       // 3-point alignment tool
    };

    struct GizmoToolbarState {
        ImGuizmo::OPERATION current_operation = ImGuizmo::TRANSLATE;
        ToolMode current_tool = ToolMode::Translate;
        bool initialized = false;
        unsigned int translation_texture = 0;
        unsigned int rotation_texture = 0;
        unsigned int scaling_texture = 0;
        unsigned int brush_texture = 0;
        unsigned int align_texture = 0;
    };

    // Initialize toolbar textures (call once during startup)
    void InitGizmoToolbar(GizmoToolbarState& state);

    // Cleanup toolbar textures
    void ShutdownGizmoToolbar(GizmoToolbarState& state);

    // Draw the gizmo toolbar
    void DrawGizmoToolbar(const UIContext& ctx, GizmoToolbarState& state,
                          const ImVec2& viewport_pos, const ImVec2& viewport_size);

} // namespace lfs::vis::gui::panels
