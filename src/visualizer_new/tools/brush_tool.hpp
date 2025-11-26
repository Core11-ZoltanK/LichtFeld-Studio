/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tool_base.hpp"
#include "core_new/tensor.hpp"
#include <glm/glm.hpp>
#include <memory>

namespace lfs::vis::tools {

    class BrushTool : public ToolBase {
    public:
        enum class BrushAction { Add, Remove };

        BrushTool();
        ~BrushTool() override = default;

        std::string_view getName() const override { return "Brush Tool"; }
        std::string_view getDescription() const override { return "Paint to select Gaussians"; }

        bool initialize(const ToolContext& ctx) override;
        void shutdown() override;
        void update(const ToolContext& ctx) override;
        void renderUI(const lfs::vis::gui::UIContext& ui_ctx, bool* p_open) override;

        bool isActive() const { return is_painting_; }
        bool handleMouseButton(int button, int action, int mods, double x, double y, const ToolContext& ctx);
        bool handleMouseMove(double x, double y, const ToolContext& ctx);
        bool handleScroll(double x_offset, double y_offset, int mods, const ToolContext& ctx);

        float getBrushRadius() const { return brush_radius_; }
        void setBrushRadius(float radius) { brush_radius_ = std::clamp(radius, 1.0f, 500.0f); }

    protected:
        void onEnabledChanged(bool enabled) override;

    private:
        BrushAction current_action_ = BrushAction::Add;
        float brush_radius_ = 20.0f;
        bool is_painting_ = false;
        glm::vec2 last_mouse_pos_{0.0f};
        const ToolContext* tool_context_ = nullptr;
        lfs::core::Tensor cumulative_selection_;
        std::shared_ptr<lfs::core::Tensor> selection_before_stroke_;

        void beginStroke(double x, double y, BrushAction action, bool clear_existing, const ToolContext& ctx);
        void endStroke();
        void clearSelection(const ToolContext& ctx);
        void updateSelectionAtPoint(double x, double y, const ToolContext& ctx);
    };

} // namespace lfs::vis::tools
