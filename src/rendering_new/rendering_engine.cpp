/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/logger.hpp"
#include "rendering_engine_impl.hpp"
#include "rendering_new/rendering.hpp"
#include "viewport_gizmo.hpp"

namespace lfs::rendering {

    std::unique_ptr<RenderingEngine> RenderingEngine::create() {
        LOG_DEBUG("Creating RenderingEngine instance");
        return std::make_unique<RenderingEngineImpl>();
    }

    glm::mat3 RenderingEngine::getAxisViewRotation(const int axis, const bool negative) {
        return ViewportGizmo::getAxisViewRotation(static_cast<GizmoAxis>(axis), negative);
    }

} // namespace lfs::rendering