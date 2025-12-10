/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <glm/glm.hpp>

namespace lfs::rendering {

    // Camera clipping planes
    constexpr float DEFAULT_NEAR_PLANE = 0.1f;
    constexpr float DEFAULT_FAR_PLANE = 100000.0f;

    // Camera defaults
    constexpr float DEFAULT_FOV = 60.0f;

    // Coordinate system transform: converts from internal camera space (+Y up, +Z forward)
    // to OpenGL clip space (-Y up, -Z forward). Used by all renderers for consistency.
    inline const glm::mat3 FLIP_YZ{1, 0, 0, 0, -1, 0, 0, 0, -1};

    // Compute view rotation from camera-to-world rotation matrix
    inline glm::mat3 computeViewRotation(const glm::mat3& camera_rotation) {
        return FLIP_YZ * glm::transpose(camera_rotation);
    }

} // namespace lfs::rendering
