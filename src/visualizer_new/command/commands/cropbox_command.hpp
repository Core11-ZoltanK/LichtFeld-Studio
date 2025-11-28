/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/command.hpp"
#include "geometry_new/euclidean_transform.hpp"
#include <glm/glm.hpp>

namespace lfs::vis {
    class RenderingManager;
}

namespace lfs::vis::command {

    struct CropBoxState {
        glm::vec3 crop_min;
        glm::vec3 crop_max;
        lfs::geometry::EuclideanTransform crop_transform;
        bool crop_inverse = false;
    };

    class CropBoxCommand : public Command {
    public:
        CropBoxCommand(RenderingManager* rendering_manager,
                       const CropBoxState& old_state,
                       const CropBoxState& new_state);

        void undo() override;
        void redo() override;
        std::string getName() const override { return "CropBox"; }

    private:
        void applyState(const CropBoxState& state);

        RenderingManager* rendering_manager_;
        CropBoxState old_state_;
        CropBoxState new_state_;
    };

} // namespace lfs::vis::command
