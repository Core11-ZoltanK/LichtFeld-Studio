/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cropbox_command.hpp"
#include "rendering/rendering_manager.hpp"

namespace lfs::vis::command {

    CropBoxCommand::CropBoxCommand(RenderingManager* rendering_manager,
                                   const CropBoxState& old_state,
                                   const CropBoxState& new_state)
        : rendering_manager_(rendering_manager)
        , old_state_(old_state)
        , new_state_(new_state) {}

    void CropBoxCommand::undo() {
        applyState(old_state_);
    }

    void CropBoxCommand::redo() {
        applyState(new_state_);
    }

    void CropBoxCommand::applyState(const CropBoxState& state) {
        if (!rendering_manager_) return;

        auto settings = rendering_manager_->getSettings();
        settings.crop_min = state.crop_min;
        settings.crop_max = state.crop_max;
        settings.crop_transform = state.crop_transform;
        settings.crop_inverse = state.crop_inverse;
        rendering_manager_->updateSettings(settings);
    }

} // namespace lfs::vis::command
