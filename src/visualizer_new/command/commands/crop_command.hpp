/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/command.hpp"
#include "core_new/tensor.hpp"
#include <string>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::command {

    // Undo/redo command for soft crop operations using deletion masks
    class CropCommand : public Command {
    public:
        CropCommand(SceneManager* scene_manager,
                    std::string node_name,
                    lfs::core::Tensor old_deleted_mask,
                    lfs::core::Tensor new_deleted_mask);

        void undo() override;
        void redo() override;
        [[nodiscard]] std::string getName() const override { return "Crop"; }

    private:
        SceneManager* const scene_manager_;
        const std::string node_name_;
        const lfs::core::Tensor old_deleted_mask_;
        const lfs::core::Tensor new_deleted_mask_;
    };

} // namespace lfs::vis::command
