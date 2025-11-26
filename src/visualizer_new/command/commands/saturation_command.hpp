/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/command.hpp"
#include "core_new/tensor.hpp"
#include <memory>
#include <string>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::command {

    class SaturationCommand : public Command {
    public:
        // node_name: name of the node whose SH0 was modified
        // old_sh0: SH0 tensor before the saturation adjustment
        // new_sh0: SH0 tensor after the saturation adjustment
        SaturationCommand(SceneManager* scene_manager,
                          std::string node_name,
                          std::shared_ptr<lfs::core::Tensor> old_sh0,
                          std::shared_ptr<lfs::core::Tensor> new_sh0);

        void undo() override;
        void redo() override;
        std::string getName() const override { return "Saturation"; }

    private:
        void applySH0(const std::shared_ptr<lfs::core::Tensor>& sh0);

        SceneManager* scene_manager_;
        std::string node_name_;
        std::shared_ptr<lfs::core::Tensor> old_sh0_;
        std::shared_ptr<lfs::core::Tensor> new_sh0_;
    };

} // namespace lfs::vis::command
