/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/command.hpp"
#include "core_new/tensor.hpp"
#include <memory>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::command {

    class SelectionCommand : public Command {
    public:
        SelectionCommand(SceneManager* scene_manager,
                         std::shared_ptr<lfs::core::Tensor> old_selection,
                         std::shared_ptr<lfs::core::Tensor> new_selection);

        void undo() override;
        void redo() override;
        std::string getName() const override { return "Selection"; }

    private:
        SceneManager* scene_manager_;
        std::shared_ptr<lfs::core::Tensor> old_selection_;
        std::shared_ptr<lfs::core::Tensor> new_selection_;
    };

} // namespace lfs::vis::command
