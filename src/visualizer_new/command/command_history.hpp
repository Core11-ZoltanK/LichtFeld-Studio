/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command.hpp"
#include <vector>

namespace lfs::vis::command {

    class CommandHistory {
    public:
        void execute(CommandPtr cmd);
        void undo();
        void redo();
        void clear();

        bool canUndo() const { return !history_.empty() && current_index_ > 0; }
        bool canRedo() const { return current_index_ < history_.size(); }
        size_t size() const { return history_.size(); }

    private:
        std::vector<CommandPtr> history_;
        size_t current_index_ = 0;
    };

} // namespace lfs::vis::command
