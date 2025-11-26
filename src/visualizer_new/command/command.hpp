/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <string>

namespace lfs::vis::command {

    class Command {
    public:
        virtual ~Command() = default;
        virtual void undo() = 0;
        virtual void redo() = 0;
        virtual std::string getName() const = 0;
    };

    using CommandPtr = std::unique_ptr<Command>;

} // namespace lfs::vis::command
