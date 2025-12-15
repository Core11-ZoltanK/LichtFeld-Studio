/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <deque>
#include <string>

namespace lfs::vis::gui {

    /// Self-contained notification popup triggered by events
    class NotificationPopup {
    public:
        enum class Type { INFO, WARNING, ERROR };

        NotificationPopup();

        void render();
        void show(Type type, const std::string& title, const std::string& message);

    private:
        void setupEventHandlers();

        struct Notification {
            Type type;
            std::string title;
            std::string message;
        };

        std::deque<Notification> pending_;
        Notification current_;
        bool popup_open_ = false;
    };

} // namespace lfs::vis::gui
