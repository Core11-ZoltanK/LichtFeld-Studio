/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/events.hpp"
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::gui {

    class ExportDialog {
    public:
        using BrowseCallback = std::function<void(
            lfs::core::ExportFormat format,
            const std::string& default_filename,
            const std::vector<std::string>& selected_nodes,
            int sh_degree)>;

        void render(bool* p_open, SceneManager* scene_manager);
        void setOnBrowse(BrowseCallback callback) { on_browse_ = std::move(callback); }

    private:
        BrowseCallback on_browse_;
        std::unordered_set<std::string> selected_nodes_;
        lfs::core::ExportFormat selected_format_ = lfs::core::ExportFormat::PLY;
        int export_sh_degree_ = 3;
        int max_sh_degree_ = 3;
        bool initialized_ = false;
    };

} // namespace lfs::vis::gui
