/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/splat_data.hpp"
#include <expected>
#include <filesystem>
#include <functional>
#include <string>

namespace lfs::vis::gui {

    struct HtmlViewerExportOptions {
        std::filesystem::path output_path;
        std::function<void(float, const std::string&)> progress_callback;
    };

    std::expected<void, std::string> export_html_viewer(
        const lfs::core::SplatData& splat_data,
        const HtmlViewerExportOptions& options);

} // namespace lfs::vis::gui
