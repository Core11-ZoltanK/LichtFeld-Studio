/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/splat_data.hpp"
#include <expected>
#include <filesystem>
#include <functional>
#include <string>

namespace lfs::core {

    // Progress callback: (progress 0.0-1.0, stage_name) -> return false to cancel
    using SogProgressCallback = std::function<bool(float progress, const std::string& stage)>;

    struct SogWriteOptions {
        int iterations = 10;
        bool use_gpu = true;
        std::filesystem::path output_path;
        SogProgressCallback progress_callback = nullptr;
    };

    std::expected<void, std::string> write_sog(
        const SplatData& splat_data,
        const SogWriteOptions& options);

} // namespace lfs::core
