/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Re-export public API
#include "io/exporter.hpp"

namespace lfs::io {

    // Internal: Loading function (not in public API)
    std::expected<SplatData, std::string> load_ply(const std::filesystem::path& filepath);

    // Alias for backward compatibility
    using SaveProgressCallback = ExportProgressCallback;

} // namespace lfs::io
