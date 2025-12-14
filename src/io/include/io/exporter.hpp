/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/point_cloud.hpp"
#include "core_new/splat_data.hpp"
#include <expected>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace lfs::io {

    using lfs::core::PointCloud;
    using lfs::core::SplatData;

    // Progress callback for export operations (returns false to cancel)
    using ExportProgressCallback = std::function<bool(float progress, const std::string& stage)>;

    // ============================================================================
    // PLY Export
    // ============================================================================

    struct PlySaveOptions {
        std::filesystem::path output_path;
        bool binary = true;
        bool async = false;
        ExportProgressCallback progress_callback = nullptr;
    };

    std::expected<void, std::string> save_ply(const SplatData& splat_data, const PlySaveOptions& options);
    std::expected<void, std::string> save_ply(const PointCloud& point_cloud, const PlySaveOptions& options);

    PointCloud to_point_cloud(const SplatData& splat_data);
    std::vector<std::string> get_ply_attribute_names(const SplatData& splat_data);

    // ============================================================================
    // Compressed PLY Export
    // ============================================================================

    struct CompressedPlyWriteOptions {
        std::filesystem::path output_path;
        bool include_sh = true;
    };

    std::expected<void, std::string> write_compressed_ply(const SplatData& splat_data,
                                                          const CompressedPlyWriteOptions& options);

    // ============================================================================
    // SOG Export (SuperSplat format)
    // ============================================================================

    struct SogSaveOptions {
        std::filesystem::path output_path;
        int kmeans_iterations = 10;
        bool use_gpu = true;
        ExportProgressCallback progress_callback = nullptr;
    };

    std::expected<void, std::string> save_sog(const SplatData& splat_data, const SogSaveOptions& options);

    // ============================================================================
    // HTML Viewer Export
    // ============================================================================

    using HtmlProgressCallback = std::function<void(float progress, const std::string& stage)>;

    struct HtmlExportOptions {
        std::filesystem::path output_path;
        int kmeans_iterations = 10;
        HtmlProgressCallback progress_callback = nullptr;
    };

    std::expected<void, std::string> export_html(const SplatData& splat_data, const HtmlExportOptions& options);

} // namespace lfs::io
