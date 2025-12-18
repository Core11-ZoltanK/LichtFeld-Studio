/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/point_cloud.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace lfs::core {

    // Forward declaration
    class SplatData;

    /**
     * @brief Save SplatData to PLY file
     * @param splat_data The splat data to save
     * @param root Directory to save to
     * @param iteration Iteration number for filename
     * @param join_threads If true, wait for save to complete; if false, save asynchronously
     * @param stem If not empty, use as filename (without .ply extension)
     */
    void save_ply(const SplatData& splat_data,
                  const std::filesystem::path& root,
                  int iteration,
                  bool join_threads = true,
                  std::string stem = "");

    /**
     * @brief Save SplatData to SOG format
     * @param splat_data The splat data to save
     * @param root Directory to save to
     * @param iteration Iteration number for filename
     * @param kmeans_iterations Number of k-means iterations for compression
     * @param join_threads If true, wait for save to complete (SOG is always sync currently)
     * @return Path to the saved SOG file
     */
    std::filesystem::path save_sog(const SplatData& splat_data,
                                   const std::filesystem::path& root,
                                   int iteration,
                                   int kmeans_iterations = 10,
                                   bool join_threads = true);

    /**
     * @brief Convert SplatData to PointCloud for export
     * @param splat_data The splat data to convert
     * @return PointCloud suitable for PLY export
     */
    PointCloud to_point_cloud(const SplatData& splat_data);

    /**
     * @brief Get attribute names for PLY format
     * @param splat_data The splat data
     * @return Vector of attribute names in PLY order
     */
    std::vector<std::string> get_attribute_names(const SplatData& splat_data);

} // namespace lfs::core
