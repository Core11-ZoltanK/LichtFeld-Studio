/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/camera.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include <tuple>

namespace lfs::rendering {

    // Import Tensor from lfs::core
    using lfs::core::Tensor;

    /**
     * @brief Rasterize Gaussians using tensor-based backend (libtorch-free)
     *
     * @param viewpoint_camera Camera parameters
     * @param gaussian_model Gaussian splat data
     * @param bg_color Background color [3]
     * @param show_rings Enable ring mode visualization
     * @param ring_width Width of the ring band
     * @param model_transforms Optional array of 4x4 transforms [num_transforms, 4, 4] (row-major)
     * @param transform_indices Optional per-Gaussian transform index [N]
     * @param selection_mask Optional per-Gaussian selection mask [N] (uint8, 1=selected/yellow)
     * @param screen_positions_out Optional output: screen positions [N, 2] for brush tool
     * @return Tuple of (rendered_image [3, H, W], depth_map [1, H, W])
     */
    std::tuple<Tensor, Tensor> rasterize_tensor(
        const lfs::core::Camera& viewpoint_camera,
        const lfs::core::SplatData& gaussian_model,
        const Tensor& bg_color,
        bool show_rings = false,
        float ring_width = 0.002f,
        const Tensor* model_transforms = nullptr,
        const Tensor* transform_indices = nullptr,
        const Tensor* selection_mask = nullptr,
        Tensor* screen_positions_out = nullptr,
        // Brush selection (computed in preprocess for coordinate consistency)
        bool brush_active = false,
        float brush_x = 0.0f,
        float brush_y = 0.0f,
        float brush_radius = 0.0f,
        bool brush_add_mode = true,
        Tensor* brush_selection_out = nullptr);

} // namespace lfs::rendering
