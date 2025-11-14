/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "forward.h"
#include "rasterization_api_tensor.h"
#include "rasterization_config.h"
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <tuple>

namespace lfs::rendering {

    // Helper to create resize function for custom Tensor
    inline std::function<char*(size_t)> resize_function_wrapper_tensor(Tensor& t, const char* buffer_name) {
        return [&t, buffer_name](size_t N) -> char* {
            if (N == 0) {
                t = Tensor::empty({0}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                return nullptr;
            }
            t = Tensor::empty({N}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
            printf("[Rendering Raster]   %s: %.2f MB (lazy, pool)\n",
                   buffer_name, N / (1024.0 * 1024.0));
            return reinterpret_cast<char*>(t.ptr<uint8_t>());
        };
    }

    // Validation for custom Tensor (similar to CHECK_INPUT)
    inline void check_tensor_input(bool debug, const Tensor& tensor, const char* name) {
        if (debug) {
            if (!tensor.is_valid() || tensor.device() != lfs::core::Device::CUDA ||
                tensor.dtype() != lfs::core::DataType::Float32 || !tensor.is_contiguous()) {
                throw std::runtime_error("Input tensor '" + std::string(name) +
                                         "' must be a contiguous CUDA float tensor.");
            }
        }
    }

    std::tuple<Tensor, Tensor>
    forward_wrapper_tensor(
        const Tensor& means,
        const Tensor& scales_raw,
        const Tensor& rotations_raw,
        const Tensor& opacities_raw,
        const Tensor& sh_coefficients_0,
        const Tensor& sh_coefficients_rest,
        const Tensor& w2c,
        const Tensor& cam_position,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane) {

        // Validate all input tensors
        check_tensor_input(config::debug, means, "means");
        check_tensor_input(config::debug, scales_raw, "scales_raw");
        check_tensor_input(config::debug, rotations_raw, "rotations_raw");
        check_tensor_input(config::debug, opacities_raw, "opacities_raw");
        check_tensor_input(config::debug, sh_coefficients_0, "sh_coefficients_0");
        check_tensor_input(config::debug, sh_coefficients_rest, "sh_coefficients_rest");

        // Extract dimensions
        const int n_primitives = static_cast<int>(means.size(0));
        const int total_bases_sh_rest = static_cast<int>(sh_coefficients_rest.size(1));

        // Allocate output tensors
        Tensor image = Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)},
                                     lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor alpha = Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)},
                                     lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        const size_t image_size = 3 * width * height * sizeof(float);
        const size_t alpha_size = width * height * sizeof(float);

        printf("[Rendering Raster] Immediate allocations (%d×%d, %d primitives):\n",
               width, height, n_primitives);
        printf("[Rendering Raster]   Output image: %.2f MB (immediate, pool)\n",
               image_size / (1024.0 * 1024.0));
        printf("[Rendering Raster]   Output alpha: %.2f MB (immediate, pool)\n",
               alpha_size / (1024.0 * 1024.0));
        printf("[Rendering Raster]   Immediate total: %.2f MB\n",
               (image_size + alpha_size) / (1024.0 * 1024.0));

        // Create buffer tensors (these will be resized by the forward function)
        Tensor per_primitive_buffers = Tensor::empty({0}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        Tensor per_tile_buffers = Tensor::empty({0}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        Tensor per_instance_buffers = Tensor::empty({0}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);

        // Create allocator functions
        const std::function<char*(size_t)> per_primitive_buffers_func =
            resize_function_wrapper_tensor(per_primitive_buffers, "PerPrimitive");
        const std::function<char*(size_t)> per_tile_buffers_func =
            resize_function_wrapper_tensor(per_tile_buffers, "PerTile");
        const std::function<char*(size_t)> per_instance_buffers_func =
            resize_function_wrapper_tensor(per_instance_buffers, "PerInstance");

        // Ensure w2c and cam_position are contiguous
        Tensor w2c_contig = w2c.is_contiguous() ? w2c : w2c.contiguous();
        Tensor cam_pos_contig = cam_position.is_contiguous() ? cam_position : cam_position.contiguous();

        // Call the actual CUDA forward function
        forward(
            per_primitive_buffers_func,
            per_tile_buffers_func,
            per_instance_buffers_func,
            reinterpret_cast<const float3*>(means.ptr<float>()),
            reinterpret_cast<const float3*>(scales_raw.ptr<float>()),
            reinterpret_cast<const float4*>(rotations_raw.ptr<float>()),
            opacities_raw.ptr<float>(),
            reinterpret_cast<const float3*>(sh_coefficients_0.ptr<float>()),
            reinterpret_cast<const float3*>(sh_coefficients_rest.ptr<float>()),
            reinterpret_cast<const float4*>(w2c_contig.ptr<float>()),
            reinterpret_cast<const float3*>(cam_pos_contig.ptr<float>()),
            image.ptr<float>(),
            alpha.ptr<float>(),
            n_primitives,
            active_sh_bases,
            total_bases_sh_rest,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            near_plane,
            far_plane);

        // Log total allocation summary
        const size_t per_prim_size = per_primitive_buffers.is_valid() ? per_primitive_buffers.numel() : 0;
        const size_t per_tile_size = per_tile_buffers.is_valid() ? per_tile_buffers.numel() : 0;
        const size_t per_inst_size = per_instance_buffers.is_valid() ? per_instance_buffers.numel() : 0;
        const size_t total_allocated = image_size + alpha_size + per_prim_size + per_tile_size + per_inst_size;

        printf("[Rendering Raster] Forward complete\n");
        printf("[Rendering Raster] Total allocated: %.2f MB (pool)\n",
               total_allocated / (1024.0 * 1024.0));
        printf("[Rendering Raster]   Breakdown: Image=%.2f Alpha=%.2f PerPrim=%.2f PerTile=%.2f PerInst=%.2f MB\n",
               image_size / (1024.0 * 1024.0),
               alpha_size / (1024.0 * 1024.0),
               per_prim_size / (1024.0 * 1024.0),
               per_tile_size / (1024.0 * 1024.0),
               per_inst_size / (1024.0 * 1024.0));
        printf("[Rendering Raster]   NO backward helpers (saves ~80 MB vs training)\n");
        printf("[Rendering Raster]   NO bucket buffers (saves ~20 MB vs training)\n");

        return {std::move(image), std::move(alpha)};
    }

} // namespace lfs::rendering
