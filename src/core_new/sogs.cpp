/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef _WIN32
#define NOMINMAX
#endif

#include "core_new/sogs.hpp"
#include "core_new/logger.hpp"
#include "core_new/splat_data.hpp"
#include "kernels/kmeans_new.cuh"
#include "kernels/morton_encoding_new.cuh"
#include <algorithm>
#include <archive.h>
#include <archive_entry.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <webp/encode.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace lfs::core {

    namespace {

#ifdef _WIN32
        using ssize_t = std::ptrdiff_t;
#endif

        // Apply log transform for better quantization
        float log_transform(float value) {
            return std::copysign(std::log(std::abs(value) + 1.0f), value);
        }

        // Pack quaternion into 8-bit values
        std::array<uint8_t, 4> pack_quaternion(float w, float x, float y, float z) {
            // Normalize
            float len = std::sqrt(w * w + x * x + y * y + z * z);
            if (len > 0) {
                w /= len;
                x /= len;
                y /= len;
                z /= len;
            } else {
                // Handle zero-length quaternion: set to identity quaternion
                w = 1.0f;
                x = 0.0f;
                y = 0.0f;
                z = 0.0f;
                LOG_WARN("pack_quaternion: Zero-length quaternion encountered, replaced with identity quaternion.");
            }

            // Find largest component (in absolute value)
            float max_val = std::abs(w);
            int max_idx = 0; // 0 = w, 1 = x, 2 = y, 3 = z

            if (std::abs(x) > max_val) {
                max_val = std::abs(x);
                max_idx = 1;
            }
            if (std::abs(y) > max_val) {
                max_val = std::abs(y);
                max_idx = 2;
            }
            if (std::abs(z) > max_val) {
                max_val = std::abs(z);
                max_idx = 3;
            }

            // Ensure largest component is positive
            if ((max_idx == 0 && w < 0) ||
                (max_idx == 1 && x < 0) ||
                (max_idx == 2 && y < 0) ||
                (max_idx == 3 && z < 0)) {
                w = -w;
                x = -x;
                y = -y;
                z = -z;
            }

            // Scale the quaternion components by sqrt(2)
            constexpr float sqrt2 = 1.41421356237f;
            w *= sqrt2;
            x *= sqrt2;
            y *= sqrt2;
            z *= sqrt2;

            // Pack the other 3 components based on which is largest
            std::array<uint8_t, 4> result;

            if (max_idx == 0) {
                // w is largest, store x, y, z
                result[0] = static_cast<uint8_t>(std::clamp((x * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[1] = static_cast<uint8_t>(std::clamp((y * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[2] = static_cast<uint8_t>(std::clamp((z * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            } else if (max_idx == 1) {
                // x is largest, store w, y, z
                result[0] = static_cast<uint8_t>(std::clamp((w * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[1] = static_cast<uint8_t>(std::clamp((y * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[2] = static_cast<uint8_t>(std::clamp((z * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            } else if (max_idx == 2) {
                // y is largest, store w, x, z
                result[0] = static_cast<uint8_t>(std::clamp((w * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[1] = static_cast<uint8_t>(std::clamp((x * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[2] = static_cast<uint8_t>(std::clamp((z * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            } else {
                // z is largest, store w, x, y
                result[0] = static_cast<uint8_t>(std::clamp((w * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[1] = static_cast<uint8_t>(std::clamp((x * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
                result[2] = static_cast<uint8_t>(std::clamp((y * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            }

            // Store which component was largest
            result[3] = 252 + max_idx;

            return result;
        }

        // Write WebP image
        bool write_webp_image(const std::filesystem::path& path,
                              const uint8_t* data,
                              int width,
                              int height,
                              int channels = 4) {

            if (!data || width <= 0 || height <= 0) {
                LOG_ERROR("Invalid input to write_webp_image: data={}, width={}, height={}",
                          (void*)data, width, height);
                return false;
            }

            uint8_t* output = nullptr;
            size_t output_size = 0;

            std::vector<uint8_t> rgba_buffer;

            if (channels == 4) {
                rgba_buffer.resize(width * height * 4);
                std::memcpy(rgba_buffer.data(), data, width * height * 4);

                output_size = WebPEncodeLosslessRGBA(
                    rgba_buffer.data(),
                    width,
                    height,
                    width * 4,
                    &output);
            } else if (channels == 3) {
                rgba_buffer.resize(width * height * 4);
                for (int i = 0; i < width * height; ++i) {
                    rgba_buffer[i * 4 + 0] = data[i * 3 + 0];
                    rgba_buffer[i * 4 + 1] = data[i * 3 + 1];
                    rgba_buffer[i * 4 + 2] = data[i * 3 + 2];
                    rgba_buffer[i * 4 + 3] = 255;
                }

                output_size = WebPEncodeLosslessRGBA(
                    rgba_buffer.data(),
                    width,
                    height,
                    width * 4,
                    &output);
            } else {
                LOG_ERROR("Unsupported number of channels: {}", channels);
                return false;
            }

            if (output_size == 0 || output == nullptr) {
                LOG_ERROR("WebP encoding failed for {} (size={})", path.string(), output_size);
                if (output)
                    WebPFree(output);
                return false;
            }

            std::ofstream file(path, std::ios::binary);
            if (!file) {
                WebPFree(output);
                LOG_ERROR("Failed to open file: {}", path.string());
                return false;
            }

            file.write(reinterpret_cast<const char*>(output), output_size);
            WebPFree(output);

            if (!file.good()) {
                LOG_ERROR("Failed to write file: {}", path.string());
                return false;
            }

            LOG_DEBUG("Successfully wrote WebP: {} ({}x{}, {} bytes)",
                      path.string(), width, height, output_size);
            return true;
        }

        // Create a ZIP archive for .sog bundle
        class SogArchive {
            struct archive* a;
            std::filesystem::path path;

        public:
            SogArchive(const std::filesystem::path& output_path) : path(output_path) {
                a = archive_write_new();
                archive_write_set_format_zip(a);
                archive_write_open_filename(a, path.string().c_str());
            }

            ~SogArchive() {
                if (a) {
                    archive_write_close(a);
                    archive_write_free(a);
                }
            }

            bool add_file(const std::string& filename, const void* data, size_t size) {
                struct archive_entry* entry = archive_entry_new();

                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);

                archive_entry_set_pathname(entry, filename.c_str());
                archive_entry_set_size(entry, size);
                archive_entry_set_filetype(entry, AE_IFREG);
                archive_entry_set_perm(entry, 0644);
                archive_entry_set_mtime(entry, time_t, 0);

                if (archive_write_header(a, entry) != ARCHIVE_OK) {
                    archive_entry_free(entry);
                    LOG_ERROR("Failed to write archive header: {}", archive_error_string(a));
                    return false;
                }

                if (archive_write_data(a, data, size) != static_cast<ssize_t>(size)) {
                    archive_entry_free(entry);
                    LOG_ERROR("Failed to write archive data: {}", archive_error_string(a));
                    return false;
                }

                archive_entry_free(entry);
                return true;
            }

            bool add_webp(const std::string& filename, const uint8_t* data,
                          int width, int height, int channels = 4) {

                if (!data || width <= 0 || height <= 0) {
                    LOG_ERROR("Invalid input to add_webp: data={}, width={}, height={}",
                              (void*)data, width, height);
                    return false;
                }

                uint8_t* output = nullptr;
                size_t output_size = 0;

                std::vector<uint8_t> rgba_buffer;

                if (channels == 4) {
                    rgba_buffer.resize(width * height * 4);
                    std::memcpy(rgba_buffer.data(), data, width * height * 4);

                    output_size = WebPEncodeLosslessRGBA(
                        rgba_buffer.data(),
                        width,
                        height,
                        width * 4,
                        &output);
                } else if (channels == 3) {
                    rgba_buffer.resize(width * height * 4);
                    for (int i = 0; i < width * height; ++i) {
                        rgba_buffer[i * 4 + 0] = data[i * 3 + 0];
                        rgba_buffer[i * 4 + 1] = data[i * 3 + 1];
                        rgba_buffer[i * 4 + 2] = data[i * 3 + 2];
                        rgba_buffer[i * 4 + 3] = 255;
                    }

                    output_size = WebPEncodeLosslessRGBA(
                        rgba_buffer.data(),
                        width,
                        height,
                        width * 4,
                        &output);
                } else {
                    LOG_ERROR("Unsupported number of channels: {}", channels);
                    return false;
                }

                if (output_size == 0 || output == nullptr) {
                    LOG_ERROR("WebP encoding failed for {} in archive", filename);
                    if (output)
                        WebPFree(output);
                    return false;
                }

                bool result = add_file(filename, output, output_size);
                WebPFree(output);

                if (result) {
                    LOG_DEBUG("Added {} to archive ({}x{}, {} bytes)",
                              filename, width, height, output_size);
                }

                return result;
            }
        };

        // Identity layout function - matches TypeScript
        int identity_layout(int index, int width) {
            return index;
        }

    } // anonymous namespace

    std::expected<void, std::string> write_sog(
        const SplatData& splat_data,
        const SogWriteOptions& options) {

        try {
            LOG_INFO("SOG write: {}", options.output_path.string());

            // Returns false if cancelled
            const auto report_progress = [&](float progress, const std::string& stage) -> bool {
                return !options.progress_callback || options.progress_callback(progress, stage);
            };

            if (!report_progress(0.0f, "Initializing")) {
                return std::unexpected("Export cancelled");
            }

            const int64_t num_splats = splat_data.size();
            if (num_splats == 0) {
                return std::unexpected("No splats to write");
            }

            // Texture dimensions (multiple of 4)
            constexpr int CHANNELS = 4;
            const int width = ((int)std::ceil(std::sqrt(num_splats) / 4.0)) * 4;
            const int height = ((int)std::ceil(num_splats / (float)width / 4.0)) * 4;

            LOG_DEBUG("SOG: {}x{} for {} splats", width, height, num_splats);

            if (!report_progress(0.02f, "Loading data")) {
                return std::unexpected("Export cancelled");
            }

            // Get data tensors - already on CUDA
            auto means = splat_data.means_raw().cuda();
            auto scales = splat_data.scaling_raw().cuda();
            auto rotations = splat_data.rotation_raw().cuda();
            auto opacities = splat_data.get_opacity().cuda(); // Apply sigmoid
            auto sh0 = splat_data.sh0_raw().cuda();
            auto shN = splat_data.shN_raw().cuda();

            // Determine SH degree from shN shape
            int sh_degree = splat_data.get_max_sh_degree();
            LOG_DEBUG("Detected SH degree: {}", sh_degree);

            if (!report_progress(0.05f, "Morton sort")) {
                return std::unexpected("Export cancelled");
            }

            // Morton encoding for spatial coherence
            auto morton_codes = morton_encode_new(means);
            auto indices_tensor = morton_sort_indices_new(morton_codes).cpu();
            const int64_t* indices = indices_tensor.ptr<int64_t>();

            const bool is_bundle = options.output_path.extension() == ".sog";
            std::unique_ptr<SogArchive> archive;
            std::filesystem::path base_path;

            if (is_bundle) {
                archive = std::make_unique<SogArchive>(options.output_path);
                base_path = options.output_path.parent_path();
            } else {
                base_path = options.output_path.parent_path();
                std::filesystem::create_directories(base_path);
            }

            // Helper lambda to write images
            auto write_image = [&](const std::string& filename,
                                   const uint8_t* data,
                                   int w = -1, int h = -1) -> bool {
                if (w == -1)
                    w = width;
                if (h == -1)
                    h = height;

                if (!data) {
                    LOG_ERROR("Null data pointer for {}", filename);
                    return false;
                }

                if (archive) {
                    LOG_DEBUG("Adding {} to archive ({}x{})", filename, w, h);
                    return archive->add_webp(filename, data, w, h, CHANNELS);
                } else {
                    auto file_path = base_path / filename;
                    auto webp_path = file_path;
                    if (webp_path.extension() != ".webp") {
                        webp_path.replace_extension(".webp");
                    }
                    LOG_DEBUG("Writing {} ({}x{})", webp_path.string(), w, h);
                    return write_webp_image(webp_path, data, w, h, CHANNELS);
                }
            };

            if (!report_progress(0.10f, "Positions")) {
                return std::unexpected("Export cancelled");
            }

            // 1. Process positions with log transform
            std::vector<uint8_t> means_l(width * height * CHANNELS, 255);
            std::vector<uint8_t> means_u(width * height * CHANNELS, 255);

            // Get raw pointer for means data
            auto means_cpu = means.cpu();
            const float* means_ptr = means_cpu.ptr<float>();

            // Apply log transform and find min/max in parallel
            std::vector<float> means_log_data(num_splats * 3);
            float min_x = std::numeric_limits<float>::max();
            float max_x = std::numeric_limits<float>::lowest();
            float min_y = std::numeric_limits<float>::max();
            float max_y = std::numeric_limits<float>::lowest();
            float min_z = std::numeric_limits<float>::max();
            float max_z = std::numeric_limits<float>::lowest();

            // First pass: log transform (parallel)
            tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_splats),
                [&](const tbb::blocked_range<int64_t>& range) {
                    for (int64_t i = range.begin(); i < range.end(); ++i) {
                        means_log_data[i * 3 + 0] = log_transform(means_ptr[i * 3 + 0]);
                        means_log_data[i * 3 + 1] = log_transform(means_ptr[i * 3 + 1]);
                        means_log_data[i * 3 + 2] = log_transform(means_ptr[i * 3 + 2]);
                    }
                });

            // Find min/max (sequential but fast)
            for (int64_t i = 0; i < num_splats; ++i) {
                min_x = std::min(min_x, means_log_data[i * 3 + 0]);
                max_x = std::max(max_x, means_log_data[i * 3 + 0]);
                min_y = std::min(min_y, means_log_data[i * 3 + 1]);
                max_y = std::max(max_y, means_log_data[i * 3 + 1]);
                min_z = std::min(min_z, means_log_data[i * 3 + 2]);
                max_z = std::max(max_z, means_log_data[i * 3 + 2]);
            }

            // Precompute scale factors
            const float scale_x = 1.0f / (max_x - min_x + 1e-10f);
            const float scale_y = 1.0f / (max_y - min_y + 1e-10f);
            const float scale_z = 1.0f / (max_z - min_z + 1e-10f);

            // Quantize to 16-bit (parallel)
            tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_splats),
                [&](const tbb::blocked_range<int64_t>& range) {
                    for (int64_t i = range.begin(); i < range.end(); ++i) {
                        const int64_t idx = indices[i];
                        const int ti = i;  // identity_layout

                        float x = (means_log_data[idx * 3 + 0] - min_x) * scale_x;
                        float y = (means_log_data[idx * 3 + 1] - min_y) * scale_y;
                        float z = (means_log_data[idx * 3 + 2] - min_z) * scale_z;

                        uint16_t x16 = static_cast<uint16_t>(65535 * std::clamp(x, 0.0f, 1.0f));
                        uint16_t y16 = static_cast<uint16_t>(65535 * std::clamp(y, 0.0f, 1.0f));
                        uint16_t z16 = static_cast<uint16_t>(65535 * std::clamp(z, 0.0f, 1.0f));

                        means_l[ti * 4 + 0] = x16 & 0xff;
                        means_l[ti * 4 + 1] = y16 & 0xff;
                        means_l[ti * 4 + 2] = z16 & 0xff;

                        means_u[ti * 4 + 0] = (x16 >> 8) & 0xff;
                        means_u[ti * 4 + 1] = (y16 >> 8) & 0xff;
                        means_u[ti * 4 + 2] = (z16 >> 8) & 0xff;
                    }
                });

            if (!write_image("means_l.webp", means_l.data())) {
                return std::unexpected("Failed to write means_l.webp");
            }
            if (!write_image("means_u.webp", means_u.data())) {
                return std::unexpected("Failed to write means_u.webp");
            }

            if (!report_progress(0.25f, "Rotations")) {
                return std::unexpected("Export cancelled");
            }

            // 2. Process quaternions (parallel)
            std::vector<uint8_t> quats(width * height * CHANNELS, 255);
            auto rotations_cpu = rotations.cpu();
            const float* rotations_ptr = rotations_cpu.ptr<float>();

            tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_splats),
                [&](const tbb::blocked_range<int64_t>& range) {
                    for (int64_t i = range.begin(); i < range.end(); ++i) {
                        const int64_t idx = indices[i];
                        const int ti = i;

                        auto quat = pack_quaternion(
                            rotations_ptr[idx * 4 + 0],
                            rotations_ptr[idx * 4 + 1],
                            rotations_ptr[idx * 4 + 2],
                            rotations_ptr[idx * 4 + 3]);

                        quats[ti * 4 + 0] = quat[0];
                        quats[ti * 4 + 1] = quat[1];
                        quats[ti * 4 + 2] = quat[2];
                        quats[ti * 4 + 3] = quat[3];
                    }
                });

            if (!write_image("quats.webp", quats.data())) {
                return std::unexpected("Failed to write quats.webp");
            }

            // 3. Cluster scales using k-means
            if (!report_progress(0.35f, "Scales k-means")) {
                return std::unexpected("Export cancelled");
            }

            // Flatten scales in column-major order using raw pointers (parallel)
            auto scales_cpu = scales.cpu();
            const float* scales_ptr = scales_cpu.ptr<float>();
            std::vector<float> scales_flat_data(num_splats * 3);

            tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_splats),
                [&](const tbb::blocked_range<int64_t>& range) {
                    for (int64_t i = range.begin(); i < range.end(); ++i) {
                        scales_flat_data[i] = scales_ptr[i * 3 + 0];
                        scales_flat_data[num_splats + i] = scales_ptr[i * 3 + 1];
                        scales_flat_data[2 * num_splats + i] = scales_ptr[i * 3 + 2];
                    }
                });

            auto scales_flat = Tensor::from_vector(scales_flat_data, {static_cast<size_t>(num_splats * 3)}, Device::CUDA);

            auto [scales_centroids, scales_labels] = cuda::kmeans_1d_new(
                scales_flat, 256, options.iterations);

            std::vector<uint8_t> scales_data(width * height * CHANNELS, 255);
            auto scales_labels_cpu = scales_labels.cpu();
            const int32_t* scales_labels_ptr = scales_labels_cpu.ptr<int32_t>();

            tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_splats),
                [&](const tbb::blocked_range<int64_t>& range) {
                    for (int64_t i = range.begin(); i < range.end(); ++i) {
                        const int64_t idx = indices[i];
                        const int ti = i;

                        scales_data[ti * 4 + 0] = static_cast<uint8_t>(scales_labels_ptr[idx]);
                        scales_data[ti * 4 + 1] = static_cast<uint8_t>(scales_labels_ptr[num_splats + idx]);
                        scales_data[ti * 4 + 2] = static_cast<uint8_t>(scales_labels_ptr[2 * num_splats + idx]);
                    }
                });

            if (!write_image("scales.webp", scales_data.data())) {
                return std::unexpected("Failed to write scales.webp");
            }

            // 4. Cluster colors using k-means
            if (!report_progress(0.50f, "Colors k-means")) {
                return std::unexpected("Export cancelled");
            }

            auto sh0_reshaped = sh0.reshape({static_cast<int>(num_splats), 3});
            auto sh0_cpu = sh0_reshaped.cpu();
            const float* sh0_ptr = sh0_cpu.ptr<float>();

            // Flatten in column-major order (parallel)
            std::vector<float> colors_flat_data(num_splats * 3);
            tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_splats),
                [&](const tbb::blocked_range<int64_t>& range) {
                    for (int64_t i = range.begin(); i < range.end(); ++i) {
                        colors_flat_data[i] = sh0_ptr[i * 3 + 0];
                        colors_flat_data[num_splats + i] = sh0_ptr[i * 3 + 1];
                        colors_flat_data[2 * num_splats + i] = sh0_ptr[i * 3 + 2];
                    }
                });

            auto colors_1d = Tensor::from_vector(colors_flat_data, {static_cast<size_t>(num_splats * 3)}, Device::CUDA);

            auto [colors_centroids, colors_labels] = cuda::kmeans_1d_new(
                colors_1d, 256, options.iterations);

            std::vector<uint8_t> sh0_data(width * height * CHANNELS, 0);
            auto colors_labels_cpu = colors_labels.cpu();
            const int32_t* colors_labels_ptr = colors_labels_cpu.ptr<int32_t>();
            auto opacities_cpu = opacities.cpu();
            const float* opacities_ptr = opacities_cpu.ptr<float>();

            tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_splats),
                [&](const tbb::blocked_range<int64_t>& range) {
                    for (int64_t i = range.begin(); i < range.end(); ++i) {
                        const int64_t idx = indices[i];
                        const int ti = i;

                        sh0_data[ti * 4 + 0] = static_cast<uint8_t>(colors_labels_ptr[idx]);
                        sh0_data[ti * 4 + 1] = static_cast<uint8_t>(colors_labels_ptr[num_splats + idx]);
                        sh0_data[ti * 4 + 2] = static_cast<uint8_t>(colors_labels_ptr[2 * num_splats + idx]);

                        // IMPORTANT: Never use alpha=0 because WebP "lossless" discards RGB values
                        // for fully transparent pixels. Clamp to at least 1/255 to preserve indices.
                        float opacity = opacities_ptr[idx];
                        sh0_data[ti * 4 + 3] = static_cast<uint8_t>(std::max(1.0f, 255.0f * opacity));
                    }
                });

            if (!write_image("sh0.webp", sh0_data.data())) {
                return std::unexpected("Failed to write sh0.webp");
            }

            // Create meta.json
            nlohmann::json meta;
            meta["version"] = 2;
            meta["count"] = num_splats;
            meta["width"] = width;
            meta["height"] = height;

            // Store means min/max (already cached above)
            meta["means"]["mins"] = {min_x, min_y, min_z};
            meta["means"]["maxs"] = {max_x, max_y, max_z};
            meta["means"]["files"] = {"means_l.webp", "means_u.webp"};

            // Convert scale centroids to vector using raw pointers
            auto scales_centroids_cpu = scales_centroids.cpu();
            const float* scales_centroids_ptr = scales_centroids_cpu.ptr<float>();
            std::vector<float> scale_codebook(scales_centroids.size(0));
            for (size_t i = 0; i < scales_centroids.size(0); ++i) {
                scale_codebook[i] = scales_centroids_ptr[i];
            }
            meta["scales"]["codebook"] = scale_codebook;
            meta["scales"]["files"] = {"scales.webp"};

            meta["quats"]["files"] = {"quats.webp"};

            // Convert color centroids to vector using raw pointers
            auto colors_centroids_cpu = colors_centroids.cpu();
            const float* colors_centroids_ptr = colors_centroids_cpu.ptr<float>();
            std::vector<float> color_codebook(colors_centroids.size(0));
            for (size_t i = 0; i < colors_centroids.size(0); ++i) {
                color_codebook[i] = colors_centroids_ptr[i];
            }

            meta["sh0"]["codebook"] = color_codebook;
            meta["sh0"]["files"] = {"sh0.webp"};

            // Handle higher-order spherical harmonics if present
            if (sh_degree > 0 && shN.is_valid() && shN.numel() > 0) {
                if (!report_progress(0.65f, "SH k-means")) {
                    return std::unexpected("Export cancelled");
                }

                // shN shape is [N, sh_coeffs, 3] where sh_coeffs is number of SH coefficients
                const int sh_coeffs = shN.size(1);

                // Flatten SH coefficients for clustering: [N, sh_coeffs, 3] -> [N, sh_coeffs * 3]
                auto shN_reshaped = shN.reshape({static_cast<int>(num_splats), sh_coeffs * 3});

                // Calculate palette size
                int palette_size = std::min(64,
                                            std::max(1, static_cast<int>(std::pow(2, std::floor(std::log2(num_splats / 1024.0)))) * 1024));
                palette_size = std::min(palette_size, static_cast<int>(num_splats));

                // Cluster SH coefficients
                auto [sh_centroids, sh_labels] = cuda::kmeans_new(
                    shN_reshaped, palette_size, options.iterations);

                if (sh_centroids.size(0) == 0) {
                    LOG_WARN("SH clustering empty, skipping");
                } else {
                    const int actual_palette_size = sh_centroids.size(0);

                    // Further cluster the centroids to create codebook
                    auto [codebook_centroids, codebook_labels] = cuda::kmeans_1d_new(
                        sh_centroids.flatten(), 256, options.iterations);

                    // Calculate dimensions for centroids texture
                    const int centroids_width = 64 * sh_coeffs;
                    const int centroids_height = (actual_palette_size + 63) / 64;

                    // Write centroids with proper band-major ordering (parallel)
                    std::vector<uint8_t> centroids_buf(centroids_width * centroids_height * CHANNELS, 255);
                    auto codebook_labels_cpu = codebook_labels.cpu();
                    const int32_t* codebook_labels_ptr = codebook_labels_cpu.ptr<int32_t>();
                    const int64_t codebook_labels_size = codebook_labels.size(0);

                    tbb::parallel_for(tbb::blocked_range<int>(0, actual_palette_size),
                        [&](const tbb::blocked_range<int>& range) {
                            for (int i = range.begin(); i < range.end(); ++i) {
                                for (int j = 0; j < sh_coeffs; ++j) {
                                    int pixel_idx = i * sh_coeffs + j;

                                    if (pixel_idx < centroids_width * centroids_height) {
                                        for (int c = 0; c < 3; ++c) {
                                            int coeff_idx = j + c * sh_coeffs;
                                            int centroid_idx = i * (sh_coeffs * 3) + coeff_idx;

                                            if (centroid_idx < codebook_labels_size) {
                                                centroids_buf[pixel_idx * 4 + c] =
                                                    static_cast<uint8_t>(codebook_labels_ptr[centroid_idx]);
                                            }
                                        }
                                    }
                                }
                            }
                        });

                    if (!write_image("shN_centroids.webp", centroids_buf.data(), centroids_width, centroids_height)) {
                        return std::unexpected("Failed to write shN_centroids.webp");
                    }

                    // Write labels (parallel)
                    std::vector<uint8_t> labels_buf(width * height * CHANNELS, 255);
                    auto sh_labels_cpu = sh_labels.cpu();
                    const int32_t* sh_labels_ptr = sh_labels_cpu.ptr<int32_t>();

                    tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_splats),
                        [&](const tbb::blocked_range<int64_t>& range) {
                            for (int64_t i = range.begin(); i < range.end(); ++i) {
                                const int64_t idx = indices[i];
                                const int32_t label = sh_labels_ptr[idx];
                                const int ti = i;

                                labels_buf[ti * 4 + 0] = label & 0xff;
                                labels_buf[ti * 4 + 1] = (label >> 8) & 0xff;
                                labels_buf[ti * 4 + 2] = 0;
                            }
                        });

                    if (!write_image("shN_labels.webp", labels_buf.data())) {
                        return std::unexpected("Failed to write shN_labels.webp");
                    }

                    // Add to meta.json (use raw pointers)
                    auto codebook_centroids_cpu = codebook_centroids.cpu();
                    const float* codebook_centroids_ptr = codebook_centroids_cpu.ptr<float>();
                    int codebook_size = std::min(256, static_cast<int>(codebook_centroids.size(0)));
                    std::vector<float> sh_codebook(codebook_size);
                    for (int i = 0; i < codebook_size; ++i) {
                        sh_codebook[i] = codebook_centroids_ptr[i];
                    }

                    meta["shN"]["codebook"] = sh_codebook;
                    meta["shN"]["palette_size"] = actual_palette_size;
                    meta["shN"]["bands"] = sh_degree;
                    meta["shN"]["coeffs"] = sh_coeffs;
                    meta["shN"]["files"] = {"shN_centroids.webp", "shN_labels.webp"};
                }
            }

            // Write meta.json
            if (!report_progress(0.90f, "Writing")) {
                return std::unexpected("Export cancelled");
            }
            const std::string meta_json = meta.dump(2);

            if (archive) {
                if (!archive->add_file("meta.json", meta_json.c_str(), meta_json.size())) {
                    return std::unexpected("Failed to write meta.json to archive");
                }
            } else {
                auto meta_path = options.output_path;
                if (meta_path.extension() != ".json") {
                    meta_path = base_path / "meta.json";
                }

                std::ofstream meta_file(meta_path);
                if (!meta_file) {
                    return std::unexpected("Failed to open meta.json");
                }
                meta_file << meta_json;
                if (!meta_file) {
                    return std::unexpected("Failed to write meta.json");
                }
            }

            LOG_INFO("SOG export: {} splats to {}", num_splats, options.output_path.string());
            report_progress(1.0f, "Complete");
            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::format("SOG export failed: {}", e.what()));
        }
    }
} // namespace lfs::core