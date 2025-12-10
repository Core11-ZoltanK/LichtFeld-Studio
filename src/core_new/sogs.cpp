/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef _WIN32
#define NOMINMAX
#endif

#include "core_new/sogs.hpp"
#include "core_new/logger.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include "core_new/cuda/kernels/kdtree_kmeans.hpp"
#include "kernels/morton_encoding_new.cuh"

using lfs::core::Tensor;
using lfs::core::Device;
using lfs::core::DataType;
#include <algorithm>
#include <archive.h>
#include <archive_entry.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <webp/encode.h>

namespace lfs::core {

namespace {

#ifdef _WIN32
    using ssize_t = std::ptrdiff_t;
#endif

    // Sign-preserving log transform
    double log_transform(const double value) {
        return std::copysign(std::log(std::abs(value) + 1.0), value);
    }

    double sigmoid(const double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Morton encoding helpers
    inline uint32_t Part1By2_cpu(const uint32_t x_in) {
        uint32_t x = x_in;
        x &= 0x000003ff;
        x = (x ^ (x << 16)) & 0xff0000ff;
        x = (x ^ (x <<  8)) & 0x0300f00f;
        x = (x ^ (x <<  4)) & 0x030c30c3;
        x = (x ^ (x <<  2)) & 0x09249249;
        return x;
    }

    inline uint32_t encodeMorton3_cpu(const uint32_t x, const uint32_t y, const uint32_t z) {
        return (Part1By2_cpu(z) << 2) + (Part1By2_cpu(y) << 1) + Part1By2_cpu(x);
    }

    // Recursive Morton ordering
    void generate_ordering_recursive(
        std::vector<int64_t>& indices,
        size_t start, size_t end,
        const float* positions) {

        const size_t count = end - start;
        if (count <= 1) return;

        double mx = std::numeric_limits<double>::infinity();
        double my = std::numeric_limits<double>::infinity();
        double mz = std::numeric_limits<double>::infinity();
        double Mx = -std::numeric_limits<double>::infinity();
        double My = -std::numeric_limits<double>::infinity();
        double Mz = -std::numeric_limits<double>::infinity();

        for (size_t i = start; i < end; ++i) {
            int64_t ri = indices[i];
            double x = static_cast<double>(positions[ri * 3 + 0]);
            double y = static_cast<double>(positions[ri * 3 + 1]);
            double z = static_cast<double>(positions[ri * 3 + 2]);

            mx = std::min(mx, x);
            my = std::min(my, y);
            mz = std::min(mz, z);
            Mx = std::max(Mx, x);
            My = std::max(My, y);
            Mz = std::max(Mz, z);
        }

        double xlen = Mx - mx;
        double ylen = My - my;
        double zlen = Mz - mz;

        if (xlen == 0 && ylen == 0 && zlen == 0) {
            return;
        }

        double xmul = (xlen == 0) ? 0.0 : 1024.0 / xlen;
        double ymul = (ylen == 0) ? 0.0 : 1024.0 / ylen;
        double zmul = (zlen == 0) ? 0.0 : 1024.0 / zlen;

        std::vector<uint32_t> morton(count);
        for (size_t i = 0; i < count; ++i) {
            int64_t ri = indices[start + i];
            double x = static_cast<double>(positions[ri * 3 + 0]);
            double y = static_cast<double>(positions[ri * 3 + 1]);
            double z = static_cast<double>(positions[ri * 3 + 2]);

            const uint32_t ix = std::min(1023u, static_cast<uint32_t>(std::floor((x - mx) * xmul)));
            const uint32_t iy = std::min(1023u, static_cast<uint32_t>(std::floor((y - my) * ymul)));
            const uint32_t iz = std::min(1023u, static_cast<uint32_t>(std::floor((z - mz) * zmul)));

            morton[i] = encodeMorton3_cpu(ix, iy, iz);
        }

        std::vector<size_t> order(count);
        for (size_t i = 0; i < count; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return morton[a] < morton[b];
        });

        std::vector<int64_t> tmp_indices(count);
        for (size_t i = 0; i < count; ++i) {
            tmp_indices[i] = indices[start + order[i]];
        }
        for (size_t i = 0; i < count; ++i) {
            indices[start + i] = tmp_indices[i];
        }

        size_t bucket_start = 0;
        while (bucket_start < count) {
            size_t bucket_end = bucket_start + 1;
            while (bucket_end < count && morton[order[bucket_end]] == morton[order[bucket_start]]) {
                ++bucket_end;
            }

            if (bucket_end - bucket_start > 256) {
                generate_ordering_recursive(indices, start + bucket_start, start + bucket_end, positions);
            }

            bucket_start = bucket_end;
        }
    }

    void generate_ordering(std::vector<int64_t>& indices, const float* positions, const size_t count) {
        for (size_t i = 0; i < count; ++i) {
            indices[i] = static_cast<int64_t>(i);
        }
        generate_ordering_recursive(indices, 0, count, positions);
    }

    // ZIP archive writer
    class SogArchive {
        struct archive* a;
    public:
        SogArchive(const std::filesystem::path& output_path) {
            a = archive_write_new();
            archive_write_set_format_zip(a);
            archive_write_open_filename(a, output_path.string().c_str());
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
                return false;
            }

            if (archive_write_data(a, data, size) != static_cast<ssize_t>(size)) {
                archive_entry_free(entry);
                return false;
            }

            archive_entry_free(entry);
            return true;
        }

        bool add_webp(const std::string& filename, const uint8_t* data, int width, int height) {
            uint8_t* output = nullptr;
            size_t output_size = WebPEncodeLosslessRGBA(data, width, height, width * 4, &output);

            if (output_size == 0 || !output) {
                if (output) WebPFree(output);
                return false;
            }

            bool result = add_file(filename, output, output_size);
            WebPFree(output);
            return result;
        }
    };

    // 1D k-means clustering with 256 clusters
    struct Cluster1dResult {
        std::vector<float> centroids;  // 256 ordered centroids
        std::vector<uint8_t> labels;   // Labels for each input point
    };

    Cluster1dResult cluster1d(const float* data, const int num_rows, const int num_columns, const int iterations) {
        const int total_points = num_rows * num_columns;
        std::vector<float> flat_data(total_points);
        for (int col = 0; col < num_columns; ++col) {
            for (int row = 0; row < num_rows; ++row) {
                flat_data[col * num_rows + row] = data[row * num_columns + col];
            }
        }

        auto data_tensor = Tensor::from_blob(flat_data.data(), {static_cast<size_t>(total_points), 1}, Device::CPU, DataType::Float32).cuda();
        auto [centroids_tensor, labels_tensor] = cuda::kmeans_kdtree(data_tensor, 256, iterations);

        auto centroids_cpu = centroids_tensor.cpu();
        auto labels_cpu = labels_tensor.cpu();
        const float* centroids_ptr = static_cast<const float*>(centroids_cpu.data_ptr());
        const int32_t* labels_ptr = static_cast<const int32_t*>(labels_cpu.data_ptr());

        std::vector<int> order(256);
        for (int i = 0; i < 256; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return centroids_ptr[a] < centroids_ptr[b];
        });

        std::vector<float> ordered_centroids(256);
        for (int i = 0; i < 256; ++i) {
            ordered_centroids[i] = centroids_ptr[order[i]];
        }

        std::vector<int> inv_order(256);
        for (int i = 0; i < 256; ++i) {
            inv_order[order[i]] = i;
        }

        Cluster1dResult result;
        result.centroids = ordered_centroids;
        result.labels.resize(total_points);
        for (int i = 0; i < total_points; ++i) {
            result.labels[i] = static_cast<uint8_t>(inv_order[labels_ptr[i]]);
        }

        return result;
    }

} // anonymous namespace

std::expected<void, std::string> write_sog(
    const SplatData& splat_data,
    const SogWriteOptions& options) {

    try {
        LOG_INFO("SOG write: {}", options.output_path.string());

        auto timer_start = std::chrono::high_resolution_clock::now();
        auto lap = [&](const char* stage) {
            auto now = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration<double, std::milli>(now - timer_start).count();
            timer_start = now;
            LOG_PERF("SOG {}: {:.1f} ms", stage, ms);
        };

        const auto report_progress = [&](float progress, const std::string& stage) -> bool {
            return !options.progress_callback || options.progress_callback(progress, stage);
        };

        if (!report_progress(0.0f, "Initializing")) {
            return std::unexpected("Export cancelled");
        }

        const int64_t num_rows = splat_data.size();
        if (num_rows == 0) {
            return std::unexpected("No splats to write");
        }

        // Texture dimensions
        const int width = static_cast<int>(std::ceil(std::sqrt(num_rows) / 4.0)) * 4;
        const int height = static_cast<int>(std::ceil(static_cast<double>(num_rows) / width / 4.0)) * 4;
        constexpr int CHANNELS = 4;

        LOG_DEBUG("SOG: {}x{} for {} splats", width, height, num_rows);

        lap("init");
        auto means_cuda = splat_data.means_raw().cuda();
        auto morton_codes = morton_encode_new(means_cuda);
        auto sort_indices_tensor = morton_sort_indices_new(morton_codes);
        auto sort_indices_cpu = sort_indices_tensor.cpu();
        const int64_t* indices = sort_indices_cpu.ptr<int64_t>();
        lap("morton");

        auto means_cpu = means_cuda.cpu();

        SogArchive archive(options.output_path);

        const auto write_webp = [&](const std::string& filename, const uint8_t* data, const int w, const int h) -> bool {
            LOG_INFO("writing '{}'...", filename);
            return archive.add_webp(filename, data, w, h);
        };

        if (!report_progress(0.10f, "Positions")) {
            return std::unexpected("Export cancelled");
        }

        const float* means_ptr = means_cpu.ptr<float>();

        std::array<std::array<double, 2>, 3> means_min_max = {{
            {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()},
            {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()},
            {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}
        }};

        for (int64_t i = 0; i < num_rows; ++i) {
            int64_t idx = indices[i];
            for (int d = 0; d < 3; ++d) {
                double v = log_transform(static_cast<double>(means_ptr[idx * 3 + d]));
                means_min_max[d][0] = std::min(means_min_max[d][0], v);
                means_min_max[d][1] = std::max(means_min_max[d][1], v);
            }
        }

        std::vector<uint8_t> means_l(width * height * CHANNELS, 0);
        std::vector<uint8_t> means_u(width * height * CHANNELS, 0);

        for (int64_t i = 0; i < num_rows; ++i) {
            int64_t idx = indices[i];

            const double x = 65535.0 * (log_transform(static_cast<double>(means_ptr[idx * 3 + 0])) - means_min_max[0][0]) /
                      (means_min_max[0][1] - means_min_max[0][0]);
            const double y = 65535.0 * (log_transform(static_cast<double>(means_ptr[idx * 3 + 1])) - means_min_max[1][0]) /
                      (means_min_max[1][1] - means_min_max[1][0]);
            const double z = 65535.0 * (log_transform(static_cast<double>(means_ptr[idx * 3 + 2])) - means_min_max[2][0]) /
                      (means_min_max[2][1] - means_min_max[2][0]);

            const uint16_t x16 = static_cast<uint16_t>(std::clamp(x, 0.0, 65535.0));
            const uint16_t y16 = static_cast<uint16_t>(std::clamp(y, 0.0, 65535.0));
            const uint16_t z16 = static_cast<uint16_t>(std::clamp(z, 0.0, 65535.0));

            const int ti = static_cast<int>(i);
            means_l[ti * 4 + 0] = x16 & 0xff;
            means_l[ti * 4 + 1] = y16 & 0xff;
            means_l[ti * 4 + 2] = z16 & 0xff;
            means_l[ti * 4 + 3] = 0xff;

            means_u[ti * 4 + 0] = (x16 >> 8) & 0xff;
            means_u[ti * 4 + 1] = (y16 >> 8) & 0xff;
            means_u[ti * 4 + 2] = (z16 >> 8) & 0xff;
            means_u[ti * 4 + 3] = 0xff;
        }

        lap("means_encode");
        if (!write_webp("means_l.webp", means_l.data(), width, height)) {
            return std::unexpected("Failed to write means_l.webp");
        }
        if (!write_webp("means_u.webp", means_u.data(), width, height)) {
            return std::unexpected("Failed to write means_u.webp");
        }
        lap("means_webp");

        if (!report_progress(0.20f, "Rotations")) {
            return std::unexpected("Export cancelled");
        }

        auto rotations = splat_data.rotation_raw().cpu();
        const float* rot_ptr = rotations.ptr<float>();

        std::vector<uint8_t> quats(width * height * CHANNELS, 0);

        for (int64_t i = 0; i < num_rows; ++i) {
            int64_t idx = indices[i];

            float q[4] = {
                rot_ptr[idx * 4 + 0],
                rot_ptr[idx * 4 + 1],
                rot_ptr[idx * 4 + 2],
                rot_ptr[idx * 4 + 3]
            };

            // Normalize
            float len = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
            for (int j = 0; j < 4; ++j) q[j] /= len;

            // Find max component
            int max_comp = 0;
            for (int j = 1; j < 4; ++j) {
                if (std::abs(q[j]) > std::abs(q[max_comp])) max_comp = j;
            }

            // Invert if max component is negative
            if (q[max_comp] < 0) {
                for (int j = 0; j < 4; ++j) q[j] *= -1;
            }

            // Scale by sqrt(2)
            constexpr float sqrt2 = 1.41421356237f;
            for (int j = 0; j < 4; ++j) q[j] *= sqrt2;

            // Get indices of other 3 components
            static const int idx_table[4][3] = {{1,2,3}, {0,2,3}, {0,1,3}, {0,1,2}};
            const int* other_idx = idx_table[max_comp];

            int ti = static_cast<int>(i);
            quats[ti * 4 + 0] = static_cast<uint8_t>(255.0f * (q[other_idx[0]] * 0.5f + 0.5f));
            quats[ti * 4 + 1] = static_cast<uint8_t>(255.0f * (q[other_idx[1]] * 0.5f + 0.5f));
            quats[ti * 4 + 2] = static_cast<uint8_t>(255.0f * (q[other_idx[2]] * 0.5f + 0.5f));
            quats[ti * 4 + 3] = 252 + max_comp;
        }

        lap("quats_encode");
        if (!write_webp("quats.webp", quats.data(), width, height)) {
            return std::unexpected("Failed to write quats.webp");
        }
        lap("quats_webp");

        if (!report_progress(0.30f, "Scales k-means")) {
            return std::unexpected("Export cancelled");
        }

        auto scales = splat_data.scaling_raw().cpu();
        const float* scales_ptr = scales.ptr<float>();

        LOG_INFO("Running k-means clustering: dims=1 points={} clusters=256 iterations={}...",
                 num_rows * 3, options.iterations);

        auto scale_result = cluster1d(scales_ptr, num_rows, 3, options.iterations);
        lap("scales_kmeans");

        std::vector<uint8_t> scales_data(width * height * CHANNELS, 0);
        for (int64_t i = 0; i < num_rows; ++i) {
            int64_t idx = indices[i];
            int ti = static_cast<int>(i);

            scales_data[ti * 4 + 0] = scale_result.labels[0 * num_rows + idx];  // scale_0
            scales_data[ti * 4 + 1] = scale_result.labels[1 * num_rows + idx];  // scale_1
            scales_data[ti * 4 + 2] = scale_result.labels[2 * num_rows + idx];  // scale_2
            scales_data[ti * 4 + 3] = 0xff;
        }

        if (!write_webp("scales.webp", scales_data.data(), width, height)) {
            return std::unexpected("Failed to write scales.webp");
        }
        lap("scales_webp");

        if (!report_progress(0.45f, "Colors k-means")) {
            return std::unexpected("Export cancelled");
        }

        auto sh0 = splat_data.sh0_raw().cpu();
        const float* sh0_ptr = sh0.ptr<float>();

        LOG_INFO("Running k-means clustering: dims=1 points={} clusters=256 iterations={}...",
                 num_rows * 3, options.iterations);

        auto color_result = cluster1d(sh0_ptr, num_rows, 3, options.iterations);
        lap("sh0_kmeans");

        // Get opacity (apply sigmoid)
        auto opacity = splat_data.opacity_raw().cpu();
        const float* opacity_ptr = opacity.ptr<float>();

        std::vector<uint8_t> sh0_data(width * height * CHANNELS, 0);
        for (int64_t i = 0; i < num_rows; ++i) {
            int64_t idx = indices[i];
            int ti = static_cast<int>(i);

            sh0_data[ti * 4 + 0] = color_result.labels[0 * num_rows + idx];  // f_dc_0
            sh0_data[ti * 4 + 1] = color_result.labels[1 * num_rows + idx];  // f_dc_1
            sh0_data[ti * 4 + 2] = color_result.labels[2 * num_rows + idx];  // f_dc_2
            // Use double precision to match JavaScript's floating point behavior
            sh0_data[ti * 4 + 3] = static_cast<uint8_t>(
                std::max(0.0, std::min(255.0, sigmoid(static_cast<double>(opacity_ptr[idx])) * 255.0)));
        }

        if (!write_webp("sh0.webp", sh0_data.data(), width, height)) {
            return std::unexpected("Failed to write sh0.webp");
        }
        lap("sh0_webp");

        const int sh_degree = splat_data.get_max_sh_degree();
        nlohmann::json sh_n_meta;

        if (sh_degree > 0) {
            if (!report_progress(0.60f, "SH k-means")) {
                return std::unexpected("Export cancelled");
            }

            auto shN = splat_data.shN_raw().cpu();
            const float* shN_ptr = shN.ptr<float>();

            static const int sh_coeffs_table[] = {0, 3, 8, 15};
            const int sh_coeffs = sh_coeffs_table[sh_degree];
            const int sh_dims = sh_coeffs * 3;

            int palette_size = std::min(64, static_cast<int>(std::pow(2, std::floor(std::log2(num_rows / 1024.0))))) * 1024;
            palette_size = std::min(palette_size, static_cast<int>(num_rows));
            palette_size = std::max(palette_size, 1024);

            LOG_INFO("Running k-means clustering: dims={} points={} clusters={} iterations={}...",
                     sh_dims, num_rows, palette_size, options.iterations);

            std::vector<float> shN_flat(num_rows * sh_dims);
            for (int64_t i = 0; i < num_rows; ++i) {
                for (int c = 0; c < 3; ++c) {
                    for (int j = 0; j < sh_coeffs; ++j) {
                        const int their_col = c * sh_coeffs + j;
                        shN_flat[i * sh_dims + their_col] = shN_ptr[i * sh_coeffs * 3 + j * 3 + c];
                    }
                }
            }

            auto shN_tensor = Tensor::from_blob(shN_flat.data(),
                {static_cast<size_t>(num_rows), static_cast<size_t>(sh_dims)}, Device::CPU, DataType::Float32).cuda();
            auto [sh_centroids, sh_labels] = cuda::kmeans_kdtree(shN_tensor, palette_size, options.iterations);
            lap("shN_kmeans");

            auto sh_centroids_cpu = sh_centroids.cpu();
            const float* sh_centroids_ptr = static_cast<const float*>(sh_centroids_cpu.data_ptr());
            const int actual_palette_size = sh_centroids.size(0);

            LOG_INFO("Running k-means clustering: dims=1 points={} clusters=256 iterations={}...",
                     actual_palette_size * sh_dims, options.iterations);

            auto codebook_result = cluster1d(sh_centroids_ptr, actual_palette_size, sh_dims, options.iterations);
            lap("shN_codebook_kmeans");

            const int centroids_width = 64 * sh_coeffs;
            const int centroids_height = (actual_palette_size + 63) / 64;

            std::vector<uint8_t> centroids_buf(centroids_width * centroids_height * CHANNELS, 0);

            for (int i = 0; i < actual_palette_size; ++i) {
                for (int j = 0; j < sh_coeffs; ++j) {
                    const int pixel_idx = i * sh_coeffs + j;
                    for (int c = 0; c < 3; ++c) {
                        const int col_idx = sh_coeffs * c + j;
                        const int label_idx = col_idx * actual_palette_size + i;
                        centroids_buf[pixel_idx * 4 + c] = codebook_result.labels[label_idx];
                    }
                    centroids_buf[pixel_idx * 4 + 3] = 0xff;
                }
            }

            if (!write_webp("shN_centroids.webp", centroids_buf.data(), centroids_width, centroids_height)) {
                return std::unexpected("Failed to write shN_centroids.webp");
            }

            auto sh_labels_cpu = sh_labels.cpu();
            const int32_t* sh_labels_ptr = static_cast<const int32_t*>(sh_labels_cpu.data_ptr());

            std::vector<uint8_t> labels_buf(width * height * CHANNELS, 0);
            for (int64_t i = 0; i < num_rows; ++i) {
                const int64_t idx = indices[i];
                const int32_t label = sh_labels_ptr[idx];
                const int ti = static_cast<int>(i);

                labels_buf[ti * 4 + 0] = label & 0xff;
                labels_buf[ti * 4 + 1] = (label >> 8) & 0xff;
                labels_buf[ti * 4 + 2] = 0;
                labels_buf[ti * 4 + 3] = 0xff;
            }

            if (!write_webp("shN_labels.webp", labels_buf.data(), width, height)) {
                return std::unexpected("Failed to write shN_labels.webp");
            }
            lap("shN_webp");

            sh_n_meta["count"] = actual_palette_size;
            sh_n_meta["bands"] = sh_degree;
            sh_n_meta["codebook"] = codebook_result.centroids;
            sh_n_meta["files"] = {"shN_centroids.webp", "shN_labels.webp"};
        }

        if (!report_progress(0.90f, "Writing meta")) {
            return std::unexpected("Export cancelled");
        }

        nlohmann::json meta;
        meta["version"] = 2;
        meta["asset"]["generator"] = "LichtFeld Studio";
        meta["count"] = num_rows;

        meta["means"]["mins"] = {means_min_max[0][0], means_min_max[1][0], means_min_max[2][0]};
        meta["means"]["maxs"] = {means_min_max[0][1], means_min_max[1][1], means_min_max[2][1]};
        meta["means"]["files"] = {"means_l.webp", "means_u.webp"};

        meta["scales"]["codebook"] = scale_result.centroids;
        meta["scales"]["files"] = {"scales.webp"};

        meta["quats"]["files"] = {"quats.webp"};

        meta["sh0"]["codebook"] = color_result.centroids;
        meta["sh0"]["files"] = {"sh0.webp"};

        if (sh_degree > 0) {
            meta["shN"] = sh_n_meta;
        }

        std::string meta_json = meta.dump();
        if (!archive.add_file("meta.json", meta_json.c_str(), meta_json.size())) {
            return std::unexpected("Failed to write meta.json");
        }

        LOG_INFO("SOG export complete: {} splats", num_rows);
        report_progress(1.0f, "Complete");
        return {};

    } catch (const std::exception& e) {
        return std::unexpected(std::format("SOG export failed: {}", e.what()));
    }
}

} // namespace lfs::core
