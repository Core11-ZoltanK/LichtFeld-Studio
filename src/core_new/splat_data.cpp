/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/splat_data.hpp"
#include "core_new/logger.hpp"
#include "core_new/parameters.hpp"
#include "core_new/point_cloud.hpp"
#include "core_new/sogs.hpp"
#include "geometry_new/bounding_box.hpp"
#include "external/nanoflann.hpp"
#include "external/tinyply.hpp"
#include <iostream>

#include <algorithm>
#include <cmath>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <numeric> // for std::iota
#include <print>
#include <random> // for std::mt19937
#include <string>
#include <thread>
#include <vector>

namespace {

    // Point cloud adaptor for nanoflann
    struct PointCloudAdaptor {
        const float* points;
        size_t num_points;

        PointCloudAdaptor(const float* pts, size_t n)
            : points(pts),
              num_points(n) {}

        inline size_t kdtree_get_point_count() const { return num_points; }

        inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
            return points[idx * 3 + dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    };

    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>,
        PointCloudAdaptor,
        3>;

    /**
     * @brief Compute mean distance to 3 nearest neighbors for each point
     */
    lfs::core::Tensor compute_mean_neighbor_distances(const lfs::core::Tensor& points) {
        auto cpu_points = points.cpu();
        const int num_points = cpu_points.size(0);

        if (cpu_points.ndim() != 2 || cpu_points.size(1) != 3) {
            LOG_ERROR("Input points must have shape [N, 3], got {}", cpu_points.shape().str());
            return lfs::core::Tensor();
        }

        if (cpu_points.dtype() != lfs::core::DataType::Float32) {
            LOG_ERROR("Input points must be float32");
            return lfs::core::Tensor();
        }

        if (num_points <= 1) {
            return lfs::core::Tensor::full({static_cast<size_t>(num_points)}, 0.01f, points.device());
        }

        const float* data = cpu_points.ptr<float>();

        PointCloudAdaptor cloud(data, num_points);
        KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index.buildIndex();

        auto result = lfs::core::Tensor::zeros({static_cast<size_t>(num_points)}, lfs::core::Device::CPU);
        float* result_data = result.ptr<float>();

#pragma omp parallel for if (num_points > 1000)
        for (int i = 0; i < num_points; i++) {
            const float query_pt[3] = {
                data[i * 3 + 0],
                data[i * 3 + 1],
                data[i * 3 + 2]};

            const size_t num_results = std::min(4, num_points);
            std::vector<size_t> ret_indices(num_results);
            std::vector<float> out_dists_sqr(num_results);

            nanoflann::KNNResultSet<float> resultSet(num_results);
            resultSet.init(&ret_indices[0], &out_dists_sqr[0]);
            index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters(10));

            float sum_dist = 0.0f;
            int valid_neighbors = 0;

            for (size_t j = 0; j < num_results && valid_neighbors < 3; j++) {
                if (out_dists_sqr[j] > 1e-8f) {
                    sum_dist += std::sqrt(out_dists_sqr[j]);
                    valid_neighbors++;
                }
            }

            result_data[i] = (valid_neighbors > 0) ? (sum_dist / valid_neighbors) : 0.01f;
        }

        return result.to(points.device());
    }

    /**
     * @brief Write PLY file implementation
     */
    void write_ply_impl(const lfs::core::PointCloud& pc,
                        const std::filesystem::path& root,
                        int iteration,
                        const std::string& stem) {
        namespace fs = std::filesystem;
        fs::create_directories(root);

        // Collect all tensors and convert to CPU
        std::vector<lfs::core::Tensor> tensors;
        tensors.push_back(pc.means.cpu().contiguous());

        if (pc.normals.is_valid()) {
            tensors.push_back(pc.normals.cpu().contiguous());
        }

        if (pc.sh0.is_valid()) {
            LOG_INFO("write_ply_impl: pc.sh0 shape: ndim={}, shape=[{}{}{}]",
                     pc.sh0.ndim(),
                     pc.sh0.shape()[0],
                     pc.sh0.ndim() >= 2 ? fmt::format(", {}", pc.sh0.shape()[1]) : "",
                     pc.sh0.ndim() >= 3 ? fmt::format(", {}", pc.sh0.shape()[2]) : "");

            if (pc.sh0.ndim() == 3) {
                // sh0 is [N, B, 3] - transpose to [N, 3, B], then flatten to [N, 3*B]
                LOG_INFO("write_ply_impl: sh0 is 3D, transposing and flattening");
                auto sh0_transposed = pc.sh0.transpose(1, 2).contiguous();
                tensors.push_back(sh0_transposed.flatten(1).cpu().contiguous());
            } else if (pc.sh0.ndim() == 2) {
                // sh0 is already [N, 3*B] - use as-is
                LOG_INFO("write_ply_impl: sh0 is 2D, using as-is");
                tensors.push_back(pc.sh0.cpu().contiguous());
            } else {
                LOG_ERROR("write_ply_impl: Unexpected sh0 ndim: {}", pc.sh0.ndim());
                tensors.push_back(pc.sh0.cpu().contiguous());
            }
        }

        if (pc.shN.is_valid()) {
            LOG_INFO("write_ply_impl: pc.shN shape: ndim={}, shape=[{}{}{}]",
                     pc.shN.ndim(),
                     pc.shN.shape()[0],
                     pc.shN.ndim() >= 2 ? fmt::format(", {}", pc.shN.shape()[1]) : "",
                     pc.shN.ndim() >= 3 ? fmt::format(", {}", pc.shN.shape()[2]) : "");

            if (pc.shN.ndim() == 3) {
                // shN is [N, B, 3] - transpose to [N, 3, B], then flatten to [N, 3*B]
                LOG_INFO("write_ply_impl: shN is 3D, transposing and flattening");
                auto shN_transposed = pc.shN.transpose(1, 2).contiguous();
                tensors.push_back(shN_transposed.flatten(1).cpu().contiguous());
            } else if (pc.shN.ndim() == 2) {
                // shN is already [N, 3*B] - use as-is
                LOG_INFO("write_ply_impl: shN is 2D, using as-is");
                tensors.push_back(pc.shN.cpu().contiguous());
            } else {
                LOG_ERROR("write_ply_impl: Unexpected shN ndim: {}", pc.shN.ndim());
                tensors.push_back(pc.shN.cpu().contiguous());
            }
        }

        if (pc.opacity.is_valid()) {
            tensors.push_back(pc.opacity.cpu().contiguous());
        }

        if (pc.scaling.is_valid()) {
            tensors.push_back(pc.scaling.cpu().contiguous());
        }

        if (pc.rotation.is_valid()) {
            tensors.push_back(pc.rotation.cpu().contiguous());
        }

        auto write_output_ply = [](const fs::path& file_path,
                                   const std::vector<lfs::core::Tensor>& data,
                                   const std::vector<std::string>& attr_names) {
            tinyply::PlyFile ply;
            size_t attr_off = 0;

            for (const auto& tensor : data) {
                const size_t cols = tensor.size(1);
                std::vector<std::string> attrs(attr_names.begin() + attr_off,
                                               attr_names.begin() + attr_off + cols);

                ply.add_properties_to_element(
                    "vertex",
                    attrs,
                    tinyply::Type::FLOAT32,
                    tensor.size(0),
                    reinterpret_cast<uint8_t*>(const_cast<float*>(tensor.ptr<float>())),
                    tinyply::Type::INVALID, 0);

                attr_off += cols;
            }

            std::filebuf fb;
            fb.open(file_path, std::ios::out | std::ios::binary);
            std::ostream out_stream(&fb);
            ply.write(out_stream, /*binary=*/true);
        };

        if (stem.empty()) {
            write_output_ply(
                root / ("splat_" + std::to_string(iteration) + ".ply"),
                tensors,
                pc.attribute_names);
        } else {
            write_output_ply(
                root / std::string(stem + ".ply"),
                tensors,
                pc.attribute_names);
        }
    }

    /**
     * @brief Write SOG format implementation
     */
    std::filesystem::path write_sog_impl(const lfs::core::SplatData& splat_data,
                                         const std::filesystem::path& root,
                                         int iteration,
                                         int kmeans_iterations) {
        namespace fs = std::filesystem;

        // Create SOG subdirectory
        fs::path sog_dir = root / "sog";
        fs::create_directories(sog_dir);

        // Set up SOG write options - use .sog extension to create bundle
        std::filesystem::path sog_out_path = sog_dir /
                                             ("splat_" + std::to_string(iteration) + "_sog.sog");

        lfs::core::SogWriteOptions options{
            .iterations = kmeans_iterations,
            .output_path = sog_out_path};

        // Write SOG format
        auto result = lfs::core::write_sog(splat_data, options);
        if (!result) {
            LOG_ERROR("Failed to write SOG format: {}", result.error());
        } else {
            LOG_DEBUG("Successfully wrote SOG format for iteration {}", iteration);
        }

        return sog_out_path;
    }

} // anonymous namespace

namespace lfs::core {

    // ========== CONSTRUCTOR & DESTRUCTOR ==========

    SplatData::SplatData(int sh_degree,
                         Tensor means_,
                         Tensor sh0_,
                         Tensor shN_,
                         Tensor scaling_,
                         Tensor rotation_,
                         Tensor opacity_,
                         float scene_scale_)
        : _max_sh_degree(sh_degree),
          _active_sh_degree(0), // Start at 0, increases during training to match old behavior
          _scene_scale(scene_scale_),
          _means(std::move(means_)),
          _sh0(std::move(sh0_)),
          _shN(std::move(shN_)),
          _scaling(std::move(scaling_)),
          _rotation(std::move(rotation_)),
          _opacity(std::move(opacity_)) {
    }

    SplatData::~SplatData() {
        wait_for_saves();
    }

    // ========== MOVE SEMANTICS ==========

    SplatData::SplatData(SplatData&& other) noexcept
        : _active_sh_degree(other._active_sh_degree),
          _max_sh_degree(other._max_sh_degree),
          _scene_scale(other._scene_scale),
          _means(std::move(other._means)),
          _sh0(std::move(other._sh0)),
          _shN(std::move(other._shN)),
          _scaling(std::move(other._scaling)),
          _rotation(std::move(other._rotation)),
          _opacity(std::move(other._opacity)),
          _densification_info(std::move(other._densification_info)) {
        // Reset the moved-from object
        other._active_sh_degree = 0;
        other._max_sh_degree = 0;
        other._scene_scale = 0.0f;
    }

    SplatData& SplatData::operator=(SplatData&& other) noexcept {
        if (this != &other) {
            // Wait for any pending saves to complete
            wait_for_saves();

            // Move scalar members
            _active_sh_degree = other._active_sh_degree;
            _max_sh_degree = other._max_sh_degree;
            _scene_scale = other._scene_scale;

            // Move tensors
            _means = std::move(other._means);
            _sh0 = std::move(other._sh0);
            _shN = std::move(other._shN);
            _scaling = std::move(other._scaling);
            _rotation = std::move(other._rotation);
            _opacity = std::move(other._opacity);
            _densification_info = std::move(other._densification_info);
        }
        return *this;
    }

    // ========== COMPUTED GETTERS ==========

    Tensor SplatData::get_means() const {
        return _means;
    }

    Tensor SplatData::get_opacity() const {
        return _opacity.sigmoid().squeeze(-1);
    }

    Tensor SplatData::get_rotation() const {
        // Normalize quaternions along the last dimension
        // _rotation is [N, 4], we want to normalize each quaternion
        // norm = sqrt(sum(x^2)) along dim=1, keepdim=true to get [N, 1]

        auto squared = _rotation.square();
        auto sum_squared = squared.sum({1}, true);    // [N, 1]
        auto norm = sum_squared.sqrt();               // [N, 1]
        return _rotation.div(norm.clamp_min(1e-12f)); // Avoid division by zero
    }

    Tensor SplatData::get_scaling() const {
        return _scaling.exp();
    }

    Tensor SplatData::get_shs() const {
        // _sh0 is [N, 1, 3], _shN is [N, coeffs, 3]
        // Concatenate along dim 1 (coeffs) to get [N, total_coeffs, 3]
        return _sh0.cat(_shN, 1);
    }

    // ========== TRANSFORMATION ==========

    SplatData& SplatData::transform(const glm::mat4& transform_matrix) {
        LOG_TIMER("SplatData::transform");

        if (!_means.is_valid() || _means.size(0) == 0) {
            LOG_WARN("Cannot transform invalid or empty SplatData");
            return *this;
        }

        const int num_points = _means.size(0);
        auto device = _means.device();

        // 1. Transform positions (_means)
        std::vector<float> transform_data = {
            transform_matrix[0][0], transform_matrix[0][1], transform_matrix[0][2], transform_matrix[0][3],
            transform_matrix[1][0], transform_matrix[1][1], transform_matrix[1][2], transform_matrix[1][3],
            transform_matrix[2][0], transform_matrix[2][1], transform_matrix[2][2], transform_matrix[2][3],
            transform_matrix[3][0], transform_matrix[3][1], transform_matrix[3][2], transform_matrix[3][3]};

        auto transform_tensor = Tensor::from_vector(transform_data, TensorShape({4, 4}), device);
        auto ones = Tensor::ones({static_cast<size_t>(num_points), 1}, device);
        auto means_homo = _means.cat(ones, 1);
        auto transformed_means = transform_tensor.mm(means_homo.t()).t();

        _means = transformed_means.slice(1, 0, 3).contiguous();

        // 2. Extract _rotation from transform matrix
        glm::mat3 rot_mat(transform_matrix);
        glm::vec3 scale;
        for (int i = 0; i < 3; ++i) {
            scale[i] = glm::length(rot_mat[i]);
            if (scale[i] > 0.0f) {
                rot_mat[i] /= scale[i];
            }
        }

        glm::quat rotation_quat = glm::quat_cast(rot_mat);

        // 3. Transform rotations (quaternions) if there's _rotation
        if (std::abs(rotation_quat.w - 1.0f) > 1e-6f) {
            std::vector<float> rot_data = {rotation_quat.w, rotation_quat.x, rotation_quat.y, rotation_quat.z};
            auto rot_tensor = Tensor::from_vector(rot_data, TensorShape({4}), device);

            auto q = _rotation;
            std::vector<int> expand_shape = {num_points, 4};
            auto q_rot = rot_tensor.unsqueeze(0).expand(std::span<const int>(expand_shape));

            auto w1 = q_rot.slice(1, 0, 1).squeeze(1);
            auto x1 = q_rot.slice(1, 1, 2).squeeze(1);
            auto y1 = q_rot.slice(1, 2, 3).squeeze(1);
            auto z1 = q_rot.slice(1, 3, 4).squeeze(1);

            auto w2 = q.slice(1, 0, 1).squeeze(1);
            auto x2 = q.slice(1, 1, 2).squeeze(1);
            auto y2 = q.slice(1, 2, 3).squeeze(1);
            auto z2 = q.slice(1, 3, 4).squeeze(1);

            auto w_new = w1.mul(w2).sub(x1.mul(x2)).sub(y1.mul(y2)).sub(z1.mul(z2));
            auto x_new = w1.mul(x2).add(x1.mul(w2)).add(y1.mul(z2)).sub(z1.mul(y2));
            auto y_new = w1.mul(y2).sub(x1.mul(z2)).add(y1.mul(w2)).add(z1.mul(x2));
            auto z_new = w1.mul(z2).add(x1.mul(y2)).sub(y1.mul(x2)).add(z1.mul(w2));

            std::vector<Tensor> components = {
                w_new.unsqueeze(1),
                x_new.unsqueeze(1),
                y_new.unsqueeze(1),
                z_new.unsqueeze(1)};
            _rotation = Tensor::cat(components, 1);
        }

        // 4. Transform _scaling
        if (std::abs(scale.x - 1.0f) > 1e-6f ||
            std::abs(scale.y - 1.0f) > 1e-6f ||
            std::abs(scale.z - 1.0f) > 1e-6f) {

            float avg_scale = (scale.x + scale.y + scale.z) / 3.0f;
            _scaling = _scaling.add(std::log(avg_scale));
        }

        // 5. Update scene scale
        Tensor scene_center = _means.mean({0}, false);
        Tensor dists = _means.sub(scene_center).norm(2.0f, {1}, false);
        auto sorted_dists = dists.sort(0, false);
        float new_scene_scale = sorted_dists.first[num_points / 2].item();

        if (std::abs(new_scene_scale - _scene_scale) > _scene_scale * 0.1f) {
            _scene_scale = new_scene_scale;
        }

        LOG_DEBUG("Transformed {} gaussians", num_points);
        return *this;
    }

    // ========== UTILITY METHODS ==========

    void SplatData::increment_sh_degree() {
        if (_active_sh_degree < _max_sh_degree) {
            _active_sh_degree++;
        }
    }

    void SplatData::set_active_sh_degree(int sh_degree) {
        if (sh_degree <= _max_sh_degree) {
            _active_sh_degree = sh_degree;
        } else {
            _active_sh_degree = _max_sh_degree;
        }
    }

    std::vector<std::string> SplatData::get_attribute_names() const {
        std::vector<std::string> a{"x", "y", "z", "nx", "ny", "nz"};

        // _sh0 attributes - calculate based on actual dimensionality
        if (_sh0.is_valid()) {
            size_t sh0_features;
            if (_sh0.ndim() == 3) {
                // 3D: [N, B, 3] -> B*3 features
                sh0_features = _sh0.shape()[1] * _sh0.shape()[2];
            } else if (_sh0.ndim() == 2) {
                // 2D: [N, 3*B] -> size(1) features
                sh0_features = _sh0.shape()[1];
            } else {
                LOG_ERROR("Unexpected sh0 ndim in get_attribute_names: {}", _sh0.ndim());
                sh0_features = 3; // fallback
            }
            for (size_t i = 0; i < sh0_features; ++i) {
                a.emplace_back("f_dc_" + std::to_string(i));
            }
        }

        // _shN attributes - calculate based on actual dimensionality
        if (_shN.is_valid()) {
            size_t shN_features;
            if (_shN.ndim() == 3) {
                // 3D: [N, B, 3] -> B*3 features
                shN_features = _shN.shape()[1] * _shN.shape()[2];
            } else if (_shN.ndim() == 2) {
                // 2D: [N, 3*B] -> size(1) features
                shN_features = _shN.shape()[1];
            } else {
                LOG_ERROR("Unexpected shN ndim in get_attribute_names: {}", _shN.ndim());
                shN_features = 45; // fallback for degree 3
            }
            for (size_t i = 0; i < shN_features; ++i) {
                a.emplace_back("f_rest_" + std::to_string(i));
            }
        }

        a.emplace_back("opacity");  // Fixed: removed underscore to match legacy

        // _scaling attributes
        if (_scaling.is_valid()) {
            for (size_t i = 0; i < _scaling.shape()[1]; ++i) {
                a.emplace_back("scale_" + std::to_string(i));
            }
        }

        // _rotation attributes
        if (_rotation.is_valid()) {
            for (size_t i = 0; i < _rotation.shape()[1]; ++i) {
                a.emplace_back("rot_" + std::to_string(i));
            }
        }

        return a;
    }

    //     bool SplatData::is_valid() const {
    //         if (!_means.is_valid()) {
    //             LOG_ERROR("SplatData: _means tensor is invalid");
    //             return false;
    //         }
    //
    //         size_t n = _means.size(0);
    //
    //         if (_means.ndim() != 2 || _means.size(1) != 3) {
    //             LOG_ERROR("SplatData: _means must be [N, 3], got {}", _means.shape().str());
    //             return false;
    //         }
    //
    //         if (_sh0.is_valid() && (_sh0.ndim() != 3 || _sh0.size(0) != n || _sh0.size(2) != 3)) {
    //             LOG_ERROR("SplatData: _sh0 must be [N, 1, 3], got {}", _sh0.shape().str());
    //             return false;
    //         }
    //
    //         if (_shN.is_valid() && (_shN.ndim() != 3 || _shN.size(0) != n || _shN.size(2) != 3)) {
    //             LOG_ERROR("SplatData: _shN must be [N, coeffs, 3], got {}", _shN.shape().str());
    //             return false;
    //         }
    //
    //         if (_scaling.is_valid() &&
    //             (_scaling.ndim() != 2 || _scaling.size(0) != n || _scaling.size(1) != 3)) {
    //             LOG_ERROR("SplatData: _scaling must be [N, 3], got {}", _scaling.shape().str());
    //             return false;
    //         }
    //
    //         if (_rotation.is_valid() &&
    //             (_rotation.ndim() != 2 || _rotation.size(0) != n || _rotation.size(1) != 4)) {
    //             LOG_ERROR("SplatData: _rotation must be [N, 4], got {}", _rotation.shape().str());
    //             return false;
    //         }
    //
    //         if (_opacity.is_valid() &&
    //             (_opacity.ndim() != 2 || _opacity.size(0) != n || _opacity.size(1) != 1)) {
    //             LOG_ERROR("SplatData: _opacity must be [N, 1], got {}", _opacity.shape().str());
    //             return false;
    //         }
    //
    //         return true;
    //     }
    //
    // ========== ASYNC SAVE MANAGEMENT ==========

    void SplatData::wait_for_saves() const {
        std::lock_guard<std::mutex> lock(_save_mutex);

        // Wait for all pending saves
        for (auto& future : _save_futures) {
            if (future.valid()) {
                try {
                    future.wait();
                } catch (const std::exception& e) {
                    LOG_ERROR("Error waiting for save to complete: {}", e.what());
                }
            }
        }
        _save_futures.clear();
    }

    void SplatData::cleanup_finished_saves() const {
        std::lock_guard<std::mutex> lock(_save_mutex);

        // Remove completed futures
        _save_futures.erase(
            std::remove_if(_save_futures.begin(), _save_futures.end(),
                           [](const std::future<void>& f) {
                               return !f.valid() ||
                                      f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                           }),
            _save_futures.end());

        // Log if we have many pending saves
        if (_save_futures.size() > 5) {
            LOG_WARN("Multiple saves pending: {} operations in queue", _save_futures.size());
        }
    }

    // ========== EXPORT METHODS ==========

    void SplatData::save_ply(const std::filesystem::path& root,
                             int iteration,
                             bool join_threads,
                             std::string stem) const {
        auto pc = to_point_cloud();

        if (join_threads) {
            // Synchronous save - wait for completion
            write_ply_impl(pc, root, iteration, stem);
        } else {
            // Asynchronous save
            cleanup_finished_saves();

            std::lock_guard<std::mutex> lock(_save_mutex);
            _save_futures.emplace_back(
                std::async(std::launch::async,
                           [pc = std::move(pc), root, iteration, stem]() {
                               try {
                                   write_ply_impl(pc, root, iteration, stem);
                               } catch (const std::exception& e) {
                                   LOG_ERROR("Failed to save PLY for iteration {}: {}",
                                             iteration, e.what());
                               }
                           }));
        }
    }

    std::filesystem::path SplatData::save_sog(const std::filesystem::path& root,
                                              int iteration,
                                              int kmeans_iterations,
                                              bool join_threads) const {
        // SOG must always be synchronous - k-_means clustering is too heavy for async
        return write_sog_impl(*this, root, iteration, kmeans_iterations);
    }

    PointCloud SplatData::to_point_cloud() const {
        PointCloud pc;

        // Basic attributes
        pc.means = _means.cpu().contiguous();
        pc.normals = Tensor::zeros_like(pc.means);

        // Gaussian attributes - SH coefficients can be either:
        // - 3D [N, B, 3] from initial load (need transpose+flatten)
        // - 2D [N, 3*B] during training after densification (already correct for PLY)
        // PLY format expects [N, 3*B]
        if (_sh0.is_valid()) {
            LOG_INFO("to_point_cloud: _sh0 shape before cpu/contiguous: ndim={}, shape=[{}{}{}]",
                     _sh0.ndim(),
                     _sh0.shape()[0],
                     _sh0.ndim() >= 2 ? fmt::format(", {}", _sh0.shape()[1]) : "",
                     _sh0.ndim() >= 3 ? fmt::format(", {}", _sh0.shape()[2]) : "");

            auto sh0_cpu = _sh0.cpu().contiguous();

            LOG_INFO("to_point_cloud: sh0_cpu shape after cpu/contiguous: ndim={}, shape=[{}{}{}]",
                     sh0_cpu.ndim(),
                     sh0_cpu.shape()[0],
                     sh0_cpu.ndim() >= 2 ? fmt::format(", {}", sh0_cpu.shape()[1]) : "",
                     sh0_cpu.ndim() >= 3 ? fmt::format(", {}", sh0_cpu.shape()[2]) : "");

            if (sh0_cpu.ndim() == 3) {
                LOG_INFO("to_point_cloud: sh0 is 3D, will transpose and flatten");
                // Transpose from [N, B, 3] to [N, 3, B], then flatten to [N, 3*B]
                auto sh0_transposed = sh0_cpu.transpose(1, 2);  // [N, B, 3] -> [N, 3, B]
                size_t N = sh0_transposed.shape()[0];
                size_t flat_dim = sh0_transposed.shape()[1] * sh0_transposed.shape()[2];
                pc.sh0 = sh0_transposed.reshape({static_cast<int>(N), static_cast<int>(flat_dim)});
                LOG_INFO("to_point_cloud: sh0 after processing: shape=[{}, {}]", N, flat_dim);
            } else if (sh0_cpu.ndim() == 2) {
                LOG_INFO("to_point_cloud: sh0 is 2D, using as-is with shape=[{}, {}]",
                         sh0_cpu.shape()[0], sh0_cpu.shape()[1]);
                // Already 2D [N, 3*B] - use as-is
                pc.sh0 = sh0_cpu;
            } else {
                LOG_ERROR("Unexpected sh0 dimensions: {}, shape: [{}]", sh0_cpu.ndim(),
                         sh0_cpu.ndim() >= 1 ? sh0_cpu.shape()[0] : 0);
                pc.sh0 = sh0_cpu;
            }
        }

        if (_shN.is_valid()) {
            LOG_INFO("to_point_cloud: _shN shape before cpu/contiguous: ndim={}, shape=[{}{}{}]",
                     _shN.ndim(),
                     _shN.shape()[0],
                     _shN.ndim() >= 2 ? fmt::format(", {}", _shN.shape()[1]) : "",
                     _shN.ndim() >= 3 ? fmt::format(", {}", _shN.shape()[2]) : "");

            auto shN_cpu = _shN.cpu().contiguous();

            LOG_INFO("to_point_cloud: shN_cpu shape after cpu/contiguous: ndim={}, shape=[{}{}{}]",
                     shN_cpu.ndim(),
                     shN_cpu.shape()[0],
                     shN_cpu.ndim() >= 2 ? fmt::format(", {}", shN_cpu.shape()[1]) : "",
                     shN_cpu.ndim() >= 3 ? fmt::format(", {}", shN_cpu.shape()[2]) : "");

            if (shN_cpu.ndim() == 3) {
                LOG_INFO("to_point_cloud: shN is 3D, will transpose and flatten");
                // Transpose from [N, B, 3] to [N, 3, B], then flatten to [N, 3*B]
                auto shN_transposed = shN_cpu.transpose(1, 2);  // [N, B, 3] -> [N, 3, B]
                size_t N = shN_transposed.shape()[0];
                size_t flat_dim = shN_transposed.shape()[1] * shN_transposed.shape()[2];
                pc.shN = shN_transposed.reshape({static_cast<int>(N), static_cast<int>(flat_dim)});
                LOG_INFO("to_point_cloud: shN after processing: shape=[{}, {}]", N, flat_dim);
            } else if (shN_cpu.ndim() == 2) {
                LOG_INFO("to_point_cloud: shN is 2D, using as-is with shape=[{}, {}]",
                         shN_cpu.shape()[0], shN_cpu.shape()[1]);
                // Already 2D [N, 3*B] - use as-is
                pc.shN = shN_cpu;
            } else {
                LOG_ERROR("Unexpected shN dimensions: {}, shape: [{}]", shN_cpu.ndim(),
                         shN_cpu.ndim() >= 1 ? shN_cpu.shape()[0] : 0);
                pc.shN = shN_cpu;
            }
        }

        if (_opacity.is_valid()) {
            pc.opacity = _opacity.cpu().contiguous();
        }

        if (_scaling.is_valid()) {
            pc.scaling = _scaling.cpu().contiguous();
        }

        if (_rotation.is_valid()) {
            // Normalize _rotation before export
            auto normalized_rotation = get_rotation(); // This already normalizes
            pc.rotation = normalized_rotation.cpu().contiguous();
        }

        // Set attribute names for PLY export
        pc.attribute_names = get_attribute_names();

        return pc;
    }

    // ========== CROPPING ==========

    SplatData SplatData::crop_by_cropbox(const lfs::geometry::BoundingBox& bounding_box) const {
        LOG_TIMER("SplatData::crop_by_cropbox");

        if (!_means.is_valid() || _means.size(0) == 0) {
            LOG_WARN("Cannot crop invalid or empty SplatData");
            return SplatData();
        }

        // Get bounding box properties
        const auto bbox_min = bounding_box.getMinBounds();
        const auto bbox_max = bounding_box.getMaxBounds();
        const auto& world2bbox_transform = bounding_box.getworld2BBox();

        const int num_points = _means.size(0);

        LOG_DEBUG("Cropping {} points with bounding box: min({}, {}, {}), max({}, {}, {})",
                  num_points, bbox_min.x, bbox_min.y, bbox_min.z,
                  bbox_max.x, bbox_max.y, bbox_max.z);

        // Get transformation matrix from the EuclideanTransform
        glm::mat4 world_to_bbox_matrix = world2bbox_transform.toMat4();

        // Convert transformation matrix to tensor (transposed for row-major)
        std::vector<float> transform_data = {
            world_to_bbox_matrix[0][0], world_to_bbox_matrix[1][0], world_to_bbox_matrix[2][0], world_to_bbox_matrix[3][0],
            world_to_bbox_matrix[0][1], world_to_bbox_matrix[1][1], world_to_bbox_matrix[2][1], world_to_bbox_matrix[3][1],
            world_to_bbox_matrix[0][2], world_to_bbox_matrix[1][2], world_to_bbox_matrix[2][2], world_to_bbox_matrix[3][2],
            world_to_bbox_matrix[0][3], world_to_bbox_matrix[1][3], world_to_bbox_matrix[2][3], world_to_bbox_matrix[3][3]};
        auto transform_tensor = Tensor::from_vector(
            transform_data,
            TensorShape({4, 4}),
            _means.device());

        // Convert _means to homogeneous coordinates [N, 4]
        auto ones = Tensor::ones({static_cast<size_t>(num_points), 1}, _means.device());
        auto means_homo = _means.cat(ones, 1);

        // Transform all points: (4x4) @ (Nx4)^T = (4xN), then transpose back to (Nx4)
        auto transformed_points = transform_tensor.mm(means_homo.t()).t();

        // Extract xyz coordinates (drop homogeneous coordinate)
        auto local_points = transformed_points.slice(1, 0, 3);

        // Create bounding box bounds tensors
        std::vector<float> bbox_min_data = {bbox_min.x, bbox_min.y, bbox_min.z};
        std::vector<float> bbox_max_data = {bbox_max.x, bbox_max.y, bbox_max.z};

        auto bbox_min_tensor = Tensor::from_vector(
            bbox_min_data,
            TensorShape({3}),
            _means.device());
        auto bbox_max_tensor = Tensor::from_vector(
            bbox_max_data,
            TensorShape({3}),
            _means.device());

        // Check which points are inside the bounding box
        auto inside_min = local_points.ge(bbox_min_tensor.unsqueeze(0)); // [N, 3]
        auto inside_max = local_points.le(bbox_max_tensor.unsqueeze(0)); // [N, 3]

        // Point is inside if all 3 coordinates satisfy both min and max constraints
        std::vector<int> reduce_dims = {1};
        auto inside_mask = (inside_min && inside_max).all(std::span<const int>(reduce_dims), false); // [N]

        // Count points inside
        int points_inside = inside_mask.sum_scalar();

        LOG_DEBUG("Found {} points inside bounding box ({:.1f}%)",
                  points_inside, (float)points_inside / num_points * 100.0f);

        if (points_inside == 0) {
            LOG_WARN("No points found inside bounding box, returning empty SplatData");
            return SplatData();
        }

        // Get indices of points inside the bounding box
        auto indices = inside_mask.nonzero(); // [points_inside, 1]

        if (indices.ndim() == 2) {
            indices = indices.squeeze(1); // [points_inside]
        }

        // Index all tensors using the indices
        auto cropped_means = _means.index_select(0, indices).contiguous();
        auto cropped_sh0 = _sh0.index_select(0, indices).contiguous();
        auto cropped_shN = _shN.index_select(0, indices).contiguous();
        auto cropped_scaling = _scaling.index_select(0, indices).contiguous();
        auto cropped_rotation = _rotation.index_select(0, indices).contiguous();
        auto cropped_opacity = _opacity.index_select(0, indices).contiguous();

        // Recalculate scene scale for the cropped data
        Tensor scene_center = cropped_means.mean({0}, false);
        Tensor dists = cropped_means.sub(scene_center).norm(2.0f, {1}, false);

        float new_scene_scale = _scene_scale;
        if (points_inside > 1) {
            auto sorted_dists = dists.sort(0, false);
            new_scene_scale = sorted_dists.first[points_inside / 2].item();
        }

        // Create new SplatData with cropped tensors
        SplatData cropped_splat(
            _max_sh_degree,
            std::move(cropped_means),
            std::move(cropped_sh0),
            std::move(cropped_shN),
            std::move(cropped_scaling),
            std::move(cropped_rotation),
            std::move(cropped_opacity),
            new_scene_scale);

        // Copy over the active SH degree
        cropped_splat._active_sh_degree = _active_sh_degree;

        // If densification info exists and has the right size, crop it too
        if (_densification_info.is_valid() && _densification_info.size(0) == num_points) {
            cropped_splat._densification_info =
                _densification_info.index_select(0, indices).contiguous();
        }

        LOG_DEBUG("Successfully cropped SplatData: {} -> {} points (scale: {:.4f} -> {:.4f})",
                  num_points, points_inside, _scene_scale, new_scene_scale);

        return cropped_splat;
    }

    // RANDOM CROP

    void SplatData::random_choose(int num_required_splat, int seed) {
        LOG_TIMER("SplatData::random_choose");

        if (!_means.is_valid() || _means.size(0) == 0) {
            LOG_WARN("Cannot choose from invalid or empty SplatData");
            return;
        }

        const int num_points = _means.size(0);

        // Clamp num_splat to valid range
        if (num_required_splat <= 0) {
            LOG_WARN("num_splat must be positive, got {}", num_required_splat);
            return;
        }

        if (num_required_splat >= num_points) {
            LOG_DEBUG("num_splat ({}) >= total points ({}), keeping all data",
                      num_required_splat, num_points);
            return;
        }

        LOG_DEBUG("Randomly selecting {} points from {} total points (seed: {})",
                  num_required_splat, num_points, seed);

        // Generate random indices
        // Create a vector of all indices [0, 1, 2, ..., num_points-1]
        std::vector<int> all_indices(num_points);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        // Shuffle the indices using the provided seed
        std::mt19937 rng(seed);
        std::shuffle(all_indices.begin(), all_indices.end(), rng);

        // Take the first num_splat indices
        std::vector<int> selected_indices(all_indices.begin(),
                                          all_indices.begin() + num_required_splat);

        // Convert to tensor for indexing
        auto indices_tensor = Tensor::from_vector(
            selected_indices,
            TensorShape({static_cast<size_t>(num_required_splat)}),
            _means.device());

        // Index all tensors in-place using the selected indices
        _means = _means.index_select(0, indices_tensor).contiguous();
        _sh0 = _sh0.index_select(0, indices_tensor).contiguous();
        _shN = _shN.index_select(0, indices_tensor).contiguous();
        _scaling = _scaling.index_select(0, indices_tensor).contiguous();
        _rotation = _rotation.index_select(0, indices_tensor).contiguous();
        _opacity = _opacity.index_select(0, indices_tensor).contiguous();

        // Update gradients if they exist
        if (_means_grad.is_valid()) {
            _means_grad = _means_grad.index_select(0, indices_tensor).contiguous();
        }
        if (_sh0_grad.is_valid()) {
            _sh0_grad = _sh0_grad.index_select(0, indices_tensor).contiguous();
        }
        if (_shN_grad.is_valid()) {
            _shN_grad = _shN_grad.index_select(0, indices_tensor).contiguous();
        }
        if (_scaling_grad.is_valid()) {
            _scaling_grad = _scaling_grad.index_select(0, indices_tensor).contiguous();
        }
        if (_rotation_grad.is_valid()) {
            _rotation_grad = _rotation_grad.index_select(0, indices_tensor).contiguous();
        }
        if (_opacity_grad.is_valid()) {
            _opacity_grad = _opacity_grad.index_select(0, indices_tensor).contiguous();
        }

        // Update densification info if it exists
        if (_densification_info.is_valid() && _densification_info.size(0) == num_points) {
            _densification_info = _densification_info.index_select(0, indices_tensor).contiguous();
        }

        // Recalculate scene scale for the selected data
        Tensor scene_center = _means.mean({0}, false);
        Tensor dists = _means.sub(scene_center).norm(2.0f, {1}, false);

        float old_scene_scale = _scene_scale;
        if (num_required_splat > 1) {
            auto sorted_dists = dists.sort(0, false);
            _scene_scale = sorted_dists.first[num_required_splat / 2].item();
        }

        LOG_DEBUG("Successfully selected {} random splats in-place (scale: {:.4f} -> {:.4f})",
                  num_required_splat, old_scene_scale, _scene_scale);
    }

    // ========== FACTORY METHOD ==========

    std::expected<SplatData, std::string> SplatData::init_model_from_pointcloud(
        const param::TrainingParameters& params,
        Tensor scene_center,
        const PointCloud& pcd,
        int capacity) {

        try {
            LOG_DEBUG("=== init_model_from_pointcloud starting ===");
            LOG_DEBUG("  capacity={}, random={}, sh_degree={}",
                      capacity, params.optimization.random, params.optimization.sh_degree);
            LOG_DEBUG("  scene_center: is_valid={}, device={}, shape={}",
                      scene_center.is_valid(),
                      scene_center.device() == Device::CUDA ? "CUDA" : "CPU",
                      scene_center.shape().str());
            LOG_DEBUG("  pcd.means: is_valid={}, device={}, shape={}, numel={}",
                      pcd.means.is_valid(),
                      pcd.means.device() == Device::CUDA ? "CUDA" : "CPU",
                      pcd.means.shape().str(), pcd.means.numel());
            LOG_DEBUG("  pcd.colors: is_valid={}, device={}, shape={}, numel={}",
                      pcd.colors.is_valid(),
                      pcd.colors.device() == Device::CUDA ? "CUDA" : "CPU",
                      pcd.colors.shape().str(), pcd.colors.numel());

            // Generate positions and colors based on init type
            Tensor positions, colors;

            if (params.optimization.random) {
                const int num_points = params.optimization.init_num_pts;
                const float extent = params.optimization.init_extent;

                LOG_DEBUG("  Using random initialization: num_points={}, extent={}", num_points, extent);
                positions = (Tensor::rand({static_cast<size_t>(num_points), 3}, Device::CUDA)
                                 .mul(2.0f)
                                 .sub(1.0f))
                                .mul(extent);
                colors = Tensor::rand({static_cast<size_t>(num_points), 3}, Device::CUDA);
                LOG_DEBUG("  Random positions created: shape={}, numel={}", positions.shape().str(), positions.numel());
                LOG_DEBUG("  Random colors created: shape={}, numel={}", colors.shape().str(), colors.numel());
            } else {
                LOG_DEBUG("  Using point cloud initialization");
                if (!pcd.means.is_valid() || !pcd.colors.is_valid()) {
                    LOG_ERROR("Point cloud has invalid means or colors: means.is_valid()={}, colors.is_valid()={}",
                              pcd.means.is_valid(), pcd.colors.is_valid());
                    return std::unexpected("Point cloud has invalid _means or colors");
                }

                LOG_DEBUG("  Converting pcd.means to CUDA...");
                positions = pcd.means.cuda();
                LOG_DEBUG("  positions after .cuda(): is_valid={}, device={}, ptr={}, shape={}, numel={}",
                          positions.is_valid(),
                          positions.device() == Device::CUDA ? "CUDA" : "CPU",
                          static_cast<void*>(positions.ptr<float>()),
                          positions.shape().str(), positions.numel());

                // Normalize colors from uint8 [0,255] to float32 [0,1] to match old behavior
                LOG_DEBUG("  Converting pcd.colors (dtype={}) to float32...",
                          pcd.colors.dtype() == DataType::UInt8 ? "UInt8" : "Float32");
                colors = pcd.colors.to(DataType::Float32).div(255.0f).cuda();
                LOG_DEBUG("  colors after conversion: is_valid={}, device={}, shape={}, numel={}",
                          colors.is_valid(),
                          colors.device() == Device::CUDA ? "CUDA" : "CPU",
                          colors.shape().str(), colors.numel());
            }

            auto scene_center_device = scene_center.to(positions.device());
            const Tensor dists = positions.sub(scene_center_device).norm(2.0f, {1}, false);

            // Get median distance for scene scale
            auto sorted_dists = dists.sort(0, false);
            const float _scene_scale = sorted_dists.first[dists.size(0) / 2].item();

            // RGB to SH conversion (DC component)
            auto rgb_to_sh = [](const Tensor& rgb) {
                constexpr float kInvSH = 0.28209479177387814f;
                return rgb.sub(0.5f).div(kInvSH);
            };

            const size_t num_points = positions.size(0);
            const int64_t feature_shape = static_cast<int64_t>(
                std::pow(params.optimization.sh_degree + 1, 2));

            // Create final tensors first to avoid pool allocations
            Tensor means_, scaling_, rotation_, opacity_, sh0_, shN_;

            if (capacity > 0 && capacity < num_points) {
                LOG_DEBUG("capacity {} was lower than num_points {}.  Matching capacity to points. ", capacity, num_points);
                capacity = num_points;
            }

            if (capacity > 0) {
                LOG_DEBUG("Creating direct tensors with capacity={}", capacity);

                means_ = Tensor::zeros_direct(TensorShape({num_points, 3}), capacity);
                LOG_DEBUG("  means_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          means_.is_valid(), static_cast<void*>(means_.ptr<float>()),
                          means_.shape().str(), means_.numel());

                scaling_ = Tensor::zeros_direct(TensorShape({num_points, 3}), capacity);
                LOG_DEBUG("  scaling_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          scaling_.is_valid(), static_cast<void*>(scaling_.ptr<float>()),
                          scaling_.shape().str(), scaling_.numel());

                rotation_ = Tensor::zeros_direct(TensorShape({num_points, 4}), capacity);
                LOG_DEBUG("  rotation_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          rotation_.is_valid(), static_cast<void*>(rotation_.ptr<float>()),
                          rotation_.shape().str(), rotation_.numel());

                opacity_ = Tensor::zeros_direct(TensorShape({num_points, 1}), capacity);
                LOG_DEBUG("  opacity_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          opacity_.is_valid(), static_cast<void*>(opacity_.ptr<float>()),
                          opacity_.shape().str(), opacity_.numel());

                sh0_ = Tensor::zeros_direct(TensorShape({num_points, 1, 3}), capacity);
                LOG_DEBUG("  sh0_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          sh0_.is_valid(), static_cast<void*>(sh0_.ptr<float>()),
                          sh0_.shape().str(), sh0_.numel());

                shN_ = Tensor::zeros_direct(TensorShape({num_points, static_cast<size_t>(feature_shape - 1), 3}), capacity);
                LOG_DEBUG("  shN_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          shN_.is_valid(), static_cast<void*>(shN_.ptr<float>()),
                          shN_.shape().str(), shN_.numel());

                LOG_DEBUG("Computing and filling values...");
            }

            // Compute parameter values on CPU to avoid pool allocations
            Tensor means_cpu, scaling_cpu, rotation_cpu, opacity_cpu, sh0_cpu, shN_cpu;

            if (capacity > 0) {
                LOG_DEBUG("Computing values on CPU");
                LOG_DEBUG("  positions tensor: is_valid={}, device={}, shape={}, numel={}",
                          positions.is_valid(), positions.device() == Device::CUDA ? "CUDA" : "CPU",
                          positions.shape().str(), positions.numel());

                // Compute means on CPU
                auto positions_cpu = positions.cpu();
                LOG_DEBUG("  positions_cpu after .cpu(): is_valid={}, ptr={}, device={}, shape={}, numel={}",
                          positions_cpu.is_valid(), static_cast<const void*>(positions_cpu.ptr<float>()),
                          positions_cpu.device() == Device::CUDA ? "CUDA" : "CPU",
                          positions_cpu.shape().str(), positions_cpu.numel());

                if (params.optimization.random) {
                    means_cpu = positions_cpu.mul(_scene_scale);
                } else {
                    means_cpu = positions_cpu;
                }
                LOG_DEBUG("  means_cpu computed: is_valid={}, ptr={}, device={}, shape={}, numel={}",
                          means_cpu.is_valid(), static_cast<const void*>(means_cpu.ptr<float>()),
                          means_cpu.device() == Device::CUDA ? "CUDA" : "CPU",
                          means_cpu.shape().str(), means_cpu.numel());

                // Compute scaling on CPU
                LOG_DEBUG("  Computing neighbor distances...");
                auto nn_dist = compute_mean_neighbor_distances(means_cpu).clamp_min(1e-7f);
                LOG_DEBUG("  nn_dist computed: is_valid={}, shape={}, numel={}",
                          nn_dist.is_valid(), nn_dist.shape().str(), nn_dist.numel());

                std::vector<int> scale_expand_shape = {static_cast<int>(num_points), 3};
                scaling_cpu = nn_dist.sqrt()
                                  .mul(params.optimization.init_scaling)
                                  .log()
                                  .unsqueeze(-1)
                                  .expand(std::span<const int>(scale_expand_shape));
                LOG_DEBUG("  scaling_cpu computed: is_valid={}, ptr={}, device={}, shape={}, numel={}",
                          scaling_cpu.is_valid(), static_cast<const void*>(scaling_cpu.ptr<float>()),
                          scaling_cpu.device() == Device::CUDA ? "CUDA" : "CPU",
                          scaling_cpu.shape().str(), scaling_cpu.numel());

                // Create identity quaternion rotations on CPU
                LOG_DEBUG("  Creating identity quaternions...");
                rotation_cpu = Tensor::zeros({num_points, 4}, Device::CPU);
                auto rot_acc = rotation_cpu.accessor<float, 2>();
                for (size_t i = 0; i < num_points; i++) {
                    rot_acc(i, 0) = 1.0f;
                }
                LOG_DEBUG("  rotation_cpu created: is_valid={}, ptr={}, shape={}, numel={}",
                          rotation_cpu.is_valid(), static_cast<const void*>(rotation_cpu.ptr<float>()),
                          rotation_cpu.shape().str(), rotation_cpu.numel());

                // Compute opacity on CPU
                LOG_DEBUG("  Computing opacity (init_val={})...", params.optimization.init_opacity);
                auto init_val = params.optimization.init_opacity;
                opacity_cpu = Tensor::full({num_points, 1}, init_val, Device::CPU).logit();
                LOG_DEBUG("  opacity_cpu computed: is_valid={}, ptr={}, shape={}, numel={}",
                          opacity_cpu.is_valid(), static_cast<const void*>(opacity_cpu.ptr<float>()),
                          opacity_cpu.shape().str(), opacity_cpu.numel());

                // Compute SH coefficients on CPU
                LOG_DEBUG("  Computing SH coefficients...");
                LOG_DEBUG("    colors tensor: is_valid={}, device={}, shape={}, numel={}",
                          colors.is_valid(), colors.device() == Device::CUDA ? "CUDA" : "CPU",
                          colors.shape().str(), colors.numel());

                auto colors_cpu = colors.cpu();
                LOG_DEBUG("    colors_cpu: is_valid={}, ptr={}, shape={}, numel={}",
                          colors_cpu.is_valid(), static_cast<const void*>(colors_cpu.ptr<float>()),
                          colors_cpu.shape().str(), colors_cpu.numel());

                auto fused_color = rgb_to_sh(colors_cpu);
                LOG_DEBUG("    fused_color: is_valid={}, shape={}, numel={}",
                          fused_color.is_valid(), fused_color.shape().str(), fused_color.numel());

                // Create SH tensor on CPU
                auto shs_cpu_tensor = Tensor::zeros(
                    {fused_color.size(0), static_cast<size_t>(feature_shape), 3},
                    Device::CPU);
                LOG_DEBUG("    shs_cpu_tensor: is_valid={}, shape={}, numel={}",
                          shs_cpu_tensor.is_valid(), shs_cpu_tensor.shape().str(), shs_cpu_tensor.numel());

                auto shs_acc = shs_cpu_tensor.accessor<float, 3>();
                auto fused_acc = fused_color.accessor<float, 2>();

                for (size_t i = 0; i < fused_color.size(0); ++i) {
                    for (size_t c = 0; c < 3; ++c) {
                        shs_acc(i, 0, c) = fused_acc(i, c); // Set DC coefficient
                    }
                }

                sh0_cpu = shs_cpu_tensor.slice(1, 0, 1).contiguous();
                if (feature_shape > 1) {
                    shN_cpu = shs_cpu_tensor.slice(1, 1, feature_shape).contiguous();
                } else {
                    // sh-degree 0: create empty shN tensor [N, 0, 3]
                    shN_cpu = Tensor::zeros({shs_cpu_tensor.size(0), 0, 3}, Device::CPU);
                }
                LOG_DEBUG("  sh0_cpu: is_valid={}, ptr={}, shape={}, numel={}",
                          sh0_cpu.is_valid(), static_cast<const void*>(sh0_cpu.ptr<float>()),
                          sh0_cpu.shape().str(), sh0_cpu.numel());
                LOG_DEBUG("  shN_cpu: is_valid={}, ptr={}, shape={}, numel={}",
                          shN_cpu.is_valid(), static_cast<const void*>(shN_cpu.ptr<float>()),
                          shN_cpu.shape().str(), shN_cpu.numel());

                // Copy CPU data to direct CUDA tensors
                LOG_DEBUG("Copying CPU values to direct CUDA tensors");
                cudaError_t err;

                // Means copy
                LOG_DEBUG("  Copying means: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(means_cpu.ptr<float>()),
                          static_cast<void*>(means_.ptr<float>()),
                          means_cpu.numel() * sizeof(float));
                err = cudaMemcpy(means_.ptr<float>(), means_cpu.ptr<float>(),
                                means_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for means:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, device={}, numel={}",
                              means_cpu.is_valid(), static_cast<const void*>(means_cpu.ptr<float>()),
                              means_cpu.device() == Device::CPU ? "CPU" : "CUDA", means_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, device={}, numel={}",
                              means_.is_valid(), static_cast<void*>(means_.ptr<float>()),
                              means_.device() == Device::CPU ? "CPU" : "CUDA", means_.numel());
                    throw TensorError("cudaMemcpy failed for means: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  Means copy successful");

                // Scaling copy
                LOG_DEBUG("  Copying scaling: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(scaling_cpu.ptr<float>()),
                          static_cast<void*>(scaling_.ptr<float>()),
                          scaling_cpu.numel() * sizeof(float));
                err = cudaMemcpy(scaling_.ptr<float>(), scaling_cpu.ptr<float>(),
                                scaling_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for scaling:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, numel={}",
                              scaling_cpu.is_valid(), static_cast<const void*>(scaling_cpu.ptr<float>()), scaling_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, numel={}",
                              scaling_.is_valid(), static_cast<void*>(scaling_.ptr<float>()), scaling_.numel());
                    throw TensorError("cudaMemcpy failed for scaling: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  Scaling copy successful");

                // Rotation copy
                LOG_DEBUG("  Copying rotation: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(rotation_cpu.ptr<float>()),
                          static_cast<void*>(rotation_.ptr<float>()),
                          rotation_cpu.numel() * sizeof(float));
                err = cudaMemcpy(rotation_.ptr<float>(), rotation_cpu.ptr<float>(),
                                rotation_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for rotation:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, numel={}",
                              rotation_cpu.is_valid(), static_cast<const void*>(rotation_cpu.ptr<float>()), rotation_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, numel={}",
                              rotation_.is_valid(), static_cast<void*>(rotation_.ptr<float>()), rotation_.numel());
                    throw TensorError("cudaMemcpy failed for rotation: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  Rotation copy successful");

                // Opacity copy
                LOG_DEBUG("  Copying opacity: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(opacity_cpu.ptr<float>()),
                          static_cast<void*>(opacity_.ptr<float>()),
                          opacity_cpu.numel() * sizeof(float));
                err = cudaMemcpy(opacity_.ptr<float>(), opacity_cpu.ptr<float>(),
                                opacity_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for opacity:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, numel={}",
                              opacity_cpu.is_valid(), static_cast<const void*>(opacity_cpu.ptr<float>()), opacity_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, numel={}",
                              opacity_.is_valid(), static_cast<void*>(opacity_.ptr<float>()), opacity_.numel());
                    throw TensorError("cudaMemcpy failed for opacity: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  Opacity copy successful");

                // SH0 copy
                LOG_DEBUG("  Copying sh0: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(sh0_cpu.ptr<float>()),
                          static_cast<void*>(sh0_.ptr<float>()),
                          sh0_cpu.numel() * sizeof(float));
                err = cudaMemcpy(sh0_.ptr<float>(), sh0_cpu.ptr<float>(),
                                sh0_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for sh0:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, numel={}",
                              sh0_cpu.is_valid(), static_cast<const void*>(sh0_cpu.ptr<float>()), sh0_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, numel={}",
                              sh0_.is_valid(), static_cast<void*>(sh0_.ptr<float>()), sh0_.numel());
                    throw TensorError("cudaMemcpy failed for sh0: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  SH0 copy successful");

                // SHN copy
                LOG_DEBUG("  Copying shN: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(shN_cpu.ptr<float>()),
                          static_cast<void*>(shN_.ptr<float>()),
                          shN_cpu.numel() * sizeof(float));
                err = cudaMemcpy(shN_.ptr<float>(), shN_cpu.ptr<float>(),
                                shN_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for shN:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, numel={}",
                              shN_cpu.is_valid(), static_cast<const void*>(shN_cpu.ptr<float>()), shN_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, numel={}",
                              shN_.is_valid(), static_cast<void*>(shN_.ptr<float>()), shN_.numel());
                    throw TensorError("cudaMemcpy failed for shN: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  SHN copy successful");

                LOG_DEBUG("All CPU to CUDA copies completed successfully");
            } else {
                // No capacity specified - use pool
                Tensor means_temp;
                if (params.optimization.random) {
                    means_temp = positions.mul(_scene_scale).cuda();
                } else {
                    means_temp = positions.cuda();
                }

                auto nn_dist = compute_mean_neighbor_distances(means_temp).clamp_min(1e-7f);
                std::vector<int> scale_expand_shape = {static_cast<int>(num_points), 3};
                auto scaling_temp = nn_dist.sqrt()
                                    .mul(params.optimization.init_scaling)
                                    .log()
                                    .unsqueeze(-1)
                                    .expand(std::span<const int>(scale_expand_shape))
                                    .cuda();

                auto ones_col = Tensor::ones({num_points, 1}, Device::CUDA);
                auto zeros_cols = Tensor::zeros({num_points, 3}, Device::CUDA);
                auto rotation_temp = ones_col.cat(zeros_cols, 1);

                auto opacity_temp = Tensor::full({num_points, 1}, params.optimization.init_opacity, Device::CUDA).logit();

                auto colors_device = colors.cuda();
                auto fused_color = rgb_to_sh(colors_device);

                auto shs = Tensor::zeros({fused_color.size(0), static_cast<size_t>(feature_shape), 3}, Device::CUDA);
                auto shs_cpu_tmp = shs.cpu();
                auto fused_cpu_tmp = fused_color.cpu();

                auto shs_acc = shs_cpu_tmp.accessor<float, 3>();
                auto fused_acc = fused_cpu_tmp.accessor<float, 2>();

                for (size_t i = 0; i < fused_color.size(0); ++i) {
                    for (size_t c = 0; c < 3; ++c) {
                        shs_acc(i, 0, c) = fused_acc(i, c);
                    }
                }

                shs = shs_cpu_tmp.cuda();
                auto sh0_temp = shs.slice(1, 0, 1).contiguous();
                Tensor shN_temp;
                if (feature_shape > 1) {
                    shN_temp = shs.slice(1, 1, feature_shape).contiguous();
                } else {
                    // sh-degree 0: create empty shN tensor [N, 0, 3]
                    shN_temp = Tensor::zeros({shs.size(0), 0, 3}, Device::CUDA);
                }

                means_ = means_temp;
                scaling_ = scaling_temp;
                rotation_ = rotation_temp;
                opacity_ = opacity_temp;
                sh0_ = sh0_temp;
                shN_ = shN_temp;
            }

            std::println("Scene scale: {}", _scene_scale);
            std::println("Initialized SplatData with:");
            std::println("  - {} points", num_points);
            std::println("  - Max SH degree: {}", params.optimization.sh_degree);
            std::println("  - Total SH coefficients: {}", feature_shape);
            std::println("  - _sh0 shape: {}", sh0_.shape().str());
            std::println("  - _shN shape: {}", shN_.shape().str());
            std::println("  - Layout: [N, channels={}, coeffs]", sh0_.size(1));

            auto result = SplatData(
                params.optimization.sh_degree,
                std::move(means_),
                std::move(sh0_),
                std::move(shN_),
                std::move(scaling_),
                std::move(rotation_),
                std::move(opacity_),
                _scene_scale);

            return result;

        } catch (const std::exception& e) {
            return std::unexpected(
                std::format("Failed to initialize SplatData: {}", e.what()));
        }
    }

    //     void SplatData::ensure_grad_allocated() {
    //         // Allocate gradient tensors with same shapes as parameters
    //         if (!means_grad.is_valid()) {
    //             means_grad = Tensor::zeros(_means.shape(), _means.device());
    //         }
    //         if (!sh0_grad.is_valid()) {
    //             sh0_grad = Tensor::zeros(_sh0.shape(), _sh0.device());
    //         }
    //         if (!shN_grad.is_valid()) {
    //             shN_grad = Tensor::zeros(_shN.shape(), _shN.device());
    //         }
    //         if (!scaling_grad.is_valid()) {
    //             scaling_grad = Tensor::zeros(_scaling.shape(), _scaling.device());
    //         }
    //         if (!rotation_grad.is_valid()) {
    //             rotation_grad = Tensor::zeros(_rotation.shape(), _rotation.device());
    //         }
    //         if (!opacity_grad.is_valid()) {
    //             opacity_grad = Tensor::zeros(_opacity.shape(), _opacity.device());
    //         }
    //     }
    //
    //     void SplatData::zero_grad() {
    //         // Zero out all gradient tensors if they exist
    //         if (means_grad.is_valid()) {
    //             means_grad.zero_();
    //         }
    //         if (sh0_grad.is_valid()) {
    //             sh0_grad.zero_();
    //         }
    //         if (shN_grad.is_valid()) {
    //             shN_grad.zero_();
    //         }
    //         if (scaling_grad.is_valid()) {
    //             scaling_grad.zero_();
    //         }
    //         if (rotation_grad.is_valid()) {
    //             rotation_grad.zero_();
    //         }
    //         if (opacity_grad.is_valid()) {
    //             opacity_grad.zero_();
    //         }
    //     }

    // ========== GRADIENT MANAGEMENT ==========

    void SplatData::allocate_gradients() {
        // Use empty() instead of zeros() - will be zeroed before first use
        if (_means.is_valid()) {
            _means_grad = Tensor::empty(_means.shape(), _means.device());
        }
        if (_sh0.is_valid()) {
            _sh0_grad = Tensor::empty(_sh0.shape(), _sh0.device());
        }
        if (_shN.is_valid()) {
            _shN_grad = Tensor::empty(_shN.shape(), _shN.device());
        }
        if (_scaling.is_valid()) {
            _scaling_grad = Tensor::empty(_scaling.shape(), _scaling.device());
        }
        if (_rotation.is_valid()) {
            _rotation_grad = Tensor::empty(_rotation.shape(), _rotation.device());
        }
        if (_opacity.is_valid()) {
            _opacity_grad = Tensor::empty(_opacity.shape(), _opacity.device());
        }
    }

    void SplatData::reserve_capacity(size_t capacity) {
        // Reserve capacity for parameters
        if (_means.is_valid()) _means.reserve(capacity);
        if (_sh0.is_valid()) _sh0.reserve(capacity);
        if (_shN.is_valid()) _shN.reserve(capacity);
        if (_scaling.is_valid()) _scaling.reserve(capacity);
        if (_rotation.is_valid()) _rotation.reserve(capacity);
        if (_opacity.is_valid()) _opacity.reserve(capacity);

        // Reserve capacity for gradients (must be same as parameters!)
        if (_means_grad.is_valid()) _means_grad.reserve(capacity);
        if (_sh0_grad.is_valid()) _sh0_grad.reserve(capacity);
        if (_shN_grad.is_valid()) _shN_grad.reserve(capacity);
        if (_scaling_grad.is_valid()) _scaling_grad.reserve(capacity);
        if (_rotation_grad.is_valid()) _rotation_grad.reserve(capacity);
        if (_opacity_grad.is_valid()) _opacity_grad.reserve(capacity);
    }

    void SplatData::zero_gradients() {
        if (_means_grad.is_valid()) {
            _means_grad.zero_();
        }
        if (_sh0_grad.is_valid()) {
            _sh0_grad.zero_();
        }
        if (_shN_grad.is_valid()) {
            _shN_grad.zero_();
        }
        if (_scaling_grad.is_valid()) {
            _scaling_grad.zero_();
        }
        if (_rotation_grad.is_valid()) {
            _rotation_grad.zero_();
        }
        if (_opacity_grad.is_valid()) {
            _opacity_grad.zero_();
        }
    }

    bool SplatData::has_gradients() const {
        return _means_grad.is_valid();
    }

} // namespace lfs::core