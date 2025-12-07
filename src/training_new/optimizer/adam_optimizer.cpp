/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_optimizer.hpp"
#include "adam_api.h"  // fast_lfs::optimizer::adam_step_raw
#include "core_new/logger.hpp"
#include "core_new/tensor/internal/tensor_serialization.hpp"
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>

// Simple CUDA error checking macro
#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t err = call;                                                      \
        if (err != cudaSuccess) {                                                    \
            throw std::runtime_error(std::string("CUDA error: ") +                  \
                                    cudaGetErrorString(err));                        \
        }                                                                            \
    } while (0)

namespace lfs::training {

    AdamOptimizer::AdamOptimizer(lfs::core::SplatData& splat_data, const AdamConfig& config)
        : splat_data_(splat_data), config_(config) {

        LOG_DEBUG("AdamOptimizer constructor: config.initial_capacity={}, config.growth_factor={}",
                  config_.initial_capacity, config_.growth_factor);

        // Ensure gradients are allocated
        if (!splat_data_.has_gradients()) {
            splat_data_.allocate_gradients();
            LOG_DEBUG("Allocated gradients for optimizer");
        }
    }

    void AdamOptimizer::step(int iteration) {
        // Optimize each parameter
        step_param(ParamType::Means, iteration);
        step_param(ParamType::Sh0, iteration);
        step_param(ParamType::ShN, iteration);
        step_param(ParamType::Scaling, iteration);
        step_param(ParamType::Rotation, iteration);
        step_param(ParamType::Opacity, iteration);
    }

    void AdamOptimizer::zero_grad(int iteration) {
        // TODO: Optional - Skip SH gradients on certain iterations (matching old behavior)
        // For now, just zero everything
        splat_data_.zero_gradients();
    }

    lfs::core::Tensor& AdamOptimizer::get_param(ParamType type) {
        switch (type) {
            case ParamType::Means: return splat_data_.means();
            case ParamType::Sh0: return splat_data_.sh0();
            case ParamType::ShN: return splat_data_.shN();
            case ParamType::Scaling: return splat_data_.scaling_raw();
            case ParamType::Rotation: return splat_data_.rotation_raw();
            case ParamType::Opacity: return splat_data_.opacity_raw();
        }
        throw std::runtime_error("Invalid param type");
    }

    lfs::core::Tensor& AdamOptimizer::get_grad(ParamType type) {
        switch (type) {
            case ParamType::Means: return splat_data_.means_grad();
            case ParamType::Sh0: return splat_data_.sh0_grad();
            case ParamType::ShN: return splat_data_.shN_grad();
            case ParamType::Scaling: return splat_data_.scaling_grad();
            case ParamType::Rotation: return splat_data_.rotation_grad();
            case ParamType::Opacity: return splat_data_.opacity_grad();
        }
        throw std::runtime_error("Invalid param type");
    }

    std::string AdamOptimizer::param_name(ParamType type) const {
        switch (type) {
            case ParamType::Means: return "means";
            case ParamType::Sh0: return "sh0";
            case ParamType::ShN: return "shN";
            case ParamType::Scaling: return "scaling";
            case ParamType::Rotation: return "rotation";
            case ParamType::Opacity: return "opacity";
        }
        return "unknown";
    }

    void AdamOptimizer::init_state(ParamType type) {
        auto& param = get_param(type);
        auto name = param_name(type);

        // Validate param before creating state
        if (!param.is_valid()) {
            throw std::runtime_error("init_state: parameter " + name + " is not valid!");
        }
        if (param.ndim() == 0) {
            throw std::runtime_error("init_state: parameter " + name + " has rank 0! This will create rank-0 optimizer state.");
        }

        auto& state = states_[name];
        size_t param_size = param.shape()[0];

        // Calculate initial capacity with pre-allocation if configured
        size_t initial_cap = compute_new_capacity(0, param_size);

        // Use zeros_direct() to bypass pool
        if (initial_cap > param_size) {
            state.exp_avg = lfs::core::Tensor::zeros_direct(param.shape(), initial_cap);
            state.exp_avg_sq = lfs::core::Tensor::zeros_direct(param.shape(), initial_cap);
            state.capacity = initial_cap;
            state.size = param_size;

            LOG_INFO("Initialized optimizer state for {} with zeros_direct() (size: {}, capacity: {})",
                      name, param_size, initial_cap);
        } else {
            state.exp_avg = lfs::core::Tensor::zeros(param.shape(), param.device());
            state.exp_avg_sq = lfs::core::Tensor::zeros(param.shape(), param.device());
            state.capacity = param_size;
            state.size = param_size;

            LOG_DEBUG("Initialized optimizer state for {}: size={}", name, param_size);
        }

        state.step_count = 0;
    }

    void AdamOptimizer::step_param(ParamType type, int iteration) {
        auto& param = get_param(type);
        auto& grad = get_grad(type);

        // Skip if no gradient or if gradient is all zeros (not yet computed)
        if (!grad.is_valid() || grad.numel() == 0) {
            return;
        }

        // Skip if parameter doesn't exist yet (lazy initialization)
        if (!param.is_valid() || param.numel() == 0) {
            return;
        }

        auto name = param_name(type);

        // Initialize state on first call
        if (states_.find(name) == states_.end()) {
            init_state(type);
        }

        auto& state = states_[name];
        state.step_count++;

        // Higher degree SH coefficients are not used in the first 1000 iterations
        if (type == ParamType::ShN && iteration <= 1000) {
            return;
        }

        // Compute bias correction factors (use double precision to avoid accumulation errors)
        double bias_correction1_rcp = 1.0 / (1.0 - std::pow(config_.beta1, state.step_count));
        double bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(config_.beta2, state.step_count));

        // Get per-parameter learning rate
        float param_lr = get_param_lr(type);

        // When fast path is used in extend_state_for_new_params, state tensors have
        // excess capacity (state.exp_avg.shape()[0] > param.shape()[0]).
        // We must ensure we only operate on the valid elements.

        size_t param_size = param.shape()[0];
        size_t state_size = state.size;

        // Verify param and state are synchronized
        if (param_size != state_size) {
            LOG_ERROR("param size ({}) != state.size ({}) for {}", param_size, state_size, name);
            throw std::runtime_error("Optimizer state desynchronization detected");
        }

        // Calculate number of elements to process
        // This MUST use state.size, not param.shape()[0], because after fast path:
        //   - param.shape()[0] reflects actual data size
        //   - state.size tracks logical size (should match param)
        //   - state.exp_avg.shape()[0] may be larger (excess capacity)
        size_t feature_dim = param.numel() / param_size;  // e.g., 3 for means, 4 for rotation
        size_t num_elements = state_size * feature_dim;

        // Optional diagnostic logging (only at trace level)
        if (iteration % 1000 == 0 && state.capacity > state.size) {
            LOG_TRACE("Optimizer capacity usage for {}: {}/{} ({:.1f}%)",
                     name, state_size, state.capacity, 100.0f * state_size / state.capacity);
        }

        // Call fused CUDA kernel - operates ONLY on valid elements
        // Uses state.size * feature_dim instead of param.numel()
        fast_lfs::optimizer::adam_step_raw(
            param.ptr<float>(),
            state.exp_avg.ptr<float>(),
            state.exp_avg_sq.ptr<float>(),
            grad.ptr<float>(),
            static_cast<int>(num_elements),  // Based on state.size!
            param_lr,  // Use per-parameter learning rate
            config_.beta1,
            config_.beta2,
            config_.eps,
            bias_correction1_rcp,
            bias_correction2_sqrt_rcp
        );
    }

    void AdamOptimizer::reset_state_at_indices(ParamType type, const std::vector<int64_t>& indices) {
        auto name = param_name(type);

        // Ensure state exists
        if (states_.find(name) == states_.end()) {
            LOG_DEBUG("State for {} not initialized yet, skipping reset", name);
            return;
        }

        if (indices.empty()) {
            return;  // Nothing to do
        }

        auto& state = states_[name];

        // Calculate row size (product of all dimensions except first)
        auto state_shape = state.exp_avg.shape();
        int row_size = 1;
        for (size_t i = 1; i < state_shape.rank(); i++) {
            row_size *= state_shape[i];
        }

        // Allocate GPU memory for indices and copy from host
        int64_t* indices_device_ptr;
        cudaMalloc(&indices_device_ptr, indices.size() * sizeof(int64_t));
        cudaMemcpy(indices_device_ptr, indices.data(), indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

        // Use batched CUDA kernel for much better performance (600x faster!)
        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg.template ptr<float>(),
            indices_device_ptr,
            indices.size(),
            row_size
        );

        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg_sq.template ptr<float>(),
            indices_device_ptr,
            indices.size(),
            row_size
        );

        cudaFree(indices_device_ptr);

        LOG_DEBUG("Reset optimizer state for {} at {} indices (batched GPU kernel)", name, indices.size());
    }

    void AdamOptimizer::extend_state_for_new_params(ParamType type, size_t n_new) {
        auto& param = get_param(type);
        auto name = param_name(type);

        // Ensure state exists
        if (states_.find(name) == states_.end()) {
            // If state doesn't exist yet, it will be initialized on first step
            LOG_DEBUG("State for {} not initialized yet, will be initialized on first step", name);
            return;
        }

        auto& state = states_[name];
        size_t new_size = state.size + n_new;

        // Defensive check: ensure param is valid
        if (!param.is_valid() || param.shape().rank() == 0) {
            throw std::runtime_error("extend_state_for_new_params: param " + name +
                                   " is invalid! is_valid=" + std::to_string(param.is_valid()) +
                                   ", rank=" + std::to_string(param.shape().rank()));
        }

        // Defensive check: ensure state tensors are valid
        if (!state.exp_avg.is_valid() || state.exp_avg.ndim() == 0) {
            throw std::runtime_error("extend_state_for_new_params: state.exp_avg for " + name +
                                   " is invalid! is_valid=" + std::to_string(state.exp_avg.is_valid()) +
                                   ", ndim=" + std::to_string(state.exp_avg.ndim()));
        }

        // Use reserved capacity if available
        if (state.exp_avg.capacity() > 0 && new_size <= state.exp_avg.capacity()) {
            LOG_DEBUG("extend_state: {} by {} params (using reserved capacity)", name, n_new);
            state.exp_avg.append_zeros(n_new);
            state.exp_avg_sq.append_zeros(n_new);
            state.size = new_size;
            state.capacity = state.exp_avg.capacity();
        } else {
            // No capacity available - allocate new tensors
            LOG_DEBUG("extend_state: {} by {} params (allocating)", name, n_new);

            auto param_shape = param.shape();
            std::vector<size_t> new_full_dims(param_shape.rank());
            new_full_dims[0] = new_size;
            for (size_t i = 1; i < param_shape.rank(); i++) {
                new_full_dims[i] = param_shape[i];
            }

            // Create empty tensors with full new size
            auto new_exp_avg = lfs::core::Tensor::empty(lfs::core::TensorShape(new_full_dims), param.device());
            auto new_exp_avg_sq = lfs::core::Tensor::empty(lfs::core::TensorShape(new_full_dims), param.device());

            // Copy old data (if any)
            if (state.size > 0 && state.exp_avg.numel() > 0) {
                size_t old_bytes = state.exp_avg.numel() * sizeof(float);
                CHECK_CUDA(cudaMemcpy(new_exp_avg.ptr<float>(), state.exp_avg.ptr<float>(),
                                     old_bytes, cudaMemcpyDeviceToDevice));
                CHECK_CUDA(cudaMemcpy(new_exp_avg_sq.ptr<float>(), state.exp_avg_sq.ptr<float>(),
                                     old_bytes, cudaMemcpyDeviceToDevice));
            }

            // Zero-fill new portion
            size_t row_size = param.numel() / param_shape[0];
            size_t new_elements = n_new * row_size;
            size_t offset_bytes = state.exp_avg.numel() * sizeof(float);
            CHECK_CUDA(cudaMemset(reinterpret_cast<char*>(new_exp_avg.ptr<float>()) + offset_bytes,
                                 0, new_elements * sizeof(float)));
            CHECK_CUDA(cudaMemset(reinterpret_cast<char*>(new_exp_avg_sq.ptr<float>()) + offset_bytes,
                                 0, new_elements * sizeof(float)));

            state.exp_avg = new_exp_avg;
            state.exp_avg_sq = new_exp_avg_sq;
            state.size = new_size;
            state.capacity = new_size;  // No extra capacity
        }

        LOG_DEBUG("Extended optimizer state for {} by {} parameters (size: {} -> {}, step_count={})",
                  name, n_new, state.size - n_new, state.size, state.step_count);

        // Verify shapes match after extend
        if (state.exp_avg.shape()[0] != new_size || state.exp_avg_sq.shape()[0] != new_size) {
            LOG_ERROR("SHAPE MISMATCH after extend_state for {}: exp_avg.shape={}, exp_avg_sq.shape={}, expected={}",
                     name, state.exp_avg.shape()[0], state.exp_avg_sq.shape()[0], new_size);
        }
    }

    size_t AdamOptimizer::compute_new_capacity(size_t current_capacity, size_t required_size) const {
        size_t new_capacity;
        if (current_capacity == 0) {
            // First allocation: use initial_capacity if set, otherwise exact fit with some growth
            if (config_.initial_capacity > 0) {
                new_capacity = std::max(config_.initial_capacity, required_size);
                LOG_DEBUG("compute_new_capacity: initial allocation with config.initial_capacity={}, required_size={} -> new_capacity={}",
                          config_.initial_capacity, required_size, new_capacity);
            } else {
                // Default: allocate 150% of required to avoid immediate reallocation
                new_capacity = static_cast<size_t>(required_size * 1.5f);
                LOG_DEBUG("compute_new_capacity: initial allocation (no initial_capacity set), required_size={} -> new_capacity={} (1.5x)",
                          required_size, new_capacity);
            }
            return new_capacity;
        }

        // Grow by growth_factor (like std::vector uses 1.5x or 2x)
        size_t grown_capacity = static_cast<size_t>(current_capacity * config_.growth_factor);
        new_capacity = std::max(grown_capacity, required_size);
        LOG_DEBUG("compute_new_capacity: growth, current_capacity={}, required_size={}, growth_factor={} -> new_capacity={}",
                  current_capacity, required_size, config_.growth_factor, new_capacity);
        return new_capacity;
    }

    const AdamParamState* AdamOptimizer::get_state(ParamType type) const {
        auto name = param_name(type);
        auto it = states_.find(name);
        if (it == states_.end()) {
            return nullptr;
        }
        // NOTE: Returns the state with full capacity tensors
        // Caller should use state->size to know the actual used size
        // The exp_avg/exp_avg_sq tensors may have shape[0] > size due to pre-allocation
        return &it->second;
    }

    int64_t AdamOptimizer::get_step_count(ParamType type) const {
        auto name = param_name(type);
        auto it = states_.find(name);
        if (it == states_.end()) {
            return 0;
        }
        return it->second.step_count;
    }

    void AdamOptimizer::set_state(ParamType type, const AdamParamState& state) {
        auto name = param_name(type);
        states_[name] = state;
        LOG_DEBUG("Set optimizer state for {} (size: {}, capacity: {})",
                  name, state.size, state.capacity);
    }

    void AdamOptimizer::add_new_params(ParamType type, const lfs::core::Tensor& new_values, bool validate) {
        auto& param = get_param(type);
        auto& grad = get_grad(type);

        // Validation: check that new_values has compatible shape
        if (validate) {
            if (new_values.ndim() != param.ndim()) {
                throw std::runtime_error(
                    "add_new_params: new_values rank (" + std::to_string(new_values.ndim()) +
                    ") doesn't match existing parameter rank (" + std::to_string(param.ndim()) + ")"
                );
            }

            // Check that all dimensions except first match
            for (size_t i = 1; i < param.ndim(); i++) {
                if (new_values.shape()[i] != param.shape()[i]) {
                    throw std::runtime_error(
                        "add_new_params: new_values shape mismatch at dimension " + std::to_string(i)
                    );
                }
            }

            // Check device matches
            if (new_values.device() != param.device()) {
                throw std::runtime_error(
                    "add_new_params: new_values device doesn't match existing parameter device"
                );
            }
        }

        size_t n_new = new_values.shape()[0];
        size_t n_current = param.shape()[0];

        // Concatenate new values to parameter
        param = lfs::core::Tensor::cat(std::vector<lfs::core::Tensor>{param, new_values}, 0);

        // Verify param is still valid after cat
        if (!param.is_valid() || param.ndim() == 0) {
            throw std::runtime_error("add_new_params: parameter became invalid after cat()!");
        }

        // Re-obtain gradient reference after modifying param
        // (in case the reference became stale)
        auto& grad_updated = get_grad(type);

        // Verify grad is valid
        if (!grad_updated.is_valid() || grad_updated.ndim() == 0) {
            throw std::runtime_error("add_new_params: gradient is invalid or rank-0! param.ndim()=" +
                                   std::to_string(param.ndim()) + ", grad.ndim()=" + std::to_string(grad_updated.ndim()));
        }

        // Zero ALL gradients to match LEGACY behavior
        // LEGACY creates new parameter tensors via torch::cat, which don't have .grad() defined,
        // then initializes .grad() = zeros for ALL parameters (see mcmc.cpp:348-365).
        // This means when add_new_gs is called, the optimizer step() uses ZERO gradients for ALL Gaussians.
        // We must replicate this exact behavior for deterministic equivalence.
        LOG_DEBUG("  add_new_params: Zeroing ALL gradients (matching LEGACY) for {}", param_name(type));
        LOG_DEBUG("    grad before: shape={}, param after: shape={}",
                  grad_updated.shape()[0], param.shape()[0]);

        grad_updated = lfs::core::Tensor::zeros(param.shape(), param.device());

        LOG_DEBUG("    grad after: shape={} (all zeros)", grad_updated.shape()[0]);

        if (grad_updated.numel() == 0) {
            LOG_ERROR("  Gradient concatenation failed! Resulting tensor is empty");
        } else {
            LOG_DEBUG("    result grad: shape[0]={}, ndim={}", grad_updated.shape()[0], grad_updated.ndim());
        }

        // Extend optimizer state (this can be optimized with capacity tracking)
        extend_state_for_new_params(type, n_new);
    }

    void AdamOptimizer::add_new_params_gather(ParamType type, const lfs::core::Tensor& indices) {
        LOG_DEBUG("add_new_params_gather for {}", param_name(type));

        // Get parameter and gradient references
        auto& param = get_param(type);
        auto& grad = get_grad(type);

        if (!param.is_valid()) {
            if (type == ParamType::ShN) {
                LOG_DEBUG("add_new_params_gather: parameter {} not initialized (likely sh-degree 0), skipping", param_name(type));
                return;
            }
            LOG_ERROR("add_new_params_gather: parameter {} not initialized", param_name(type));
            return;
        }

        // Skip parameters with empty feature dimension (e.g., shN when sh-degree is 0)
        if (param.ndim() >= 2 && param.shape()[1] == 0) {
            LOG_DEBUG("add_new_params_gather: parameter {} has empty feature dimension (size 0), skipping", param_name(type));
            return;
        }

        if (indices.device() != param.device()) {
            LOG_ERROR("add_new_params_gather: indices device doesn't match parameter device");
            return;
        }

        size_t n_new = indices.numel();
        size_t n_current = param.shape()[0];
        size_t param_capacity = param.capacity();

        // Use fused append_gather() operation - NO INTERMEDIATE ALLOCATION (if capacity available)!
        size_t old_size = param.shape()[0];
        param.append_gather(indices);
        size_t new_size = param.shape()[0];

        if (new_size != old_size + n_new) {
            LOG_ERROR("append_gather FAILED for {}: expected new_size={}, got new_size={} (unchanged!)",
                      param_name(type), old_size + n_new, new_size);
        }

        // Extend gradient tensor to match parameter size
        LOG_DEBUG("  add_new_params_gather: Extending gradient for {}", param_name(type));
        if (grad.capacity() > 0 && grad.shape()[0] + n_new <= grad.capacity()) {
            // Use append_zeros() to extend gradient using reserved capacity
            LOG_DEBUG("    grad: using append_zeros({}) [capacity={}]", n_new, grad.capacity());
            grad.append_zeros(n_new);
        } else {
            // No capacity available, create new tensor
            LOG_DEBUG("    grad: creating new zeros tensor");
            grad = lfs::core::Tensor::zeros(param.shape(), param.device());
        }
        LOG_DEBUG("    grad after: shape={}", grad.shape()[0]);

        // Extend optimizer state (this can be optimized with capacity tracking)
        extend_state_for_new_params(type, n_new);
    }

    void AdamOptimizer::relocate_params_at_indices(ParamType type, const std::vector<int64_t>& indices) {
        if (indices.empty()) return;

        auto& param = get_param(type);

        // Validation: check indices are in bounds
        for (auto idx : indices) {
            if (idx < 0 || static_cast<size_t>(idx) >= param.shape()[0]) {
                throw std::runtime_error(
                    "relocate_params_at_indices: index " + std::to_string(idx) +
                    " out of bounds [0, " + std::to_string(param.shape()[0]) + ")"
                );
            }
        }

        // Copy indices to GPU once, then use fast GPU version
        int64_t* indices_device_ptr;
        cudaMalloc(&indices_device_ptr, indices.size() * sizeof(int64_t));
        cudaMemcpy(indices_device_ptr, indices.data(), indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

        relocate_params_at_indices_gpu(type, indices_device_ptr, indices.size());

        cudaFree(indices_device_ptr);
    }

    void AdamOptimizer::relocate_params_at_indices_gpu(ParamType type, const int64_t* indices_device, size_t n_indices) {
        if (n_indices == 0) return;

        auto& param = get_param(type);
        auto name = param_name(type);

        // Ensure optimizer state exists
        if (states_.find(name) == states_.end()) {
            LOG_DEBUG("State for {} not initialized yet, skipping reset", name);
            return;
        }

        auto& state = states_[name];

        // Calculate row size for optimizer state
        auto state_shape = state.exp_avg.shape();
        int state_row_size = 1;
        for (size_t i = 1; i < state_shape.rank(); i++) {
            state_row_size *= state_shape[i];
        }

        // Zero out optimizer state using batched GPU kernel (FAST!)
        // NOTE: We do NOT zero gradients here! Relocate is called in post_backward(),
        // BEFORE the optimizer step. The optimizer needs the current gradients to update
        // parameters. Gradients will be zeroed by zero_grad() after the optimizer step.
        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg.template ptr<float>(),
            indices_device,
            n_indices,
            state_row_size
        );

        fast_lfs::optimizer::zero_rows_at_indices(
            state.exp_avg_sq.template ptr<float>(),
            indices_device,
            n_indices,
            state_row_size
        );

        LOG_DEBUG("relocate_params_at_indices_gpu: Reset optimizer state for {} at {} indices (batched GPU kernel)",
                  name, n_indices);
    }

    // ===== Serialization =====

    namespace {
        constexpr uint32_t ADAM_STATE_MAGIC = 0x4C464144;  // "LFAD"
        constexpr uint32_t ADAM_STATE_VERSION = 1;
    }

    void AdamOptimizer::serialize(std::ostream& os) const {
        // Write header
        os.write(reinterpret_cast<const char*>(&ADAM_STATE_MAGIC), sizeof(ADAM_STATE_MAGIC));
        os.write(reinterpret_cast<const char*>(&ADAM_STATE_VERSION), sizeof(ADAM_STATE_VERSION));

        // Write config
        os.write(reinterpret_cast<const char*>(&config_.lr), sizeof(config_.lr));
        os.write(reinterpret_cast<const char*>(&config_.beta1), sizeof(config_.beta1));
        os.write(reinterpret_cast<const char*>(&config_.beta2), sizeof(config_.beta2));
        os.write(reinterpret_cast<const char*>(&config_.eps), sizeof(config_.eps));
        os.write(reinterpret_cast<const char*>(&config_.growth_factor), sizeof(config_.growth_factor));
        os.write(reinterpret_cast<const char*>(&config_.initial_capacity), sizeof(config_.initial_capacity));

        // Write per-param LRs count and data
        uint32_t num_param_lrs = static_cast<uint32_t>(config_.param_lrs.size());
        os.write(reinterpret_cast<const char*>(&num_param_lrs), sizeof(num_param_lrs));
        for (const auto& [name, lr] : config_.param_lrs) {
            uint32_t name_len = static_cast<uint32_t>(name.size());
            os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            os.write(name.data(), name_len);
            os.write(reinterpret_cast<const char*>(&lr), sizeof(lr));
        }

        // Count valid states (those with valid tensors)
        uint32_t num_states = 0;
        for (const auto& [name, state] : states_) {
            if (state.exp_avg.is_valid() && state.exp_avg_sq.is_valid()) {
                ++num_states;
            }
        }
        os.write(reinterpret_cast<const char*>(&num_states), sizeof(num_states));

        // Write each valid state
        for (const auto& [name, state] : states_) {
            // Skip invalid states
            if (!state.exp_avg.is_valid() || !state.exp_avg_sq.is_valid()) {
                continue;
            }

            // Write name
            uint32_t name_len = static_cast<uint32_t>(name.size());
            os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            os.write(name.data(), name_len);

            // Write state metadata
            os.write(reinterpret_cast<const char*>(&state.step_count), sizeof(state.step_count));
            os.write(reinterpret_cast<const char*>(&state.capacity), sizeof(state.capacity));
            os.write(reinterpret_cast<const char*>(&state.size), sizeof(state.size));

            // Write tensors using tensor serialization
            os << state.exp_avg;
            os << state.exp_avg_sq;
        }

        LOG_DEBUG("Serialized AdamOptimizer: {} states", num_states);
    }

    void AdamOptimizer::deserialize(std::istream& is) {
        // Read and verify header
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != ADAM_STATE_MAGIC) {
            throw std::runtime_error("Invalid AdamOptimizer checkpoint: wrong magic");
        }
        if (version != ADAM_STATE_VERSION) {
            throw std::runtime_error("Unsupported AdamOptimizer checkpoint version: " + std::to_string(version));
        }

        // Read config
        is.read(reinterpret_cast<char*>(&config_.lr), sizeof(config_.lr));
        is.read(reinterpret_cast<char*>(&config_.beta1), sizeof(config_.beta1));
        is.read(reinterpret_cast<char*>(&config_.beta2), sizeof(config_.beta2));
        is.read(reinterpret_cast<char*>(&config_.eps), sizeof(config_.eps));
        is.read(reinterpret_cast<char*>(&config_.growth_factor), sizeof(config_.growth_factor));
        is.read(reinterpret_cast<char*>(&config_.initial_capacity), sizeof(config_.initial_capacity));

        // Read per-param LRs
        uint32_t num_param_lrs;
        is.read(reinterpret_cast<char*>(&num_param_lrs), sizeof(num_param_lrs));
        config_.param_lrs.clear();
        for (uint32_t i = 0; i < num_param_lrs; ++i) {
            uint32_t name_len;
            is.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            std::string name(name_len, '\0');
            is.read(name.data(), name_len);
            double lr;
            is.read(reinterpret_cast<char*>(&lr), sizeof(lr));
            config_.param_lrs[name] = lr;
        }

        // Read number of states
        uint32_t num_states;
        is.read(reinterpret_cast<char*>(&num_states), sizeof(num_states));

        // Read each state
        states_.clear();
        for (uint32_t i = 0; i < num_states; ++i) {
            // Read name
            uint32_t name_len;
            is.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            std::string name(name_len, '\0');
            is.read(name.data(), name_len);

            // Read state metadata
            AdamParamState state;
            is.read(reinterpret_cast<char*>(&state.step_count), sizeof(state.step_count));
            is.read(reinterpret_cast<char*>(&state.capacity), sizeof(state.capacity));
            is.read(reinterpret_cast<char*>(&state.size), sizeof(state.size));

            // Read tensors (loaded on CPU, move to CUDA)
            is >> state.exp_avg;
            is >> state.exp_avg_sq;
            state.exp_avg = state.exp_avg.cuda();
            state.exp_avg_sq = state.exp_avg_sq.cuda();

            // Reserve capacity for future growth (MCMC densification)
            // Use saved capacity or compute based on config if larger
            const size_t target_capacity = std::max(state.capacity,
                compute_new_capacity(state.size, state.size));
            if (target_capacity > state.size) {
                state.exp_avg.reserve(target_capacity);
                state.exp_avg_sq.reserve(target_capacity);
                state.capacity = target_capacity;
                LOG_DEBUG("Reserved capacity {} for optimizer state '{}' (size: {})",
                         target_capacity, name, state.size);
            }

            states_[name] = std::move(state);
        }

        LOG_DEBUG("Deserialized AdamOptimizer: {} states", num_states);
    }

    void AdamOptimizer::reserve_capacity(size_t capacity) {
        for (auto& [name, state] : states_) {
            if (capacity > state.capacity) {
                state.exp_avg.reserve(capacity);
                state.exp_avg_sq.reserve(capacity);
                state.capacity = capacity;
                LOG_DEBUG("Reserved capacity {} for optimizer state '{}' (current size: {})",
                         capacity, name, state.size);
            }
        }
    }

} // namespace lfs::training
