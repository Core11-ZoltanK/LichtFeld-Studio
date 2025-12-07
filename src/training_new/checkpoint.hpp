/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/parameters.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>

namespace lfs::training {

    class IStrategy;

    /*
     * ============================================================================
     * LichtFeld Studio Checkpoint Format (.resume) - Version 1
     * ============================================================================
     *
     * Binary format for storing complete training state including Gaussian
     * parameters, optimizer state, and training configuration.
     *
     * FILE STRUCTURE
     * ─────────────────────────────────────────────────────────────────────────
     *
     * ┌─────────────────────────────────────────────────────────────────────┐
     * │                         CHECKPOINT HEADER                           │
     * │                          (40 bytes fixed)                           │
     * ├─────────────────────────────────────────────────────────────────────┤
     * │ Offset │ Size │ Type     │ Field              │ Description         │
     * │────────┼──────┼──────────┼────────────────────┼─────────────────────│
     * │ 0x00   │ 4    │ uint32   │ magic              │ 0x4C464B50 "LFKP"   │
     * │ 0x04   │ 4    │ uint32   │ version            │ Format version (1)  │
     * │ 0x08   │ 4    │ int32    │ iteration          │ Training iteration  │
     * │ 0x0C   │ 4    │ uint32   │ num_gaussians      │ Gaussian count      │
     * │ 0x10   │ 4    │ int32    │ sh_degree          │ Max SH degree       │
     * │ 0x14   │ 4    │ uint32   │ flags              │ Feature flags       │
     * │ 0x18   │ 8    │ uint64   │ params_json_offset │ JSON params offset  │
     * │ 0x20   │ 8    │ uint64   │ params_json_size   │ JSON params size    │
     * └─────────────────────────────────────────────────────────────────────┘
     *
     * ┌─────────────────────────────────────────────────────────────────────┐
     * │                        STRATEGY TYPE STRING                         │
     * ├─────────────────────────────────────────────────────────────────────┤
     * │ 4 bytes  │ uint32  │ String length                                  │
     * │ N bytes  │ char[]  │ Strategy name ("mcmc" or "default")            │
     * └─────────────────────────────────────────────────────────────────────┘
     *
     * ┌─────────────────────────────────────────────────────────────────────┐
     * │                          SPLATDATA BLOCK                            │
     * │                    (Gaussian parameters + state)                    │
     * ├─────────────────────────────────────────────────────────────────────┤
     * │ 4 bytes  │ uint32  │ Magic: 0x4C465350 "LFSP"                       │
     * │ 4 bytes  │ uint32  │ Version (3)                                    │
     * │ 4 bytes  │ int32   │ Active SH degree                               │
     * │ 4 bytes  │ int32   │ Max SH degree                                  │
     * │ 4 bytes  │ float   │ Scene scale                                    │
     * │ ─────────┴─────────┴──────────────────────────────────────────────  │
     * │ TENSOR   │ means      [N, 3]     │ float32  │ Gaussian positions   │
     * │ TENSOR   │ sh0        [N, 1, 3]  │ float32  │ DC spherical harmonic│
     * │ TENSOR   │ scaling    [N, 3]     │ float32  │ Log-scale factors    │
     * │ TENSOR   │ rotation   [N, 4]     │ float32  │ Quaternions (wxyz)   │
     * │ TENSOR   │ opacity    [N, 1]     │ float32  │ Logit opacity        │
     * │ TENSOR   │ shN        [N, K, 3]  │ float32  │ Higher SH (if deg>0) │
     * │ ─────────┴─────────┴──────────────────────────────────────────────  │
     * │ 1 byte   │ uint8   │ has_deleted flag                               │
     * │ TENSOR   │ deleted    [N]        │ bool     │ (optional)           │
     * │ 1 byte   │ uint8   │ has_densification flag                         │
     * │ TENSOR   │ densif_info [N]       │ float32  │ (optional)           │
     * └─────────────────────────────────────────────────────────────────────┘
     *
     * ┌─────────────────────────────────────────────────────────────────────┐
     * │                         STRATEGY STATE                              │
     * │              (Optimizer + Scheduler, strategy-specific)             │
     * ├─────────────────────────────────────────────────────────────────────┤
     * │ MCMC Strategy:                                                      │
     * │   • Adam optimizer state (6 param groups)                           │
     * │   • Scheduler state                                                 │
     * │   • Binoms tensor [num_gaussians]                                   │
     * │   • Dead count tensor [num_gaussians]                               │
     * │                                                                     │
     * │ Default Strategy:                                                   │
     * │   • Adam optimizer state (6 param groups)                           │
     * │   • Scheduler state                                                 │
     * └─────────────────────────────────────────────────────────────────────┘
     *
     * ┌─────────────────────────────────────────────────────────────────────┐
     * │                     TRAINING PARAMETERS (JSON)                      │
     * │              (at params_json_offset, params_json_size)              │
     * ├─────────────────────────────────────────────────────────────────────┤
     * │ {                                                                   │
     * │   "optimization": { ... },   // Learning rates, iterations, etc.   │
     * │   "dataset": {                                                      │
     * │     "data_path": "...",      // Original dataset path              │
     * │     "output_path": "...",    // Output directory                   │
     * │     ...                                                             │
     * │   }                                                                 │
     * │ }                                                                   │
     * └─────────────────────────────────────────────────────────────────────┘
     *
     * TENSOR SERIALIZATION FORMAT
     * ─────────────────────────────────────────────────────────────────────────
     * Each tensor is stored as:
     *   • 4 bytes: uint32 magic (0x4C465354 "LFST")
     *   • 4 bytes: uint32 version
     *   • 4 bytes: uint32 dtype enum
     *   • 4 bytes: uint32 ndim
     *   • ndim×8 bytes: uint64[] shape
     *   • N bytes: raw data (CPU, contiguous)
     *
     * FLAGS (CheckpointFlags)
     * ─────────────────────────────────────────────────────────────────────────
     *   Bit 0 (0x01): HAS_BILATERAL_GRID - Reserved for future use
     *
     * NOTES
     * ─────────────────────────────────────────────────────────────────────────
     *   • All multi-byte values are little-endian
     *   • Tensors are stored in CPU memory, loaded to GPU on deserialize
     *   • Strategy type must match when loading (mcmc ↔ mcmc only)
     *   • CLI parameters (--iterations, --data, --output) override checkpoint
     *
     * ============================================================================
     */

    // Checkpoint file magic and version
    constexpr uint32_t CHECKPOINT_MAGIC = 0x4C464B50;    // "LFKP" (LichtFeld CheckPoint)
    constexpr uint32_t CHECKPOINT_VERSION = 1;

    // Flags for optional components
    enum class CheckpointFlags : uint32_t {
        NONE = 0,
        HAS_BILATERAL_GRID = 1 << 0,
    };

    inline CheckpointFlags operator|(CheckpointFlags a, CheckpointFlags b) {
        return static_cast<CheckpointFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }

    inline CheckpointFlags operator&(CheckpointFlags a, CheckpointFlags b) {
        return static_cast<CheckpointFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    }

    inline bool has_flag(CheckpointFlags flags, CheckpointFlags flag) {
        return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(flag)) != 0;
    }

    /**
     * @brief Checkpoint file header (40 bytes)
     */
    struct CheckpointHeader {
        uint32_t magic = CHECKPOINT_MAGIC;
        uint32_t version = CHECKPOINT_VERSION;
        int32_t iteration = 0;
        uint32_t num_gaussians = 0;
        int32_t sh_degree = 0;
        CheckpointFlags flags = CheckpointFlags::NONE;
        uint64_t params_json_offset = 0;
        uint64_t params_json_size = 0;
    };

    /**
     * @brief Save a complete training checkpoint
     *
     * Saves SplatData, optimizer state (Adam moments), scheduler state,
     * and training parameters as JSON.
     *
     * @param path Output directory (checkpoint saved to path/checkpoints/)
     * @param iteration Current training iteration
     * @param strategy Strategy containing model and optimizer
     * @param params Training parameters
     * @return Error message on failure
     */
    std::expected<void, std::string> save_checkpoint(
        const std::filesystem::path& path,
        int iteration,
        const IStrategy& strategy,
        const lfs::core::param::TrainingParameters& params);

    /**
     * @brief Load checkpoint header only (for inspection)
     *
     * @param path Path to checkpoint file
     * @return Header or error message
     */
    std::expected<CheckpointHeader, std::string> load_checkpoint_header(
        const std::filesystem::path& path);

    /**
     * @brief Load a complete training checkpoint
     *
     * Restores:
     * - SplatData into the provided model
     * - Optimizer state
     * - Scheduler state
     * - Training parameters
     *
     * @param path Path to checkpoint file
     * @param strategy Strategy to restore state into
     * @param params Output: loaded training parameters
     * @return Iteration number on success, error message on failure
     */
    std::expected<int, std::string> load_checkpoint(
        const std::filesystem::path& path,
        IStrategy& strategy,
        lfs::core::param::TrainingParameters& params);

    /**
     * @brief Load only SplatData from checkpoint (for viewing without training)
     *
     * This extracts just the Gaussian parameters, skipping optimizer/scheduler state.
     * Useful for viewing checkpoints without full training resumption.
     *
     * @param path Path to checkpoint file
     * @return SplatData on success, error message on failure
     */
    std::expected<lfs::core::SplatData, std::string> load_checkpoint_splat_data(
        const std::filesystem::path& path);

    /**
     * @brief Load only training parameters from checkpoint
     *
     * This extracts the stored training parameters (including dataset path)
     * without loading the model or optimizer state.
     *
     * @param path Path to checkpoint file
     * @return TrainingParameters on success, error message on failure
     */
    std::expected<lfs::core::param::TrainingParameters, std::string> load_checkpoint_params(
        const std::filesystem::path& path);

} // namespace lfs::training
