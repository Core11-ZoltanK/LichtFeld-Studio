/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/parameters.hpp"
#include <expected>
#include <string>
#include <string_view>

namespace lfs::vis {

// Single source of truth for training parameters in GUI mode.
// Stores GT (from JSON, immutable) and current (user-editable) params per strategy.
class ParameterManager {
public:
    // Lazy-load params from JSON. No-op after first successful call.
    std::expected<void, std::string> ensureLoaded();

    // Get current (editable) params for strategy ("mcmc" or "default")
    [[nodiscard]] lfs::core::param::OptimizationParameters& getCurrentParams(std::string_view strategy);
    [[nodiscard]] const lfs::core::param::OptimizationParameters& getCurrentParams(std::string_view strategy) const;

    // Get GT (original JSON) params for strategy
    [[nodiscard]] const lfs::core::param::OptimizationParameters& getGTParams(std::string_view strategy) const;

    // Get/set loading params
    [[nodiscard]] lfs::core::param::LoadingParams& getLoadingParams() { return loading_params_; }
    [[nodiscard]] const lfs::core::param::LoadingParams& getLoadingParams() const { return loading_params_; }

    // Reset current to GT. Empty strategy resets both.
    void resetToDefaults(std::string_view strategy = "");

    // Apply CLI overrides to current params (CLI > config > defaults)
    void applyCLIOverrides(const lfs::core::param::OptimizationParameters& cli_params);

    // Active strategy accessors
    [[nodiscard]] const std::string& getActiveStrategy() const { return active_strategy_; }
    void setActiveStrategy(std::string_view strategy);

    [[nodiscard]] lfs::core::param::OptimizationParameters& getActiveParams();
    [[nodiscard]] const lfs::core::param::OptimizationParameters& getActiveParams() const;

    // Create TrainingParameters with current active params and given paths
    [[nodiscard]] lfs::core::param::TrainingParameters createForDataset(
        const std::filesystem::path& data_path,
        const std::filesystem::path& output_path) const;

    [[nodiscard]] bool isLoaded() const { return loaded_; }

private:
    bool loaded_ = false;
    std::string active_strategy_ = "mcmc";

    // GT params (from JSON, immutable after load)
    lfs::core::param::OptimizationParameters mcmc_gt_;
    lfs::core::param::OptimizationParameters default_gt_;

    // Current params (user-editable)
    lfs::core::param::OptimizationParameters mcmc_current_;
    lfs::core::param::OptimizationParameters default_current_;

    lfs::core::param::LoadingParams loading_params_;
};

} // namespace lfs::vis
