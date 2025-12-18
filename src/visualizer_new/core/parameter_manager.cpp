/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "parameter_manager.hpp"
#include "core_new/logger.hpp"

namespace lfs::vis {

namespace {
    constexpr const char* MCMC_CONFIG_FILE = "mcmc_optimization_params.json";
    constexpr const char* DEFAULT_CONFIG_FILE = "default_optimization_params.json";
    constexpr const char* LOADING_CONFIG_FILE = "loading_params.json";
} // namespace

std::expected<void, std::string> ParameterManager::ensureLoaded() {
    if (loaded_) return {};

    const auto mcmc_path = lfs::core::param::get_parameter_file_path(MCMC_CONFIG_FILE);
    auto mcmc_result = lfs::core::param::read_optim_params_from_json(mcmc_path);
    if (!mcmc_result) {
        return std::unexpected("Failed to load MCMC params: " + mcmc_result.error());
    }
    mcmc_gt_ = *mcmc_result;
    mcmc_current_ = mcmc_gt_;

    const auto default_path = lfs::core::param::get_parameter_file_path(DEFAULT_CONFIG_FILE);
    auto default_result = lfs::core::param::read_optim_params_from_json(default_path);
    if (!default_result) {
        return std::unexpected("Failed to load default params: " + default_result.error());
    }
    default_gt_ = *default_result;
    default_current_ = default_gt_;

    const auto loading_path = lfs::core::param::get_parameter_file_path(LOADING_CONFIG_FILE);
    auto loading_result = lfs::core::param::read_loading_params_from_json(loading_path);
    if (!loading_result) {
        return std::unexpected("Failed to load loading params: " + loading_result.error());
    }
    loading_params_ = *loading_result;

    loaded_ = true;
    LOG_DEBUG("Parameters loaded from JSON config files");
    return {};
}

lfs::core::param::OptimizationParameters& ParameterManager::getCurrentParams(const std::string_view strategy) {
    return (strategy == "mcmc") ? mcmc_current_ : default_current_;
}

const lfs::core::param::OptimizationParameters& ParameterManager::getCurrentParams(const std::string_view strategy) const {
    return (strategy == "mcmc") ? mcmc_current_ : default_current_;
}

const lfs::core::param::OptimizationParameters& ParameterManager::getGTParams(const std::string_view strategy) const {
    return (strategy == "mcmc") ? mcmc_gt_ : default_gt_;
}

void ParameterManager::resetToDefaults(const std::string_view strategy) {
    if (strategy.empty() || strategy == "mcmc") {
        mcmc_current_ = mcmc_gt_;
    }
    if (strategy.empty() || strategy == "default") {
        default_current_ = default_gt_;
    }
    LOG_DEBUG("Reset params to defaults{}", strategy.empty() ? "" : " for " + std::string(strategy));
}

void ParameterManager::applyCLIOverrides(const lfs::core::param::OptimizationParameters& cli_params) {
    if (const auto result = ensureLoaded(); !result) {
        LOG_ERROR("Failed to load params: {}", result.error());
        return;
    }

    if (!cli_params.strategy.empty()) {
        setActiveStrategy(cli_params.strategy);
    }

    getActiveParams() = cli_params;

    LOG_INFO("CLI overrides: strategy={}, iter={}, max_cap={}, sh_degree={}",
             cli_params.strategy, cli_params.iterations, cli_params.max_cap, cli_params.sh_degree);
}

void ParameterManager::setActiveStrategy(const std::string_view strategy) {
    if (strategy == "mcmc" || strategy == "default") {
        active_strategy_ = std::string(strategy);
    } else {
        LOG_WARN("Invalid strategy '{}', keeping '{}'", strategy, active_strategy_);
    }
}

lfs::core::param::OptimizationParameters& ParameterManager::getActiveParams() {
    return getCurrentParams(active_strategy_);
}

const lfs::core::param::OptimizationParameters& ParameterManager::getActiveParams() const {
    return getCurrentParams(active_strategy_);
}

lfs::core::param::TrainingParameters ParameterManager::createForDataset(
    const std::filesystem::path& data_path,
    const std::filesystem::path& output_path) const {

    lfs::core::param::TrainingParameters params;
    params.optimization = getActiveParams();
    params.dataset.loading_params = loading_params_;
    params.dataset.data_path = data_path;
    params.dataset.output_path = output_path;
    return params;
}

} // namespace lfs::vis
