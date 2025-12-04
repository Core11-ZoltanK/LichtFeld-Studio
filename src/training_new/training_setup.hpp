/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "dataset.hpp"
#include "trainer.hpp"
#include "core_new/tensor.hpp"
#include <expected>
#include <memory>

// Forward declaration
namespace lfs::vis {
    class Scene;
}

namespace lfs::training {
    struct TrainingSetup {
        std::unique_ptr<Trainer> trainer;
        std::shared_ptr<CameraDataset> dataset;
        lfs::core::Tensor scene_center;
    };

    // Legacy: Set up training from parameters (creates Trainer internally)
    std::expected<TrainingSetup, std::string> setupTraining(const lfs::core::param::TrainingParameters& params);

    /**
     * @brief Load training data into Scene
     *
     * This is the unified loading path for both headless and GUI modes.
     * Loads cameras and point cloud, initializes SplatData, and stores everything in Scene.
     *
     * After calling this, create a Trainer with: Trainer(scene)
     *
     * @param params Training parameters (including data path)
     * @param scene Scene to populate with training data
     * @return Error message on failure
     */
    std::expected<void, std::string> loadTrainingDataIntoScene(
        const lfs::core::param::TrainingParameters& params,
        lfs::vis::Scene& scene);
} // namespace lfs::training
