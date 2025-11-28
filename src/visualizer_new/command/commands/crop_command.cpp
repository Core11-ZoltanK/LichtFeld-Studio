/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "crop_command.hpp"
#include "scene/scene_manager.hpp"
#include "core_new/logger.hpp"

namespace lfs::vis::command {

    CropCommand::CropCommand(SceneManager* scene_manager,
                             std::string node_name,
                             lfs::core::Tensor old_deleted_mask,
                             lfs::core::Tensor new_deleted_mask)
        : scene_manager_(scene_manager),
          node_name_(std::move(node_name)),
          old_deleted_mask_(std::move(old_deleted_mask)),
          new_deleted_mask_(std::move(new_deleted_mask)) {
    }

    void CropCommand::undo() {
        if (!scene_manager_) return;

        auto* node = scene_manager_->getScene().getMutableNode(node_name_);
        if (!node || !node->model) {
            LOG_WARN("CropCommand::undo - node '{}' not found", node_name_);
            return;
        }

        node->model->deleted() = old_deleted_mask_.clone();
    }

    void CropCommand::redo() {
        if (!scene_manager_) return;

        auto* node = scene_manager_->getScene().getMutableNode(node_name_);
        if (!node || !node->model) {
            LOG_WARN("CropCommand::redo - node '{}' not found", node_name_);
            return;
        }

        node->model->deleted() = new_deleted_mask_.clone();
    }

} // namespace lfs::vis::command
