/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "scene/scene.hpp"
#include "core_new/logger.hpp"
#include "core_new/splat_data_transform.hpp"

#include <algorithm>
#include <limits>
#include <array>
#include <cmath>
#include <functional>
#include <glm/gtc/quaternion.hpp>
#include <numeric>
#include <ranges>

namespace lfs::vis {

    Scene::Scene() {
        // Create default selection group
        addSelectionGroup("Group 1", glm::vec3(0.0f));
    }

    // Helper to compute centroid from model (GPU computation, single copy)
    static glm::vec3 computeCentroid(const lfs::core::SplatData* model) {
        if (!model || model->size() == 0) {
            return glm::vec3(0.0f);
        }
        const auto& means = model->means_raw();
        if (!means.is_valid() || means.size(0) == 0) {
            return glm::vec3(0.0f);
        }
        // Compute mean on GPU, copy only 3 floats back
        auto centroid_tensor = means.mean({0}, false);
        glm::vec3 result(
            centroid_tensor.slice(0, 0, 1).item<float>(),
            centroid_tensor.slice(0, 1, 2).item<float>(),
            centroid_tensor.slice(0, 2, 3).item<float>());
        // Guard against NaN from empty or invalid data
        if (std::isnan(result.x) || std::isnan(result.y) || std::isnan(result.z)) {
            return glm::vec3(0.0f);
        }
        return result;
    }

    void Scene::addNode(const std::string& name, std::unique_ptr<lfs::core::SplatData> model) {
        // Calculate gaussian count and centroid before moving
        const size_t gaussian_count = static_cast<size_t>(model->size());
        const glm::vec3 centroid = computeCentroid(model.get());

        // Check if name already exists
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });

        if (it != nodes_.end()) {
            // Replace existing model
            it->model = std::move(model);
            it->gaussian_count = gaussian_count;
            it->centroid = centroid;
        } else {
            // Add new splat node
            const NodeId id = next_node_id_++;
            Node node;
            node.id = id;
            node.type = NodeType::SPLAT;
            node.name = name;
            node.model = std::move(model);
            node.gaussian_count = gaussian_count;
            node.centroid = centroid;

            id_to_index_[id] = nodes_.size();
            nodes_.push_back(std::move(node));
        }

        invalidateCache();
        LOG_DEBUG("Added node '{}': {} gaussians", name, gaussian_count);
    }

    void Scene::removeNode(const std::string& name, const bool keep_children) {
        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });
        if (it == nodes_.end()) return;

        const NodeId id = it->id;
        const NodeId parent_id = it->parent_id;

        // Remove from parent's children list
        if (parent_id != NULL_NODE) {
            if (auto* parent = getNodeById(parent_id)) {
                auto& children = parent->children;
                children.erase(std::remove(children.begin(), children.end(), id), children.end());
            }
        }

        if (keep_children) {
            // Reparent children to the removed node's parent (or root if no parent)
            for (const NodeId child_id : it->children) {
                if (auto* child = getNodeById(child_id)) {
                    child->parent_id = parent_id;
                    child->transform_dirty = true;
                    // Add to new parent's children list
                    if (parent_id != NULL_NODE) {
                        if (auto* new_parent = getNodeById(parent_id)) {
                            new_parent->children.push_back(child_id);
                        }
                    }
                }
            }
        } else {
            // Recursively remove children
            // Copy children list since removeNode modifies it
            const std::vector<NodeId> children_copy = it->children;
            for (const NodeId child_id : children_copy) {
                if (const auto* child = getNodeById(child_id)) {
                    removeNode(child->name, false);
                }
            }
        }

        // Remove from lookup and vector (re-find iterator since recursive calls may have invalidated it)
        const auto it_final = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });
        if (it_final == nodes_.end()) return; // Already removed somehow

        id_to_index_.erase(id);
        const size_t removed_index = static_cast<size_t>(std::distance(nodes_.begin(), it_final));
        nodes_.erase(it_final);

        // Update indices for nodes after removed one
        for (auto& [node_id, index] : id_to_index_) {
            if (index > removed_index) --index;
        }

        invalidateCache();
        LOG_DEBUG("Removed node '{}'{}", name, keep_children ? " (children kept)" : "");
    }

    void Scene::replaceNodeModel(const std::string& name, std::unique_ptr<lfs::core::SplatData> model) {
        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });

        if (it != nodes_.end()) {
            const size_t gaussian_count = static_cast<size_t>(model->size());
            const glm::vec3 centroid = computeCentroid(model.get());
            LOG_DEBUG("replaceNodeModel '{}': {} -> {} gaussians", name, it->gaussian_count, gaussian_count);
            it->model = std::move(model);
            it->gaussian_count = gaussian_count;
            it->centroid = centroid;
            invalidateCache();
        } else {
            LOG_WARN("replaceNodeModel: node '{}' not found", name);
        }
    }

    void Scene::setNodeVisibility(const std::string& name, bool visible) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });

        if (it != nodes_.end() && it->visible != visible) {
            it->visible = visible;
            invalidateCache();
        }
    }

    void Scene::setNodeLocked(const std::string& name, const bool locked) {
        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& n) { return n.name == name; });
        if (it != nodes_.end()) {
            it->locked = locked;
        }
    }

    bool Scene::isNodeLocked(const std::string& name) const {
        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });
        return it != nodes_.end() && it->locked;
    }

    void Scene::setNodeTransform(const std::string& name, const glm::mat4& transform) {
        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });

        if (it != nodes_.end()) {
            it->local_transform = transform;
            markTransformDirty(it->id);
            invalidateCache();
        }
    }

    glm::mat4 Scene::getNodeTransform(const std::string& name) const {
        const auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });

        if (it != nodes_.end()) {
            return it->local_transform;
        }
        return glm::mat4(1.0f);
    }

    void Scene::clear() {
        nodes_.clear();
        id_to_index_.clear();
        next_node_id_ = 0;
        cached_combined_.reset();
        cache_valid_ = false;
    }

    std::pair<std::string, std::string> Scene::cycleVisibilityWithNames() {
        static constexpr std::pair<const char*, const char*> EMPTY_PAIR = {"", ""};

        if (nodes_.size() <= 1) {
            return EMPTY_PAIR;
        }

        std::string hidden_name, shown_name;

        // Find first visible node using modular arithmetic as suggested
        auto visible = std::find_if(nodes_.begin(), nodes_.end(),
                                    [](const Node& n) { return n.visible; });

        if (visible != nodes_.end()) {
            visible->visible = false;
            hidden_name = visible->name;

            auto next_index = (std::distance(nodes_.begin(), visible) + 1) % nodes_.size();
            auto next = nodes_.begin() + next_index;

            next->visible = true;
            shown_name = next->name;
        } else {
            // No visible nodes, show first
            nodes_[0].visible = true;
            shown_name = nodes_[0].name;
        }

        invalidateCache();
        return {hidden_name, shown_name};
    }

    const lfs::core::SplatData* Scene::getCombinedModel() const {
        rebuildCacheIfNeeded();
        return cached_combined_.get();
    }

    size_t Scene::getTotalGaussianCount() const {
        size_t total = 0;
        for (const auto& node : nodes_) {
            if (node.visible) {
                total += node.gaussian_count;
            }
        }
        return total;
    }

    std::vector<const Scene::Node*> Scene::getNodes() const {
        std::vector<const Node*> result;
        result.reserve(nodes_.size());
        for (const auto& node : nodes_) {
            result.push_back(&node);
        }
        return result;
    }

    std::vector<const Scene::Node*> Scene::getVisibleNodes() const {
        std::vector<const Node*> visible;
        for (const auto& node : nodes_) {
            if (node.visible && node.model) {
                visible.push_back(&node);
            }
        }
        return visible;
    }

    const Scene::Node* Scene::getNode(const std::string& name) const {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });
        return (it != nodes_.end()) ? &(*it) : nullptr;
    }

    Scene::Node* Scene::getMutableNode(const std::string& name) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&name](const Node& node) { return node.name == name; });
        if (it != nodes_.end()) {
            invalidateCache();
            return &(*it);
        }
        return nullptr;
    }

    void Scene::rebuildCacheIfNeeded() const {
        if (cache_valid_)
            return;

        LOG_DEBUG("rebuildCacheIfNeeded - rebuilding");

        // Collect visible nodes (we need both model and transform)
        // Use effective visibility which considers parent hierarchy
        auto visible_nodes = nodes_ | std::views::filter([this](const auto& node) {
                                 return node.model && isNodeEffectivelyVisible(node.id);
                             }) |
                             std::ranges::to<std::vector<std::reference_wrapper<const Node>>>();

        if (visible_nodes.empty()) {
            cached_combined_.reset();
            cached_transform_indices_.reset();
            cached_transforms_.clear();
            cache_valid_ = true;
            return;
        }

        // All transforms are now handled by the kernel for proper covariance transformation.
        // We just combine the raw Gaussian data and create transform indices.

        // Calculate totals and find max SH degree in one pass
        struct ModelStats {
            size_t total_gaussians = 0;
            int max_sh_degree = 0;
            float total_scene_scale = 0.0f;
            bool has_shN = false;
        };

        auto stats = std::accumulate(
            visible_nodes.begin(), visible_nodes.end(), ModelStats{},
            [](ModelStats acc, const std::reference_wrapper<const Node>& node_ref) {
                const auto* model = node_ref.get().model.get();
                acc.total_gaussians += model->size();

                // Calculate SH degree from shN dimensions
                int sh_degree = 0;
                const auto& shN_tensor = model->shN_raw();
                if (shN_tensor.is_valid() && shN_tensor.ndim() >= 2 && shN_tensor.size(1) > 0) {
                    int shN_coeffs = static_cast<int>(shN_tensor.size(1));
                    sh_degree = static_cast<int>(std::round(std::sqrt(shN_coeffs + 1))) - 1;
                    sh_degree = std::clamp(sh_degree, 0, 3);
                }

                acc.max_sh_degree = std::max(acc.max_sh_degree, sh_degree);
                acc.total_scene_scale += model->get_scene_scale();
                acc.has_shN = acc.has_shN || (shN_tensor.numel() > 0 && shN_tensor.size(1) > 0);
                return acc;
            });

        // Get device from first model (all should be on CUDA)
        lfs::core::Device device = visible_nodes[0].get().model->means_raw().device();

        // Calculate SH dimensions
        int sh0_coeffs = 1;
        int shN_coeffs = (stats.max_sh_degree > 0) ? ((stats.max_sh_degree + 1) * (stats.max_sh_degree + 1) - 1) : 0;

        // Pre-allocate all tensors
        using lfs::core::Tensor;
        Tensor means = Tensor::empty({static_cast<size_t>(stats.total_gaussians), 3}, device);
        Tensor sh0 = Tensor::empty({static_cast<size_t>(stats.total_gaussians), static_cast<size_t>(sh0_coeffs), 3}, device);
        Tensor shN = (shN_coeffs > 0) ? Tensor::zeros({static_cast<size_t>(stats.total_gaussians), static_cast<size_t>(shN_coeffs), 3}, device) : Tensor::empty({static_cast<size_t>(stats.total_gaussians), 0, 3}, device);
        Tensor opacity = Tensor::empty({static_cast<size_t>(stats.total_gaussians), 1}, device);
        Tensor scaling = Tensor::empty({static_cast<size_t>(stats.total_gaussians), 3}, device);
        Tensor rotation = Tensor::empty({static_cast<size_t>(stats.total_gaussians), 4}, device);

        // Check if any node has a deletion mask
        const bool has_any_deleted = std::any_of(visible_nodes.begin(), visible_nodes.end(),
            [](const auto& node_ref) { return node_ref.get().model->has_deleted_mask(); });

        Tensor deleted = has_any_deleted
            ? Tensor::zeros({static_cast<size_t>(stats.total_gaussians)}, device, lfs::core::DataType::Bool)
            : Tensor();

        // Create transform indices tensor on CPU first (will be moved to CUDA later)
        std::vector<int> transform_indices_data(stats.total_gaussians);

        // Collect transforms for visible nodes
        cached_transforms_.clear();
        cached_transforms_.reserve(visible_nodes.size());

        // Copy data from each node (no CPU-side transforms - kernel handles them)
        size_t offset = 0;
        int node_index = 0;
        for (const auto& node_ref : visible_nodes) {
            const auto& node = node_ref.get();
            const auto* model = node.model.get();
            const auto size = model->size();

            // Store this node's world transform (includes parent hierarchy)
            cached_transforms_.push_back(getWorldTransform(node.id));

            // Fill transform indices for this node's Gaussians
            std::fill(transform_indices_data.begin() + offset,
                     transform_indices_data.begin() + offset + size,
                     node_index);

            // Direct copy of all Gaussian data (no CPU-side transform)
            means.slice(0, offset, offset + size) = model->means_raw();
            scaling.slice(0, offset, offset + size) = model->scaling_raw();
            rotation.slice(0, offset, offset + size) = model->rotation_raw();
            sh0.slice(0, offset, offset + size) = model->sh0_raw();
            opacity.slice(0, offset, offset + size) = model->opacity_raw();

            // Copy shN if we have coefficients
            if (shN_coeffs > 0) {
                const auto& model_shN = model->shN_raw();
                int model_shN_coeffs = (model_shN.is_valid() && model_shN.ndim() >= 2) ? static_cast<int>(model_shN.size(1)) : 0;

                if (model_shN_coeffs > 0) {
                    int coeffs_to_copy = std::min(model_shN_coeffs, shN_coeffs);

                    // Slice in both dimensions: rows and coefficients
                    shN.slice(0, offset, offset + size).slice(1, 0, coeffs_to_copy) =
                        model_shN.slice(1, 0, coeffs_to_copy);
                }
            }

            // Copy deletion mask if present
            if (has_any_deleted && model->has_deleted_mask()) {
                deleted.slice(0, offset, offset + size) = model->deleted();
            }

            offset += size;
            node_index++;
        }

        // Create transform indices tensor on CUDA
        // from_vector for int automatically creates Int32 tensor
        cached_transform_indices_ = std::make_shared<Tensor>(
            Tensor::from_vector(transform_indices_data, {stats.total_gaussians}, lfs::core::Device::CPU).cuda());

        // Create the combined model
        cached_combined_ = std::make_unique<lfs::core::SplatData>(
            stats.max_sh_degree,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            stats.total_scene_scale / visible_nodes.size());

        if (has_any_deleted) {
            cached_combined_->deleted() = std::move(deleted);
        }

        cache_valid_ = true;
    }

    std::vector<glm::mat4> Scene::getVisibleNodeTransforms() const {
        rebuildCacheIfNeeded();
        return cached_transforms_;
    }

    std::shared_ptr<lfs::core::Tensor> Scene::getTransformIndices() const {
        rebuildCacheIfNeeded();
        return cached_transform_indices_;
    }

    int Scene::getVisibleNodeIndex(const std::string& name) const {
        int index = 0;
        for (const auto& node : nodes_) {
            if (!node.visible || !node.model) { continue; }
            if (node.name == name) { return index; }
            ++index;
        }
        return -1;
    }

    std::shared_ptr<lfs::core::Tensor> Scene::getSelectionMask() const {
        if (!has_selection_) {
            return nullptr;
        }
        return selection_mask_;
    }

    void Scene::setSelection(const std::vector<size_t>& selected_indices) {
        // Get total gaussian count
        size_t total = getTotalGaussianCount();
        if (total == 0) {
            clearSelection();
            return;
        }

        // Create or resize selection mask
        if (!selection_mask_ || selection_mask_->size(0) != total) {
            // Create new mask (all zeros on CPU first, then move to GPU)
            selection_mask_ = std::make_shared<lfs::core::Tensor>(
                lfs::core::Tensor::zeros({total}, lfs::core::Device::CPU, lfs::core::DataType::UInt8));
        } else {
            // Clear existing mask (set all to 0)
            auto mask_cpu = selection_mask_->cpu();
            std::memset(mask_cpu.ptr<uint8_t>(), 0, total);
            *selection_mask_ = mask_cpu;
        }

        if (!selected_indices.empty()) {
            auto mask_cpu = selection_mask_->cpu();
            uint8_t* mask_data = mask_cpu.ptr<uint8_t>();
            for (size_t idx : selected_indices) {
                if (idx < total) {
                    mask_data[idx] = 1;
                }
            }
            *selection_mask_ = mask_cpu.cuda();
            has_selection_ = true;
        } else {
            has_selection_ = false;
        }
    }

    void Scene::setSelectionMask(std::shared_ptr<lfs::core::Tensor> mask) {
        selection_mask_ = std::move(mask);
        has_selection_ = selection_mask_ && selection_mask_->is_valid() && selection_mask_->numel() > 0;
    }

    void Scene::clearSelection() {
        selection_mask_.reset();
        has_selection_ = false;
    }

    bool Scene::hasSelection() const {
        return has_selection_;
    }

    bool Scene::renameNode(const std::string& old_name, const std::string& new_name) {
        // Check if new name already exists (case-sensitive)
        if (old_name == new_name) {
            return true; // Same name, consider it successful
        }

        // Check if new name already exists
        auto existing_it = std::find_if(nodes_.begin(), nodes_.end(),
                                        [&new_name](const Node& node) {
                                            return node.name == new_name;
                                        });

        if (existing_it != nodes_.end()) {
            LOG_INFO("Scene: Cannot rename '{}' to '{}' - name already exists", old_name, new_name);
            return false; // Name already exists
        }

        // Find the node to rename
        auto it = std::find_if(nodes_.begin(), nodes_.end(),
                               [&old_name](const Node& node) {
                                   return node.name == old_name;
                               });

        if (it != nodes_.end()) {
            std::string prev_name = it->name;
            it->name = new_name;
            invalidateCache();
            LOG_INFO("Scene: Renamed node '{}' to '{}'", prev_name, new_name);
            return true;
        }

        LOG_WARN("Scene: Cannot find node '{}' to rename", old_name);
        return false;
    }

    size_t Scene::applyDeleted() {
        size_t total_removed = 0;

        for (auto& node : nodes_) {
            if (node.model && node.model->has_deleted_mask()) {
                const size_t removed = node.model->apply_deleted();
                if (removed > 0) {
                    node.gaussian_count = node.model->size();
                    node.centroid = computeCentroid(node.model.get());
                    total_removed += removed;
                }
            }
        }

        if (total_removed > 0) {
            invalidateCache();
            clearSelection();
        }

        return total_removed;
    }

    // Selection group color palette
    static constexpr std::array<glm::vec3, 8> GROUP_COLOR_PALETTE = {{
        {1.0f, 0.3f, 0.3f},  // Red
        {0.3f, 1.0f, 0.3f},  // Green
        {0.3f, 0.5f, 1.0f},  // Blue
        {1.0f, 1.0f, 0.3f},  // Yellow
        {1.0f, 0.5f, 0.0f},  // Orange
        {0.8f, 0.3f, 1.0f},  // Purple
        {0.3f, 1.0f, 1.0f},  // Cyan
        {1.0f, 0.5f, 0.8f},  // Pink
    }};

    SelectionGroup* Scene::findGroup(const uint8_t id) {
        const auto it = std::find_if(selection_groups_.begin(), selection_groups_.end(),
                                     [id](const SelectionGroup& g) { return g.id == id; });
        return (it != selection_groups_.end()) ? &(*it) : nullptr;
    }

    const SelectionGroup* Scene::findGroup(const uint8_t id) const {
        const auto it = std::find_if(selection_groups_.begin(), selection_groups_.end(),
                                     [id](const SelectionGroup& g) { return g.id == id; });
        return (it != selection_groups_.end()) ? &(*it) : nullptr;
    }

    uint8_t Scene::addSelectionGroup(const std::string& name, const glm::vec3& color) {
        if (next_group_id_ == 0) {
            LOG_WARN("Maximum selection groups reached");
            return 0;
        }

        SelectionGroup group;
        group.id = next_group_id_++;
        group.name = name.empty() ? "Group " + std::to_string(group.id) : name;
        group.color = (color == glm::vec3(0.0f))
            ? GROUP_COLOR_PALETTE[(group.id - 1) % GROUP_COLOR_PALETTE.size()]
            : color;
        group.count = 0;

        selection_groups_.push_back(group);
        active_selection_group_ = group.id;

        LOG_DEBUG("Added selection group '{}' (ID {})", group.name, group.id);
        return group.id;
    }

    void Scene::removeSelectionGroup(const uint8_t id) {
        const auto it = std::find_if(selection_groups_.begin(), selection_groups_.end(),
                                     [id](const SelectionGroup& g) { return g.id == id; });
        if (it == selection_groups_.end()) return;

        clearSelectionGroup(id);
        const std::string name = it->name;
        selection_groups_.erase(it);

        if (active_selection_group_ == id) {
            active_selection_group_ = selection_groups_.empty() ? 0 : selection_groups_.back().id;
        }

        LOG_DEBUG("Removed selection group '{}' (ID {})", name, id);
    }

    void Scene::renameSelectionGroup(const uint8_t id, const std::string& name) {
        if (auto* group = findGroup(id)) {
            group->name = name;
        }
    }

    void Scene::setSelectionGroupColor(const uint8_t id, const glm::vec3& color) {
        if (auto* group = findGroup(id)) {
            group->color = color;
        }
    }

    void Scene::setSelectionGroupLocked(const uint8_t id, const bool locked) {
        if (auto* group = findGroup(id)) {
            group->locked = locked;
        }
    }

    bool Scene::isSelectionGroupLocked(const uint8_t id) const {
        const auto* group = findGroup(id);
        return group ? group->locked : false;
    }

    const SelectionGroup* Scene::getSelectionGroup(const uint8_t id) const {
        return findGroup(id);
    }

    void Scene::updateSelectionGroupCounts() {
        for (auto& group : selection_groups_) {
            group.count = 0;
        }

        if (!selection_mask_ || !selection_mask_->is_valid()) return;

        const auto mask_cpu = selection_mask_->cpu();
        const uint8_t* data = mask_cpu.ptr<uint8_t>();
        const size_t n = mask_cpu.numel();

        for (size_t i = 0; i < n; ++i) {
            const uint8_t group_id = data[i];
            if (auto* group = findGroup(group_id)) {
                group->count++;
            }
        }
    }

    void Scene::clearSelectionGroup(const uint8_t id) {
        if (!selection_mask_ || !selection_mask_->is_valid()) return;

        auto mask_cpu = selection_mask_->cpu();
        uint8_t* data = mask_cpu.ptr<uint8_t>();
        const size_t n = mask_cpu.numel();

        bool any_remaining = false;
        for (size_t i = 0; i < n; ++i) {
            if (data[i] == id) {
                data[i] = 0;
            } else if (data[i] > 0) {
                any_remaining = true;
            }
        }

        *selection_mask_ = mask_cpu.cuda();
        has_selection_ = any_remaining;

        if (auto* group = findGroup(id)) {
            group->count = 0;
        }
    }

    void Scene::resetSelectionState() {
        selection_mask_.reset();
        has_selection_ = false;
        selection_groups_.clear();
        next_group_id_ = 1;
        addSelectionGroup("Group 1", glm::vec3(0.0f));
    }

    // ========== Scene Graph Operations ==========

    NodeId Scene::addGroup(const std::string& name, const NodeId parent) {
        const NodeId id = next_node_id_++;
        Node node;
        node.id = id;
        node.parent_id = parent;
        node.type = NodeType::GROUP;
        node.name = name;

        // Add to parent's children
        if (parent != NULL_NODE) {
            if (auto* p = getNodeById(parent)) {
                p->children.push_back(id);
            }
        }

        id_to_index_[id] = nodes_.size();
        nodes_.push_back(std::move(node));
        invalidateCache();

        LOG_DEBUG("Added group node '{}' (id={})", name, id);
        return id;
    }

    NodeId Scene::addSplat(const std::string& name, std::unique_ptr<lfs::core::SplatData> model, const NodeId parent) {
        const size_t gaussian_count = static_cast<size_t>(model->size());
        const glm::vec3 centroid = computeCentroid(model.get());

        const NodeId id = next_node_id_++;
        Node node;
        node.id = id;
        node.parent_id = parent;
        node.type = NodeType::SPLAT;
        node.name = name;
        node.model = std::move(model);
        node.gaussian_count = gaussian_count;
        node.centroid = centroid;

        // Add to parent's children
        if (parent != NULL_NODE) {
            if (auto* p = getNodeById(parent)) {
                p->children.push_back(id);
            }
        }

        id_to_index_[id] = nodes_.size();
        nodes_.push_back(std::move(node));
        invalidateCache();

        LOG_DEBUG("Added splat node '{}' (id={}, {} gaussians)", name, id, gaussian_count);
        return id;
    }

    std::string Scene::duplicateNode(const std::string& name) {
        const auto* src_node = getNode(name);
        if (!src_node) return "";

        // Helper to generate unique name
        auto generate_unique_name = [this](const std::string& base_name) -> std::string {
            std::string new_name = base_name + "_copy";
            int counter = 2;
            while (std::any_of(nodes_.begin(), nodes_.end(),
                               [&new_name](const Node& n) { return n.name == new_name; })) {
                new_name = base_name + "_copy_" + std::to_string(counter++);
            }
            return new_name;
        };

        // Recursive helper to duplicate node and children
        // IMPORTANT: Pass NodeId, not reference, because nodes_ vector may reallocate!
        std::function<NodeId(NodeId, NodeId)> duplicate_recursive =
            [&](const NodeId src_id, const NodeId parent_id) -> NodeId {
            // Must re-lookup node each time as vector may have reallocated
            const auto* src = getNodeById(src_id);
            if (!src) return NULL_NODE;

            // Copy data we need BEFORE calling addGroup/addSplat (which may reallocate)
            const std::string src_name_copy = src->name;
            const NodeType src_type = src->type;
            const glm::mat4 src_transform = src->local_transform;
            const bool src_visible = src->visible;
            const bool src_locked = src->locked;
            const std::vector<NodeId> src_children = src->children;  // Copy children list

            const std::string new_name = generate_unique_name(src_name_copy);

            NodeId new_id = NULL_NODE;
            if (src_type == NodeType::GROUP) {
                new_id = addGroup(new_name, parent_id);
            } else {
                // Re-lookup src after potential reallocation check
                const auto* src_for_model = getNodeById(src_id);
                if (src_for_model && src_for_model->model) {
                    // Clone SplatData
                    const auto& model = *src_for_model->model;
                    auto cloned = std::make_unique<lfs::core::SplatData>(
                        model.get_max_sh_degree(),
                        model.means_raw().clone(), model.sh0_raw().clone(), model.shN_raw().clone(),
                        model.scaling_raw().clone(), model.rotation_raw().clone(), model.opacity_raw().clone(),
                        model.get_scene_scale());
                    cloned->set_active_sh_degree(model.get_active_sh_degree());
                    new_id = addSplat(new_name, std::move(cloned), parent_id);
                }
            }

            // Copy transform and visibility (re-lookup new node as vector may have changed)
            if (auto* new_node = getNodeById(new_id)) {
                new_node->local_transform = src_transform;
                new_node->visible = src_visible;
                new_node->locked = src_locked;
                new_node->transform_dirty = true;
            }

            // Recursively duplicate children (using copied children list)
            for (const NodeId child_id : src_children) {
                duplicate_recursive(child_id, new_id);
            }

            return new_id;
        };

        // Store source info before any modifications
        const NodeId src_id = src_node->id;
        const NodeId src_parent_id = src_node->parent_id;
        const std::string result_name = generate_unique_name(src_node->name);

        duplicate_recursive(src_id, src_parent_id);

        invalidateCache();
        LOG_INFO("Duplicated node '{}' as '{}'", name, result_name);
        return result_name;
    }

    void Scene::reparent(const NodeId node_id, const NodeId new_parent) {
        auto* node = getNodeById(node_id);
        if (!node) return;

        // Prevent circular references
        if (new_parent != NULL_NODE) {
            NodeId check = new_parent;
            while (check != NULL_NODE) {
                if (check == node_id) {
                    LOG_WARN("Cannot reparent: would create cycle");
                    return;
                }
                const auto* check_node = getNodeById(check);
                check = check_node ? check_node->parent_id : NULL_NODE;
            }
        }

        // Remove from old parent
        if (node->parent_id != NULL_NODE) {
            if (auto* old_parent = getNodeById(node->parent_id)) {
                auto& children = old_parent->children;
                children.erase(std::remove(children.begin(), children.end(), node_id), children.end());
            }
        }

        // Add to new parent
        node->parent_id = new_parent;
        if (new_parent != NULL_NODE) {
            if (auto* p = getNodeById(new_parent)) {
                p->children.push_back(node_id);
            }
        }

        markTransformDirty(node_id);
        invalidateCache();
    }

    const glm::mat4& Scene::getWorldTransform(const NodeId node_id) const {
        const auto* node = getNodeById(node_id);
        if (!node) {
            static const glm::mat4 IDENTITY{1.0f};
            return IDENTITY;
        }
        updateWorldTransform(*node);
        return node->world_transform;
    }

    std::vector<NodeId> Scene::getRootNodes() const {
        std::vector<NodeId> roots;
        for (const auto& node : nodes_) {
            if (node.parent_id == NULL_NODE) {
                roots.push_back(node.id);
            }
        }
        return roots;
    }

    Scene::Node* Scene::getNodeById(const NodeId id) {
        const auto it = id_to_index_.find(id);
        if (it == id_to_index_.end()) return nullptr;
        return &nodes_[it->second];
    }

    const Scene::Node* Scene::getNodeById(const NodeId id) const {
        const auto it = id_to_index_.find(id);
        if (it == id_to_index_.end()) return nullptr;
        return &nodes_[it->second];
    }

    bool Scene::isNodeEffectivelyVisible(const NodeId id) const {
        const auto* node = getNodeById(id);
        if (!node) return false;

        // Check this node's visibility
        if (!node->visible) return false;

        // Recursively check parent visibility
        if (node->parent_id != NULL_NODE) {
            return isNodeEffectivelyVisible(node->parent_id);
        }

        return true;
    }

    void Scene::markTransformDirty(const NodeId node_id) {
        auto* node = getNodeById(node_id);
        if (!node || node->transform_dirty) return;

        node->transform_dirty = true;
        for (const NodeId child_id : node->children) {
            markTransformDirty(child_id);
        }
    }

    void Scene::updateWorldTransform(const Node& node) const {
        if (!node.transform_dirty) return;

        if (node.parent_id == NULL_NODE) {
            node.world_transform = node.local_transform;
        } else {
            const auto* parent = getNodeById(node.parent_id);
            if (parent) {
                updateWorldTransform(*parent);
                node.world_transform = parent->world_transform * node.local_transform;
            } else {
                node.world_transform = node.local_transform;
            }
        }
        node.transform_dirty = false;
    }

    bool Scene::getNodeBounds(const NodeId id, glm::vec3& out_min, glm::vec3& out_max) const {
        const auto* node = getNodeById(id);
        if (!node) return false;

        bool has_bounds = false;
        glm::vec3 total_min(std::numeric_limits<float>::max());
        glm::vec3 total_max(std::numeric_limits<float>::lowest());

        // Helper to expand bounds
        const auto expand_bounds = [&](const glm::vec3& min_b, const glm::vec3& max_b) {
            total_min = glm::min(total_min, min_b);
            total_max = glm::max(total_max, max_b);
            has_bounds = true;
        };

        // If this node has a model, include its bounds
        if (node->model && node->model->size() > 0) {
            glm::vec3 model_min, model_max;
            if (lfs::core::compute_bounds(*node->model, model_min, model_max)) {
                expand_bounds(model_min, model_max);
            }
        }

        // Recursively include children bounds
        for (const NodeId child_id : node->children) {
            glm::vec3 child_min, child_max;
            if (getNodeBounds(child_id, child_min, child_max)) {
                // Transform child bounds by child's local transform relative to this node
                const auto* child = getNodeById(child_id);
                if (child) {
                    // Transform the 8 corners of child bounding box
                    const glm::mat4& child_transform = child->local_transform;
                    glm::vec3 corners[8] = {
                        {child_min.x, child_min.y, child_min.z},
                        {child_max.x, child_min.y, child_min.z},
                        {child_min.x, child_max.y, child_min.z},
                        {child_max.x, child_max.y, child_min.z},
                        {child_min.x, child_min.y, child_max.z},
                        {child_max.x, child_min.y, child_max.z},
                        {child_min.x, child_max.y, child_max.z},
                        {child_max.x, child_max.y, child_max.z}
                    };
                    for (const auto& corner : corners) {
                        const glm::vec3 transformed = glm::vec3(child_transform * glm::vec4(corner, 1.0f));
                        expand_bounds(transformed, transformed);
                    }
                }
            }
        }

        if (has_bounds) {
            out_min = total_min;
            out_max = total_max;
        }
        return has_bounds;
    }

    glm::vec3 Scene::getNodeBoundsCenter(const NodeId id) const {
        glm::vec3 min_bounds, max_bounds;
        if (getNodeBounds(id, min_bounds, max_bounds)) {
            return (min_bounds + max_bounds) * 0.5f;
        }
        return glm::vec3(0.0f);
    }

} // namespace lfs::vis