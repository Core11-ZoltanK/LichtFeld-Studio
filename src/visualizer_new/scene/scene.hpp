/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::vis {

    // Node identifier (-1 = invalid/root)
    using NodeId = int32_t;
    constexpr NodeId NULL_NODE = -1;

    // Node types
    enum class NodeType : uint8_t {
        SPLAT,   // Contains gaussian splat data
        GROUP    // Empty transform node for organization
    };

    // Selection group with ID, name, and color
    struct SelectionGroup {
        uint8_t id = 0;              // 1-255, 0 means unselected
        std::string name;
        glm::vec3 color{1.0f, 0.0f, 0.0f};
        size_t count = 0;            // Number of selected Gaussians
        bool locked = false;         // If true, painting with other groups won't overwrite
    };

    class Scene {
    public:
        struct Node {
            NodeId id = NULL_NODE;
            NodeId parent_id = NULL_NODE;          // NULL_NODE = root level
            std::vector<NodeId> children;
            NodeType type = NodeType::SPLAT;

            std::string name;
            std::unique_ptr<lfs::core::SplatData> model;  // Only for SPLAT type
            glm::mat4 local_transform{1.0f};
            mutable glm::mat4 world_transform{1.0f};
            mutable bool transform_dirty = true;
            bool visible = true;
            bool locked = false;
            size_t gaussian_count = 0;
            glm::vec3 centroid{0.0f};

            // Legacy accessor (transform is now local_transform)
            [[nodiscard]] glm::mat4& transform() { return local_transform; }
            [[nodiscard]] const glm::mat4& transform() const { return local_transform; }
        };

        Scene();
        ~Scene() = default;

        // Delete copy operations
        Scene(const Scene&) = delete;
        Scene& operator=(const Scene&) = delete;

        // Allow move operations
        Scene(Scene&&) = default;
        Scene& operator=(Scene&&) = default;

        // Node management (by name - legacy API)
        void addNode(const std::string& name, std::unique_ptr<lfs::core::SplatData> model);
        void removeNode(const std::string& name, bool keep_children = false);
        void replaceNodeModel(const std::string& name, std::unique_ptr<lfs::core::SplatData> model);
        void setNodeVisibility(const std::string& name, bool visible);
        void setNodeLocked(const std::string& name, bool locked);
        [[nodiscard]] bool isNodeLocked(const std::string& name) const;
        void setNodeTransform(const std::string& name, const glm::mat4& transform);
        glm::mat4 getNodeTransform(const std::string& name) const;
        bool renameNode(const std::string& old_name, const std::string& new_name);
        void clear();
        std::pair<std::string, std::string> cycleVisibilityWithNames();

        // Scene graph operations
        NodeId addGroup(const std::string& name, NodeId parent = NULL_NODE);
        NodeId addSplat(const std::string& name, std::unique_ptr<lfs::core::SplatData> model, NodeId parent = NULL_NODE);
        void reparent(NodeId node, NodeId new_parent);
        // Duplicate a node (and all children recursively for groups)
        // Returns new node name (original name with "_copy" or "_copy_N" suffix)
        [[nodiscard]] std::string duplicateNode(const std::string& name);
        [[nodiscard]] const glm::mat4& getWorldTransform(NodeId node) const;
        [[nodiscard]] std::vector<NodeId> getRootNodes() const;
        [[nodiscard]] Node* getNodeById(NodeId id);
        [[nodiscard]] const Node* getNodeById(NodeId id) const;

        // Check if node is effectively visible (considers parent hierarchy)
        [[nodiscard]] bool isNodeEffectivelyVisible(NodeId id) const;

        // Get bounding box center for a node (for groups: includes all descendants)
        [[nodiscard]] glm::vec3 getNodeBoundsCenter(NodeId id) const;
        [[nodiscard]] bool getNodeBounds(NodeId id, glm::vec3& out_min, glm::vec3& out_max) const;

        // Get combined model for rendering
        const lfs::core::SplatData* getCombinedModel() const;

        // Get transforms for visible nodes (for kernel-based transform)
        std::vector<glm::mat4> getVisibleNodeTransforms() const;

        // Get per-Gaussian transform indices tensor (for kernel-based transform)
        // Returns nullptr if no transforms needed (single node with identity transform)
        std::shared_ptr<lfs::core::Tensor> getTransformIndices() const;

        // Get node index in combined model (-1 if not found or not visible)
        [[nodiscard]] int getVisibleNodeIndex(const std::string& name) const;

        // Selection mask for highlighting selected Gaussians
        // Returns nullptr if no selection (all zeros = no selection)
        std::shared_ptr<lfs::core::Tensor> getSelectionMask() const;

        // Set selection for Gaussians (indices into combined model)
        void setSelection(const std::vector<size_t>& selected_indices);

        // Set selection mask directly from GPU tensor (for GPU-based brush selection)
        void setSelectionMask(std::shared_ptr<lfs::core::Tensor> mask);

        // Clear all selection
        void clearSelection();

        // Check if any Gaussians are selected
        bool hasSelection() const;

        // Selection groups management
        uint8_t addSelectionGroup(const std::string& name, const glm::vec3& color);
        void removeSelectionGroup(uint8_t id);
        void renameSelectionGroup(uint8_t id, const std::string& name);
        void setSelectionGroupColor(uint8_t id, const glm::vec3& color);
        void setSelectionGroupLocked(uint8_t id, bool locked);
        [[nodiscard]] bool isSelectionGroupLocked(uint8_t id) const;
        void setActiveSelectionGroup(uint8_t id) { active_selection_group_ = id; }
        [[nodiscard]] uint8_t getActiveSelectionGroup() const { return active_selection_group_; }
        [[nodiscard]] const std::vector<SelectionGroup>& getSelectionGroups() const { return selection_groups_; }
        [[nodiscard]] const SelectionGroup* getSelectionGroup(uint8_t id) const;
        void updateSelectionGroupCounts();
        void clearSelectionGroup(uint8_t id);
        void resetSelectionState();  // Full reset: clear mask, remove all groups, create default

        // Direct queries
        size_t getNodeCount() const { return nodes_.size(); }
        size_t getTotalGaussianCount() const;
        std::vector<const Node*> getNodes() const;
        const Node* getNode(const std::string& name) const;
        Node* getMutableNode(const std::string& name);
        bool hasNodes() const { return !nodes_.empty(); }

        // Get visible nodes for split view
        std::vector<const Node*> getVisibleNodes() const;

        // Mark scene data as changed (e.g., after modifying a node's deleted mask)
        void markDirty() { invalidateCache(); }

        // Permanently remove soft-deleted gaussians from all nodes
        // Returns total number of gaussians removed
        size_t applyDeleted();

    private:
        std::vector<Node> nodes_;
        std::unordered_map<NodeId, size_t> id_to_index_;  // NodeId -> index in nodes_
        NodeId next_node_id_ = 0;

        // Caching for combined model
        mutable std::unique_ptr<lfs::core::SplatData> cached_combined_;
        mutable std::shared_ptr<lfs::core::Tensor> cached_transform_indices_;
        mutable std::vector<glm::mat4> cached_transforms_;
        mutable bool cache_valid_ = false;

        // Selection mask: UInt8 [N], value = group ID (0=unselected, 1-255=group ID)
        mutable std::shared_ptr<lfs::core::Tensor> selection_mask_;
        mutable bool has_selection_ = false;

        // Selection groups (ID 0 is reserved for "unselected")
        std::vector<SelectionGroup> selection_groups_;
        uint8_t active_selection_group_ = 1;  // Default to group 1
        uint8_t next_group_id_ = 1;

        void invalidateCache() { cache_valid_ = false; }
        void rebuildCacheIfNeeded() const;

        // Scene graph helpers
        void markTransformDirty(NodeId node);
        void updateWorldTransform(const Node& node) const;

        // Helper to find group by ID
        SelectionGroup* findGroup(uint8_t id);
        const SelectionGroup* findGroup(uint8_t id) const;
    };

} // namespace lfs::vis