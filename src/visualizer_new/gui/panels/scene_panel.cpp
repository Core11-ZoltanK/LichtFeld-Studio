/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/scene_panel.hpp"
#include "core_new/logger.hpp"
#include "gui/utils/windows_utils.hpp"
#include "gui/windows/image_preview.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <imgui.h>
#include <stdexcept>

namespace lfs::vis::gui {

    using namespace lfs::core::events;

    // ScenePanel Implementation
    ScenePanel::ScenePanel(std::shared_ptr<const TrainerManager> trainer_manager) : m_trainer_manager(trainer_manager) {
        // Create image preview window
        m_imagePreview = std::make_unique<ImagePreview>();
        setupEventHandlers();
        LOG_DEBUG("ScenePanel created");
    }

    ScenePanel::~ScenePanel() {
        // Cleanup handled automatically
    }

    void ScenePanel::setupEventHandlers() {
        // Subscribe to events using the new event system
        state::SceneLoaded::when([this](const auto& event) {
            handleSceneLoaded(event);
        });

        state::SceneCleared::when([this](const auto&) {
            handleSceneCleared();
        });

        state::PLYAdded::when([this](const auto& event) {
            handlePLYAdded(event);
        });

        state::PLYRemoved::when([this](const auto& event) {
            handlePLYRemoved(event);
        });

        // Listen for PLY visibility changes to update checkboxes
        cmd::SetPLYVisibility::when([this](const auto& event) {
            // Update the visibility state in our local PLY nodes
            auto it = std::find_if(m_plyNodes.begin(), m_plyNodes.end(),
                                   [&event](const PLYNode& node) { return node.name == event.name; });
            if (it != m_plyNodes.end()) {
                it->visible = event.visible;
                LOG_TRACE("Updated PLY '{}' visibility in scene panel to: {}", event.name, event.visible);
            }
        });

        // Listen for GoToCamView to sync selection
        cmd::GoToCamView::when([this](const auto& event) {
            handleGoToCamView(event);
        });

        cmd::RenamePLY::when([this](const auto& event) {
            handlePLYRenamed(event);
        });

        state::NodeReparented::when([this](const auto& event) {
            const auto it = std::find_if(m_plyNodes.begin(), m_plyNodes.end(),
                [&event](const PLYNode& n) { return n.name == event.name; });
            if (it != m_plyNodes.end()) {
                it->parent_name = event.new_parent;
            }
        });

        ui::NodeSelected::when([this](const auto& event) {
            if (event.type == "PLY") {
                for (size_t i = 0; i < m_plyNodes.size(); ++i) {
                    if (m_plyNodes[i].name == event.path) {
                        for (auto& n : m_plyNodes) n.selected = false;
                        m_plyNodes[i].selected = true;
                        m_selectedPLYIndex = static_cast<int>(i);
                        break;
                    }
                }
            }
        });
    }

    void ScenePanel::handleSceneLoaded(const state::SceneLoaded& event) {
        LOG_DEBUG("Scene loaded event - type: {}",
                  event.type == state::SceneLoaded::Type::PLY ? "PLY" : "Dataset");

        if (event.type == state::SceneLoaded::Type::PLY) {
            m_currentMode = DisplayMode::PLYSceneGraph;
            m_plyNodes.clear();
            m_selectedPLYIndex = -1;
            m_activeTab = TabType::PLYs; // Switch to PLY tab
            LOG_TRACE("Switched to PLY scene graph mode");
        } else if (event.type == state::SceneLoaded::Type::Dataset) {
            m_currentMode = DisplayMode::DatasetImages;
            m_plyNodes.clear();
            m_selectedPLYIndex = -1;
            m_activeTab = TabType::Images; // Switch to Images tab
            LOG_TRACE("Switched to dataset images mode");
            if (!event.path.empty()) {
                loadImageCams(event.path);
            }
        }
    }

    void ScenePanel::handleSceneCleared() {
        LOG_DEBUG("Clearing scene panel data");
        // Clear all data
        m_imagePaths.clear();
        m_selectedImageIndex = -1;
        m_plyNodes.clear();
        m_selectedPLYIndex = -1;
        m_currentMode = DisplayMode::Empty;
        // Keep the active tab as is - user might want to stay on the same tab
    }

    void ScenePanel::handlePLYAdded(const state::PLYAdded& event) {
        const auto it = std::find_if(m_plyNodes.begin(), m_plyNodes.end(),
                               [&event](const PLYNode& n) { return n.name == event.name; });

        if (it != m_plyNodes.end()) {
            it->gaussian_count = event.node_gaussians;
            it->parent_name = event.parent_name;
            it->is_group = event.is_group;
        } else {
            m_plyNodes.push_back({
                .name = event.name,
                .parent_name = event.parent_name,
                .is_group = event.is_group,
                .visible = event.is_visible,
                .selected = false,
                .locked = false,
                .gaussian_count = event.node_gaussians
            });
        }
        updateModeFromTab();
    }

    void ScenePanel::handlePLYRemoved(const state::PLYRemoved& event) {
        if (event.children_kept) {
            for (auto& n : m_plyNodes) {
                if (n.parent_name == event.name) {
                    n.parent_name = event.parent_of_removed;
                }
            }
        }

        const auto it = std::find_if(m_plyNodes.begin(), m_plyNodes.end(),
                               [&event](const PLYNode& n) { return n.name == event.name; });
        if (it != m_plyNodes.end()) {
            m_plyNodes.erase(it);
            if (m_selectedPLYIndex >= static_cast<int>(m_plyNodes.size())) {
                m_selectedPLYIndex = -1;
            }
        }
        updateModeFromTab();
    }

    void ScenePanel::handleGoToCamView(const cmd::GoToCamView& event) {
        // Find the image path for this camera ID
        for (const auto& [path, cam_id] : m_PathToCamId) {
            if (cam_id == event.cam_id) {
                // Find index in sorted image list
                if (auto it = std::find(m_imagePaths.begin(), m_imagePaths.end(), path); it != m_imagePaths.end()) {
                    m_selectedImageIndex = static_cast<int>(std::distance(m_imagePaths.begin(), it));
                    m_needsScrollToSelection = true; // Mark that we need to scroll
                    LOG_TRACE("Synced image selection to camera ID {} (index {})",
                              event.cam_id, m_selectedImageIndex);
                }
                break;
            }
        }
    }

    void ScenePanel::updatePLYNodes() {
        LOG_TRACE("Updating PLY nodes");
        // For now, we'll rebuild the node list when we get events
        // In a more sophisticated implementation, we'd query the scene directly
    }

    void ScenePanel::updateModeFromTab() {
        // Update display mode based on active tab and available data
        // Prioritize PLYs if available and active tab is PLYs
        if (m_activeTab == TabType::PLYs && !m_plyNodes.empty()) {
            m_currentMode = DisplayMode::PLYSceneGraph;
            LOG_TRACE("Display mode set to PLYSceneGraph");
        } else if (m_activeTab == TabType::Images && !m_imagePaths.empty()) {
            m_currentMode = DisplayMode::DatasetImages;
            LOG_TRACE("Display mode set to DatasetImages");
        } else if (!m_plyNodes.empty()) {
            // Fall back to PLYs if available (even if Images tab was selected but no images)
            m_currentMode = DisplayMode::PLYSceneGraph;
            m_activeTab = TabType::PLYs;
            LOG_TRACE("Fallback to PLYSceneGraph mode");
        } else if (!m_imagePaths.empty()) {
            // Fall back to Images if PLYs not available
            m_currentMode = DisplayMode::DatasetImages;
            m_activeTab = TabType::Images;
            LOG_TRACE("Fallback to DatasetImages mode");
        } else {
            m_currentMode = DisplayMode::Empty;
            LOG_TRACE("Display mode set to Empty");
        }
    }

    bool ScenePanel::hasImages() const {
        return !m_imagePaths.empty();
    }

    bool ScenePanel::hasPLYs() const {
        return !m_plyNodes.empty();
    }

    void ScenePanel::render(bool* p_open, const UIContext* /*ctx*/) {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.5f, 0.5f, 0.5f, 0.8f));

        if (!ImGui::Begin("Scene", p_open)) {
            ImGui::End();
            ImGui::PopStyleColor();
            return;
        }

        // Make buttons smaller to fit the narrow panel
        float button_width = ImGui::GetContentRegionAvail().x;

        if (ImGui::Button("Import dataset", ImVec2(button_width, 0))) {
            // Request to show file browser
            LOG_DEBUG("Opening file browser from scene panel");

            // Fire the callback to open file browser with empty path
            if (m_onDatasetLoad) {
                m_onDatasetLoad(std::filesystem::path("")); // Empty path signals to open browser
            }
#ifdef WIN32
            // show native windows file dialog for folder selection
            OpenDatasetFolderDialog();

            // hide the file browser
            cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
        }

        if (ImGui::Button("Open .ply", ImVec2(button_width, 0))) {
            // Request to show file browser
            LOG_DEBUG("Opening file browser from scene panel");

            // Fire the callback to open file browser with empty path
            if (m_onDatasetLoad) {
                m_onDatasetLoad(std::filesystem::path("")); // Empty path signals to open browser
            }
#ifdef WIN32
            // show native windows file dialog for folder selection
            OpenPlyFileDialog();

            // hide the file browser
            cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif // WIN32
        }

        if (ImGui::Button("Refresh", ImVec2(button_width * 0.48f, 0))) {
            if (m_currentMode == DisplayMode::DatasetImages && !m_currentDatasetPath.empty()) {
                LOG_DEBUG("Refreshing dataset images");
                loadImageCams(m_currentDatasetPath);
            } else if (m_currentMode == DisplayMode::PLYSceneGraph) {
                LOG_DEBUG("Refreshing PLY nodes");
                updatePLYNodes();
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Clear", ImVec2(button_width * 0.48f, 0))) {
            LOG_INFO("Clearing scene from panel");
            // Clear everything
            handleSceneCleared();

            // Also clear the actual scene data
            cmd::ClearScene{}.emit();
        }

        ImGui::Separator();

        // Render tabs if we have any data
        if (hasImages() || hasPLYs()) {
            if (ImGui::BeginTabBar("SceneTabs", ImGuiTabBarFlags_None)) {

                // PLYs tab - show first if we have PLYs (prioritize PLYs)
                if (hasPLYs()) {
                    bool plys_tab_selected = ImGui::BeginTabItem("PLYs");
                    if (plys_tab_selected) {
                        if (m_activeTab != TabType::PLYs) {
                            m_activeTab = TabType::PLYs;
                            m_currentMode = DisplayMode::PLYSceneGraph;
                            LOG_TRACE("Switched to PLYs tab");
                        }
                        renderPLYSceneGraph();
                        ImGui::EndTabItem();
                    }
                }

                // Images tab - show second
                if (hasImages()) {
                    bool images_tab_selected = ImGui::BeginTabItem("Images");
                    if (images_tab_selected) {
                        if (m_activeTab != TabType::Images) {
                            m_activeTab = TabType::Images;
                            m_currentMode = DisplayMode::DatasetImages;
                            LOG_TRACE("Switched to Images tab");
                        }
                        renderImageList();
                        ImGui::EndTabItem();
                    }
                }

                ImGui::EndTabBar();
            }
        } else {
            // No data loaded - show empty state
            ImGui::Text("No data loaded.");
        }

        ImGui::End();
        ImGui::PopStyleColor();

        // Render image preview window if open
        if (m_showImagePreview && m_imagePreview) {
            m_imagePreview->render(&m_showImagePreview);
        }
    }
    void ScenePanel::startRenaming(int nodeIndex) {
        if (nodeIndex < 0 || nodeIndex >= static_cast<int>(m_plyNodes.size())) {
            return;
        }

        m_renameState.is_renaming = true;
        m_renameState.renaming_index = nodeIndex;
        m_renameState.focus_input = true;

        // Copy current name to buffer
        const std::string& current_name = m_plyNodes[nodeIndex].name;
        strncpy(m_renameState.buffer, current_name.c_str(), sizeof(m_renameState.buffer) - 1);
        m_renameState.buffer[sizeof(m_renameState.buffer) - 1] = '\0';

        LOG_DEBUG("Started renaming PLY at index {} ('{}')", nodeIndex, current_name);
    }

    void ScenePanel::finishRenaming() {
        if (!m_renameState.is_renaming || m_renameState.renaming_index < 0) {
            return;
        }

        std::string new_name(m_renameState.buffer);
        new_name = new_name.substr(0, new_name.find_last_not_of(" \t\n\r") + 1); // trim whitespace

        if (!new_name.empty()) {
            const std::string old_name = m_plyNodes[m_renameState.renaming_index].name;

            if (new_name != old_name) {
                // Check if name already exists
                bool name_exists = std::any_of(m_plyNodes.begin(), m_plyNodes.end(),
                                               [&new_name](const PLYNode& node) {
                                                   return node.name == new_name;
                                               });

                if (name_exists) {
                    LOG_WARN("Name '{}' already exists, keeping original name '{}'", new_name, old_name);
                } else {
                    // Emit rename command
                    cmd::RenamePLY{
                        .old_name = old_name,
                        .new_name = new_name}
                        .emit();
                    LOG_INFO("Emitted rename command: '{}' -> '{}'", old_name, new_name);
                }
            }
        }

        cancelRenaming();
    }

    void ScenePanel::cancelRenaming() {
        m_renameState.is_renaming = false;
        m_renameState.renaming_index = -1;
        m_renameState.focus_input = false;
        m_renameState.input_was_active = false; // Reset the tracking flag
        m_renameState.escape_pressed = false;
        memset(m_renameState.buffer, 0, sizeof(m_renameState.buffer));
    }

    void ScenePanel::renderPLYSceneGraph() {
        ImGui::BeginChild("SceneGraph", ImVec2(0, 0), ImGuiChildFlags_None);

        // Keyboard shortcuts
        if (ImGui::IsWindowFocused() && !m_renameState.is_renaming) {
            if (ImGui::IsKeyPressed(ImGuiKey_F2) && m_selectedPLYIndex >= 0 &&
                m_selectedPLYIndex < static_cast<int>(m_plyNodes.size())) {
                startRenaming(m_selectedPLYIndex);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Escape) && m_selectedPLYIndex >= 0) {
                for (auto& n : m_plyNodes) n.selected = false;
                m_selectedPLYIndex = -1;
                ui::NodeDeselected{}.emit();
            }
        }

        constexpr ImGuiTreeNodeFlags ROOT_FLAGS = ImGuiTreeNodeFlags_DefaultOpen |
                                                  ImGuiTreeNodeFlags_OpenOnArrow |
                                                  ImGuiTreeNodeFlags_SpanAvailWidth |
                                                  ImGuiTreeNodeFlags_Framed;

        if (ImGui::TreeNodeEx("Scene", ROOT_FLAGS)) {
            renderModelsFolder();
            ImGui::TreePop();
        }

        // Summary
        if (!m_plyNodes.empty()) {
            ImGui::Separator();
            size_t total = 0;
            for (const auto& node : m_plyNodes) {
                if (node.visible) total += node.gaussian_count;
            }
            ImGui::TextDisabled("Visible: %zu gaussians", total);
        }

        ImGui::EndChild();
    }

    void ScenePanel::renderModelsFolder() {
        constexpr ImGuiTreeNodeFlags FOLDER_FLAGS = ImGuiTreeNodeFlags_DefaultOpen |
                                                    ImGuiTreeNodeFlags_OpenOnArrow |
                                                    ImGuiTreeNodeFlags_SpanAvailWidth;

        const std::string label = std::format("Models ({})", m_plyNodes.size());
        if (!ImGui::TreeNodeEx(label.c_str(), FOLDER_FLAGS)) return;

        // Drop target for moving nodes to root
        handleDragDrop("", true);

        // Context menu for folder
        if (ImGui::BeginPopupContextItem("##ModelsMenu")) {
            if (ImGui::MenuItem("Add PLY...")) {
                cmd::ShowWindow{.window_name = "file_browser", .show = true}.emit();
#ifdef WIN32
                OpenPlyFileDialog();
                cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif
            }
            if (ImGui::MenuItem("Add Group")) {
                cmd::AddGroup{.name = "New Group", .parent_name = ""}.emit();
            }
            ImGui::EndPopup();
        }

        // Only render root-level nodes (parent_name is empty)
        for (size_t i = 0; i < m_plyNodes.size(); ++i) {
            if (m_plyNodes[i].parent_name.empty()) {
                renderModelNode(i);
            }
        }

        if (m_plyNodes.empty()) {
            ImGui::TextDisabled("No models loaded");
            ImGui::TextDisabled("Right-click to add...");
        }

        ImGui::TreePop();
    }

    void ScenePanel::renderModelNode(const size_t index) {
        auto& node = m_plyNodes[index];
        const int idx = static_cast<int>(index);
        ImGui::PushID(idx);

        // Visibility toggle
        if (ImGui::SmallButton(node.visible ? "[*]" : "[ ]")) {
            node.visible = !node.visible;
            cmd::SetPLYVisibility{.name = node.name, .visible = node.visible}.emit();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(node.visible ? "Hide" : "Show");
        }
        ImGui::SameLine();

        const bool is_renaming = m_renameState.is_renaming && m_renameState.renaming_index == idx;

        if (is_renaming) {
            if (m_renameState.focus_input) {
                ImGui::SetKeyboardFocusHere();
                m_renameState.focus_input = false;
            }

            constexpr ImGuiInputTextFlags INPUT_FLAGS = ImGuiInputTextFlags_AutoSelectAll |
                                                        ImGuiInputTextFlags_EnterReturnsTrue;
            const bool entered = ImGui::InputText("##rename", m_renameState.buffer,
                                                  sizeof(m_renameState.buffer), INPUT_FLAGS);
            const bool is_focused = ImGui::IsItemFocused();
            if (ImGui::IsItemActive()) m_renameState.input_was_active = true;

            if (entered) {
                finishRenaming();
            } else if (ImGui::IsKeyPressed(ImGuiKey_Escape) ||
                       (m_renameState.input_was_active && !is_focused)) {
                cancelRenaming();
            }
        } else {
            const bool has_children = node.is_group && std::ranges::any_of(m_plyNodes,
                [&node](const PLYNode& n) { return n.parent_name == node.name; });

            constexpr ImGuiTreeNodeFlags BASE_FLAGS = ImGuiTreeNodeFlags_OpenOnArrow;
            ImGuiTreeNodeFlags flags = BASE_FLAGS;
            if (node.selected) flags |= ImGuiTreeNodeFlags_Selected;
            if (!has_children) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            if (node.is_group) flags |= ImGuiTreeNodeFlags_DefaultOpen;

            const std::string label = node.is_group
                ? node.name
                : std::format("{} ({:L})", node.name, node.gaussian_count);

            const bool is_open = ImGui::TreeNodeEx(label.c_str(), flags);

            // Capture state before adding lock button
            const bool hovered = ImGui::IsItemHovered();
            const bool clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
            const bool toggled = ImGui::IsItemToggledOpen();

            if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                ImGui::OpenPopup(("##ctx_" + node.name).c_str());
            }

            // Lock button
            ImGui::SameLine();
            static constexpr ImVec4 TRANSPARENT{0, 0, 0, 0};
            static constexpr ImVec4 HOVER_COLOR{0.3f, 0.3f, 0.3f, 0.5f};
            static constexpr ImVec4 LOCKED_TEXT_COLOR{1.0f, 0.4f, 0.4f, 1.0f};
            ImGui::PushStyleColor(ImGuiCol_Button, TRANSPARENT);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, HOVER_COLOR);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, HOVER_COLOR);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 0));
            if (node.locked) ImGui::PushStyleColor(ImGuiCol_Text, LOCKED_TEXT_COLOR);

            if (ImGui::SmallButton((std::string(node.locked ? "L##" : "U##") + node.name).c_str())) {
                node.locked = !node.locked;
                cmd::SetNodeLocked{.name = node.name, .locked = node.locked}.emit();
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip(node.locked ? "Unlock" : "Lock");

            if (node.locked) ImGui::PopStyleColor();
            ImGui::PopStyleVar();
            ImGui::PopStyleColor(3);

            // Drag source
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                ImGui::SetDragDropPayload("SCENE_NODE", node.name.c_str(), node.name.size() + 1);
                ImGui::Text("Move: %s", node.name.c_str());
                ImGui::EndDragDropSource();
            }

            if (node.is_group) handleDragDrop(node.name, true);

            // Selection
            if (clicked && !toggled) {
                if (node.selected) {
                    node.selected = false;
                    m_selectedPLYIndex = -1;
                    ui::NodeDeselected{}.emit();
                } else {
                    for (auto& n : m_plyNodes) n.selected = false;
                    node.selected = true;
                    m_selectedPLYIndex = idx;
                    ui::NodeSelected{
                        .path = node.name,
                        .type = node.is_group ? "Group" : "PLY",
                        .metadata = {{"name", node.name},
                                     {"gaussians", std::to_string(node.gaussian_count)},
                                     {"visible", node.visible ? "true" : "false"}}}.emit();
                }
            }

            // Context menu
            if (ImGui::BeginPopup(("##ctx_" + node.name).c_str())) {
                if (node.is_group && ImGui::MenuItem("Add Group...")) {
                    cmd::AddGroup{.name = "New Group", .parent_name = node.name}.emit();
                }
                if (node.is_group) ImGui::Separator();
                if (!node.is_group && ImGui::MenuItem("Save As...")) {
                    cmd::SavePLYAs{.name = node.name}.emit();
                }
                if (ImGui::MenuItem("Rename")) startRenaming(idx);
                if (ImGui::MenuItem("Duplicate")) cmd::DuplicateNode{.name = node.name}.emit();

                if (ImGui::BeginMenu("Move to")) {
                    if (!node.parent_name.empty()) {
                        if (ImGui::MenuItem("Root")) {
                            cmd::ReparentNode{.node_name = node.name, .new_parent_name = ""}.emit();
                        }
                        ImGui::Separator();
                    }
                    bool found_group = false;
                    for (const auto& t : m_plyNodes) {
                        if (t.is_group && t.name != node.name && t.name != node.parent_name) {
                            found_group = true;
                            if (ImGui::MenuItem(t.name.c_str())) {
                                cmd::ReparentNode{.node_name = node.name, .new_parent_name = t.name}.emit();
                            }
                        }
                    }
                    if (!found_group && node.parent_name.empty()) {
                        ImGui::TextDisabled("No groups available");
                    }
                    ImGui::EndMenu();
                }

                if (ImGui::MenuItem(node.locked ? "Unlock" : "Lock")) {
                    node.locked = !node.locked;
                    cmd::SetNodeLocked{.name = node.name, .locked = node.locked}.emit();
                }

                ImGui::Separator();
                if (has_children) {
                    if (ImGui::MenuItem("Delete group only")) {
                        cmd::RemovePLY{.name = node.name, .keep_children = true}.emit();
                    }
                    if (ImGui::MenuItem("Delete with children")) {
                        cmd::RemovePLY{.name = node.name, .keep_children = false}.emit();
                    }
                } else {
                    if (ImGui::MenuItem("Delete")) cmd::RemovePLY{.name = node.name}.emit();
                }
                ImGui::EndPopup();
            }

            if (is_open && has_children) {
                renderNodeChildren(node.name);
                ImGui::TreePop();
            }
        }

        ImGui::PopID();
    }

    void ScenePanel::renderNodeChildren(const std::string& parent_name) {
        for (size_t i = 0; i < m_plyNodes.size(); ++i) {
            if (m_plyNodes[i].parent_name == parent_name) {
                renderModelNode(i);
            }
        }
    }

    bool ScenePanel::handleDragDrop(const std::string& target_name, const bool is_folder) {
        if (!ImGui::BeginDragDropTarget()) return false;

        bool handled = false;
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE_NODE")) {
            const char* dragged_name = static_cast<const char*>(payload->Data);

            // Don't allow dropping on self or own children
            if (dragged_name != target_name) {
                cmd::ReparentNode{
                    .node_name = std::string(dragged_name),
                    .new_parent_name = is_folder ? target_name : ""
                }.emit();
                LOG_INFO("Reparented '{}' to '{}'", dragged_name, target_name.empty() ? "root" : target_name);
                handled = true;
            }
        }

        ImGui::EndDragDropTarget();
        return handled;
    }

    void ScenePanel::renderImageList() {
        // Image list view
        ImGui::BeginChild("ImageList", ImVec2(0, 0), true);

        if (!m_imagePaths.empty()) {
            ImGui::Text("Images (%zu):", m_imagePaths.size());
            ImGui::Separator();

            // Track if we need to scroll to the selected item
            bool should_scroll = false;

            for (size_t i = 0; i < m_imagePaths.size(); ++i) {
                const auto& imagePath = m_imagePaths[i];
                std::string filename = imagePath.filename().string();

                // Create unique ID for ImGui by combining filename with index
                std::string unique_id = std::format("{}##{}", filename, i);

                // Check if this item is selected
                bool is_selected = (m_selectedImageIndex == static_cast<int>(i));

                // Push a different color for selected items to make them more visible
                if (is_selected) {
                    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.5f, 0.8f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.4f, 0.6f, 0.9f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.3f, 0.5f, 0.8f, 1.0f));
                }

                if (ImGui::Selectable(unique_id.c_str(), is_selected)) {
                    m_selectedImageIndex = static_cast<int>(i);
                    onImageSelected(imagePath);
                }

                // Scroll to this item if it's selected and we need to scroll
                if (is_selected && m_needsScrollToSelection) {
                    ImGui::SetScrollHereY(0.5f); // Center the selected item
                    m_needsScrollToSelection = false;
                    should_scroll = true;
                }

                if (is_selected) {
                    ImGui::PopStyleColor(3);
                }

                // Handle double-click to open image preview
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    onImageDoubleClicked(i);
                }

                // Context menu for right-click - use unique ID
                std::string context_menu_id = std::format("context_menu_{}", i);
                if (ImGui::BeginPopupContextItem(context_menu_id.c_str())) {
                    if (ImGui::MenuItem("Go to Cam View")) {
                        // Get the camera data for this image
                        auto cam_data_it = m_PathToCamId.find(imagePath);
                        if (cam_data_it != m_PathToCamId.end()) {
                            // Emit the new GoToCamView command event with camera data
                            cmd::GoToCamView{
                                .cam_id = cam_data_it->second}
                                .emit();

                            LOG_INFO("Going to camera view for: {} (Camera ID: {})",
                                     imagePath.filename().string(),
                                     cam_data_it->second);
                        } else {
                            // Log warning if camera data not found
                            LOG_WARN("Camera data not found for: {}", imagePath.filename().string());
                        }
                    }
                    ImGui::EndPopup();
                }
            }

            if (should_scroll) {
                LOG_TRACE("Scrolled to selected image at index {}", m_selectedImageIndex);
            }
        } else {
            ImGui::Text("No images loaded.");
            ImGui::Text("Use 'Open File Browser' to load a dataset.");
        }

        ImGui::EndChild();
    }

    void ScenePanel::loadImageCams(const std::filesystem::path& path) {
        LOG_TIMER_TRACE("ScenePanel::loadImageCams");

        m_currentDatasetPath = path;
        m_imagePaths.clear();
        m_PathToCamId.clear();
        m_selectedImageIndex = -1;

        if (!m_trainer_manager) {
            LOG_ERROR("m_trainer_manager was not set");
            return;
        }

        LOG_DEBUG("Loading camera list from dataset: {}", path.string());
        auto cams = m_trainer_manager->getCamList();
        LOG_DEBUG("Found {} cameras", cams.size());

        for (const auto& cam : cams) {
            m_imagePaths.emplace_back(cam->image_path());
            m_PathToCamId[cam->image_path()] = cam->uid();
            LOG_TRACE("Added camera: {} (ID: {})", cam->image_path().filename().string(), cam->uid());
        }

        // Sort paths for consistent ordering
        std::ranges::sort(m_imagePaths, [](const auto& a, const auto& b) {
            return a.filename() < b.filename();
        });

        LOG_INFO("Loaded {} images from dataset: {}", m_imagePaths.size(), path.string());
    }

    void ScenePanel::setOnDatasetLoad(std::function<void(const std::filesystem::path&)> callback) {
        m_onDatasetLoad = std::move(callback);
    }

    void ScenePanel::onImageSelected(const std::filesystem::path& imagePath) {
        LOG_DEBUG("Selected image: {}", imagePath.filename().string());

        // Publish NodeSelectedEvent for other components to react
        ui::NodeSelected{
            .path = imagePath.string(),
            .type = "Images",
            .metadata = {{"filename", imagePath.filename().string()}, {"path", imagePath.string()}}}
            .emit();
    }

    void ScenePanel::onImageDoubleClicked(size_t imageIndex) {
        if (imageIndex >= m_imagePaths.size()) {
            LOG_WARN("Invalid image index for double-click: {}", imageIndex);
            return;
        }

        const auto& imagePath = m_imagePaths[imageIndex];

        // Open the image preview with all images and current index
        if (m_imagePreview) {
            m_imagePreview->open(m_imagePaths, imageIndex);
            m_showImagePreview = true;
            LOG_INFO("Opening image preview: {} (index {})",
                     imagePath.filename().string(), imageIndex);
        }
    }

    void ScenePanel::handlePLYRenamed(const cmd::RenamePLY& event) {
        const auto it = std::find_if(m_plyNodes.begin(), m_plyNodes.end(),
                               [&event](const PLYNode& n) { return n.name == event.old_name; });
        if (it != m_plyNodes.end()) {
            it->name = event.new_name;
            for (auto& n : m_plyNodes) {
                if (n.parent_name == event.old_name) {
                    n.parent_name = event.new_name;
                }
            }
        }
        cancelRenaming();
    }

} // namespace lfs::vis::gui