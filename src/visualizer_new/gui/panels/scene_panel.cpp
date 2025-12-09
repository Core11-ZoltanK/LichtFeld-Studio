/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/scene_panel.hpp"
#include "core_new/logger.hpp"
#include "gui/utils/windows_utils.hpp"
#include "gui/windows/image_preview.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "visualizer_impl.hpp"

#include <algorithm>
#include <format>
#include <imgui.h>
#include <ranges>

namespace lfs::vis::gui {

    using namespace lfs::core::events;

    ScenePanel::ScenePanel(std::shared_ptr<const TrainerManager> trainer_manager)
        : m_trainerManager(std::move(trainer_manager)) {
        m_imagePreview = std::make_unique<ImagePreview>();
        setupEventHandlers();
        LOG_DEBUG("ScenePanel created");
    }

    ScenePanel::~ScenePanel() = default;

    void ScenePanel::setupEventHandlers() {
        cmd::GoToCamView::when([this](const auto& e) { handleGoToCamView(e); });

        state::SceneCleared::when([this](const auto&) {
            m_imagePaths.clear();
            m_pathToCamId.clear();
            m_currentDatasetPath.clear();
            m_selectedImageIndex = -1;
        });

        state::DatasetLoadCompleted::when([this](const auto& e) {
            if (e.success) { loadImageCams(e.path); }
        });
    }

    void ScenePanel::handleGoToCamView(const cmd::GoToCamView& event) {
        for (const auto& [path, cam_id] : m_pathToCamId) {
            if (cam_id == event.cam_id) {
                if (auto it = std::find(m_imagePaths.begin(), m_imagePaths.end(), path); it != m_imagePaths.end()) {
                    m_selectedImageIndex = static_cast<int>(std::distance(m_imagePaths.begin(), it));
                    m_needsScrollToSelection = true;
                    LOG_TRACE("Synced image selection to camera ID {} (index {})", event.cam_id, m_selectedImageIndex);
                }
                break;
            }
        }
    }

    bool ScenePanel::hasImages() const {
        return !m_imagePaths.empty();
    }

    bool ScenePanel::hasPLYs(const UIContext* ctx) const {
        if (!ctx || !ctx->viewer) return false;
        const auto* sm = ctx->viewer->getSceneManager();
        if (!sm) return false;
        return sm->getScene().hasNodes();
    }

    void ScenePanel::render(bool* p_open, const UIContext* ctx) {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, withAlpha(theme().palette.surface_bright, 0.8f));

        if (!ImGui::Begin("Scene", p_open)) {
            ImGui::End();
            ImGui::PopStyleColor();
            return;
        }

        const float button_width = ImGui::GetContentRegionAvail().x;

        if (ImGui::Button("Import dataset", ImVec2(button_width, 0))) {
            LOG_DEBUG("Opening file browser from scene panel");
            if (m_onDatasetLoad) {
                m_onDatasetLoad(std::filesystem::path(""));
            }
#ifdef WIN32
            OpenDatasetFolderDialog();
            cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif
        }

        if (ImGui::Button("Open .ply", ImVec2(button_width, 0))) {
            LOG_DEBUG("Opening file browser from scene panel");
            if (m_onDatasetLoad) {
                m_onDatasetLoad(std::filesystem::path(""));
            }
#ifdef WIN32
            OpenPlyFileDialog();
            cmd::ShowWindow{.window_name = "file_browser", .show = false}.emit();
#endif
        }

        if (ImGui::Button("Refresh", ImVec2(button_width * 0.48f, 0))) {
            if (!m_currentDatasetPath.empty()) {
                LOG_DEBUG("Refreshing dataset images");
                loadImageCams(m_currentDatasetPath);
            }
        }

        ImGui::SameLine();

        if (ImGui::Button("Clear", ImVec2(button_width * 0.48f, 0))) {
            LOG_INFO("Clearing scene from panel");
            m_imagePaths.clear();
            m_selectedImageIndex = -1;
            cmd::ClearScene{}.emit();
        }

        ImGui::Separator();

        const bool has_plys = hasPLYs(ctx);

        if (has_plys) {
            renderPLYSceneGraph(ctx);
        } else {
            ImGui::Text("No data loaded.");
        }

        ImGui::End();
        ImGui::PopStyleColor();

        if (m_showImagePreview && m_imagePreview) {
            m_imagePreview->render(&m_showImagePreview);
        }
    }

    void ScenePanel::renderPLYSceneGraph(const UIContext* ctx) {
        if (!ctx || !ctx->viewer) return;

        auto* scene_manager = ctx->viewer->getSceneManager();
        if (!scene_manager) return;

        const auto& scene = scene_manager->getScene();

        // Get selection state from SceneManager (single source of truth)
        const auto selected_names_vec = scene_manager->getSelectedNodeNames();
        std::unordered_set<std::string> selected_names(selected_names_vec.begin(), selected_names_vec.end());

        ImGui::BeginChild("SceneGraph", ImVec2(0, 0), ImGuiChildFlags_None);

        // Keyboard shortcuts
        if (ImGui::IsWindowFocused() && !m_renameState.is_renaming) {
            if (ImGui::IsKeyPressed(ImGuiKey_F2) && !selected_names.empty()) {
                startRenaming(*selected_names.begin());
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Escape) && !selected_names.empty()) {
                scene_manager->clearSelection();
                ui::NodeDeselected{}.emit();
            }
        }

        static constexpr ImGuiTreeNodeFlags ROOT_NODE_FLAGS =
            ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
            ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_Framed;

        if (ImGui::TreeNodeEx("Scene", ROOT_NODE_FLAGS)) {
            renderModelsFolder(scene, selected_names);
            ImGui::TreePop();
        }

        // Summary
        if (scene.hasNodes()) {
            ImGui::Separator();
            size_t total = 0;
            for (const auto* node : scene.getVisibleNodes()) {
                total += node->gaussian_count;
            }
            ImGui::TextDisabled("Visible: %zu gaussians", total);
        }

        ImGui::EndChild();
    }

    void ScenePanel::renderModelsFolder(const Scene& scene, const std::unordered_set<std::string>& selected_names) {
        static constexpr ImGuiTreeNodeFlags FOLDER_NODE_FLAGS =
            ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow |
            ImGuiTreeNodeFlags_SpanAvailWidth;

        // Count only splat nodes
        const auto nodes = scene.getNodes();
        const size_t splat_count = std::ranges::count_if(nodes,
            [](const SceneNode* n) { return n->type == NodeType::SPLAT; });

        const std::string label = std::format("Models ({})", splat_count);
        if (!ImGui::TreeNodeEx(label.c_str(), FOLDER_NODE_FLAGS)) return;

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
            ImGui::Separator();
            if (ImGui::MenuItem("Save All Merged...", nullptr, false, splat_count > 0)) {
                cmd::SaveAllMergedAs{}.emit();
            }
            ImGui::EndPopup();
        }

        // Render root-level nodes (parent_id == NULL_NODE)
        for (const auto* node : nodes) {
            if (node->parent_id == NULL_NODE) {
                renderModelNode(*node, scene, selected_names);
            }
        }

        if (!scene.hasNodes()) {
            ImGui::TextDisabled("No models loaded");
            ImGui::TextDisabled("Right-click to add...");
        }

        ImGui::TreePop();
    }

    void ScenePanel::renderModelNode(const SceneNode& node, const Scene& scene,
                                     const std::unordered_set<std::string>& selected_names) {
        ImGui::PushID(node.id);

        const bool is_visible = node.visible.get();
        const bool is_selected = selected_names.contains(node.name);
        const bool is_group = (node.type == NodeType::GROUP);
        const bool is_cropbox = (node.type == NodeType::CROPBOX);
        const bool is_dataset = (node.type == NodeType::DATASET);
        const bool is_camera_group = (node.type == NodeType::CAMERA_GROUP);
        const bool is_camera = (node.type == NodeType::CAMERA);
        const bool is_pointcloud = (node.type == NodeType::POINTCLOUD);
        const bool has_children = !node.children.empty();

        // Check if parent is a dataset (for "Cameras" group and "Model" splat inside dataset)
        const auto* parent_node = scene.getNodeById(node.parent_id);
        const bool parent_is_dataset = parent_node && parent_node->type == NodeType::DATASET;

        // Visibility toggle
        if (ImGui::SmallButton(is_visible ? "[*]" : "[ ]")) {
            cmd::SetPLYVisibility{.name = node.name, .visible = !is_visible}.emit();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(is_visible ? "Hide" : "Show");
        }
        ImGui::SameLine();

        const bool is_renaming = m_renameState.is_renaming && m_renameState.renaming_node_name == node.name;

        if (is_renaming) {
            if (m_renameState.focus_input) {
                ImGui::SetKeyboardFocusHere();
                m_renameState.focus_input = false;
            }

            static constexpr ImGuiInputTextFlags RENAME_INPUT_FLAGS =
                ImGuiInputTextFlags_AutoSelectAll | ImGuiInputTextFlags_EnterReturnsTrue;
            const bool entered = ImGui::InputText("##rename", m_renameState.buffer,
                                                  sizeof(m_renameState.buffer), RENAME_INPUT_FLAGS);
            const bool is_focused = ImGui::IsItemFocused();
            if (ImGui::IsItemActive()) m_renameState.input_was_active = true;

            if (entered) {
                // Need non-const scene_manager for rename - get from context indirectly via event
                finishRenaming(nullptr);
            } else if (ImGui::IsKeyPressed(ImGuiKey_Escape) ||
                       (m_renameState.input_was_active && !is_focused)) {
                cancelRenaming();
            }
        } else {
            static constexpr ImGuiTreeNodeFlags BASE_NODE_FLAGS = ImGuiTreeNodeFlags_OpenOnArrow;
            ImGuiTreeNodeFlags flags = BASE_NODE_FLAGS;
            if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;
            if (!has_children) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
            if (is_group || is_dataset) flags |= ImGuiTreeNodeFlags_DefaultOpen;

            // Build label based on node type
            std::string label;
            if (is_cropbox) {
                label = std::format("[Crop] {}", node.name);
            } else if (is_dataset) {
                label = std::format("[Dataset] {}", node.name);
            } else if (is_camera_group) {
                label = node.name;  // Already includes count like "Training (185)"
            } else if (is_camera) {
                label = node.name;  // Camera name (image filename)
            } else if (is_group) {
                label = node.name;
            } else if (is_pointcloud) {
                // POINTCLOUD node - show point count
                const size_t point_count = node.point_cloud ? node.point_cloud->size() : 0;
                label = std::format("[Points] {} ({:L})", node.name, point_count);
            } else {
                // SPLAT node
                label = std::format("{} ({:L})", node.name, node.gaussian_count);
            }

            const bool is_open = ImGui::TreeNodeEx(label.c_str(), flags);
            const bool hovered = ImGui::IsItemHovered();
            const bool clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
            const bool toggled = ImGui::IsItemToggledOpen();

            if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                ImGui::OpenPopup(("##ctx_" + node.name).c_str());
            }

            // Drag source
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                ImGui::SetDragDropPayload("SCENE_NODE", node.name.c_str(), node.name.size() + 1);
                ImGui::Text("Move: %s", node.name.c_str());
                ImGui::EndDragDropSource();
            }

            if (is_group) handleDragDrop(node.name, true);

            // Selection - emit event, let SceneManager handle state
            // Camera nodes don't participate in selection - they have their own interactions
            if (clicked && !toggled && !is_camera) {
                if (is_selected) {
                    ui::NodeDeselected{}.emit();
                } else {
                    // Determine node type string for event
                    std::string type_str = "PLY";
                    if (is_group) type_str = "Group";
                    else if (is_dataset) type_str = "Dataset";
                    else if (is_camera_group) type_str = "CameraGroup";
                    else if (is_pointcloud) type_str = "PointCloud";

                    ui::NodeSelected{
                        .path = node.name,
                        .type = type_str,
                        .metadata = {{"name", node.name},
                                     {"gaussians", std::to_string(node.gaussian_count)},
                                     {"visible", is_visible ? "true" : "false"}}}.emit();
                }
            }

            // Double-click on camera opens image preview with arrow key navigation
            if (is_camera && hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                if (!node.image_path.empty() && m_imagePreview) {
                    std::vector<std::filesystem::path> camera_paths;
                    size_t current_idx = 0;

                    for (const auto* n : scene.getNodes()) {
                        if (n->type == NodeType::CAMERA && !n->image_path.empty()) {
                            if (n->name == node.name) current_idx = camera_paths.size();
                            camera_paths.push_back(n->image_path);
                        }
                    }

                    if (!camera_paths.empty()) {
                        m_imagePreview->open(camera_paths, current_idx);
                        m_showImagePreview = true;
                    }
                }
            }

            // Helper lambda to close context menu and finish node rendering
            const auto closeContextAndFinish = [&]() {
                ImGui::EndPopup();
                if (is_open && has_children) {
                    renderNodeChildren(node.id, scene, selected_names);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            };

            // Context menu
            if (ImGui::BeginPopup(("##ctx_" + node.name).c_str())) {
                if (is_camera) {
                    if (ImGui::MenuItem("Go to Camera View")) {
                        cmd::GoToCamView{.cam_id = node.camera_uid}.emit();
                    }
                    closeContextAndFinish();
                    return;
                }

                if (is_camera_group || parent_is_dataset) {
                    ImGui::TextDisabled("(No actions)");
                    closeContextAndFinish();
                    return;
                }

                if (is_dataset) {
                    if (ImGui::MenuItem("Delete")) {
                        cmd::RemovePLY{.name = node.name, .keep_children = false}.emit();
                    }
                    closeContextAndFinish();
                    return;
                }

                if (is_cropbox) {
                    if (ImGui::MenuItem("Fit to Scene")) {
                        cmd::FitCropBoxToScene{.use_percentile = false}.emit();
                    }
                    if (ImGui::MenuItem("Fit to Scene (Trimmed)")) {
                        cmd::FitCropBoxToScene{.use_percentile = true}.emit();
                    }
                    ImGui::EndPopup();
                    if (is_open && has_children) {
                        renderNodeChildren(node.id, scene, selected_names);
                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                    return;
                }

                if (is_group) {
                    if (ImGui::MenuItem("Add Group...")) {
                        cmd::AddGroup{.name = "New Group", .parent_name = node.name}.emit();
                    }
                    if (ImGui::MenuItem("Merge to Single PLY")) {
                        cmd::MergeGroup{.name = node.name}.emit();
                    }
                    ImGui::Separator();
                }
                if (!is_group && ImGui::MenuItem("Save As...")) {
                    cmd::SavePLYAs{.name = node.name}.emit();
                }
                if (ImGui::MenuItem("Rename")) startRenaming(node.name);
                if (ImGui::MenuItem("Duplicate")) cmd::DuplicateNode{.name = node.name}.emit();

                if (ImGui::BeginMenu("Move to")) {
                    const auto* parent_node = scene.getNodeById(node.parent_id);
                    if (parent_node) {
                        if (ImGui::MenuItem("Root")) {
                            cmd::ReparentNode{.node_name = node.name, .new_parent_name = ""}.emit();
                        }
                        ImGui::Separator();
                    }
                    bool found_group = false;
                    for (const auto* other : scene.getNodes()) {
                        if (other->type == NodeType::GROUP && other->name != node.name &&
                            (parent_node == nullptr || other->name != parent_node->name)) {
                            found_group = true;
                            if (ImGui::MenuItem(other->name.c_str())) {
                                cmd::ReparentNode{.node_name = node.name, .new_parent_name = other->name}.emit();
                            }
                        }
                    }
                    if (!found_group && !parent_node) {
                        ImGui::TextDisabled("No groups available");
                    }
                    ImGui::EndMenu();
                }

                ImGui::Separator();
                if (ImGui::MenuItem("Delete")) {
                    cmd::RemovePLY{.name = node.name, .keep_children = false}.emit();
                }
                ImGui::EndPopup();
            }

            if (is_open && has_children) {
                renderNodeChildren(node.id, scene, selected_names);
                ImGui::TreePop();
            }
        }

        ImGui::PopID();
    }

    void ScenePanel::renderNodeChildren(NodeId parent_id, const Scene& scene,
                                        const std::unordered_set<std::string>& selected_names) {
        const auto* parent = scene.getNodeById(parent_id);
        if (!parent) return;

        for (const NodeId child_id : parent->children) {
            const auto* child = scene.getNodeById(child_id);
            if (child) {
                renderModelNode(*child, scene, selected_names);
            }
        }
    }

    bool ScenePanel::handleDragDrop(const std::string& target_name, const bool is_folder) {
        if (!ImGui::BeginDragDropTarget()) return false;

        bool handled = false;
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE_NODE")) {
            const char* dragged_name = static_cast<const char*>(payload->Data);
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

    void ScenePanel::startRenaming(const std::string& node_name) {
        m_renameState.is_renaming = true;
        m_renameState.renaming_node_name = node_name;
        m_renameState.focus_input = true;
        strncpy(m_renameState.buffer, node_name.c_str(), sizeof(m_renameState.buffer) - 1);
        m_renameState.buffer[sizeof(m_renameState.buffer) - 1] = '\0';
        LOG_DEBUG("Started renaming node '{}'", node_name);
    }

    void ScenePanel::finishRenaming(SceneManager* /*scene_manager*/) {
        if (!m_renameState.is_renaming) return;

        std::string new_name(m_renameState.buffer);
        // Trim whitespace
        if (const auto pos = new_name.find_last_not_of(" \t\n\r"); pos != std::string::npos) {
            new_name = new_name.substr(0, pos + 1);
        }

        if (!new_name.empty() && new_name != m_renameState.renaming_node_name) {
            cmd::RenamePLY{
                .old_name = m_renameState.renaming_node_name,
                .new_name = new_name
            }.emit();
            LOG_INFO("Emitted rename command: '{}' -> '{}'", m_renameState.renaming_node_name, new_name);
        }

        cancelRenaming();
    }

    void ScenePanel::cancelRenaming() {
        m_renameState.is_renaming = false;
        m_renameState.renaming_node_name.clear();
        m_renameState.focus_input = false;
        m_renameState.input_was_active = false;
        m_renameState.escape_pressed = false;
        memset(m_renameState.buffer, 0, sizeof(m_renameState.buffer));
    }

    void ScenePanel::renderImageList() {
        ImGui::BeginChild("ImageList", ImVec2(0, 0), true);

        if (!m_imagePaths.empty()) {
            ImGui::Text("Images (%zu):", m_imagePaths.size());
            ImGui::Separator();

            for (size_t i = 0; i < m_imagePaths.size(); ++i) {
                const auto& imagePath = m_imagePaths[i];
                const std::string filename = imagePath.filename().string();
                const std::string unique_id = std::format("{}##{}", filename, i);
                const bool is_selected = (m_selectedImageIndex == static_cast<int>(i));

                if (is_selected) {
                    const auto& t = theme();
                    ImGui::PushStyleColor(ImGuiCol_Header, withAlpha(t.palette.info, 0.8f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, withAlpha(t.palette.info, 0.9f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive, withAlpha(t.palette.info, 0.8f));
                }

                if (ImGui::Selectable(unique_id.c_str(), is_selected)) {
                    m_selectedImageIndex = static_cast<int>(i);
                    onImageSelected(imagePath);
                }

                if (is_selected && m_needsScrollToSelection) {
                    ImGui::SetScrollHereY(0.5f);
                    m_needsScrollToSelection = false;
                }

                if (is_selected) {
                    ImGui::PopStyleColor(3);
                }

                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    onImageDoubleClicked(i);
                }

                const std::string context_menu_id = std::format("context_menu_{}", i);
                if (ImGui::BeginPopupContextItem(context_menu_id.c_str())) {
                    if (ImGui::MenuItem("Go to Cam View")) {
                        if (auto cam_it = m_pathToCamId.find(imagePath); cam_it != m_pathToCamId.end()) {
                            cmd::GoToCamView{.cam_id = cam_it->second}.emit();
                            LOG_INFO("Going to camera view for: {} (Camera ID: {})",
                                     imagePath.filename().string(), cam_it->second);
                        }
                    }
                    ImGui::EndPopup();
                }
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
        m_pathToCamId.clear();
        m_selectedImageIndex = -1;

        if (!m_trainerManager) {
            LOG_ERROR("m_trainerManager was not set");
            return;
        }

        LOG_DEBUG("Loading camera list from dataset: {}", path.string());
        auto cams = m_trainerManager->getCamList();
        LOG_DEBUG("Found {} cameras", cams.size());

        for (const auto& cam : cams) {
            m_imagePaths.emplace_back(cam->image_path());
            m_pathToCamId[cam->image_path()] = cam->uid();
        }

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
        ui::NodeSelected{
            .path = imagePath.string(),
            .type = "Images",
            .metadata = {{"filename", imagePath.filename().string()}, {"path", imagePath.string()}}
        }.emit();
    }

    void ScenePanel::onImageDoubleClicked(size_t imageIndex) {
        if (imageIndex >= m_imagePaths.size()) return;

        const auto& imagePath = m_imagePaths[imageIndex];
        if (m_imagePreview) {
            m_imagePreview->open(m_imagePaths, imageIndex);
            m_showImagePreview = true;
            LOG_INFO("Opening image preview: {} (index {})", imagePath.filename().string(), imageIndex);
        }
    }

} // namespace lfs::vis::gui
