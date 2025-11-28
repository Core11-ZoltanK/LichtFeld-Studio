/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/crop_box_panel.hpp"
#include "command/command_history.hpp"
#include "command/commands/cropbox_command.hpp"
#include "gui/ui_widgets.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer_impl.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <ImGuizmo.h>
#include <optional>

namespace lfs::vis::gui::panels {

    using namespace lfs::core::events;

    // Static state for tracking undo state in UI
    static std::optional<command::CropBoxState> s_state_before_edit;

    // Helper to capture current cropbox state
    static command::CropBoxState captureState(RenderingManager* render_manager) {
        auto settings = render_manager->getSettings();
        return command::CropBoxState{
            .crop_min = settings.crop_min,
            .crop_max = settings.crop_max,
            .crop_transform = settings.crop_transform,
            .crop_inverse = settings.crop_inverse
        };
    }

    // Helper to create and execute undo command if state changed
    static void commitUndoIfChanged(VisualizerImpl* viewer, RenderingManager* render_manager) {
        if (!s_state_before_edit.has_value()) return;

        auto new_state = captureState(render_manager);

        // Check if state actually changed
        bool changed = (s_state_before_edit->crop_min != new_state.crop_min ||
                        s_state_before_edit->crop_max != new_state.crop_max ||
                        s_state_before_edit->crop_inverse != new_state.crop_inverse);

        // Also check transform - compare translation and rotation matrix
        if (!changed) {
            auto old_trans = s_state_before_edit->crop_transform.getTranslation();
            auto new_trans = new_state.crop_transform.getTranslation();
            auto old_rot = s_state_before_edit->crop_transform.getRotationMat();
            auto new_rot = new_state.crop_transform.getRotationMat();

            changed = (old_trans != new_trans || old_rot != new_rot);
        }

        if (changed) {
            auto cmd = std::make_unique<command::CropBoxCommand>(
                render_manager, *s_state_before_edit, new_state);
            viewer->getCommandHistory().execute(std::move(cmd));
        }

        s_state_before_edit.reset();
    }

    // Apply rotation delta to crop box transform (in local space)
    static void updateRotationMatrix(lfs::geometry::EuclideanTransform& transform,
                                     float delta_rot_x, float delta_rot_y, float delta_rot_z) {
        // Extract current rotation and translation
        glm::mat3 currentRot = transform.getRotationMat();
        glm::vec3 translation = transform.getTranslation();

        // Create rotation delta
        float rad_x = glm::radians(delta_rot_x);
        float rad_y = glm::radians(delta_rot_y);
        float rad_z = glm::radians(delta_rot_z);

        glm::mat3 deltaRot = glm::mat3(
            glm::rotate(glm::mat4(1.0f), rad_z, glm::vec3(0, 0, 1)) *
            glm::rotate(glm::mat4(1.0f), rad_y, glm::vec3(0, 1, 0)) *
            glm::rotate(glm::mat4(1.0f), rad_x, glm::vec3(1, 0, 0))
        );

        // Apply delta rotation in local space
        glm::mat3 newRot = currentRot * deltaRot;

        // Update transform with new rotation, keep translation
        transform = lfs::geometry::EuclideanTransform(newRot, translation);
    }

    void DrawCropBoxControls(const UIContext& ctx) {
        auto render_manager = ctx.viewer->getRenderingManager();
        if (!render_manager)
            return;

        if (!ImGui::CollapsingHeader("Crop Box")) {
            return;
        }

        auto settings = render_manager->getSettings();
        bool settings_changed = false;

        if (settings.show_crop_box) {
            // Appearance controls
            float bbox_color[3] = {settings.crop_color.x, settings.crop_color.y, settings.crop_color.z};
            if (ImGui::ColorEdit3("Box Color", bbox_color)) {
                settings.crop_color = glm::vec3(bbox_color[0], bbox_color[1], bbox_color[2]);
                settings_changed = true;
            }

            if (ImGui::SliderFloat("Line Width", &settings.crop_line_width, 0.5f, 10.0f)) {
                settings_changed = true;
            }

            // Rotation controls
            if (ImGui::TreeNode("Rotation")) {
                ImGui::Text("Ctrl+click for faster steps");
                ImGui::Text("Rotation around crop box axes:");

                const float rotation_step = 1.0f;
                const float rotation_step_fast = 15.0f;

                static float rotate_timer_x = 0.0f;
                static float rotate_timer_y = 0.0f;
                static float rotate_timer_z = 0.0f;
                static bool rotation_active = false;

                float step = ImGui::GetIO().KeyCtrl ? rotation_step_fast : rotation_step;
                float repeat_rate = 0.05f;

                float diff_x = 0, diff_y = 0, diff_z = 0;
                bool any_rotation_button_active = false;

                // X-axis rotation
                ImGui::Text("X-axis:");
                ImGui::SameLine();
                ImGui::Text("RotX");

                if (ImGui::ArrowButton("##RotX_Up", ImGuiDir_Up)) {
                    diff_x = step;
                    rotate_timer_x = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    any_rotation_button_active = true;
                    rotate_timer_x += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_x >= repeat_rate) {
                        diff_x = step;
                        rotate_timer_x = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotX_Down", ImGuiDir_Down)) {
                    diff_x = -step;
                    rotate_timer_x = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    any_rotation_button_active = true;
                    rotate_timer_x += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_x >= repeat_rate) {
                        diff_x = -step;
                        rotate_timer_x = 0.0f;
                    }
                }

                // Y-axis rotation
                ImGui::Text("Y-axis:");
                ImGui::SameLine();
                ImGui::Text("RotY");

                if (ImGui::ArrowButton("##RotY_Up", ImGuiDir_Up)) {
                    diff_y = step;
                    rotate_timer_y = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    any_rotation_button_active = true;
                    rotate_timer_y += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_y >= repeat_rate) {
                        diff_y = step;
                        rotate_timer_y = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotY_Down", ImGuiDir_Down)) {
                    diff_y = -step;
                    rotate_timer_y = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    any_rotation_button_active = true;
                    rotate_timer_y += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_y >= repeat_rate) {
                        diff_y = -step;
                        rotate_timer_y = 0.0f;
                    }
                }

                // Z-axis rotation
                ImGui::Text("Z-axis:");
                ImGui::SameLine();
                ImGui::Text("RotZ");

                if (ImGui::ArrowButton("##RotZ_Up", ImGuiDir_Up)) {
                    diff_z = step;
                    rotate_timer_z = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    any_rotation_button_active = true;
                    rotate_timer_z += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_z >= repeat_rate) {
                        diff_z = step;
                        rotate_timer_z = 0.0f;
                    }
                }

                ImGui::SameLine();
                if (ImGui::ArrowButton("##RotZ_Down", ImGuiDir_Down)) {
                    diff_z = -step;
                    rotate_timer_z = 0.0f;
                }
                if (ImGui::IsItemActive()) {
                    any_rotation_button_active = true;
                    rotate_timer_z += ImGui::GetIO().DeltaTime;
                    if (rotate_timer_z >= repeat_rate) {
                        diff_z = -step;
                        rotate_timer_z = 0.0f;
                    }
                }

                // Capture state when rotation starts
                if (any_rotation_button_active && !rotation_active) {
                    rotation_active = true;
                    s_state_before_edit = captureState(render_manager);
                }

                // Commit undo when rotation ends
                if (!any_rotation_button_active && rotation_active) {
                    rotation_active = false;
                    // Apply final settings first
                    if (diff_x != 0 || diff_y != 0 || diff_z != 0) {
                        updateRotationMatrix(settings.crop_transform, diff_x, diff_y, diff_z);
                        render_manager->updateSettings(settings);
                    }
                    commitUndoIfChanged(ctx.viewer, render_manager);
                }

                if (diff_x != 0 || diff_y != 0 || diff_z != 0) {
                    updateRotationMatrix(settings.crop_transform, diff_x, diff_y, diff_z);
                    settings_changed = true;
                }

                ImGui::TreePop();
            }

            // Bounds controls (local space)
            if (ImGui::TreeNode("Local Bounds")) {
                float min_bounds[3] = {settings.crop_min.x, settings.crop_min.y, settings.crop_min.z};
                float max_bounds[3] = {settings.crop_max.x, settings.crop_max.y, settings.crop_max.z};

                bool bounds_changed = false;
                static bool bounds_editing_active = false;

                const float bound_step = 0.01f;
                const float bound_step_fast = 0.1f;

                ImGui::Text("Ctrl+click for faster steps");
                ImGui::Text("Local Min Bounds:");

                // calculate the exact width to hold 0000.000 string in the text box + extra
                float text_width = ImGui::CalcTextSize("0000.000").x + ImGui::GetStyle().FramePadding.x * 2.0f + 50.0f;

                // Track if any input becomes active (for undo capture)
                bool any_input_active = false;
                bool any_input_deactivated = false;

                // Min bounds
                ImGui::Text("X:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MinX", &min_bounds[0], bound_step, bound_step_fast, "%.3f");
                any_input_active |= ImGui::IsItemActive();
                any_input_deactivated |= ImGui::IsItemDeactivatedAfterEdit();
                min_bounds[0] = std::min(min_bounds[0], max_bounds[0]);

                ImGui::Text("Y:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MinY", &min_bounds[1], bound_step, bound_step_fast, "%.3f");
                any_input_active |= ImGui::IsItemActive();
                any_input_deactivated |= ImGui::IsItemDeactivatedAfterEdit();
                min_bounds[1] = std::min(min_bounds[1], max_bounds[1]);

                ImGui::Text("Z:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MinZ", &min_bounds[2], bound_step, bound_step_fast, "%.3f");
                any_input_active |= ImGui::IsItemActive();
                any_input_deactivated |= ImGui::IsItemDeactivatedAfterEdit();
                min_bounds[2] = std::min(min_bounds[2], max_bounds[2]);

                ImGui::Separator();
                ImGui::Text("Local Max Bounds:");

                // Max bounds
                ImGui::Text("X:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MaxX", &max_bounds[0], bound_step, bound_step_fast, "%.3f");
                any_input_active |= ImGui::IsItemActive();
                any_input_deactivated |= ImGui::IsItemDeactivatedAfterEdit();
                max_bounds[0] = std::max(max_bounds[0], min_bounds[0]);

                ImGui::Text("Y:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MaxY", &max_bounds[1], bound_step, bound_step_fast, "%.3f");
                any_input_active |= ImGui::IsItemActive();
                any_input_deactivated |= ImGui::IsItemDeactivatedAfterEdit();
                max_bounds[1] = std::max(max_bounds[1], min_bounds[1]);

                ImGui::Text("Z:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(text_width);
                bounds_changed |= ImGui::InputFloat("##MaxZ", &max_bounds[2], bound_step, bound_step_fast, "%.3f");
                any_input_active |= ImGui::IsItemActive();
                any_input_deactivated |= ImGui::IsItemDeactivatedAfterEdit();
                max_bounds[2] = std::max(max_bounds[2], min_bounds[2]);

                // Capture state when editing starts
                if (any_input_active && !bounds_editing_active) {
                    bounds_editing_active = true;
                    s_state_before_edit = captureState(render_manager);
                }

                if (bounds_changed) {
                    settings.crop_min = glm::vec3(min_bounds[0], min_bounds[1], min_bounds[2]);
                    settings.crop_max = glm::vec3(max_bounds[0], max_bounds[1], max_bounds[2]);
                    settings_changed = true;

                    // Emit event for bounds change
                    ui::CropBoxChanged{
                        .min_bounds = settings.crop_min,
                        .max_bounds = settings.crop_max,
                        .enabled = settings.use_crop_box}
                        .emit();
                }

                // Commit undo when editing ends
                if (any_input_deactivated && bounds_editing_active) {
                    bounds_editing_active = false;
                    // Apply settings first
                    render_manager->updateSettings(settings);
                    commitUndoIfChanged(ctx.viewer, render_manager);
                }

                // Also handle case where user clicks away without making changes
                if (!any_input_active && bounds_editing_active) {
                    bounds_editing_active = false;
                    commitUndoIfChanged(ctx.viewer, render_manager);
                }

                // Display info
                glm::vec3 translation = settings.crop_transform.getTranslation();
                glm::vec3 size = settings.crop_max - settings.crop_min;

                ImGui::Text("Center: (%.3f, %.3f, %.3f)", translation.x, translation.y, translation.z);
                ImGui::Text("Size: (%.3f, %.3f, %.3f)", size.x, size.y, size.z);

                ImGui::TreePop();
            }
        }

        if (settings_changed) {
            render_manager->updateSettings(settings);
        }
    }

    const CropBoxState& getCropBoxState() {
        return CropBoxState::getInstance();
    }
} // namespace lfs::vis::gui::panels