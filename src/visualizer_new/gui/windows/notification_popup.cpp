/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "notification_popup.hpp"
#include "core_new/events.hpp"
#include <cmath>
#include <format>
#include <imgui.h>

namespace lfs::vis::gui {

    namespace {
        constexpr float BUTTON_WIDTH = 100.0f;
        constexpr float TEXT_WRAP_WIDTH = 30.0f;
        constexpr ImVec4 COLOR_ERROR{0.9f, 0.3f, 0.3f, 1.0f};
        constexpr ImVec4 COLOR_WARNING{0.9f, 0.7f, 0.2f, 1.0f};
        constexpr ImVec4 COLOR_SUCCESS{0.3f, 0.8f, 0.4f, 1.0f};
        constexpr ImVec4 COLOR_BG{0.15f, 0.15f, 0.15f, 0.95f};
        constexpr ImVec4 COLOR_TITLE_BG{0.2f, 0.2f, 0.2f, 1.0f};
        constexpr ImVec4 COLOR_TITLE_BG_ACTIVE{0.25f, 0.25f, 0.25f, 1.0f};
        constexpr ImGuiWindowFlags POPUP_FLAGS = ImGuiWindowFlags_AlwaysAutoResize |
                                                 ImGuiWindowFlags_NoCollapse |
                                                 ImGuiWindowFlags_NoDocking;
    } // namespace

    using namespace lfs::core::events;

    NotificationPopup::NotificationPopup() {
        setupEventHandlers();
    }

    std::string NotificationPopup::formatDuration(const float seconds) {
        const float clamped = std::max(0.0f, seconds);
        const int total = static_cast<int>(std::round(clamped));
        const int hours = total / 3600;
        const int minutes = (total % 3600) / 60;
        const int secs = total % 60;

        if (hours > 0) {
            return std::format("{}h {}m {}s", hours, minutes, secs);
        }
        if (minutes > 0) {
            return std::format("{}m {}s", minutes, secs);
        }
        if (clamped >= 1.0f) {
            return std::format("{}s", secs);
        }
        return std::format("{:.1f}s", clamped);
    }

    void NotificationPopup::setupEventHandlers() {
        state::DatasetLoadCompleted::when([this](const auto& e) {
            if (!e.success && e.error.has_value()) {
                show(Type::FAILURE, "Failed to Load Dataset", *e.error);
            }
        });

        state::TrainingCompleted::when([this](const auto& e) {
            if (e.success) {
                const auto message = std::format(
                    "Training completed successfully!\n\n"
                    "Iterations: {}\n"
                    "Final loss: {:.6f}\n"
                    "Duration: {}",
                    e.iteration, e.final_loss, formatDuration(e.elapsed_seconds));

                show(Type::INFO, "Training Complete", message,
                     []() { cmd::SwitchToLatestCheckpoint{}.emit(); });
            } else {
                show(Type::FAILURE, "Training Failed",
                     e.error.value_or("Unknown error occurred during training."));
            }
        });
    }

    void NotificationPopup::show(const Type type, const std::string& title,
                                 const std::string& message, Callback on_close) {
        pending_.push_back({type, title, message, std::move(on_close)});
    }

    void NotificationPopup::render() {
        if (!popup_open_ && !pending_.empty()) {
            current_ = std::move(pending_.front());
            pending_.pop_front();
            popup_open_ = true;
            ImGui::OpenPopup(current_.title.c_str());
        }

        if (!popup_open_) {
            return;
        }

        ImVec4 accent_color;
        const char* type_label;
        switch (current_.type) {
            case Type::FAILURE:
                accent_color = COLOR_ERROR;
                type_label = "Error";
                break;
            case Type::WARNING:
                accent_color = COLOR_WARNING;
                type_label = "Warning";
                break;
            case Type::INFO:
            default:
                accent_color = COLOR_SUCCESS;
                type_label = "Success";
                break;
        }

        const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        ImGui::PushStyleColor(ImGuiCol_WindowBg, COLOR_BG);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, COLOR_TITLE_BG);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, COLOR_TITLE_BG_ACTIVE);
        ImGui::PushStyleColor(ImGuiCol_Border, accent_color);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 2.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.0f, 12.0f));

        if (ImGui::BeginPopupModal(current_.title.c_str(), nullptr, POPUP_FLAGS)) {
            ImGui::TextColored(accent_color, "%s", type_label);
            ImGui::SameLine();
            ImGui::TextDisabled("|");
            ImGui::SameLine();
            ImGui::TextUnformatted(current_.title.c_str());

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::PushTextWrapPos(ImGui::GetFontSize() * TEXT_WRAP_WIDTH);
            ImGui::TextUnformatted(current_.message.c_str());
            ImGui::PopTextWrapPos();

            ImGui::Spacing();
            ImGui::Spacing();

            const float avail = ImGui::GetContentRegionAvail().x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - BUTTON_WIDTH) * 0.5f);

            if (ImGui::Button("OK", ImVec2(BUTTON_WIDTH, 0)) ||
                ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                popup_open_ = false;
                if (current_.on_close) {
                    current_.on_close();
                }
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(4);
    }

} // namespace lfs::vis::gui
