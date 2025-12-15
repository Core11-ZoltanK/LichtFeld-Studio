/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "notification_popup.hpp"
#include "core_new/events.hpp"
#include <imgui.h>

namespace lfs::vis::gui {

    namespace {
        constexpr float BUTTON_WIDTH = 100.0f;
        constexpr float TEXT_WRAP_WIDTH = 30.0f;
        constexpr ImVec4 COLOR_ERROR{0.9f, 0.3f, 0.3f, 1.0f};
        constexpr ImVec4 COLOR_WARNING{0.9f, 0.7f, 0.2f, 1.0f};
        constexpr ImVec4 COLOR_INFO{0.3f, 0.6f, 0.9f, 1.0f};
        constexpr ImVec4 COLOR_BG{0.15f, 0.15f, 0.15f, 0.95f};
        constexpr ImVec4 COLOR_TITLE_BG{0.2f, 0.2f, 0.2f, 1.0f};
        constexpr ImVec4 COLOR_TITLE_BG_ACTIVE{0.25f, 0.25f, 0.25f, 1.0f};
    } // namespace

    using namespace lfs::core::events;

    NotificationPopup::NotificationPopup() {
        setupEventHandlers();
    }

    void NotificationPopup::setupEventHandlers() {
        state::DatasetLoadCompleted::when([this](const auto& e) {
            if (!e.success && e.error.has_value()) {
                show(Type::ERROR, "Failed to Load Dataset", *e.error);
            }
        });
    }

    void NotificationPopup::show(Type type, const std::string& title, const std::string& message) {
        pending_.push_back({type, title, message});
    }

    void NotificationPopup::render() {
        if (!popup_open_ && !pending_.empty()) {
            current_ = pending_.front();
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
            case Type::ERROR:
                accent_color = COLOR_ERROR;
                type_label = "Error";
                break;
            case Type::WARNING:
                accent_color = COLOR_WARNING;
                type_label = "Warning";
                break;
            case Type::INFO:
            default:
                accent_color = COLOR_INFO;
                type_label = "Info";
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

        constexpr ImGuiWindowFlags FLAGS = ImGuiWindowFlags_AlwaysAutoResize |
                                           ImGuiWindowFlags_NoCollapse |
                                           ImGuiWindowFlags_NoDocking;

        if (ImGui::BeginPopupModal(current_.title.c_str(), nullptr, FLAGS)) {
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
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(4);
    }

} // namespace lfs::vis::gui
