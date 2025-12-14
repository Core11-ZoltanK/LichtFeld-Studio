/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "html.hpp"
#include "core_new/logger.hpp"
#include "html_viewer_resources.hpp"
#include "sogs.hpp"

#include <fstream>
#include <sstream>

namespace lfs::io {

    namespace {

        constexpr char BASE64_CHARS[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        std::string base64_encode(const std::vector<uint8_t>& data) {
            std::string result;
            result.reserve(((data.size() + 2) / 3) * 4);

            for (size_t i = 0; i < data.size(); i += 3) {
                const uint32_t b0 = data[i];
                const uint32_t b1 = (i + 1 < data.size()) ? data[i + 1] : 0;
                const uint32_t b2 = (i + 2 < data.size()) ? data[i + 2] : 0;

                result += BASE64_CHARS[(b0 >> 2) & 0x3F];
                result += BASE64_CHARS[((b0 << 4) | (b1 >> 4)) & 0x3F];
                result += (i + 1 < data.size()) ? BASE64_CHARS[((b1 << 2) | (b2 >> 6)) & 0x3F] : '=';
                result += (i + 2 < data.size()) ? BASE64_CHARS[b2 & 0x3F] : '=';
            }
            return result;
        }

        std::vector<uint8_t> read_file_binary(const std::filesystem::path& path) {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file)
                return {};

            const auto size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<uint8_t> buffer(size);
            file.read(reinterpret_cast<char*>(buffer.data()), size);
            return buffer;
        }

        std::string replace_placeholder(std::string_view input, std::string_view placeholder, std::string_view replacement) {
            std::string result;
            result.reserve(input.size() + replacement.size());

            size_t pos = 0;
            while (pos < input.size()) {
                const size_t found = input.find(placeholder, pos);
                if (found == std::string_view::npos) {
                    result.append(input.substr(pos));
                    break;
                }
                result.append(input.substr(pos, found - pos));
                result.append(replacement);
                pos = found + placeholder.size();
            }
            return result;
        }

        std::string pad_text(std::string_view text, int spaces) {
            std::string result;
            std::string whitespace(spaces, ' ');
            size_t pos = 0;
            while (pos < text.size()) {
                const size_t newline = text.find('\n', pos);
                if (newline == std::string_view::npos) {
                    result += whitespace;
                    result.append(text.substr(pos));
                    break;
                }
                result += whitespace;
                result.append(text.substr(pos, newline - pos + 1));
                pos = newline + 1;
            }
            return result;
        }

        std::string generate_html(const std::string& base64_sog) {
            const auto tmpl = get_viewer_template();
            const auto css = get_viewer_css();
            const auto js = get_viewer_js();

            std::string html{tmpl};

            const std::string style_link = R"(<link rel="stylesheet" href="./index.css">)";
            const std::string inline_style = "<style>\n" + pad_text(css, 12) + "\n        </style>";
            html = replace_placeholder(html, style_link, inline_style);

            const std::string js_import = "import { main } from './index.js';";
            html = replace_placeholder(html, js_import, js);

            const std::string settings_fetch = "settings: fetch(settingsUrl).then(response => response.json())";
            const std::string inline_settings = R"(settings: {"camera":{"fov":50,"position":[2,2,-2],"target":[0,0,0],"startAnim":"none"},"background":{"color":[0,0,0]},"animTracks":[]})";
            html = replace_placeholder(html, settings_fetch, inline_settings);

            const std::string content_fetch = "fetch(contentUrl)";
            const std::string base64_fetch = "fetch(\"data:application/octet-stream;base64," + base64_sog + "\")";
            html = replace_placeholder(html, content_fetch, base64_fetch);

            html = replace_placeholder(html, ".compressed.ply", ".sog");

            return html;
        }

    } // anonymous namespace

    std::expected<void, std::string> export_html(const SplatData& splat_data, const HtmlExportOptions& options) {
        if (options.progress_callback) {
            options.progress_callback(0.0f, "Exporting SOG...");
        }

        const auto temp_sog = std::filesystem::temp_directory_path() / "lfs_html_export_temp.sog";
        const SogSaveOptions sog_options{
            .output_path = temp_sog,
            .kmeans_iterations = options.kmeans_iterations,
            .use_gpu = true,
            .progress_callback = [&](float p, const std::string& stage) {
                if (options.progress_callback) {
                    options.progress_callback(p * 0.5f, stage);
                }
                return true;
            }};

        if (auto result = save_sog(splat_data, sog_options); !result) {
            return std::unexpected("Failed to write SOG: " + result.error());
        }

        if (options.progress_callback) {
            options.progress_callback(0.5f, "Encoding data...");
        }

        const auto sog_data = read_file_binary(temp_sog);
        std::filesystem::remove(temp_sog);

        if (sog_data.empty()) {
            return std::unexpected("Failed to read temporary SOG file");
        }

        const auto base64_data = base64_encode(sog_data);

        if (options.progress_callback) {
            options.progress_callback(0.8f, "Generating HTML...");
        }

        const auto html = generate_html(base64_data);

        std::ofstream out(options.output_path);
        if (!out) {
            return std::unexpected("Failed to open output file: " + options.output_path.string());
        }
        out << html;
        out.close();

        if (options.progress_callback) {
            options.progress_callback(1.0f, "Done");
        }

        LOG_INFO("Exported HTML viewer: {} ({:.1f} MB)",
                 options.output_path.string(),
                 static_cast<float>(html.size()) / (1024 * 1024));

        return {};
    }

} // namespace lfs::io
