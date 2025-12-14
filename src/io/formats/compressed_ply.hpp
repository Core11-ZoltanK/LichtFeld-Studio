/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Re-export public API
#include "io/exporter.hpp"

namespace lfs::io {

    /**
     * @brief Check if a PLY file is in compressed format
     * @param filepath Path to the PLY file
     * @return true if the file is a compressed PLY
     */
    bool is_compressed_ply(const std::filesystem::path& filepath);

    /**
     * @brief Load compressed PLY file and return SplatData
     *
     * Compressed PLY format uses chunk-based quantization:
     * - 256 splats per chunk with min/max bounds
     * - Position: 11-10-11 bit packed (uint32)
     * - Rotation: 10-10-10-2 bit packed (uint32)
     * - Scale: 11-10-11 bit packed (uint32)
     * - Color+Opacity: 8-8-8-8 bit packed (uint32)
     * - Optional SH: uint8 per coefficient
     *
     * @param filepath Path to the compressed PLY file
     * @return SplatData on success, error string on failure
     */
    std::expected<SplatData, std::string>
    load_compressed_ply(const std::filesystem::path& filepath);

} // namespace lfs::io
