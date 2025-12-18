/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/parameters.hpp"

namespace lfs::core {

    // Returns 0 on success, 1 on failure
    int run_converter(const param::ConvertParameters& params);

} // namespace lfs::core
