// Copyright (c) 2026. Licensed under the BSD 3-Clause License.
#pragma once

#include "exe_graph/runtime/shape.h"
#include "op_common/log/log.h"
#include <sstream>
#include <string>

namespace QLIHost {
// Diagnostics only: use the SDK logging macros without vendoring opbase.
inline std::string ShapeString(const gert::Shape& shape)
{
    std::ostringstream text;
    text << "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i) text << ", ";
        text << shape.GetDim(i);
    }
    text << "]";
    return text.str();
}
} // namespace QLIHost
