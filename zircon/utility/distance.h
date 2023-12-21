// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
// Created by jeff on 23-12-21.
//

#ifndef ZIRCON_UTILITY_DISTANCE_H_
#define ZIRCON_UTILITY_DISTANCE_H_

#include "turbo/simd/simd.h"
#include "turbo/meta/span.h"
#include "turbo/log/logging.h"
#include "turbo/base/bits.h"
#include "zircon/core/metric_type.h"

namespace zircon {

    template<MetricType>
    struct VectorDistance {
        float distance(turbo::Span<float> a, turbo::Span<float> b) const;

        void normalize(turbo::Span<float> a) const;
    };


}  // namespace zircon

#endif  // ZIRCON_UTILITY_DISTANCE_H_
