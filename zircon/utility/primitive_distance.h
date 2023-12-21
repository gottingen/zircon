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

#ifndef ZIRCON_UTILITY_PRIMITIVE_DISTANCE_H_
#define ZIRCON_UTILITY_PRIMITIVE_DISTANCE_H_

#include "turbo/simd/simd.h"
#include "turbo/meta/span.h"
#include "zircon//core/allocator.h"
#include "turbo/log/logging.h"
#include "turbo/memory/prefetch.h"

namespace zircon::distance {

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the L1 distance between two vectors.
     *        SUM(|a[i] - b[i]|), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The L1 distance between the two vectors.
     */
    float simple_distance_l1(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the L1 distance between two vectors.
     *        SUM(|a[i] - b[i]|), SIMD implementation.
     *        simd implementation is faster than simple_distance_l1
     * @param a The first vector.
     * @param b The second vector.
     * @return The L1 distance between the two vectors.
     */
    float distance_l1(turbo::Span<float> a, turbo::Span<float> b);
}  // namespace zircon::distance

#endif // ZIRCON_UTILITY_PRIMITIVE_DISTANCE_H_
