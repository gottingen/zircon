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

#include "zircon/utility/primitive_distance.h"

namespace zircon::distance {

    float simple_distance_l1(turbo::Span<float> a, turbo::Span<float> b) {
        float distance = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            distance += std::abs(a[i] - b[i]);
        }
        return distance;
    }

    float distance_l1(turbo::Span<float> a, turbo::Span<float> b) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");
        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        float sum = 0.0;
        b_type sum_vec = b_type::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&b[i], turbo::aligned_mode());
            sum_vec += turbo::simd::fabs(avec - bvec);
        }

        sum += turbo::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum += std::fabs(a[i] - b[i]);
        }
        return sum;
    }
}  // namespace zircon::distance