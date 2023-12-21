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

#ifndef ZIRCON_UTILITY_POPCOUNT_H_
#define ZIRCON_UTILITY_POPCOUNT_H_

#include "turbo/simd/simd.h"
#include "turbo/base/bits.h"
#include "turbo/format/print.h"
#include "turbo/meta/type_traits.h"

namespace zircon::distance {

    template<typename T, typename A, size_t N>
    struct popcount_impl {
        uint32_t operator()(const turbo::simd::batch<T, A> &a);
        static_assert(sizeof(T) == 4 || sizeof(T) == 8, "sizeof T must equal 4 or 8");
        static_assert(std::is_unsigned<T>::value, "T must be unsigned");
    };

    // A = avx512 T = uint32_t
    template<typename T, typename A>
    struct popcount_impl<T, A, 16> {
        uint32_t operator()(const turbo::simd::batch<T, A> &a) {
            return turbo::popcount(a.get(0)) + turbo::popcount(a.get(1)) + turbo::popcount(a.get(2)) +
                   turbo::popcount(a.get(3)) + turbo::popcount(a.get(4)) + turbo::popcount(a.get(5)) +
                   turbo::popcount(a.get(6)) + turbo::popcount(a.get(7)) + turbo::popcount(a.get(8)) +
                   turbo::popcount(a.get(9)) + turbo::popcount(a.get(10)) + turbo::popcount(a.get(11)) +
                   turbo::popcount(a.get(12)) + turbo::popcount(a.get(13)) + turbo::popcount(a.get(14)) +
                   turbo::popcount(a.get(15));
        }
    };

    // A = avx2 T = uint32_t
    // A = avx512 T = uint64_t
    template<typename T, typename A>
    struct popcount_impl<T, A, 8> {
        uint32_t operator()(const turbo::simd::batch<T, A> &a) {
            return turbo::popcount(a.get(0)) + turbo::popcount(a.get(1)) + turbo::popcount(a.get(2)) +
                   turbo::popcount(a.get(3)) + turbo::popcount(a.get(4)) + turbo::popcount(a.get(5)) +
                   turbo::popcount(a.get(6)) + turbo::popcount(a.get(7));
        }
    };

    // A = avx2 T = uint64_t
    template<typename T, typename A>
    struct popcount_impl<T, A, 4> {

        uint32_t operator()(const turbo::simd::batch<T, A> &a) {
            return turbo::popcount(a.get(0)) + turbo::popcount(a.get(1)) + turbo::popcount(a.get(2)) +
                   turbo::popcount(a.get(3));
        }
    };


    template<typename T, typename A>
    inline uint32_t popcount(const turbo::simd::batch<T, A> &v) {
        return popcount_impl<T, A, turbo::simd::batch<T, A>::size>()(v);
    }
}  // namespace zircon::distance

#endif // ZIRCON_UTILITY_POPCOUNT_H_
