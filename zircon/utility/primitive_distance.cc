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
#include "zircon/utility/popcount.h"
#include "turbo/base/bits.h"
#include "turbo/format/print.h"

namespace zircon::distance {

    ////////////////////////////// L1 ////////////////////////////////
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

    float simple_norm_l1(turbo::Span<float> a) {
        float norm = 0.0f;
        for (float i: a) {
            norm += std::abs(i);
        }
        return norm;
    }

    float norm_l1(turbo::Span<float> a) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");
        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        float sum = 0.0;
        b_type sum_vec = b_type::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            sum_vec += turbo::simd::fabs(avec);
        }

        sum += turbo::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum += std::fabs(a[i]);
        }
        return sum;
    }

    void simple_normalize_l1(turbo::Span<float> a, float norm, turbo::Span<float> out) {
        for (std::size_t i = 0; i < a.size(); ++i) {
            out[i] = a[i] / norm;
        }
    }

    void simple_normalize_l1(turbo::Span<float> a, turbo::Span<float> out) {
        float norm = simple_norm_l1(a);
        simple_normalize_l1(a, norm, out);
    }

    void normalize_l1(turbo::Span<float> a, turbo::Span<float> out) {
        float norm = norm_l1(a);
        normalize_l1(a, norm, out);
    }

    void normalize_l1(turbo::Span<float> a, float norm, turbo::Span<float> out) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(out.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");
        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            avec /= norm;
            avec.store(&out[i], turbo::aligned_mode());
        }
        for (std::size_t i = vec_size; i < size; ++i) {
            out[i] = a[i] / norm;
        }
    }

    void simple_normalize_l1(turbo::Span<float> a) {
        float norm = simple_norm_l1(a);
        simple_normalize_l1(a, norm, a);
    }

    void simple_normalize_l1(turbo::Span<float> a, float norm) {
        simple_normalize_l1(a, norm, a);
    }

    void normalize_l1(turbo::Span<float> a) {
        float norm = norm_l1(a);
        normalize_l1(a, norm, a);
    }

    void normalize_l1(turbo::Span<float> a, float norm) {
        normalize_l1(a, norm, a);
    }

    ////////////////////////////// IP ////////////////////////////////
    float simple_distance_ip(turbo::Span<float> a, turbo::Span<float> b) {
        float distance = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            distance += a[i] * b[i];
        }
        return distance;
    }

    float distance_ip(turbo::Span<float> a, turbo::Span<float> b) {
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
            sum_vec += avec * bvec;
        }

        sum += turbo::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum += a[i] * b[i];
        }
        return sum;

    }

    /////////////////////////// L2 ////////////////////////////////
    float simple_distance_l2(turbo::Span<float> a, turbo::Span<float> b) {
        float distance = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            distance += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(distance);
    }

    float distance_l2(turbo::Span<float> a, turbo::Span<float> b) {
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
            sum_vec += (avec - bvec) * (avec - bvec);
        }

        sum += turbo::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum);
    }

    float simple_distance_normalized_l2(turbo::Span<float> a, turbo::Span<float> b) {
        auto ip = simple_distance_ip(a, b);
        return sqrt(2.0f * (1.0f - ip));
    }

    float distance_normalized_l2(turbo::Span<float> a, turbo::Span<float> b) {
        auto ip = distance_ip(a, b);
        return sqrt(2.0f * (1.0f - ip));
    }

    float simple_norm_l2(turbo::Span<float> a) {
        float norm = 0.0f;
        for (float i: a) {
            norm += i * i;
        }
        return std::sqrt(norm);
    }

    float norm_l2(turbo::Span<float> a) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");
        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        float sum = 0.0;
        b_type sum_vec = b_type::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            sum_vec += avec * avec;
        }

        sum += turbo::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum += a[i] * a[i];
        }
        return std::sqrt(sum);
    }

    void simple_normalize_l2(turbo::Span<float> a, float norm, turbo::Span<float> out) {
        for (std::size_t i = 0; i < a.size(); ++i) {
            out[i] = a[i] / norm;
        }
    }

    void simple_normalize_l2(turbo::Span<float> a, turbo::Span<float> out) {
        float norm = simple_norm_l2(a);
        simple_normalize_l2(a, norm, out);
    }

    void normalize_l2(turbo::Span<float> a, float norm, turbo::Span<float> out) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(out.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");
        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            avec /= norm;
            avec.store(&out[i], turbo::aligned_mode());
        }
        for (std::size_t i = vec_size; i < size; ++i) {
            out[i] = a[i] / norm;
        }
    }

    void normalize_l2(turbo::Span<float> a, turbo::Span<float> out) {
        float norm = norm_l2(a);
        normalize_l2(a, norm, out);
    }

    void simple_normalize_l2(turbo::Span<float> a) {
        float norm = simple_norm_l2(a);
        simple_normalize_l2(a, norm, a);
    }

    void normalize_l2(turbo::Span<float> a) {
        float norm = norm_l2(a);
        normalize_l2(a, norm, a);
    }

    void simple_normalize_l2(turbo::Span<float> a, float norm) {
        simple_normalize_l2(a, norm, a);
    }

    void normalize_l2(turbo::Span<float> a, float norm) {
        normalize_l2(a, norm, a);
    }

    /////////////////////////// Cosine ////////////////////////////////
    float simple_distance_cosine(turbo::Span<float> a, turbo::Span<float> b) {
        float distance = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            distance += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return distance / std::sqrt(norm_a * norm_b);
    }

    float distance_cosine(turbo::Span<float> a, turbo::Span<float> b) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");
        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        float sum = 0.0;
        b_type sum_vec = b_type::broadcast(0.0f);
        b_type norm_a_vec = b_type::broadcast(0.0f);
        b_type norm_b_vec = b_type::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&b[i], turbo::aligned_mode());
            sum_vec += avec * bvec;
            norm_a_vec += avec * avec;
            norm_b_vec += bvec * bvec;
        }

        sum += turbo::simd::reduce_add(sum_vec);
        float norm_a = turbo::simd::reduce_add(norm_a_vec);
        float norm_b = turbo::simd::reduce_add(norm_b_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return sum / std::sqrt(norm_a * norm_b);
    }

    float simple_distance_normalized_cosine(turbo::Span<float> a, turbo::Span<float> b) {
        auto ip = simple_distance_ip(a, b);
        return 1.0f - ip;
    }

    float distance_normalized_cosine(turbo::Span<float> a, turbo::Span<float> b) {
        auto ip = distance_ip(a, b);
        return 1.0f - ip;
    }

    /////////////////////////// Jacard ////////////////////////////////

    float simple_distance_min_max_jaccard(turbo::Span<float> a, turbo::Span<float> b) {
        float sum_interact = 0.0f;
        float sum_union = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            sum_interact += std::min(a[i], b[i]);
            sum_union += std::max(a[i], b[i]);
        }
        return 1.0f - sum_interact / sum_union;
    }

    float distance_min_max_jaccard(turbo::Span<float> a, turbo::Span<float> b) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");
        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        b_type sum_interact_vec = b_type::broadcast(0.0f);
        b_type sum_union_vec = b_type::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&b[i], turbo::aligned_mode());
            sum_interact_vec += turbo::simd::min(avec, bvec);
            sum_union_vec += turbo::simd::max(avec, bvec);
        }
        float sum_interact = turbo::simd::reduce_add(sum_interact_vec);
        float sum_union = turbo::simd::reduce_add(sum_union_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum_interact += std::min(a[i], b[i]);
            sum_union += std::max(a[i], b[i]);
        }
        return 1.0f - sum_interact / sum_union;
    }

    float simple_distance_bits_jaccard(turbo::Span<float> a, turbo::Span<float> b) {
        float sum_interact = 0.0f;
        float sum_union = 0.0f;
        uint32_t *u32_ptr_a = reinterpret_cast<uint32_t *>(a.data());
        uint32_t *u32_ptr_b = reinterpret_cast<uint32_t *>(b.data());
        for (std::size_t i = 0; i < a.size(); ++i) {
            sum_interact += turbo::popcount(u32_ptr_a[i] & u32_ptr_b[i]);
            sum_union += turbo::popcount(u32_ptr_a[i] | u32_ptr_b[i]);
        }
        return 1.0f - sum_interact / sum_union;
    }

    float distance_bits_jaccard(turbo::Span<float> a, turbo::Span<float> b) {
        using e_type = uint64_t;
        using b_type = turbo::simd::batch<e_type, turbo::simd::default_arch>;
        auto u64_size = a.size() * sizeof(float) / sizeof(e_type);
        auto ai = turbo::Span<e_type>(reinterpret_cast<e_type *>(a.data()), u64_size);
        auto bi = turbo::Span<e_type>(reinterpret_cast<e_type *>(b.data()), u64_size);
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");
        std::size_t inc = b_type::size;
        std::size_t size = ai.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        uint32_t sum_interact = 0;
        uint32_t sum_union = 0;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&ai[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&bi[i], turbo::aligned_mode());
            sum_interact += popcount(avec & bvec);
            sum_union += popcount(avec | bvec);
        }
        auto undo_index = vec_size * sizeof(e_type) / sizeof(float);
        auto *u32_ptr_a = reinterpret_cast<uint32_t *>(a.data());
        auto *u32_ptr_b = reinterpret_cast<uint32_t *>(b.data());
        for (std::size_t i = undo_index; i < size; ++i) {
            sum_interact += turbo::popcount(u32_ptr_a[i] & u32_ptr_b[i]);
            sum_union += turbo::popcount(u32_ptr_a[i] | u32_ptr_b[i]);
        }
        return 1.0f - static_cast<float>(sum_interact) / static_cast<float>(sum_union);
    }

    /////////////////////////// Hamming ////////////////////////////////

    float simple_distance_hamming(turbo::Span<float> a, turbo::Span<float> b) {
        float sum_interact = 0.0f;
        uint32_t *u32_ptr_a = reinterpret_cast<uint32_t *>(a.data());
        uint32_t *u32_ptr_b = reinterpret_cast<uint32_t *>(b.data());
        for (std::size_t i = 0; i < a.size(); ++i) {
            sum_interact += turbo::popcount(u32_ptr_a[i] ^ u32_ptr_b[i]);
        }
        return sum_interact;
    }

    float distance_hamming(turbo::Span<float> a, turbo::Span<float> b) {
        using e_type = uint64_t;
        using b_type = turbo::simd::batch<e_type, turbo::simd::default_arch>;
        auto u64_size = a.size() * sizeof(float) / sizeof(e_type);
        auto ai = turbo::Span<e_type>(reinterpret_cast<e_type *>(a.data()), u64_size);
        auto bi = turbo::Span<e_type>(reinterpret_cast<e_type *>(b.data()), u64_size);
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");
        std::size_t inc = b_type::size;
        std::size_t size = ai.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;
        uint32_t sum_interact = 0;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&ai[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&bi[i], turbo::aligned_mode());
            sum_interact += popcount(avec ^ bvec);
        }
        auto undo_index = vec_size * sizeof(e_type) / sizeof(float);
        auto *u32_ptr_a = reinterpret_cast<uint32_t *>(a.data());
        auto *u32_ptr_b = reinterpret_cast<uint32_t *>(b.data());
        for (std::size_t i = undo_index; i < size; ++i) {
            sum_interact += turbo::popcount(u32_ptr_a[i] ^ u32_ptr_b[i]);
        }
        return sum_interact;
    }

    /////////////////////////// Canberra ////////////////////////////////

    float simple_distance_canberra(turbo::Span<float> a, turbo::Span<float> b) {
        float distance = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (TURBO_UNLIKELY(a[i] == 0.0f && b[i] == 0.0f)) {
                continue;
            }
            distance += std::abs(a[i] - b[i]) / (std::abs(a[i]) + std::abs(b[i]));
        }
        return distance;
    }

    float distance_canberra(turbo::Span<float> a, turbo::Span<float> b) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");
        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;

        float sum = 0.0;
        b_type sum_vec = b_type::broadcast(0.0f);
        static const b_type zero_vec = b_type::broadcast(0.0f);
        static const b_type one_vec = b_type::broadcast(1.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&b[i], turbo::aligned_mode());
            b_type abs_sum = turbo::simd::abs(avec) + turbo::simd::abs(bvec);
            b_type abs_diff = turbo::simd::abs(avec - bvec);
            auto mask = turbo::simd::abs(abs_sum) > zero_vec;
            // if |q| + |p| == 0 means that both are zero
            // in this case we let the |q -p| / (|q| + |p|) to be zero
            auto fixed_abs_sum = turbo::simd::select(mask, abs_sum, one_vec);
            sum_vec += abs_diff / fixed_abs_sum;
        }

        sum += turbo::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            if (TURBO_UNLIKELY(a[i] == 0.0f && b[i] == 0.0f)) {
                continue;
            }
            sum += std::abs(a[i] - b[i]) / (std::abs(a[i]) + std::abs(b[i]));
        }
        return sum;
    }

    /////////////////////////// Lp ////////////////////////////////
    float simple_distance_lp(turbo::Span<float> a, turbo::Span<float> b, float p) {
        float distance = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            distance += std::pow(std::abs(a[i] - b[i]), p);
        }
        return std::pow(distance, 1.0f / p);
    }

    float distance_lp(turbo::Span<float> a, turbo::Span<float> b, float p) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");
        TLOG_CHECK(p > 0.0f, "p must be greater than 0");
        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;

        float sum = 0.0;
        b_type sum_vec = b_type::broadcast(0.0f);
        static const b_type zero_vec = b_type::broadcast(0.0f);
        static const b_type one_vec = b_type::broadcast(0.0f);
        b_type p_vec = b_type::broadcast(p);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&b[i], turbo::aligned_mode());
            b_type abs_diff = turbo::simd::abs(avec - bvec);
            auto mask = turbo::simd::abs(abs_diff) > zero_vec;
            auto fixed_abs_diff = turbo::simd::select(mask, abs_diff, one_vec);
            sum_vec += turbo::simd::pow(fixed_abs_diff, p_vec);
        }

        sum += turbo::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum += std::pow(std::abs(a[i] - b[i]), p);
        }
        return std::pow(sum, 1.0f / p);
    }

    /////////////////////////// Bray Curtis ////////////////////////////////
    float simple_distance_bray_curtis(turbo::Span<float> a, turbo::Span<float> b) {
        float sum_interact = 0.0f;
        float sum_union = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            sum_interact += std::abs(a[i] - b[i]);
            sum_union += std::abs(a[i] + b[i]);
        }
        // if sum_union is zero then both vectors are zero
        // in this case we let the sum_interact / sum_union to be zero
        if (sum_union == 0.0f) {
            return 0.0f;
        }
        return sum_interact / sum_union;
    }

    float distance_bray_curtis(turbo::Span<float> a, turbo::Span<float> b) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");

        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;

        b_type acc_vec = b_type::broadcast(0.0f);
        b_type den_vec = b_type::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&b[i], turbo::aligned_mode());
            acc_vec += turbo::simd::abs(avec - bvec);
            den_vec += turbo::simd::abs(avec + bvec);
        }

        auto sum_acc = turbo::simd::reduce_add(acc_vec);
        auto sum_den = turbo::simd::reduce_add(den_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum_acc += std::abs(a[i] - b[i]);
            sum_den += std::abs(a[i] + b[i]);
        }
        // if sum_union is zero then both vectors are zero
        // in this case we let the sum_interact / sum_union to be zero
        if (sum_den == 0.0f) {
            return 0.0f;
        }
        return sum_acc / sum_den;
    }

    /////////////////////////// Jensen Shannon ////////////////////////////////

    float simple_distance_jensen_shannon(turbo::Span<float> a, turbo::Span<float> b) {
        float acc = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            float m = 0.5f * (a[i] + b[i]);
            acc += a[i] * std::log(a[i] / m);
            acc += b[i] * std::log(b[i] / m);
        }
        return 0.5f * acc;
    }

    float distance_jensen_shannon(turbo::Span<float> a, turbo::Span<float> b) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");

        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;

        b_type acc_vec = b_type::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&b[i], turbo::aligned_mode());
            b_type mvec = 0.5 * (avec + bvec);

            b_type kl1 = avec * turbo::simd::log(avec/mvec);
            b_type kl2 = bvec * turbo::simd::log(bvec/mvec);
            acc_vec += kl1 + kl2;
        }

        auto sum_acc = turbo::simd::reduce_add(acc_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            float m = 0.5f * (a[i] + b[i]);
            sum_acc += a[i] * std::log(a[i] / m);
            sum_acc += b[i] * std::log(b[i] / m);
        }
        return 0.5f * sum_acc;
    }

    /////////////////////////// Linf ////////////////////////////////
    float simple_distance_linf(turbo::Span<float> a, turbo::Span<float> b) {
        float distance = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            distance = std::max(distance, std::abs(a[i] - b[i]));
        }
        return distance;
    }

    float distance_linf(turbo::Span<float> a, turbo::Span<float> b) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");

        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;

        b_type acc_vec = b_type::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&b[i], turbo::aligned_mode());
            acc_vec = turbo::simd::max(acc_vec, turbo::simd::abs(avec - bvec));
        }

        auto max_acc = turbo::simd::reduce_max(acc_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            max_acc = std::max(max_acc, std::abs(a[i] - b[i]));
        }
        return max_acc;
    }

    /////////////////////////// Cross Entropy ////////////////////////////////

    float simple_distance_cross_entropy(turbo::Span<float> a, turbo::Span<float> b) {
        float distance = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            distance += a[i] * std::log(b[i]);
        }
        return -distance;
    }

    float distance_cross_entropy(turbo::Span<float> a, turbo::Span<float> b) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");

        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;

        b_type acc_vec = b_type::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&b[i], turbo::aligned_mode());
            acc_vec += avec * turbo::simd::log(bvec);
        }

        auto sum_acc = turbo::simd::reduce_add(acc_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            sum_acc += a[i] * std::log(b[i]);
        }
        return -sum_acc;
    }

    /////////////////////////// Kullback Leibler (KLD) ////////////////////////////////

    float simple_distance_kld(turbo::Span<float> a, turbo::Span<float> b) {
        float distance = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            distance += a[i] * std::log(a[i] / b[i]);
        }
        return distance;
    }

    float distance_kld(turbo::Span<float> a, turbo::Span<float> b) {
        using b_type = turbo::simd::batch<float, turbo::simd::default_arch>;
        bool is_aligned = turbo::is_aligned(a.data(), 64) && turbo::is_aligned(b.data(), 64);
        TLOG_CHECK(is_aligned, "the memory must be aligned");

        std::size_t inc = b_type::size;
        std::size_t size = a.size();
        // size for which the vectorization is possible
        std::size_t vec_size = size - size % inc;

        b_type acc_vec = b_type::broadcast(0.0f);
        static const b_type zero_vec = b_type::broadcast(0.0f);
        b_type eps_vec = b_type::broadcast(1e-7f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type avec = b_type::load(&a[i], turbo::aligned_mode());
            b_type bvec = b_type::load(&b[i], turbo::aligned_mode());
            b_type fixed_avec = turbo::simd::select(avec > zero_vec, avec, eps_vec);
            b_type fixed_bvec = turbo::simd::select(bvec > zero_vec, bvec, eps_vec);
            acc_vec += fixed_avec * (turbo::simd::log(fixed_avec) - turbo::simd::log(fixed_bvec));
        }

        auto sum_acc = turbo::simd::reduce_add(acc_vec);
        for (std::size_t i = vec_size; i < size; ++i) {
            float fixed_a = a[i] > 0.0f ? a[i] : 1e-7f;
            float fixed_b = b[i] > 0.0f ? b[i] : 1e-7f;
            sum_acc += fixed_a * (std::log(fixed_a) - std::log(fixed_b));
        }
        return sum_acc;
    }

    /////////////////////////// Angle ////////////////////////////////

    float simple_distance_angle(turbo::Span<float> a, turbo::Span<float> b) {
        double cosine = simple_distance_cosine(a, b);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return acos(-1.0);
        } else {
            return acos(cosine);
        }
    }

    float distance_angle(turbo::Span<float> a, turbo::Span<float> b) {
        double cosine = distance_cosine(a, b);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return acos(-1.0);
        } else {
            return acos(cosine);
        }
    }

    /////////////////////////// Normalized Angle ////////////////////////////////

    float simple_distance_normalized_angle(turbo::Span<float> a, turbo::Span<float> b) {
        double cosine = simple_distance_normalized_cosine(a, b);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return acos(-1.0);
        } else {
            return acos(cosine);
        }
    }

    float distance_normalized_angle(turbo::Span<float> a, turbo::Span<float> b) {
        double cosine = distance_normalized_cosine(a, b);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return acos(-1.0);
        } else {
            return acos(cosine);
        }
    }

}  // namespace zircon::distance