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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// L1 distance
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the L1 norm of a vector.
     *       SUM(|a[i]|), simple version.
     *       for testing purposes.
     * @param a The vector.
     * @return The L1 norm of the vector.
     */
    float simple_norm_l1(turbo::Span<float> a);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the L1 norm of a vector.
     *       SUM(|a[i]|), SIMD version.
     *       simd implementation is faster than simple_norm_l1
     * @param a The vector.
     * @return The L1 norm of the vector.
     */
    float norm_l1(turbo::Span<float> a);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize l1 norm of a vector.
     *        simple version.
     * @param a [input] The vector.
     * @param norm [input] The norm of the vector.
     * @param out [output] The normalized vector.
     */
    void simple_normalize_l1(turbo::Span<float> a, float norm, turbo::Span<float> out);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize l1 norm of a vector.
     *        simple version.
     * @param a [input] The vector.
     * @param out [output] The normalized vector.
     */
    void simple_normalize_l1(turbo::Span<float> a, turbo::Span<float> out);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize l1 norm of a vector. simd version
     * @param a [input] The vector.
     * @param norm [input] The norm of the vector.
     * @param out [output] The normalized vector.
     */
    void normalize_l1(turbo::Span<float> a, float norm, turbo::Span<float> out);
    /**
     * @ingroup zircon_utility_distance
     * @brief normalize l1 norm of a vector.
     *        simple version.
     * @param a [input] The vector.
     * @param norm [input] The norm of the vector.
     * @param out [output] The normalized vector.
     */
    void normalize_l1(turbo::Span<float> a, turbo::Span<float> out);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize a vector inplace. simple version
     * @param a [input/output] The vector.
     */
    void simple_normalize_l1(turbo::Span<float> a);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize a vector inplace. simd version
     * @param a [input/output] The vector.
     */
    void normalize_l1(turbo::Span<float> a);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize a vector inplace. simple version
     * @param a [input/output] The vector.
     * @param norm [input] The norm of the vector.
     */
    void simple_normalize_l1(turbo::Span<float> a, float norm);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize a vector inplace. simd version
     * @param a [input/output] The vector.
     * @param norm [input] The norm of the vector.
     */
    void normalize_l1(turbo::Span<float> a, float norm);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Inner product
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the inner product between two vectors.
     *        SUM(a[i] * b[i]), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The inner product between the two vectors.
     */
    float simple_distance_ip(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the inner product between two vectors.
     *        SUM(a[i] * b[i]), SIMD implementation.
     *        simd implementation is faster than simple_distance_ip
     * @param a The first vector.
     * @param b The second vector.
     * @return The inner product between the two vectors.
     */
    float distance_ip(turbo::Span<float> a, turbo::Span<float> b);


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// L2 distance
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the L2 distance between two vectors.
     *        SUM((a[i] - b[i])^2), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The L2 distance between the two vectors.
     */
    float simple_distance_l2(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the L2 distance between two vectors.
     *        SUM((a[i] - b[i])^2), SIMD implementation.
     *        simd implementation is faster than simple_distance_l2
     * @param a The first vector.
     * @param b The second vector.
     * @return The L2 distance between the two vectors.
     */
    float distance_l2(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the L2 distance between two vectors.
     *        SUM((a[i] - b[i])^2), in this case, assume that
     *        a and b all normalized to 1.0f, so we can simplify
     *        the formula to 2 - 2 * (a * b), this is a simple
     *        version, for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The L2 distance between the two vectors.
     */
    float simple_distance_normalized_l2(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the L2 distance between two vectors.
     *        SUM((a[i] - b[i])^2), in this case, assume that
     *        a and b all normalized to 1.0f, so we can simplify
     *        the formula to 2 - 2 * (a * b), this is a simd
     *        implementation, simd implementation is faster than
     *        simple_distance_normalized_l2
     * @param a The first vector.
     * @param b The second vector.
     * @return The L2 distance between the two vectors.
     */
    float distance_normalized_l2(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the L2 norm of a vector.
     *       SUM((a[i])^2), simple version.
     *       for testing purposes.
     * @param a The vector.
     * @return The L2 norm of the vector.
     */
    float simple_norm_l2(turbo::Span<float> a);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the L2 norm of a vector.
     *       SUM((a[i])^2), SIMD version.
     *       simd implementation is faster than simple_norm_l2
     * @param a The vector.
     * @return The L2 norm of the vector.
     */
    float norm_l2(turbo::Span<float> a);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize l2 norm of a vector.
     *        simple version.
     * @param a [input] The vector.
     * @param norm [input] The norm of the vector.
     * @param out [output] The normalized vector.
     */
    void simple_normalize_l2(turbo::Span<float> a, float norm, turbo::Span<float> out);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize l2 norm of a vector.
     *        simple version.
     * @param a [input] The vector.
     * @param out [output] The normalized vector.
     */
    void simple_normalize_l2(turbo::Span<float> a, turbo::Span<float> out);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize l2 norm of a vector. simd version
     * @param a [input] The vector.
     * @param norm [input] The norm of the vector.
     * @param out [output] The normalized vector.
     */
    void normalize_l2(turbo::Span<float> a, float norm, turbo::Span<float> out);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize l2 norm of a vector.
     *        simple version.
     * @param a [input] The vector.
     * @param norm [input] The norm of the vector.
     * @param out [output] The normalized vector.
     */
    void normalize_l2(turbo::Span<float> a, turbo::Span<float> out);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize a vector inplace. simple version
     * @param a [input/output] The vector.
     */
    void simple_normalize_l2(turbo::Span<float> a);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize a vector inplace. simd version
     * @param a [input/output] The vector.
     */
    void normalize_l2(turbo::Span<float> a);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize a vector inplace. simple version
     * @param a [input/output] The vector.
     * @param norm [input] The norm of the vector.
     */
    void simple_normalize_l2(turbo::Span<float> a, float norm);

    /**
     * @ingroup zircon_utility_distance
     * @brief normalize a vector inplace. simd version
     * @param a [input/output] The vector.
     * @param norm [input] The norm of the vector.
     */
    void normalize_l2(turbo::Span<float> a, float norm);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Cosine distance
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the cosine distance between two vectors.
     *        1 - (a * b) / (|a| * |b|), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The cosine distance between the two vectors.
     */
    float simple_distance_cosine(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the cosine distance between two vectors.
     *        1 - (a * b) / (|a| * |b|), SIMD implementation.
     *        simd implementation is faster than simple_distance_cosine
     * @param a The first vector.
     * @param b The second vector.
     * @return The cosine distance between the two vectors.
     */
    float distance_cosine(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the cosine distance between two vectors. assume that
     *        vector `a` and `b` are all normalized to 1.0f, so we can simplify
     *        the formula to 1 - (a * b), a * b is the inner product of vector
     *        `a` and `b`, this is a simple version, for testing purposes.
     * @param a the first vector
     * @param b the second vector
     * @return the distance of cosine.
     */
    float simple_normalized_distance_cosine(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the cosine distance between two vectors. assume that
     *        vector `a` and `b` are all normalized to 1.0f, so we can simplify
     *        the formula to 1 - (a * b), a * b is the inner product of vector
     *        `a` and `b`, this is a simd version.
     * @param a the first vector
     * @param b the second vector
     * @return the distance of cosine.
     */
    float normalized_distance_cosine(turbo::Span<float> a, turbo::Span<float> b);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Jaccard distance
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the jaccard distance between two vectors.
     *        1 - SUM( std::min*a[i], b[i])) / SUM(std::max(a[i], b[i] )), SIMD implementation.
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The jaccard distance between the two vectors.
     */
    float simple_distance_min_max_jaccard(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the jaccard distance between two vectors.
     *        1 - SUM( std::min*a[i], b[i])) / SUM(std::max(a[i], b[i] )), SIMD implementation.
     *        simd implementation is faster than simple_distance_min_max_jaccard
     * @param a The first vector.
     * @param b The second vector.
     * @return The jaccard distance between the two vectors.
     */
    float distance_min_max_jaccard(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the jaccard distance between two vectors.
     *        1 - SUM( count_bits(a[i]&b[i])) / SUM(count_bits(a[i]|b[i])), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The jaccard distance between the two vectors.
     */
    float simple_distance_bits_jaccard(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the jaccard distance between two vectors.
     *        1 - SUM( count_bits(a[i]&b[i])) / SUM(count_bits(a[i]|b[i])), SIMD implementation.
     *        simd implementation is faster than simple_distance_bits_jaccard
     * @param a The first vector.
     * @param b The second vector.
     * @return The jaccard distance between the two vectors.
     */
    float distance_bits_jaccard(turbo::Span<float> a, turbo::Span<float> b);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Hamming distance
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the hamming distance between two vectors.
     *        SUM( count_bits(a[i]^b[i])), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The hamming distance between the two vectors.
     */
    float simple_distance_hamming(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the hamming distance between two vectors.
     *        SUM( count_bits(a[i]^b[i])), SIMD implementation.
     *        simd implementation is faster than simple_distance_hamming
     * @param a The first vector.
     * @param b The second vector.
     * @return The hamming distance between the two vectors.
     */
    float distance_hamming(turbo::Span<float> a, turbo::Span<float> b);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Distance Canberra
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Canberra distance between two vectors.
     *        SUM(|a[i] - b[i]| / (|a[i]| + |b[i]|)), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The Canberra distance between the two vectors.
     */
    float simple_distance_canberra(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Canberra distance between two vectors.
     *        SUM(|a[i] - b[i]| / (|a[i]| + |b[i]|)), SIMD implementation.
     *        simd implementation is faster than simple_distance_canberra
     * @param a The first vector.
     * @param b The second vector.
     * @return The Canberra distance between the two vectors.
     */
    float distance_canberra(turbo::Span<float> a, turbo::Span<float> b);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Distance LP
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the LP distance between two vectors.
     *        SUM(|a[i] - b[i]|^p)^(1/p), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @param p The p value.
     * @return The LP distance between the two vectors.
     */
    float simple_distance_lp(turbo::Span<float> a, turbo::Span<float> b, float p);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the LP distance between two vectors.
     *        SUM(|a[i] - b[i]|^p)^(1/p), SIMD implementation.
     *        simd implementation is faster than simple_distance_lp
     * @param a The first vector.
     * @param b The second vector.
     * @param p The p value.
     * @return The LP distance between the two vectors.
     */
    float distance_lp(turbo::Span<float> a, turbo::Span<float> b, float p);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Distance Bray Curtis
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Bray Curtis distance between two vectors.
     *        SUM(|a[i] - b[i]|) / SUM(|a[i] + b[i]|), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The Bray Curtis distance between the two vectors.
     */
    float simple_distance_bray_curtis(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Bray Curtis distance between two vectors.
     *        SUM(|a[i] - b[i]|) / SUM(|a[i] + b[i]|), SIMD implementation.
     *        simd implementation is faster than simple_distance_bray_curtis
     * @param a The first vector.
     * @param b The second vector.
     * @return The Bray Curtis distance between the two vectors.
     */
    float distance_bray_curtis(turbo::Span<float> a, turbo::Span<float> b);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Distance Jensen Shannon
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Jensen Shannon distance between two vectors.
     *        SUM(a[i] * log(2 * a[i] / (a[i] + b[i])) + b[i] * log(2 * b[i] / (a[i] + b[i]))), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The Jensen Shannon distance between the two vectors.
     */
    float simple_distance_jensen_shannon(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Jensen Shannon distance between two vectors.
     *        SUM(a[i] * log(2 * a[i] / (a[i] + b[i])) + b[i] * log(2 * b[i] / (a[i] + b[i]))), SIMD implementation.
     *        simd implementation is faster than simple_distance_jensen_shannon
     * @param a The first vector.
     * @param b The second vector.
     * @return The Jensen Shannon distance between the two vectors.
     */
    float distance_jensen_shannon(turbo::Span<float> a, turbo::Span<float> b);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Distance LInf
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the LInf distance between two vectors.
     *        MAX(|a[i] - b[i]|), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The LInf distance between the two vectors.
     */
    float simple_distance_linf(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the LInf distance between two vectors.
     *        MAX(|a[i] - b[i]|), SIMD implementation.
     *        simd implementation is faster than simple_distance_linf
     * @param a The first vector.
     * @param b The second vector.
     * @return The LInf distance between the two vectors.
     */
    float distance_linf(turbo::Span<float> a, turbo::Span<float> b);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Distance cross entropy
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the cross entropy distance between two vectors.
     *        SUM(a[i] * log(b[i])), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The cross entropy distance between the two vectors.
     */
    float simple_distance_cross_entropy(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the cross entropy distance between two vectors.
     *        SUM(a[i] * log(b[i])), SIMD implementation.
     *        simd implementation is faster than simple_distance_cross_entropy
     * @param a The first vector.
     * @param b The second vector.
     * @return The cross entropy distance between the two vectors.
     */
    float distance_cross_entropy(turbo::Span<float> a, turbo::Span<float> b);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Distance Kullback Leibler (KLD)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Kullback Leibler divergence between two vectors.
     *        SUM(a[i] * log(a[i] / b[i])), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The Kullback Leibler divergence between the two vectors.
     */
    float simple_distance_kld(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Kullback Leibler divergence between two vectors.
     *        SUM(a[i] * log(a[i] / b[i])), SIMD implementation.
     *        simd implementation is faster than simple_distance_kld
     * @param a The first vector.
     * @param b The second vector.
     * @return The Kullback Leibler divergence between the two vectors.
     */
    float distance_kld(turbo::Span<float> a, turbo::Span<float> b);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Distance Angle
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Angle distance between two vectors.
     *        arccos(sum(a[i] * b[i]) / (sqrt(sum(a[i]^2)) * sqrt(sum(b[i]^2)))), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The Angle distance between the two vectors.
     */
    float simple_distance_angle(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Angle distance between two vectors.
     *        arccos(sum(a[i] * b[i]) / (sqrt(sum(a[i]^2)) * sqrt(sum(b[i]^2)))), SIMD implementation.
     *        simd implementation is faster than simple_distance_angle
     * @param a The first vector.
     * @param b The second vector.
     * @return The Angle distance between the two vectors.
     */
    float distance_angle(turbo::Span<float> a, turbo::Span<float> b);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Distance Normalized Angle
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Normalized Angle distance between two vectors.
     *        1 - sum(a[i] * b[i]) / (sqrt(sum(a[i]^2)) * sqrt(sum(b[i]^2)))), simple and slow implementation
     *        for testing purposes.
     * @param a The first vector.
     * @param b The second vector.
     * @return The Normalized Angle distance between the two vectors.
     */
    float simple_distance_normalized_angle(turbo::Span<float> a, turbo::Span<float> b);

    /**
     * @ingroup zircon_utility_distance
     * @brief Compute the Normalized Angle distance between two vectors.
     *        1 - sum(a[i] * b[i]) / (sqrt(sum(a[i]^2)) * sqrt(sum(b[i]^2)))), SIMD implementation.
     *        simd implementation is faster than simple_distance_normalized_angle
     * @param a The first vector.
     * @param b The second vector.
     * @return The Normalized Angle distance between the two vectors.
     */
    float distance_normalized_angle(turbo::Span<float> a, turbo::Span<float> b);

}  // namespace zircon::distance

#endif // ZIRCON_UTILITY_PRIMITIVE_DISTANCE_H_
