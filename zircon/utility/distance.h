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
#include "zircon/utility/primitive_distance.h"

namespace zircon {

    /**
     * @ingroup zircon_utility_distance
     * @brief The VectorDistance struct defines the different distance metrics that can be used.
     *        Interfaces are divided into two categories:
     *        1. distance: compute the distance between two vectors.
     *        2. the vector before add to index conditionally to preprocess the vector. e.g.
     *           you want to using cosine distance, using normalized cosine distance is better.
     *           for it make less computation every query. so you need to normalize the vector.
     *           Every distance metric has a ``has_normalize`` function to check if it has a
     *           normalize function. and a ``need_normalize`` function to check if it need to
     *           normalize the vector before add to index.
     *           The normalize function has two interfaces, one is in-place, the other is out-place.
     *           The out-place interface is used when you want to keep the original vector.
     *           The in-place interface is used when you don't need to keep the original vector.
     *        The LP metric need a argument to define the p value. so it has a ``metric_arg`` member.
     *        Let us see an example:
     *        @code
     *        auto distancer = VectorDistance<MetricType::METRIC_NORMALIZED_COSINE>();
     *        // when add to index
     *        auto vec = turbo::Span<float>(...); // the vector you want to add to index
     *        auto out = turbo::Span<float>(...); // the vector you want to store
     *        if(distancer.need_normalize()) {
     *            distancer.normalize(vec, out);
     *         } else {
     *              out = vec;
     *         }
     *         // add out to index
     *         // when query
     *         auto query_vec = turbo::Span<float>(...); // the vector you want to query
     *         if(distancer.need_normalize()) {
     *              distancer.normalize(query_vec);
     *         }
     *         // query to index
     *         @endcode
     *         It is not the exact api exposed to user, thus to the developers. the above workflow
     *         shold be implemented in the index class. the user only need to call ``add`` and
     *         ``query`` function. but need to know the stage that what we have done to the vector.
     *         friendly for user, may be we need to provide a option to known if the vector has been
     *         normalized or not. so we can avoid the unnecessary computation.
     *         eg.
     *         @code
     *         // assume that, we provide a option to known if the vector has been normalized or not.
     *         // when add to index
     *         if(!distancer.has_normalize() && distancer.need_normalize()) {
     *              distancer.normalize(vec, out);
*               } else {
     *               out = vec;
     *          }
     *          // add out to index
     *          // when query
     *          if(!distancer.has_normalize() && distancer.need_normalize()) {
     *               distancer.normalize(query_vec);
     *          }
     *          // query to index
     *          @endcode
     *          The above code is more friendly for user. but it is not the exact api exposed to user.
     *          the user only need to call ``add`` and ``query`` function. but need to know the stage,
     *          when user neel original vector, they should known that, the vector has been normalized.
     *          Another important thing is that, the vector should be store in a aligned memory. the
     *          alignedment defines in ``turbo::simd::batch``. the default alignedment is 64 bytes, which
     *          has defined in ``core/allocator.h``. it is easy to make the storage. using a vector with
     *          ``turbo::aligned_allocator``. eg.
     *          @code
     *          std::vector<float, turbo::aligned_allocator<float, 64>> vec;
     *          @endcode
     *          The better way is to using the ``Alloctor::alignment`` to make the code more portable.
     *          eg.
     *          @code
     *          std::vector<float, turbo::aligned_allocator<float, Allocator::alignment>> vec;
     *          @endcode
     * @note if test fail ``has_normalize``, please do not call ``normalize`` function. it will
     *           cause undefined behavior. usually, it raise a core dump.
     * @tparam MetricType the distance metric type.
     */
    template<MetricType>
    struct VectorDistance {

        /**
         * @brief the argument of the metric. only used in LP metric.
         */
        float metric_arg{0.0f};

        /**
         * @brief compute the distance between two vectors.
         *         this is the production environment interface.
         * @param a the first vector.
         * @param b the second vector.
         * @return the distance between two vectors.
         */
        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const;

        /**
         * @brief compute the distance between two vectors.
         *         this is the test environment interface.
         *         it works slower than ``distance`` function.
         * @param a the first vector.
         * @param b the second vector.
         * @return the distance between two vectors.
         */
        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const;

        /**
         * @brief compute the norm of the vector.
         *        be careful, calling this function will cause undefined behavior if
         *        the metric is not L1, L2, normalized L2, cosine, normalized cosine.
         *        It is a better way to check if ``has_normalize`` before calling this function.
         * @param a the vector.
         * @return the norm of the vector.
         */
        [[nodiscard]] float norm(turbo::Span<float> a) const;

        /**
         * @brief normalize the vector in-place.
         *        be careful, calling this function will cause undefined behavior if
         *        the metric is not L1, L2, normalized L2, cosine, normalized cosine.
         *        It is a better way to check if ``has_normalize`` before calling this function.
         * @param a the vector.
         */
        void normalize(turbo::Span<float> a) const;

        /**
         * @brief normalize the vector out-place.
         *        be careful, calling this function will cause undefined behavior if
         *        the metric is not L1, L2, normalized L2, cosine, normalized cosine.
         *        It is a better way to check if ``has_normalize`` before calling this function.
         * @param a the vector.
         * @param out the output vector.
         */
        void normalize(turbo::Span<float> a, turbo::Span<float> out) const;

        /**
         * @brief check if the vector need to be normalized before add to index.
         * @return true if need to be normalized, otherwise false.
         */
        [[nodiscard]] bool need_normalize() const;

        /**
         * @brief check if the vector has a normalize function.
         * @return true if has normalize function, otherwise false.
         */
        [[nodiscard]] bool has_normalize() const;
    };

    /**
     * @brief  Specialization of VectorDistance for L1 metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_L1> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_l1(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_l1(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            return distance::norm_l1(a);
        }

        void normalize(turbo::Span<float> a) const {
            distance::normalize_l1(a);
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            distance::normalize_l1(a, out);
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for L2 metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_L2> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_l2(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_l2(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            return distance::norm_l2(a);
        }

        void normalize(turbo::Span<float> a) const {
            distance::normalize_l2(a);
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            distance::normalize_l2(a, out);
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for normalized L2 metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_NORMALIZED_L2> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_normalized_l2(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_normalized_l2(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            return distance::norm_l2(a);
        }

        void normalize(turbo::Span<float> a) const {
            distance::normalize_l2(a);
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            distance::normalize_l2(a, out);
        }

        [[nodiscard]] bool need_normalize() const {
            return true;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for inner product metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_IP> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_ip(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_ip(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            TURBO_ASSERT(false, "not implemented");
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return false;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for cosine metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_COSINE> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_cosine(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_cosine(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            return distance::norm_l2(a);
        }

        void normalize(turbo::Span<float> a) const {
            distance::normalize_l2(a);
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            distance::normalize_l2(a, out);
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return false;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for normalized cosine metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_NORMALIZED_COSINE> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_normalized_cosine(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_normalized_cosine(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            return distance::norm_l2(a);
        }

        void normalize(turbo::Span<float> a) const {
            distance::normalize_l2(a);
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            distance::normalize_l2(a, out);
        }

        [[nodiscard]] bool need_normalize() const {
            return true;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for min max jaccard metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_MIN_MAX_JACCARD> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_min_max_jaccard(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_min_max_jaccard(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            TURBO_ASSERT(false, "not implemented");
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }
        [[nodiscard]] bool has_normalize() const {
            return false;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for bits jaccard metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_BITS_JACCARD> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_bits_jaccard(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_bits_jaccard(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            TURBO_ASSERT(false, "not implemented");
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for hamming metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_HAMMING> {
        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_hamming(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_hamming(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            TURBO_ASSERT(false, "not implemented");
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for Canberra metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_CANBERRA> {
        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_canberra(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_canberra(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            TURBO_ASSERT(false, "not implemented");
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for LP metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_LP> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_lp(a, b, metric_arg);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_lp(a, b, metric_arg);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            TURBO_ASSERT(false, "not implemented");
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for Bray Curtis metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_BRAY_CURTIS> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_bray_curtis(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_bray_curtis(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            TURBO_ASSERT(false, "not implemented");
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }
        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for Jensen Shannon metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_JENSEN_SHANNON> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_jensen_shannon(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_jensen_shannon(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            TURBO_ASSERT(false, "not implemented");
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for Linfinity metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_LINF> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_linf(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_linf(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            TURBO_ASSERT(false, "not implemented");
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for cross entropy metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_CROSS_ENTROPY> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_cross_entropy(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_cross_entropy(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            TURBO_ASSERT(false, "not implemented");
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }

    };

    /**
     * @brief  Specialization of VectorDistance for Kullback Leibler divergence metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_KLD> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_kld(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_kld(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a) const {
            TURBO_ASSERT(false, "not implemented");
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            TURBO_ASSERT(false, "not implemented");
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }

    };

    /**
     * @brief  Specialization of VectorDistance for angle metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_ANGLE> {

        float metric_arg{0.0f};

        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_angle(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_angle(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            return distance::norm_l2(a);
        }

        void normalize(turbo::Span<float> a) const {
            distance::normalize_l2(a);
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
            distance::normalize_l2(a, out);
        }

        [[nodiscard]] bool need_normalize() const {
            return false;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }
    };

    /**
     * @brief  Specialization of VectorDistance for normalized angle metric.
     */
    template<>
    struct VectorDistance<MetricType::METRIC_NORMALIZED_ANGLE> {

        float metric_arg{0.0f};


        [[nodiscard]] float distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::distance_normalized_angle(a, b);
        }

        [[nodiscard]] float simple_distance(turbo::Span<float> a, turbo::Span<float> b) const {
            return distance::simple_distance_normalized_angle(a, b);
        }

        [[nodiscard]] float norm(turbo::Span<float> a) const {
            return distance::norm_l2(a);
        }

        void normalize(turbo::Span<float> a) const {
            distance::normalize_l2(a);
        }

        void normalize(turbo::Span<float> a, turbo::Span<float> out) const {
             distance::normalize_l2(a, out);
        }

        [[nodiscard]] bool need_normalize() const {
            return true;
        }

        [[nodiscard]] bool has_normalize() const {
            return true;
        }

    };


}  // namespace zircon

#endif  // ZIRCON_UTILITY_DISTANCE_H_
