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


#ifndef ZIRCON_CORE_METRIC_TYPE_H_
#define ZIRCON_CORE_METRIC_TYPE_H_

namespace zircon {

    /**
     * @ingroup zircon_core
     * @brief The MetricType enum defines the different distance metrics that can be used.
     */
    enum class MetricType {
        UNDEFINED = 0,
        /**
         * @brief The L1 norm distance is also known as the Manhattan distance.
         *        wikipedia: https://en.wikipedia.org/wiki/Taxicab_geometry
         *        It is defined as:
         *        d(x,y) = sum(|x_i - y_i|)
         */
        METRIC_L1 = 1,
        /**
         * @brief The L2 norm distance is also known as the Euclidean distance.
         *        wikipedia: https://en.wikipedia.org/wiki/Euclidean_distance
         *        It is defined as:
         *        d(x,y) = sqrt(sum((x_i - y_i)^2))
         */
        METRIC_L2 = 2,
        /**
         * @brief The normalized L2 norm distance that the same as above
         *        but requires that the vectors are normalized. so it can
         *        be computed as:
         *        d(x,y) = 2 - 2 * sum(x_i * y_i)
         */
        METRIC_NORMALIZED_L2 = 2,

        /**
         * @brief The inner product distance is also known as the dot product.
         *         wikipedia: https://en.wikipedia.org/wiki/Dot_product
         *         It is defined as:
         *         d(x,y) = sum(x_i * y_i)
         */
        METRIC_IP = 4,

        /**
         * @brief The Cosine distance is also known as the cosine similarity.
         *        wikipedia: https://en.wikipedia.org/wiki/Cosine_similarity
         *        It is defined as:
         *        d(x,y) = 1 - sum(x_i * y_i) / (sqrt(sum(x_i^2)) * sqrt(sum(y_i^2)))
         */
        METRIC_COSINE = 5,

        /**
         * @brief The normalized cosine distance that the same as above
         *        but requires that the vectors are normalized. so it can
         *        be computed as:
         *        d(x,y) = 1 - sum(x_i * y_i) = 1 - distance_ip(x,y)
         */
        METRIC_NORMALIZED_COSINE,

        /**
         * @brief The Jaccard distance is also known as the Jaccard index.
         *        wikipedia: https://en.wikipedia.org/wiki/Jaccard_index
         *        It is defined as:
         *        d(x,y) = 1 - sum(min(x_i, y_i)) / sum(max(x_i, y_i))
         */
        METRIC_MIN_MAX_JACCARD = 6,

        /**
         * @brief The Jaccard distance binary version.
         *         it is defined as:
         *         d(x,y) = 1 - sum(x_i and y_i) / sum(x_i or y_i)
         */
        METRIC_BINARY_JACCARD = 7,

        /**
         * @brief The Hamming distance is a metric for comparing two binary data strings.
         *        wikipedia: https://en.wikipedia.org/wiki/Hamming_distance
         *        It is defined as:
         *        d(x,y) = sum(x_i != y_i) means the number of different bits
         */
        METRIC_HAMMING = 8,
        /**
         * @brief The canberra distance is a weighted version of the Manhattan distance.
         *        wikipedia: https://en.wikipedia.org/wiki/Canberra_distance
         *        It is also known as the L1 norm distance. It is defined as:
         *        d(x,y) = sum(|x_i - y_i| / (|x_i| + |y_i|))
         */
        METRIC_CANBERRA = 9,

        /**
         * @brief The LP distance is a generalization of the L1 and L2 norm distances.
         *       wikipedia: https://en.wikipedia.org/wiki/Lp_space
         *       It is defined as:
         *       d(x,y) = sum(|x_i - y_i|^p)^(1/p)
         */
        METRIC_LP = 10,

        /**
         * @brief The Bray Curtis distance is a metric for comparing two vectors.
         *        wikipedia: https://en.wikipedia.org/wiki/Bray%E2%80%93Curtis_dissimilarity
         *        It is defined as:
         *        d(x,y) = sum(|x_i - y_i|) / sum(|x_i + y_i|)
         */
        METRIC_BRAY_CURTIS = 11,

        /**
         * @brief The Jensen Shannon distance is a metric for comparing two probability distributions.
         *        wikipedia: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
         *        It is defined as:
         *        d(x,y) = sum(x_i * log(2 * x_i / (x_i + y_i)) + y_i * log(2 * y_i / (x_i + y_i)))
         */
        METRIC_JENSEN_SHANNON = 12,

        /**
         * @brief The Linfinity distance is a metric for comparing two vectors.
         *       wikipedia: https://en.wikipedia.org/wiki/Chebyshev_distance
         *       It is defined as:
         *       d(x,y) = max(|x_i - y_i|)
         */

        METRIC_LINF = 13,

        /**
         * @brief The cross entropy distance is a metric for comparing two vectors.
         *     wikipedia: https://en.wikipedia.org/wiki/Cross_entropy
         *     It is defined as:
         *     d(x,y) = sum(x_i * log(y_i))
         */
        METRIC_CROSS_ENTROPY= 14,

        /**
         * @brief The Kullback Leibler divergence is a metric for comparing two probability distributions.
         *        wikipedia: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
         *        It is defined as:
         *        d(x,y) = sum(x_i * log(x_i / y_i))
         */
        METRIC_KLD = 15,

        /**
         * @brief The Angle distance is a metric for comparing two vectors.
         *       wikipedia: https://en.wikipedia.org/wiki/Angle
         *       It is defined as:
         *       d(x,y) = arccos(sum(x_i * y_i) / (sqrt(sum(x_i^2)) * sqrt(sum(y_i^2))))
         */
        METRIC_ANGLE = 16,

        /**
         * @brief The Normalized Angle distance is a metric for comparing two vectors.
         *       wikipedia: https://en.wikipedia.org/wiki/Angle
         *       It is defined as:
         *       d(x,y) = 1 - sum(x_i * y_i) / (sqrt(sum(x_i^2)) * sqrt(sum(y_i^2))))
         */
        METRIC_NORMALIZED_ANGLE = 17,

        /**
         * @note not implemented yet
         * @brief The Poincare distance is a metric for comparing two vectors.
         *       wikipedia: https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model
         *       It is defined as:
         *       d(x,y) = arccosh(1 + 2 * sum((x_i - y_i)^2) / ((1 - sum(x_i^2)) * (1 - sum(y_i^2))))
         */
        METRIC_POINCARE = 18,

        /**
         * @note not implemented yet
         * @brief The Lorentz distance is a metric for comparing two vectors.
         *       wikipedia: https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model
         *       It is defined as:
         *       d(x,y) = arccosh(1 + 2 * sum((x_i - y_i)^2) / ((1 + sum(x_i^2)) * (1 + sum(y_i^2))))
         */
        METRIC_LORENTZ = 19,
    };
}  // namespace zircon
#endif  // ZIRCON_CORE_METRIC_TYPE_H_
