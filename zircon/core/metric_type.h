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

    enum class MetricType {
        UNDEFINED = 0,
        METRIC_L1,
        METRIC_L2,
        METRIC_IP,
        METRIC_HAMMING,
        METRIC_JACCARD,
        METRIC_COSINE,
        METRIC_ANGLE,
        METRIC_NORMALIZED_COSINE,
        METRIC_NORMALIZED_ANGLE,
        METRIC_NORMALIZED_L2,
        METRIC_POINCARE,
        METRIC_LORENTZ,
    };
}  // namespace zircon
#endif  // ZIRCON_CORE_METRIC_TYPE_H_
