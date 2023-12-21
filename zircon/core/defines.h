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


#ifndef ZIRCON_CORE_DEFINES_H_
#define ZIRCON_CORE_DEFINES_H_

#include <cstdint>
#include <limits>
#include <cstddef>

namespace zircon {

    typedef uint32_t location_t;
    typedef size_t label_type;
    typedef double distance_type;

    namespace constants {
        static constexpr location_t INVALID_LOCATION = std::numeric_limits<location_t>::max();

        /// for common
        static constexpr size_t kUnknownSize = std::numeric_limits<size_t>::max();
        static constexpr size_t kMaxElements = 100000;
        static constexpr size_t kBatchSize = 256;
        static constexpr size_t kLockSlots = 65536;

        static constexpr location_t kUnknownLocation = std::numeric_limits<location_t>::max();
        static constexpr label_type kUnknownLabel = std::numeric_limits<label_type>::max();
        /// for hnsw
        static constexpr size_t kHnswM = 16;
        static constexpr size_t kHnswEf = 50;
        static constexpr size_t kHnswEfConstruction = 200;
        static constexpr size_t kHnswRandomSeed = 100;
    }

    struct VectorStoreOption {
        uint32_t batch_size{constants::kBatchSize};
        uint32_t max_elements{constants::kMaxElements};
        uint32_t vector_byte_size{0};
        bool     enable_replace_vacant{true};
    };

    struct SerializeOption {
        //DataType data_type;
        std::size_t n_vectors{constants::kUnknownSize};
        std::size_t dimension;
    };

}  // namespace zircon

#endif  // ZIRCON_CORE_DEFINES_H_
