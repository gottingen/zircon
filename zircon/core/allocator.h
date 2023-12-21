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

#ifndef ZIRCON_CORE_ALLOCATOR_H_
#define ZIRCON_CORE_ALLOCATOR_H_

#include "turbo/memory/aligned_allocator.h"
#include "turbo/simd/simd.h"

namespace zircon {

    class Allocator {
    public:
        ~Allocator() = default;

        static constexpr size_t alignment = 64;
        static_assert(alignment % turbo::simd::default_arch::alignment() == 0, "must be align to simd");

        static Allocator &get_instance() {
            static Allocator instance;
            return instance;
        }

        uint8_t *allocate(size_t n) {
            return _allocator.allocate(n);
        }

        void deallocate(uint8_t *p, size_t n) {
            _allocator.deallocate(p, n);
        }

    private:
        turbo::aligned_allocator<uint8_t, Allocator::alignment> _allocator;
        Allocator() = default;
    };
}
#endif  // ZIRCON_CORE_ALLOCATOR_H_
