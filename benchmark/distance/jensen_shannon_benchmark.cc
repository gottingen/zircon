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
#include "benchmark/benchmark.h"
#include "zircon/utility/primitive_distance.h"
#include "zircon/utility/primitive_distance.h"
#include "zircon/core/allocator.h"
#include "turbo/random/random.h"
#include <vector>

using vector_type = std::vector<float, turbo::aligned_allocator<float, 64>>;

static void BM_JENSEN_SHANNON(benchmark::State &state) {

    auto length = state.range(0);
    vector_type a(length, 1.0f);
    vector_type b(length, 2.0f);

    turbo::Span<float> a_span(a.data(), a.size());
    turbo::Span<float> b_span(b.data(), b.size());

    for (auto _ : state) {
        benchmark::DoNotOptimize(zircon::distance::simple_distance_jensen_shannon(a_span, b_span));
    }
}

static void BM_JENSEN_SHANNON_SIMD(benchmark::State &state) {

    auto length = state.range(0);
    vector_type a(length, 1.0f);
    vector_type b(length, 2.0f);

    turbo::Span<float> a_span(a.data(), a.size());
    turbo::Span<float> b_span(b.data(), b.size());

    for (auto _ : state) {
        benchmark::DoNotOptimize(zircon::distance::distance_jensen_shannon(a_span, b_span));
    }
}


BENCHMARK(BM_JENSEN_SHANNON)->RangeMultiplier(2)->Range(1 << 7, 1 << 11);
BENCHMARK(BM_JENSEN_SHANNON_SIMD)->RangeMultiplier(2)->Range(1 << 7, 1 << 11);