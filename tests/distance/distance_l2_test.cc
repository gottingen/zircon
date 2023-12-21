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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include "zircon/utility/primitive_distance.h"
#include "zircon/utility/distance.h"
#include "zircon/core/allocator.h"
#include "turbo/random/random.h"
#include <vector>

class DistanceL1Test {
public:
    DistanceL1Test() {
        a_vec.resize(256);
        b_vec.resize(256);
        for (size_t i = 0; i < a_vec.size(); ++i) {
            a_vec[i] = turbo::uniform(1.0f, 100.0f);
            b_vec[i] = turbo::uniform(1.0f, 100.0f);
        }
    }

    ~DistanceL1Test() = default;

    typedef std::vector<float, turbo::aligned_allocator<float, 64>> vector_type;

    vector_type a_vec;
    vector_type b_vec;

};


TEST_CASE_FIXTURE(DistanceL1Test, "distance l2 varify") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto distance = zircon::distance::distance_l2(a_span, b_span);
    auto simple_distance = zircon::distance::simple_distance_l2(a_span, b_span);
    CHECK_EQ(distance, doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance norm") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto norm_a = zircon::distance::norm_l2(a_span);
    auto simple_norm_a = zircon::distance::simple_norm_l2(a_span);
    CHECK_EQ(norm_a, doctest::Approx(simple_norm_a));
    auto norm_b = zircon::distance::norm_l2(b_span);
    auto simple_norm_b = zircon::distance::simple_norm_l2(b_span);
    CHECK_EQ(norm_b, doctest::Approx(simple_norm_b));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance normalization") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    vector_type a_out(a_vec.size());
    vector_type b_out(b_vec.size());
    vector_type a_save = a_vec;
    vector_type b_save = b_vec;
    auto a_out_span = turbo::Span<float>(a_out.data(), a_out.size());
    auto b_out_span = turbo::Span<float>(b_out.data(), b_out.size());
    zircon::distance::normalize_l2(a_span, a_out_span);
    zircon::distance::normalize_l2(a_span);
    CHECK_EQ(a_out_span, a_span);
    auto norm_b = zircon::distance::norm_l2(b_span);
    zircon::distance::normalize_l2(b_span,norm_b, b_out_span);
    zircon::distance::normalize_l2(b_span,norm_b);
    CHECK_EQ(b_out_span, b_span);
    auto norm_a = zircon::distance::norm_l2(a_span);
    CHECK_EQ(norm_a, doctest::Approx(1.0f));
    auto distance = zircon::distance::distance_l2(a_span, b_span);
    auto distance1 = zircon::distance::distance_normalized_l2(a_span, b_span);
    CHECK_EQ(distance, doctest::Approx(distance1));
}