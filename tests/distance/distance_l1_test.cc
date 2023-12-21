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

    std::vector<float, turbo::aligned_allocator<float, 64>> a_vec;
    std::vector<float, turbo::aligned_allocator<float, 64>> b_vec;

};

TEST_CASE_FIXTURE(DistanceL1Test, "distance l1") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto distance = zircon::distance::distance_l1(a_span, b_span);
    auto simple_distance = zircon::distance::simple_distance_l1(a_span, b_span);
    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance l2") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto distance = zircon::distance::distance_l2(a_span, b_span);
    auto simple_distance = zircon::distance::simple_distance_l2(a_span, b_span);
    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance ip") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto distance = zircon::distance::distance_ip(a_span, b_span);
    auto simple_distance = zircon::distance::simple_distance_ip(a_span, b_span);
    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance cosine") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto distance = zircon::distance::distance_cosine(a_span, b_span);
    auto simple_distance = zircon::distance::simple_distance_cosine(a_span, b_span);
    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance jaccard") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto distance = zircon::distance::distance_min_max_jaccard(a_span, b_span);
    auto simple_distance = zircon::distance::simple_distance_min_max_jaccard(a_span, b_span);
    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance bits jaccard") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto simple_distance = zircon::distance::simple_distance_bits_jaccard(a_span, b_span);
    auto distance = zircon::distance::distance_bits_jaccard(a_span, b_span);

    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance hamming") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto simple_distance = zircon::distance::simple_distance_hamming(a_span, b_span);
    auto distance = zircon::distance::distance_hamming(a_span, b_span);

    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance Canberra") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto simple_distance = zircon::distance::simple_distance_canberra(a_span, b_span);
    auto distance = zircon::distance::distance_canberra(a_span, b_span);

    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance LP") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto simple_distance = zircon::distance::simple_distance_lp(a_span, b_span, 3.0f);
    auto distance = zircon::distance::distance_lp(a_span, b_span, 3.0);

    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance Bray Curtis") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto simple_distance = zircon::distance::simple_distance_bray_curtis(a_span, b_span);
    auto distance = zircon::distance::distance_bray_curtis(a_span, b_span);

    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance Jensen Shannon") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto simple_distance = zircon::distance::simple_distance_jensen_shannon(a_span, b_span);
    auto distance = zircon::distance::distance_jensen_shannon(a_span, b_span);

    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance linf") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto simple_distance = zircon::distance::simple_distance_linf(a_span, b_span);
    auto distance = zircon::distance::distance_linf(a_span, b_span);

    CHECK(distance == doctest::Approx(simple_distance));

}


class DistanceEntropyTest {
public:
    DistanceEntropyTest() {
        a_vec.resize(256);
        b_vec.resize(256);
        for (size_t i = 0; i < a_vec.size(); ++i) {
            a_vec[i] = turbo::uniform(0.01f, 0.9f);
            b_vec[i] = 1- a_vec[i];
        }
    }
    ~DistanceEntropyTest() = default;

    std::vector<float, turbo::aligned_allocator<float, 64>> a_vec;
    std::vector<float, turbo::aligned_allocator<float, 64>> b_vec;

};

TEST_CASE_FIXTURE(DistanceEntropyTest, "distance cross entropy") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto simple_distance = zircon::distance::simple_distance_cross_entropy(a_span, b_span);
    auto distance = zircon::distance::distance_cross_entropy(a_span, b_span);

    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance kld") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto simple_distance = zircon::distance::simple_distance_kld(a_span, b_span);
    auto distance = zircon::distance::distance_kld(a_span, b_span);

    CHECK(distance == doctest::Approx(simple_distance));

}

TEST_CASE_FIXTURE(DistanceL1Test, "distance angle") {

    auto a_span = turbo::Span<float>(a_vec.data(), a_vec.size());
    auto b_span = turbo::Span<float>(b_vec.data(), b_vec.size());
    auto simple_distance = zircon::distance::simple_distance_angle(a_span, b_span);
    auto distance = zircon::distance::distance_angle(a_span, b_span);

    CHECK(distance == doctest::Approx(simple_distance));

}