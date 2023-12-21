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


#ifndef ZIRCON_UTILITY_ID_FILTER_H_
#define ZIRCON_UTILITY_ID_FILTER_H_

#include "zircon/core/defines.h"
#include "turbo/container/flat_hash_set.h"
#include "bluebird/bits/bitmap.h"


namespace zircon {

    struct IdFilter {
        virtual ~IdFilter() = default;

        virtual bool is_member(label_type id) const = 0;
    };

    struct IdFilterRange {
        label_type min_id;
        label_type max_id;

        IdFilterRange(label_type min_id, label_type max_id) : min_id(min_id), max_id(max_id) {}

        bool is_member(label_type id) const noexcept{
            return id >= min_id && id <= max_id;
        }
    };

    struct IdFilterSet : public IdFilter {
        turbo::flat_hash_set<label_type> id_set;

        IdFilterSet() = default;

        IdFilterSet(const turbo::flat_hash_set<label_type> &id_set) : id_set(id_set) {}

        IdFilterSet(const std::initializer_list<label_type> &id_list) : id_set(id_list) {}

        IdFilterSet(const std::vector<label_type> &id_list) : id_set(id_list.begin(), id_list.end()) {}

        IdFilterSet(const std::set<label_type> &id_list) : id_set(id_list.begin(), id_list.end()) {}

        IdFilterSet(const std::unordered_set<label_type> &id_list) : id_set(id_list.begin(), id_list.end()) {}

        IdFilterSet(const label_type* id_list, size_t size) : id_set(id_list, id_list + size) {}

        IdFilterSet(const IdFilterSet &other) : id_set(other.id_set) {}

        IdFilterSet(IdFilterSet &&other) noexcept : id_set(std::move(other.id_set)) {}

        IdFilterSet &operator=(const IdFilterSet &other) {
            id_set = other.id_set;
            return *this;
        }

        IdFilterSet &operator=(IdFilterSet &&other) noexcept {
            id_set = std::move(other.id_set);
            return *this;
        }


        bool is_member(label_type id) const noexcept override {
            return id_set.contains(id);
        }
    };

    struct IdFilterBitmap : public IdFilter {
        bluebird::Bitmap bitmap;

        template<typename It>
        IdFilterBitmap(It begin, It end) {
            for(auto it = begin; it != end; ++it) {
                bitmap.add(*it);
            }
        }

        IdFilterBitmap(const std::vector<label_type> &id_list) {
            for(auto id : id_list) {
                bitmap.add(id);
            }
        }

        IdFilterBitmap(const std::initializer_list<label_type> &list) {
            for(auto id : list) {
                bitmap.add(id);
            }
        }

        IdFilterBitmap(const std::set<label_type> &list) {
            for(auto id : list) {
                bitmap.add(id);
            }
        }

        bool is_member(label_type lb) const noexcept override {
            return bitmap.contains(lb);
        }
    };

    struct IdFilterAnd : public IdFilter {
        IdFilter* _a;
        IdFilter* _b;
        IdFilterAnd(IdFilter *a, IdFilter*b) :_a(a), _b(b){}
        bool is_member(label_type lb) const noexcept override{
            return _a->is_member(lb) && _b->is_member(lb);
        }
    };

    struct IdFilterOr : public IdFilter {
        IdFilter* _a;
        IdFilter* _b;
        IdFilterOr(IdFilter *a, IdFilter*b) :_a(a), _b(b){}
        bool is_member(label_type lb) const noexcept override{
            return _a->is_member(lb) || _b->is_member(lb);
        }
    };

    struct IdFilterXor : public IdFilter {
        IdFilter* _a;
        IdFilter* _b;
        IdFilterXor(IdFilter *a, IdFilter*b) :_a(a), _b(b){}
        bool is_member(label_type lb) const noexcept override{
            return _a->is_member(lb) ^ _b->is_member(lb);
        }
    };


}  // namespace zircon

#endif  // ZIRCON_UTILITY_ID_FILTER_H_
