/* Copyright 2023 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/python/ifrt/memory.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/device.h"

namespace xla {
namespace ifrt {

namespace {

// Global state that keeps a stable copy of memory kind strings for `MemoryKind`
// instances.
struct MemoryKindsSet {
  absl::Mutex mu;
  absl::node_hash_set<std::string> memory_kinds_set ABSL_GUARDED_BY(mu);
};

absl::string_view InternMemoryKind(absl::string_view memory_kind) {
  if (memory_kind.data() == MemoryKind::kDefault.data() &&
      memory_kind.size() == MemoryKind::kDefault.size()) {
    memory_kind = MemoryKind::kDefault;
  }
  static auto* const global_set = []() {
    auto* const set = new MemoryKindsSet();
    absl::MutexLock lock(set->mu);
    set->memory_kinds_set.insert(std::string(MemoryKind::kDefault));
    return set;
  }();
  absl::MutexLock lock(global_set->mu);
  auto it = global_set->memory_kinds_set.find(memory_kind);
  if (it == global_set->memory_kinds_set.end()) {
    return *global_set->memory_kinds_set.insert(std::string(memory_kind)).first;
  }
  return *it;
}

}  // namespace

MemoryKind::MemoryKind() : memory_kind_(InternMemoryKind(kDefault)) {}

MemoryKind::MemoryKind(std::optional<absl::string_view> memory_kind)
    : memory_kind_(InternMemoryKind(memory_kind.value_or(kDefault))) {}

std::string MemoryKind::ToString() const { return std::string(memory_kind_); }

MemoryKind CanonicalizeMemoryKind(MemoryKind memory_kind,
                                  const Device* device) {
  return memory_kind;
}

char Memory::ID = 0;

}  // namespace ifrt
}  // namespace xla
