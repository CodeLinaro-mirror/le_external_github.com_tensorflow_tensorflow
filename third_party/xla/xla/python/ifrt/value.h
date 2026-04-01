/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_VALUE_H_
#define XLA_PYTHON_IFRT_VALUE_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

class Client;

// Semantics for computing the byte size of a value.
enum class ValueByteSizeSemantics : int {
  // On-device byte size on a single shard (device).
  kPerShard = 0,
  // On-device byte size on a single process. If the runtime does not recognize
  // processes, falls back to per-shard.
  kPerProcess,
  // Total on-device byte sizes across all shards.
  kAllShards,
};

// Abstract superclass of values such as arrays.
class Value : public tsl::ReferenceCounted<Value>,
              public llvm::RTTIExtends<Value, llvm::RTTIRoot> {
 public:
  Value() = default;

  // Not copyable or movable.
  Value(const Value&) = delete;
  Value(Value&&) = delete;
  Value& operator=(const Value&) = delete;
  Value& operator=(Value&&) = delete;

  virtual Client* client() const = 0;

  // Returns the user context associated with the creation of this array.
  virtual UserContextRef user_context() const = 0;

  // Returns a byte size of a value. It takes into account the sharding and
  // layout of the value. This is often called the "on-device" size. Since this
  // size is typically computed from the metadata of the value, it is
  // essentially an estimation, but is expected to be close enough to the
  // actual size for many practical purposes.
  //
  // Returns `std::nullopt` if there is no well-defined byte size, e.g., using a
  // non-array `DType` that has no byte size, or requesting a per-process/-shard
  // for an array that has multiple byte sizes across shards.
  //
  // Returns an error if essential information for the size calculation could
  // not be obtained, e.g., failing to determine the concrete layout for a value
  // using a default layout.
  virtual absl::StatusOr<std::optional<int64_t>> GetByteSize(
      ValueByteSizeSemantics semantics) const = 0;

  // Returns a future that becomes ready when the buffer is computed or has an
  // error.
  virtual tsl::Future<> GetReadyFuture() const = 0;

  // Deletes the value from the devices. The operation may be asynchronous. The
  // returned future will have the result of the deletion on the devices, and
  // will be triggered after all values have been deleted.
  // Implementations that do not track the completion of the deletion operation
  // may make the future immediately ready with an OK status.
  //
  // Deletion is idempotent. Deleting an already deleted value is allowed, and
  // all the futures returned by different calls to Delete() will become ready
  // with the same status.
  virtual tsl::Future<> Delete() = 0;

  // Returns whether the value has been enqueued for deletion from the devices.
  virtual bool IsDeleted() const = 0;

  virtual std::string DebugString() const = 0;

  static char ID;  // NOLINT
};

using ValueRef = tsl::RCReference<Value>;

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_VALUE_H_
