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

#ifndef XLA_PYTHON_IFRT_MEMORY_H_
#define XLA_PYTHON_IFRT_MEMORY_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {
namespace ifrt {

// `MemoryKind` uniquely identifies a group of memory spaces with a
// platform-dependent string. When no specific memory kind is chosen,
// the universal default "device" memory kind is used.
class MemoryKind {
 public:
  static constexpr absl::string_view kDefault = "device";

  // Dual-compatibility adapter acting as both
  // `absl::string_view` and `std::optional<absl::string_view>`.
  //
  // TODO(b/360374577): Remove this adapter and return `absl::string_view`
  // directly once all callsites are migrated to use `absl::string_view`.
  class MemoryKindView {
   public:
    using value_type = absl::string_view;

    constexpr MemoryKindView() : sv_(kDefault) {}
    // NOLINTNEXTLINE(*-explicit-constructor)
    constexpr MemoryKindView(absl::string_view sv) : sv_(sv) {}
    // NOLINTNEXTLINE(*-explicit-constructor)
    MemoryKindView(std::nullopt_t) : sv_(kDefault) {}

    // `absl::string_view` behavior.
    // NOLINTNEXTLINE(*-explicit-constructor)
    constexpr operator absl::string_view() const { return sv_; }
    constexpr absl::string_view string_view() const { return sv_; }
    constexpr const char* data() const { return sv_.data(); }
    constexpr size_t size() const { return sv_.size(); }
    constexpr size_t length() const { return sv_.length(); }
    constexpr bool empty() const { return sv_.empty(); }
    constexpr const char& operator[](size_t n) const { return sv_[n]; }
    constexpr const char& front() const { return sv_.front(); }
    constexpr const char& back() const { return sv_.back(); }
    constexpr const char* begin() const { return sv_.begin(); }
    constexpr const char* end() const { return sv_.end(); }
    constexpr const char* cbegin() const { return sv_.cbegin(); }
    constexpr const char* cend() const { return sv_.cend(); }
    constexpr absl::string_view substr(
        size_t pos = 0, size_t n = absl::string_view::npos) const {
      return sv_.substr(pos, n);
    }

    // Legacy `std::optional<absl::string_view>` compatibility behavior.
    constexpr bool has_value() const { return true; }
    constexpr explicit operator bool() const { return true; }
    constexpr absl::string_view operator*() const { return sv_; }
    constexpr const absl::string_view* operator->() const { return &sv_; }
    constexpr absl::string_view value() const { return sv_; }
    constexpr absl::string_view value_or(absl::string_view) const {
      return sv_;
    }
    // NOLINTNEXTLINE(*-explicit-constructor)
    operator std::optional<absl::string_view>() const { return sv_; }

    // Stringification, hashing, streams.
    template <typename H>
    friend H AbslHashValue(H h, const MemoryKindView& v) {
      return H::combine(std::move(h), v.sv_);
    }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const MemoryKindView& v) {
      sink.Append(v.sv_);
    }

    template <typename OStream>
    friend OStream& operator<<(OStream& os, const MemoryKindView& v) {
      return os << v.sv_;
    }

    // Comparisons.
    friend bool operator==(const MemoryKindView& lhs,
                           const MemoryKindView& rhs) {
      return lhs.sv_ == rhs.sv_;
    }
    friend bool operator==(const MemoryKindView& lhs, absl::string_view rhs) {
      return lhs.sv_ == rhs;
    }
    friend bool operator==(absl::string_view lhs, const MemoryKindView& rhs) {
      return lhs == rhs.sv_;
    }
    friend bool operator==(const MemoryKindView&, std::nullopt_t) {
      return false;
    }
    friend bool operator==(std::nullopt_t, const MemoryKindView&) {
      return false;
    }

    friend bool operator!=(const MemoryKindView& lhs,
                           const MemoryKindView& rhs) {
      return !(lhs == rhs);
    }
    friend bool operator!=(const MemoryKindView& lhs, absl::string_view rhs) {
      return !(lhs == rhs);
    }
    friend bool operator!=(absl::string_view lhs, const MemoryKindView& rhs) {
      return !(lhs == rhs);
    }
    friend bool operator!=(const MemoryKindView&, std::nullopt_t) {
      return true;
    }
    friend bool operator!=(std::nullopt_t, const MemoryKindView&) {
      return true;
    }

   private:
    absl::string_view sv_;
  };

  // Creates `MemoryKind` with default "device" memory kind.
  MemoryKind();

  // Creates `MemoryKind` from a platform-dependent identifier of a memory kind.
  // `MemoryKind` will be stable even after the string referenced by
  // `memory_kind` is deallocated.
  explicit MemoryKind(std::optional<absl::string_view> memory_kind);

  bool operator==(const MemoryKind& other) const {
    // Use a pointer comparison. `memory_kind_` always points to an interned
    // string.
    return memory_kind_.data() == other.memory_kind_.data() &&
           memory_kind_.size() == other.memory_kind_.size();
  }
  bool operator!=(const MemoryKind& other) const { return !(*this == other); }

  template <typename H>
  friend H AbslHashValue(H h, const MemoryKind& memory_kind) {
    return H::combine(std::move(h), memory_kind.memory_kind_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const MemoryKind& memory_kind) {
    sink.Append(memory_kind.memory_kind_);
  }

  bool is_default() const { return memory_kind_ == kDefault; }

  MemoryKindView memory_kind() const { return MemoryKindView(memory_kind_); }

 private:
  std::string ToString() const;

  absl::string_view memory_kind_;
};

// Canonicalizes `MemoryKind`. If `MemoryKind` has no memory kind chosen,
// returns a default `MemoryKind` chosen for the device. If there is no default
// indicated by the device, simply returns `MemoryKind` with no memory kind
// chosen.
//
// TODO(b/356623715): Remove this function.
MemoryKind CanonicalizeMemoryKind(MemoryKind memory_kind, const Device* device);

TSL_LIB_GTL_DEFINE_INT_TYPE(MemoryId, int32_t);

// `Memory` represents a memory space that one or more devices can be attached
// to. A platform may have multiple memory spaces with different backing
// hardware or memory region types.
class Memory : public RTTIExtends<Memory, RTTIRoot> {
 public:
  Memory() = default;

  // Not copyable or movable.
  Memory(const Memory&) = delete;
  Memory(Memory&&) = delete;
  Memory& operator=(const Memory&) = delete;
  Memory& operator=(Memory&&) = delete;

  virtual MemoryId Id() const = 0;

  // A platform-dependent string that uniquely identifies the kind of the
  // memory.
  virtual const MemoryKind& Kind() const = 0;

  // Debug string suitable for reading by end users, should be reasonably terse.
  virtual absl::string_view ToString() const = 0;

  // Debug string suitable for logging when errors occur. Should be verbose
  // enough to describe the current device unambiguously.
  //
  // TODO(hyeontaek): Remove this method in favor of AbslStringify.
  ABSL_DEPRECATED("Memory implements AbslStringify; rely on that instead.")
  virtual absl::string_view DebugString() const = 0;

  // The devices to which this memory space is attached.
  virtual absl::Span<Device* const> Devices() const = 0;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Memory& memory) {
    sink.Append(memory.DebugString());
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Memory* memory) {
    if (memory == nullptr) {
      sink.Append("<nullptr>");
    } else {
      sink.Append(memory->DebugString());
    }
  }

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_MEMORY_H_
