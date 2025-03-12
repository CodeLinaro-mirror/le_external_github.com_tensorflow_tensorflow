/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/tfrt/tracked_tfrt_gpu_device_buffer.h"

#include <stdlib.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/mem.h"

namespace xla {
namespace {

using ::tsl::BlockUntilReady;
using ::tsl::MakeConstructedAsyncValueRef;

class TestAllocator : public tsl::Allocator {
 public:
  std::string Name() override { return "test_allocator"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return tsl::port::AlignedMalloc(num_bytes, alignment);
  }
  void DeallocateRaw(void* ptr) override { return tsl::port::AlignedFree(ptr); }
};

TEST(MaybeOwningGpuMemoryTest, NonOwning) {
  auto memory = MaybeOwningGpuMemory(se::DeviceMemoryBase());
  EXPECT_FALSE(memory.owns_data());
  EXPECT_EQ(memory.allocator(), nullptr);
}

TEST(TrackedTfrtGpuDeviceBufferTest, Basic) {
  std::string expected = "tracked_tfrt_gpu_device_buffer_test";
  auto usage_event = MakeConstructedAsyncValueRef<GpuEvent>();

  auto allocator = std::make_unique<TestAllocator>();
  auto test_buffer = MakeConstructedAsyncValueRef<MaybeOwningGpuMemory>(
      allocator.get(), se::DeviceMemoryBase(expected.data(), expected.size()));

  auto definition_event = MakeConstructedAsyncValueRef<GpuEvent>();

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                      "tracked_buffer_test",
                                      /*num_threads=*/4);

  TrackedTfrtGpuDeviceBuffer tracked_buffer(test_buffer, definition_event,
                                            /*on_delete_callback_=*/nullptr);
  {
    MarkEventReadyOnExit ready_on_exit(usage_event);
    tracked_buffer.AddUsageEvents(absl::MakeSpan(&usage_event, 1));
    // Mimic transfer event in a thread pool.
    thread_pool.Schedule([&]() {
      std::memcpy(test_buffer->buffer().opaque(), expected.data(),
                  expected.size());
      definition_event.SetStateConcrete();
      test_buffer.SetStateConcrete();
    });
    BlockUntilReady(tracked_buffer.definition_event().GetAsyncValue());
    EXPECT_EQ(tracked_buffer.buffer()->size(), expected.size());
    auto result = tracked_buffer.buffer();
    ASSERT_TRUE(result.IsAvailable());
    EXPECT_EQ(std::string(static_cast<const char*>(result->buffer().opaque()),
                          result->size()),
              expected);
  }
  BlockUntilReady(tracked_buffer.AfterAllUsageEvents());
}
}  // namespace

}  // namespace xla
