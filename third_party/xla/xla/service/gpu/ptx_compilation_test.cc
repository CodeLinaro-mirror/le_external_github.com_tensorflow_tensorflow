/* Copyright 2024 The OpenXLA Authors.

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

#include <memory>
#include <string_view>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/ptx_compilation_method.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"
#include "xla/stream_executor/cuda/ptx_linking_method.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

using stream_executor::PtxCompilationMethod;
using stream_executor::PtxLinkingMethod;

class NVPTXCompilationTests
    : public HloTestBase,
      public ::testing::WithParamInterface<std::tuple<
          std::string_view, PtxCompilationMethod, PtxLinkingMethod>> {
 public:
  void SkipTestIfUnsupported(PtxCompilationMethod compilation_method,
                             PtxLinkingMethod linking_method) {
    if (!stream_executor::IsLibNvPtxCompilerSupported() &&
        compilation_method == PtxCompilationMethod::kNvPtxCompiler) {
      // Compiled without libnvptxcompiler support
      GTEST_SKIP() << "libnvptxcompiler is not supported in this build.";
    }
  }

  void SetDebugOptionsFromPtxSettings(DebugOptions* debug_options,
                                      PtxCompilationMethod compilation_method,
                                      PtxLinkingMethod linking_method) {
    debug_options->set_xla_gpu_enable_libnvptxcompiler(
        compilation_method == PtxCompilationMethod::kNvPtxCompiler);

    debug_options->set_xla_gpu_enable_llvm_module_compilation_parallelism(
        linking_method != PtxLinkingMethod::kNone);
    debug_options->set_xla_gpu_force_compilation_parallelism(12);

    if (linking_method == PtxLinkingMethod::kDriver) {
      debug_options->set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(
          true);
      debug_options->set_xla_gpu_cuda_data_dir("/does/not/exist");
    }

    tsl::setenv("TF_USE_NVLINK_FOR_PARALLEL_COMPILATION",
                linking_method == PtxLinkingMethod::kNvLink ? "true" : "false",
                1);

    // We need individual functions to test parallel compilation.
    debug_options->set_xla_llvm_force_inline_before_split(false);
  }

  void SetUp() override {
    HloTestBase::SetUp();
    PtxCompilationMethod compilation_method = std::get<1>(GetParam());
    PtxLinkingMethod linking_method = std::get<2>(GetParam());
    SkipTestIfUnsupported(compilation_method, linking_method);
  }

  absl::StatusOr<std::unique_ptr<Executable>> CompileExecutable(
      std::unique_ptr<HloModule> module) {
    NVPTXCompiler compiler{};

    return compiler.RunBackend(std::move(module),
                               backend().default_stream_executor(),
                               {/*device_allocator=*/nullptr,
                                /*thread_pool=*/nullptr,
                                /*layout_canonicalization_callback=*/{},
                                /*is_autotuning_compilation=*/false});
  }
};

TEST_P(NVPTXCompilationTests, CompileProgram) {
  auto hlo_text = std::get<0>(GetParam());
  auto module = ParseAndReturnVerifiedModule(hlo_text).value();

  HloModuleConfig hlo_module_config = module->config();
  DebugOptions debug_options = hlo_module_config.debug_options();
  PtxCompilationMethod compilation_method = std::get<1>(GetParam());
  PtxLinkingMethod linking_method = std::get<2>(GetParam());
  SetDebugOptionsFromPtxSettings(&debug_options, compilation_method,
                                 linking_method);
  hlo_module_config.set_debug_options(debug_options);
  module->set_config(hlo_module_config);

  EXPECT_THAT(CompileExecutable(std::move(module)),
              tsl::testing::IsOkAndHolds(::testing::NotNull()));
}

TEST_P(NVPTXCompilationTests, CompareBinaryOutput) {
  auto hlo_text = std::get<0>(GetParam());
  auto compile = [&](PtxCompilationMethod compilation_method,
                     PtxLinkingMethod linking_method) {
    auto module = ParseAndReturnVerifiedModule(hlo_text).value();

    HloModuleConfig hlo_module_config = module->config();
    DebugOptions debug_options = hlo_module_config.debug_options();
    SetDebugOptionsFromPtxSettings(&debug_options, compilation_method,
                                   linking_method);
    hlo_module_config.set_debug_options(debug_options);
    module->set_config(hlo_module_config);

    return CompileExecutable(std::move(module));
  };

  PtxCompilationMethod compilation_method = std::get<1>(GetParam());
  PtxLinkingMethod linking_method = std::get<2>(GetParam());
  absl::StatusOr<std::unique_ptr<Executable>> executable =
      compile(compilation_method, linking_method);
  constexpr PtxLinkingMethod kReferenceLinkingMethod = PtxLinkingMethod::kNone;
  absl::StatusOr<std::unique_ptr<Executable>> reference =
      compile(PtxCompilationMethod::kPtxas, kReferenceLinkingMethod);

  EXPECT_THAT(executable, tsl::testing::IsOkAndHolds(::testing::NotNull()));
  EXPECT_THAT(reference, tsl::testing::IsOkAndHolds(::testing::NotNull()));

  if (linking_method == kReferenceLinkingMethod) {
    // Different linking methods produce different orderings of symbols in
    // the rel.debug_frame section. That's why we only compare the full binary
    // if we use the same linking method. In all other cases a comparison
    // of the size of the binary provides a smoke test.
    EXPECT_THAT(
        static_cast<GpuExecutable*>(executable.value().get())->binary(),
        ::testing::Eq(
            static_cast<GpuExecutable*>(reference.value().get())->binary()));
  } else {
    EXPECT_THAT(
        static_cast<GpuExecutable*>(executable.value().get())->binary().size(),
        ::testing::Eq(static_cast<GpuExecutable*>(reference.value().get())
                          ->binary()
                          .size()));
  }
}

constexpr std::string_view kHlo1 = R"(
HloModule simple

ENTRY main {
  p = f32[10]{0} parameter(0)
  ROOT neg = f32[10]{0} negate(p)
}
)";
constexpr std::string_view kHlo2 = R"(
HloModule parallel_compilation

ENTRY main {
  p1 = f32[10,20,30] parameter(0)
  p2 = f32[40,30,10] parameter(1)
  // With the new MLIR emitters, each indexing change leads to a new function.
  // So adding 2 transposes and a concatenate will results in 3 LLVM IR
  // functions that can be compiled in parallel.
  t1 = f32[20,10,30] transpose(p1), dimensions={1,0,2}
  t2 = f32[40,10,30] transpose(p2), dimensions={0,2,1}
  ROOT c = f32[60,10,30] concatenate(t1, t2), dimensions={0}
}
)";

INSTANTIATE_TEST_SUITE_P(
    NVPTXCompilationTest, NVPTXCompilationTests,
    ::testing::Combine(::testing::Values(kHlo1, kHlo2),
                       ::testing::Values(PtxCompilationMethod::kNvPtxCompiler,
                                         PtxCompilationMethod::kPtxas),
                       ::testing::Values(PtxLinkingMethod::kNone,
                                         PtxLinkingMethod::kNvLink,
                                         PtxLinkingMethod::kDriver)),
    [](const ::testing::TestParamInfo<std::tuple<
           std::string_view, PtxCompilationMethod, PtxLinkingMethod>>& info) {
      std::string_view hlo = std::get<0>(info.param);
      std::string_view hlo_module_line = *(++absl::StrSplit(hlo, '\n').begin());
      std::string_view name = *(++absl::StrSplit(hlo_module_line, ' ').begin());
      return absl::StrFormat("%v_CompilationMethod_%v_LinkingMethod_%v", name,
                             std::get<1>(info.param), std::get<2>(info.param));
    });

}  // namespace
}  // namespace xla::gpu
