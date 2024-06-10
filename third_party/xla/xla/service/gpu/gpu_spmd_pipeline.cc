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

#include "xla/service/gpu/gpu_spmd_pipeline.h"

#include <cstdint>

#include "absl/log/check.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/transforms/hlo_constant_splitter.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/bitcast_dtypes_expander.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gpu/gpu_algebraic_simplifier.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/sort_simplifier.h"
#include "xla/service/spmd/collective_permute_motion.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

#ifdef PLATFORM_GOOGLE
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#endif  // PLATFORM_GOOGLE

namespace xla {
namespace gpu {

void AddSPMDPasses(
    const HloModule* hlo_module,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts,
    const se::GpuComputeCapability& compute_capability, int core_count,
    HloPassPipeline& spmd_pipeline) {
  const int64_t num_partitions = hlo_module->config().num_partitions();
  CHECK_GE(num_partitions, 1);
  bool auto_sharding = hlo_module->config().use_auto_spmd_partitioning();

  HloPassPipeline& spmd_simplify =
      spmd_pipeline.AddPass<HloPassFix<HloPassPipeline>>("spmd-simplify");

  spmd_simplify.AddPass<GpuAlgebraicSimplifier>(layout_insensitive_algsimp_opts,
                                                compute_capability);
  spmd_simplify.AddPass<SortSimplifier>();
  spmd_simplify.AddPass<TupleSimplifier>();
  spmd_simplify.AddPass<ScatterExpander>(
      ScatterExpander::kEliminateSimpleScatters);
  spmd_simplify.AddPass<GatherExpander>(
      GatherExpander::kEliminateSimpleGathers);
  spmd_simplify.AddPass<WhileLoopConstantSinking>();
  spmd_simplify.AddPass<WhileLoopSimplifier>();

  ReshapeMoverOptions reshape_mover_options;
  reshape_mover_options.reshape_of_1d_broadcast_is_cheap = true;
  spmd_simplify.AddPass<ReshapeMover>(reshape_mover_options);
  // Run AlgebraicSimplifier directly before HloConstantFolding, because we
  // need to simplify DynamicSlice(Broadcast) away. Constant folding of
  // DynamicSlice can be quite costly, as the whole operand will be evaluated.
  // We run AlgebraicSimplifier as HloPassFix to make sure all simplifications
  // have been done before running HloConstantFolding. This is necessary
  // because simplifications create new instructions which may not be visited
  // in the same iteration of AlgebraicSimplifier.
  spmd_simplify.AddPass<HloPassFix<GpuAlgebraicSimplifier>>(
      layout_insensitive_algsimp_opts, compute_capability);
  spmd_simplify.AddPass<HloConstantFolding>();
  spmd_simplify.AddPass<ConditionalSimplifier>();

  spmd_pipeline.AddPass<HloConstantSplitter>();
  spmd_simplify.AddPass<HloDCE>();

#ifdef PLATFORM_GOOGLE
  if (auto_sharding) {
    AutoShardingOption option;
    option.enable = true;
    if (!hlo_module->config().auto_spmd_partitioning_mesh_shape().empty()) {
      option.device_mesh_shape =
          hlo_module->config().auto_spmd_partitioning_mesh_shape();
    } else {
      // Use a simple mesh shape if not specified.
      option.device_mesh_shape = {core_count, 1};
    }
    if (!hlo_module->config().auto_spmd_partitioning_mesh_ids().empty()) {
      option.device_mesh_ids =
          hlo_module->config().auto_spmd_partitioning_mesh_ids();
    }
    option.memory_budget_per_device =
        hlo_module->config()
            .debug_options()
            .xla_gpu_auto_spmd_partitioning_memory_budget_gb() *
        1024 * 1024 * 1024;
    option.memory_budget_ratio =
        hlo_module->config()
            .debug_options()
            .xla_gpu_auto_spmd_partitioning_memory_budget_ratio();
    spmd_pipeline.AddPass<AutoSharding>(option);
  }
#endif  // PLATFORM_GOOGLE

  spmd_pipeline.AddPass<ShardingPropagation>(
      /*is_spmd=*/true, /*propagate_metadata=*/false,
      hlo_module->config().allow_spmd_sharding_propagation_to_output());
  spmd_pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(
      num_partitions, hlo_module->config().replica_count(),
      hlo_module->config()
          .debug_options()
          .xla_gpu_threshold_for_windowed_einsum_mib(),
      hlo_module->config()
          .debug_options()
          .xla_gpu_multi_streamed_windowed_einsum(),
      /*skip_checking_windowed_einsum_users=*/true,
      /*disable_ag_rewrite_for_multiple_consumers=*/true);
  spmd_pipeline.AddPass<CollectivePermuteMotion>();
}

}  // namespace gpu
}  // namespace xla
