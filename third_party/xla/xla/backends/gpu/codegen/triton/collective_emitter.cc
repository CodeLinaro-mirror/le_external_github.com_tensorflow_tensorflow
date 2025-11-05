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

#include "xla/backends/gpu/codegen/triton/collective_emitter.h"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "third_party/triton/include/triton/Dialect/Triton/IR/Dialect.h"
#include "third_party/triton/include/triton/Dialect/Triton/IR/Types.h"

namespace xla::gpu {

namespace {
using ::mlir::ShapedType;
using ::mlir::Value;
using se::gpu::AllReduceStrategy;
using ::xla::gpu::triton::ScalarOrTensor;
using ::xla::gpu::triton::TileInfo;

namespace ttir = ::mlir::triton;
namespace mtx = ::mlir::triton::xla;
namespace arith = ::mlir::arith;

// The main memory space on a device (HBM).
static constexpr auto kGlobalAddressSpace =
    static_cast<std::underlying_type_t<mlir::NVVM::NVVMMemorySpace>>(
        mlir::NVVM::NVVMMemorySpace::Global);

bool CanAllReduceBeEmitted(const HloAllReduceInstruction* all_reduce,
                           ReductionKind reduction_kind, int64_t num_devices,
                           int64_t num_elements, PrimitiveType element_type,
                           se::gpu::AllReduceStrategy all_reduce_strategy) {
  if (!all_reduce->GetModule()
           ->config()
           .debug_options()
           .xla_gpu_unsupported_use_all_reduce_one_shot_kernel()) {
    return false;
  }
  // TODO(b/383125489): Support variadic all-reduce.
  if (all_reduce->operand_count() > 1) {
    return false;
  }
  const int64_t byte_size =
      num_elements * ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  // TODO(b/457333991): Support twoShot for codegen.
  if (byte_size >
      GetMaxSupportedAllReduceSizeBytes(AllReduceStrategy::kOneShot)) {
    return false;
  }
  return IsAllReduceKernelSupported(num_devices, num_elements, element_type,
                                    reduction_kind, all_reduce_strategy);
}

int64_t RoundUpToPowerOf2(int64_t tile_size) {
  return (tile_size & (tile_size - 1)) ? llvm::PowerOf2Ceil(tile_size)
                                       : tile_size;
};

// The logic here is very naive and assumes a monotonic layout
// where only the last dimension is used as a tiling dimension.
absl::StatusOr<std::optional<BlockLevelFusionConfig>>
GetBlockLevelFusionConfigForAllReduce(const se::DeviceDescription& device_info,
                                      const HloInstruction* root) {
  if (root->opcode() != HloOpcode::kAllReduceStart) {
    return std::nullopt;
  }
  const auto* const all_reduce = Cast<HloAllReduceInstruction>(root);
  const std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(all_reduce->called_computations().front());
  if (!reduction_kind.has_value()) {
    return absl::InternalError(
        "Reduction computation not found for all-reduce.");
  }
  const int64_t num_devices =
      all_reduce->device_list().num_devices_per_group() *
      all_reduce->device_list().num_replica_groups();
  const int64_t num_elements =
      ShapeUtil::ElementsIn(all_reduce->operand(0)->shape());
  const PrimitiveType element_type =
      all_reduce->operand(0)->shape().element_type();
  // NB: We do not codegen multimem kernels for now.
  const AllReduceStrategy all_reduce_strategy =
      GetAllReduceStrategy(num_elements, /*is_multimem_enabled=*/false);
  if (!CanAllReduceBeEmitted(all_reduce, reduction_kind.value(), num_devices,
                             num_elements, element_type, all_reduce_strategy)) {
    return std::nullopt;
  }
  const Shape& output_shape = root->shape();
  const LaunchDimensions launch_dims =
      AllReduceLaunchDimensions(num_elements, num_devices, all_reduce_strategy);
  const int64_t num_blocks = llvm::PowerOf2Ceil(launch_dims.num_blocks());
  BlockLevelFusionConfig block_level_config;
  block_level_config.set_num_warps(launch_dims.num_threads_per_block() /
                                   WarpSize(device_info));
  block_level_config.set_num_ctas(1);    // No block-level clustering.
  block_level_config.set_num_stages(1);  // No pipelining of loops.
  Tile* output_tile = block_level_config.add_output_tiles();
  const int64_t rank = output_shape.dimensions().size();

  // Tile sizes are rolled up to power of 2 because this is what the emitter
  // infra expects.
  for (int i = 0; i < rank - 1; ++i) {
    output_tile->add_sizes(RoundUpToPowerOf2(output_shape.dimensions(i)));
  }
  // The last dimension is divided amongst blocks.
  if (rank > 0) {
    const int64_t tile_size =
        CeilOfRatio(output_shape.dimensions(rank - 1),
                    absl::implicit_cast<int64_t>(num_blocks));
    output_tile->add_sizes(RoundUpToPowerOf2(tile_size));
  }
  return block_level_config;
}
}  // namespace

absl::StatusOr<std::optional<BlockLevelFusionConfig>>
GetCollectiveBlockLevelFusionConfig(const se::DeviceDescription& device_info,
                                    const HloFusionInstruction* fusion_instr) {
  const HloInstruction* root = fusion_instr->fused_expression_root();
  switch (root->opcode()) {
    case HloOpcode::kAllReduceStart:
      return GetBlockLevelFusionConfigForAllReduce(device_info, root);
    default:
      return std::nullopt;
  }
}

absl::StatusOr<bool> TrySetGpuBackendConfigForCollective(
    const se::DeviceDescription& device_info,
    HloFusionInstruction* fusion_instr) {
  TF_ASSIGN_OR_RETURN(
      const std::optional<BlockLevelFusionConfig> block_config,
      GetCollectiveBlockLevelFusionConfig(device_info, fusion_instr));
  if (!block_config.has_value()) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      fusion_instr->backend_config<GpuBackendConfig>());
  gpu_backend_config.mutable_fusion_backend_config()->set_kind(
      kTritonCollectiveFusionKind);
  *gpu_backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() = *std::move(block_config);
  TF_RETURN_IF_ERROR(
      fusion_instr->set_backend_config(std::move(gpu_backend_config)));
  return true;
}

absl::StatusOr<int32_t> AddCollectiveMetadataArguments(
    llvm::SmallVector<mlir::Type>& fn_arg_types, EmitterLocOpBuilder& b,
    const HloComputation* hlo_computation) {
  // rank: i32
  fn_arg_types.push_back(b.getI32Type());
  // signal_value: i32
  fn_arg_types.push_back(b.getI32Type());
  // signal_buffers: !tt.ptr<!tt.ptr<i32>>
  fn_arg_types.push_back(ttir::PointerType::get(
      ttir::PointerType::get(b.getI32Type(), kGlobalAddressSpace),
      kGlobalAddressSpace));
  int32_t num_metadata_arguments = 3;
  for (HloInstruction* p : hlo_computation->parameter_instructions()) {
    PrimitiveType type = p->shape().element_type();
    mlir::Type ir_type;
    if (type == U16) {
      ir_type = b.getI16Type();
    } else if (type == S4) {
      ir_type = b.getI4Type();
    } else {
      TF_ASSIGN_OR_RETURN(ir_type, triton::TritonType(b, type));
    }
    // Also add the remote/scratch buffers for collectives.
    // !tt.ptr<!tt.ptr<type>>
    fn_arg_types.push_back(ttir::PointerType::get(
        ttir::PointerType::get(ir_type, kGlobalAddressSpace),
        kGlobalAddressSpace));
    ++num_metadata_arguments;
  }
  return num_metadata_arguments;
}

absl::StatusOr<ScalarOrTensor> EmitCollective(
    EmitterLocOpBuilder b, const se::DeviceDescription& device_info,
    const HloFusionInstruction* fusion,
    const TiledHloInstruction& tiled_hlo_reduce,
    const BlockLevelParameters& block_level_parameters,
    mlir::FunctionOpInterface fn, mlir::Value pid,
    absl::flat_hash_map<const TiledHloInstruction*,
                        ScalarOrTensor::TensorValue>& values) {
  // Check that the fusion is an all-reduce collective.
  const HloComputation* computation = fusion->fused_instructions_computation();
  const HloInstruction* root = computation->root_instruction();
  if (root->opcode() != HloOpcode::kAllReduceStart) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected an all-reduce fusion, got: ", root->ToString()));
  }
  const auto& all_reduce = *xla::Cast<HloAllReduceInstruction>(root);
  const TiledHloInstruction* tiled_input_hlo = tiled_hlo_reduce.operand(0);
  ScalarOrTensor::TensorValue input_tile = values[tiled_input_hlo];

  // Variadics are not supported yet so we can fix inputs to 1.
  // Which means 2 arguments for input/output one for scratch buffers and 3
  // metadata arguments. Plus 1 for the tile index.
  TF_RET_CHECK(fn.getNumArguments() == 7);
  const int32_t start_idx = computation->num_parameters() * 2;
  mlir::Value device_rank = fn.getArgument(start_idx);
  TF_RET_CHECK(device_rank.getType().isInteger(32));
  mlir::Value signal_value = fn.getArgument(start_idx + 1);
  TF_RET_CHECK(signal_value.getType().isInteger(32));
  // !tt.ptr<!tt.ptr<i32>>
  mlir::Value signal_buffers = fn.getArgument(start_idx + 2);
  // !tt.ptr<!tt.ptr<i64>>
  mlir::Value remote_input_buffers = fn.getArgument(start_idx + 3);

  TF_ASSIGN_OR_RETURN(
      TileInfo tile_info,
      TileInfo::Construct(b, pid, /*runtime_values=*/{}, *tiled_input_hlo));

  // 1. Scatter phase: Copy local tile to the remote buffer of the current rank.
  const auto ptr_to_i64_type =
      ttir::PointerType::get(b.getI64Type(), kGlobalAddressSpace);
  auto remote_input_buffers_i64 =
      b.create<ttir::BitcastOp>(ptr_to_i64_type, remote_input_buffers);
  Value remote_buf_ptr_addr = b.create<ttir::AddPtrOp>(
      ptr_to_i64_type, remote_input_buffers_i64, device_rank);
  Value remote_buf_i64 =
      b.create<ttir::LoadOp>(remote_buf_ptr_addr,
                             ttir::CacheModifier::NONE,     //
                             ttir::EvictionPolicy::NORMAL,  //
                             false);                        // isVolatile
  const auto elem_type =
      mlir::cast<ShapedType>(input_tile.getType()).getElementType();
  const auto ptr_to_elem_type =
      ttir::PointerType::get(elem_type, kGlobalAddressSpace);
  Value remote_buf_ptr =
      b.create<ttir::IntToPtrOp>(ptr_to_elem_type, remote_buf_i64);
  mlir::ArrayRef<int64_t> remote_shape = tile_info.original_shape();
  const mlir::MemRefType remote_memref_type =
      mlir::MemRefType::get(remote_shape, elem_type);
  mlir::Value remote_buf_memref =
      b.create<mtx::PtrToMemrefOp>(remote_memref_type, remote_buf_ptr);
  b.create<xtile::InsertTileOp>(
      input_tile, remote_buf_memref, tile_info.offsets(),
      tile_info.padded_tile_sizes(), tile_info.tile_strides());

  // 2. Synchronization phase: Wait for all ranks to complete the scatter.
  int64_t world_size = all_reduce.replica_groups()[0].replica_ids_size();
  b.create<mtx::BlockBarrierOp>(signal_buffers, device_rank, signal_value,
                                b.getI32IntegerAttr(world_size));

  // 3. Reduce phase: Load tiles from all ranks and reduce them.
  // Load from rank 0 as the initial accumulator value.
  HloComputation* reduction_computation = all_reduce.to_apply();
  llvm::SmallVector<const HloInstruction*> to_emit;
  // There is really only one instruction in the computation.
  for (const HloInstruction* instr : reduction_computation->instructions()) {
    if (instr->opcode() != HloOpcode::kParameter) {
      to_emit.push_back(instr);
    }
  }
  ScalarOrTensor accumulator;
  for (int rank = 0; rank < world_size; ++rank) {
    Value rank_idx =
        b.create<arith::ConstantOp>(b.getI64Type(), b.getI64IntegerAttr(rank));
    Value remote_buf_ptr_addr = remote_input_buffers_i64;
    if (rank > 0) {  // No need to add rank 0 as it was already loaded.
      remote_buf_ptr_addr = b.create<ttir::AddPtrOp>(
          ptr_to_i64_type, remote_input_buffers_i64, rank_idx);
    }
    Value remote_buf_i64 =
        b.create<ttir::LoadOp>(remote_buf_ptr_addr,
                               ttir::CacheModifier::NONE,     //
                               ttir::EvictionPolicy::NORMAL,  //
                               false);                        // isVolatile
    Value remote_buf_ptr =
        b.create<ttir::IntToPtrOp>(ptr_to_elem_type, remote_buf_i64);
    Value remote_buf_memref =
        b.create<mtx::PtrToMemrefOp>(remote_memref_type, remote_buf_ptr);
    if (rank == 0) {  // For rank 0 directly extract into accumulator.
      accumulator = EmitParameterExtract(b, tile_info, remote_buf_memref);
    } else {  // Otherwise extract into a new tile and reduce to accumulator.
      ScalarOrTensor next_tile =
          EmitParameterExtract(b, tile_info, remote_buf_memref);

      absl::flat_hash_map<const HloInstruction*, ScalarOrTensor> region_values;
      region_values[reduction_computation->parameter_instruction(0)] =
          accumulator;
      region_values[reduction_computation->parameter_instruction(1)] =
          next_tile;
      TF_ASSIGN_OR_RETURN(
          accumulator,
          triton::EmitScope(b, device_info,
                            /*analysis=*/nullptr, /*instructions=*/to_emit,
                            /*values=*/region_values));
    }
  }
  return accumulator;
}

}  // namespace xla::gpu
