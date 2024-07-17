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

#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_ATTRS
#define XLA_SERVICE_GPU_FUSIONS_MLIR_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "xla/service/gpu/fusions/mlir/ir/xla_gpu_dialect.td"

class XLAGPU_Attr<string name, list<Trait> traits = []> :
      AttrDef<XlaGpuDialect, name, traits> {
}

def XLAGPU_AffineMapParameter :
    AttrOrTypeParameter<"::mlir::AffineMap", ""> {
}

def XLAGPU_IndexingMapAttr : XLAGPU_Attr<"IndexingMap"> {
  let mnemonic = "indexing_map";
  let summary = "An Attribute representing an indexing map.";
  let description = [{
    This attribute stores an indexing map, used for lowering HLO to block level
    MLIR.
  }];
  let parameters = (ins XLAGPU_AffineMapParameter:$map,
                        ArrayRefParameter<"DimVar">:$dimensions,
                        ArrayRefParameter<"RangeVar">:$symbols);

  let assemblyFormat = [{
    `<` `map` `=` $map `,`
        `dimensions` `=` $dimensions `,`
        `symbols` `=` $symbols `>`
  }];
}

#endif // MLIR_HLO_DIALECT_MHLO_IR_HLO_ATTRS
