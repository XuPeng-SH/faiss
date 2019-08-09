/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


 #pragma once

 #include "../GpuIndicesOptions.h"
 #include "../utils/Tensor.cuh"
 #include "../../IndexScalarQuantizer.h"
 #include <thrust/device_vector.h>
 
 namespace faiss { namespace gpu {
 
 class GpuResources;

 struct GpuScalarQuantizer {
    GpuScalarQuantizer(float vmin, float vdiff) {
        this->vmin = vmin;
        this->vdiff = vdiff;
    }
    float vmin = 0;
    float vdiff = 0;
 };
 
 void runIVFScalarQuantizerScan(Tensor<float, 2, true>& queries,
                     Tensor<int, 2, true>& listIds,
                     thrust::device_vector<void*>& listData,
                     thrust::device_vector<void*>& listIndices,
                     IndicesOptions indicesOptions,
                     thrust::device_vector<int>& listLengths,
                     int maxListLength,
                     int k,
                     bool l2Distance,
                     // output
                     Tensor<float, 2, true>& outDistances,
                     // output
                     Tensor<long, 2, true>& outIndices,
                     GpuResources* res,
                     GpuScalarQuantizer& gsq);
 
 } } // namespace
 