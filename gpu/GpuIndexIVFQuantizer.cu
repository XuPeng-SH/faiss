/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "GpuIndexIVFQuantizer.h"
#include "utils/DeviceUtils.h"
#include "utils/CopyUtils.cuh"
#include "impl/IVFFlat.cuh"

namespace faiss { namespace gpu {

GpuIndexIVFQuantizer::GpuIndexIVFQuantizer(GpuResources* resources,
                         int dims,
                         faiss::MetricType metric,
                         int nlist,
                         GpuIndexIVFConfig config) :
    GpuIndexIVF(resources, dims, metric, nlist, config) {

}

void
GpuIndexIVFQuantizer::reset() {
}

void
GpuIndexIVFQuantizer::addImpl_(int n,
                          const float* x,
                          const Index::idx_t* xids) {
}

void
GpuIndexIVFQuantizer::searchImpl_(int n,
                             const float* x,
                             int k,
                             float* distances,
                             Index::idx_t* labels) const {
  FAISS_ASSERT(n > 0);

  Tensor<float, 2, true> queries(const_cast<float*>(x), {n, (int) this->d});
  Tensor<float, 2, true> outDistances(distances, {n, k});

  static_assert(sizeof(long) == sizeof(Index::idx_t), "size mismatch");
  Tensor<long, 2, true> outLabels(const_cast<long*>(labels), {n, k});

  quantizer_->getGpuData()->query(queries, nprobe_, outDistances, outLabels, false);
}

} } // namespace
