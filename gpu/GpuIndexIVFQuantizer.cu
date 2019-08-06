/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "GpuIndexIVFQuantizer.h"

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
}

} } // namespace
