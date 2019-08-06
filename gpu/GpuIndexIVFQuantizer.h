/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "GpuIndexIVF.h"

namespace faiss { namespace gpu {

class GpuIndexIVFQuantizer : public GpuIndexIVF {
 public:
  void reset() override;

  /* void train(Index::idx_t n, const float* x) override; */

 protected:
  /* void searchImpl_(int n, */
  /*                  const float* x, */
  /*                  int k, */
  /*                  float* distances, */
  /*                  Index::idx_t* labels) const override; */
};

} } // namespace
