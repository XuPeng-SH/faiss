/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "GpuIndexIVF.h"

namespace faiss { struct IndexIVFScalarQuantizer; }

namespace faiss { namespace gpu {

class IVFSQ;

struct GpuIndexIVFSQConfig : public GpuIndexIVFConfig {
  inline GpuIndexIVFSQConfig()
      : useFloat16IVFStorage(false) {
  }

  /// Whether or not IVFFlat inverted list storage is in float16;
  /// supported on all architectures
  bool useFloat16IVFStorage;
};

class GpuIndexIVFSQ : public GpuIndexIVF {
 public:
  using BaseT = GpuIndexIVF;
  using CpuIndexT = faiss::IndexIVFScalarQuantizer;
  using ConfigT = GpuIndexIVFSQConfig;
  using MetricT = faiss::MetricType;
  using ImplT = IVFSQ;
  using VecT = uint8_t;

  GpuIndexIVFSQ(GpuResources* resources,
          const CpuIndexT* index,
          ConfigT config = ConfigT());

  GpuIndexIVFSQ(GpuResources* resources,
                  int dims,
                  int nlist,
                  MetricT metric,
                  ConfigT config = ConfigT());

  ~GpuIndexIVFSQ() override;

  /// Reserve GPU memory in our inverted lists for this number of vectors
  void reserveMemory(size_t numVecs);

  void copyFrom(const CpuIndexT* index);

  void copyTo(CpuIndexT* index) const;

  /// After adding vectors, one can call this to reclaim device memory
  /// to exactly the amount needed. Returns space reclaimed in bytes
  size_t reclaimMemory();

  void reset() override;

  void train(Index::idx_t n, const float* x) override;

  void dump();

 protected:
  /// Called from GpuIndex for add/add_with_ids
  void addImpl_(int n,
                const float* x,
                const Index::idx_t* ids) override;

  /// Called from GpuIndex for search
  void searchImpl_(int n,
                   const float* x,
                   int k,
                   float* distances,
                   Index::idx_t* labels) const override;

 private:
  ConfigT config_;

  /// Desired inverted list memory reservation
  size_t reserveMemoryVecs_;

  /// Instance that we own; contains the inverted list
  ImplT* index_ = nullptr;
};

} } // namespace
