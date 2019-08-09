/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "GpuIndexIVF.h"
#include "impl/IVFScalarQuantizer.cuh"

namespace faiss { struct IndexIVFScalarQuantizer; }

namespace faiss { namespace gpu {

// class IVFScalarQuantizer;

struct GpuIndexIVFScalarQuantizerConfig : public GpuIndexIVFConfig {
  inline GpuIndexIVFScalarQuantizerConfig()
      : useFloat16IVFStorage(false) {
  }

  /// Whether or not IVFFlat inverted list storage is in float16;
  /// supported on all architectures
  bool useFloat16IVFStorage;
};

/// Wrapper around the GPU implementation that looks like
/// faiss::IndexIVFFlat
class GpuIndexIVFScalarQuantizer : public GpuIndexIVF {
 public:
  /// Construct from a pre-existing faiss::IndexIVFFlat instance, copying
  /// data over to the given GPU, if the input index is trained.
  GpuIndexIVFScalarQuantizer(GpuResources* resources,
                  const faiss::IndexIVFScalarQuantizer* index,
                  GpuIndexIVFScalarQuantizerConfig config = GpuIndexIVFScalarQuantizerConfig());

  /// Constructs a new instance with an empty flat quantizer; the user
  /// provides the number of lists desired.
  GpuIndexIVFScalarQuantizer(GpuResources* resources,
                  int dims,
                  int nlist,
                  faiss::MetricType metric,
                  GpuIndexIVFScalarQuantizerConfig config = GpuIndexIVFScalarQuantizerConfig());

  ~GpuIndexIVFScalarQuantizer() override;

  /// Reserve GPU memory in our inverted lists for this number of vectors
  void reserveMemory(size_t numVecs);

  /// Initialize ourselves from the given CPU index; will overwrite
  /// all data in ourselves
  void copyFrom(const IndexIVFScalarQuantizer* index);

  /// Copy ourselves to the given CPU index; will overwrite all data
  /// in the index instance
  void copyTo(IndexIVFScalarQuantizer* index) const;

  /// After adding vectors, one can call this to reclaim device memory
  /// to exactly the amount needed. Returns space reclaimed in bytes
  size_t reclaimMemory();

  void reset() override;

  void train(Index::idx_t n, const float* x) override;

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
  GpuIndexIVFScalarQuantizerConfig ivfSQConfig_;

  /// Desired inverted list memory reservation
  size_t reserveMemoryVecs_;

  /// Instance that we own; contains the inverted list
  IVFScalarQuantizer* index_ = nullptr;
};

} } // namespace
