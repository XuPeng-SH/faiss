/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include "GpuIndexIVFSQ.h"
#include "../IndexFlat.h"
#include "../IndexIVFFlat.h"
#include "GpuIndexFlat.h"
#include "GpuResources.h"
#include "impl/IVFSQ.cuh"
#include "utils/CopyUtils.cuh"
#include "utils/DeviceUtils.h"

#include <limits>
#include <memory>

namespace faiss { namespace gpu {

GpuIndexIVFSQ::GpuIndexIVFSQ(GpuResources* resources,
        const CpuIndexT* index,
        ConfigT config) :
    BaseT(resources, index->d, index->metric_type, index->nlist, config),
    config_(config),
    reserveMemoryVecs_(0),
    index_(nullptr) {
    copyFrom(index);
}

GpuIndexIVFSQ::GpuIndexIVFSQ(GpuResources* resources,
                                 int dims,
                                 int nlist,
                                 MetricT metric,
                                 ConfigT config) :
    BaseT(resources, dims, metric, nlist, config),
    config_(config),
    reserveMemoryVecs_(0),
    index_(nullptr) {
  this->is_trained = false;
}

GpuIndexIVFSQ::~GpuIndexIVFSQ() {
  delete index_;
}

void
GpuIndexIVFSQ::reserveMemory(size_t numVecs) {
  reserveMemoryVecs_ = numVecs;
  if (index_) {
    index_->reserveMemory(numVecs);
  }
}

void
GpuIndexIVFSQ::copyFrom(const CpuIndexT* index) {
  DeviceScope scope(device_);

  BaseT::copyFrom(index);

  // Clear out our old data
  delete index_;
  index_ = nullptr;

  // The other index might not be trained
  if (!index->is_trained) {
    return;
  }

  // Otherwise, we can populate ourselves from the other index
  this->is_trained = true;

  float vmin = index->sq.trained[0];
  float vdiff = index->sq.trained[1];


  index_ = new ImplT(resources_,
          quantizer_->getGpuData(),
          index->sq.code_size,
          index->metric_type == faiss::METRIC_L2,
          config_.indicesOptions,
          memorySpace_,
          vmin,
          vdiff);

  InvertedLists *ivf_lists = index->invlists;

  for (size_t i = 0; i < ivf_lists->nlist; ++i) {
    auto numVecs = ivf_lists->list_size(i);

    FAISS_THROW_IF_NOT_FMT(numVecs <=
                       (size_t) std::numeric_limits<int>::max(),
                       "GPU inverted list can only support "
                       "%zu entries; %zu found",
                       (size_t) std::numeric_limits<int>::max(),
                       numVecs);

    index_->addCodeVectorsFromCpu(
             i, ivf_lists->get_codes(i),
             ivf_lists->get_ids(i), numVecs);
  }
}

void
GpuIndexIVFSQ::dump() {
    for (auto i=0; i<index_->getNumLists(); ++i) {
        std::cout << "Size Of Buckets[" << i <<  "] = " << index_->getListLength(i) << std::endl;
    }
    for (auto i=0; i<index_->getNumLists(); ++i) {
        std::cout << "GpuIndice[" << i << "] = ";
        auto indices = index_->getListIndices(i);
        for (auto& id : indices) {
            std::cout << id << " | ";
        }
        std::cout << std::endl;
    }

    std::cout << "Trained data size = "  << index_->getTrainedData()->capacity() << std::endl;
}

void
GpuIndexIVFSQ::copyTo(CpuIndexT* index) const {
  FAISS_THROW_MSG("GpuIndexIVFSQ train not supported");
}

size_t
GpuIndexIVFSQ::reclaimMemory() {
  if (index_) {
    DeviceScope scope(device_);

    return index_->reclaimMemory();
  }

  return 0;
}

void
GpuIndexIVFSQ::reset() {
  if (index_) {
    DeviceScope scope(device_);

    index_->reset();
    this->ntotal = 0;
  } else {
    FAISS_ASSERT(this->ntotal == 0);
  }
}

void
GpuIndexIVFSQ::train(Index::idx_t n, const float* x) {
  FAISS_THROW_MSG("GpuIndexIVFSQ train not supported");
}

void
GpuIndexIVFSQ::addImpl_(int n,
                          const float* x,
                          const Index::idx_t* xids) {
  FAISS_THROW_MSG("GpuIndexIVFSQ addImpl_ not supported");
}

void
GpuIndexIVFSQ::searchImpl_(int n,
                             const float* x,
                             int k,
                             float* distances,
                             Index::idx_t* labels) const {
  // Device is already set in GpuIndex::search
  FAISS_ASSERT(index_);
  FAISS_ASSERT(n > 0);

  // Data is already resident on the GPU
  Tensor<float, 2, true> queries(const_cast<float*>(x), {n, (int) this->d});
  Tensor<float, 2, true> outDistances(distances, {n, k});

  static_assert(sizeof(long) == sizeof(Index::idx_t), "size mismatch");
  Tensor<long, 2, true> outLabels(const_cast<long*>(labels), {n, k});

  index_->query(queries, nprobe_, k, outDistances, outLabels);
}

} } // namespace
