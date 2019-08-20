
#include "IVFSQ.cuh"
#include "../GpuResources.h"
#include "FlatIndex.cuh"
#include "InvertedListAppend.cuh"
#include "IVFSQScan.cuh"
#include "RemapIndices.h"
#include "../utils/CopyUtils.cuh"
#include "../utils/DeviceDefs.cuh"
#include "../utils/DeviceUtils.h"

#include "../utils/HostTensor.cuh"
#include "../utils/Transpose.cuh"
#include "../../utils.h"
#include <limits>
#include <thrust/host_vector.h>
#include <unordered_map>
#include <chrono>

namespace faiss { namespace gpu {

IVFSQ::IVFSQ(GpuResources* resources,
                CoarseQuantizerT* coarse_quantizer,
                int code_size,
                bool l2Distance,
                IndicesOptions indicesOptions,
                MemorySpace space,
                float vdiff,
                float vmin) :
    IVFBase(resources,
            coarse_quantizer,
            code_size,
            indicesOptions,
            space),
    l2Distance_(l2Distance),
    gpu_scalar_quantizer_(vdiff, vmin) {
}

IVFSQ::~IVFSQ() {
}

void
IVFSQ::copyCodeVectorsFromCpu(const VecT* vecs, const long* indices,
        const std::vector<size_t>& list_length) {
    FAISS_ASSERT_FMT(list_length.size() == this->getNumLists(), "Expect list size %zu but %zu received!",
          this->getNumLists(), list_length.size());

    auto numVecs = std::accumulate(list_length.begin(), list_length.end(), 0);
    if (numVecs == 0) {
        return;
    }

    auto stream = resources_->getDefaultStreamCurrentDevice();
    deviceListLengths_ = list_length;
    size_t lengthInBytes = numVecs * bytesPerVector_;

    FAISS_ASSERT(deviceData_->size() + lengthInBytes <=
         (size_t) std::numeric_limits<int>::max());

    deviceData_->reserve(lengthInBytes, stream);
    deviceData_->append((unsigned char*) vecs,
            lengthInBytes,
            stream,
            true);

    copyIndicesFromCpu_(indices, list_length);

    maxListLength_ = 0;

    size_t listId = 0;
    size_t pos = 0;
    /* INIT_TIMER; */
    /* std::vector<size_t> counts = {0,0}; */
    thrust::host_vector<void*> hostPointers;
    hostPointers.resize(deviceListData_.size(), nullptr);
    for (auto& device_data : deviceListData_) {
        auto data = deviceData_->data() + pos;
        device_data->reset(data, list_length[listId]*bytesPerVector_, list_length[listId]*bytesPerVector_);
        /* START_TIMER; */
        hostPointers[listId] = device_data->data();
        /* counts[0] += GET_TIMER; */
        maxListLength_ = std::max(maxListLength_, (int)list_length[listId]);
        pos += list_length[listId]*bytesPerVector_;
        listId++;
    }

    /* START_TIMER; */
    deviceListDataPointers_ = hostPointers;
    /* counts[1] += GET_TIMER; */

    /* for (auto& t : counts) { */
    /*     std::cout << "ts: " << t << std::endl; */
    /* } */

    // device_vector add is potentially happening on a different stream
    // than our default stream
    if (stream != 0) {
        streamWait({stream}, {0});
    }
}

void
IVFSQ::addCodeVectorsFromCpu(int listId,
        const VecT* vecs, const long* indices, size_t numVecs) {
    FAISS_ASSERT(listId < deviceListData_.size());
    auto stream = resources_->getDefaultStreamCurrentDevice();

    if (numVecs == 0) {
        return;
    }

    size_t lengthInBytes = numVecs * bytesPerVector_;

    auto& listData = deviceListData_[listId];
    auto prevData = listData->data();

    FAISS_ASSERT(listData->size() + lengthInBytes <= (size_t)std::numeric_limits<int>::max());

    listData->append((unsigned char*) vecs,
            lengthInBytes,
            stream,
            true);

    addIndicesFromCpu_(listId, indices, numVecs);

    if (prevData != listData->data()) {
        deviceListDataPointers_[listId] = listData->data();
    }

    int listLength = listData->size() / bytesPerVector_;
    deviceListLengths_[listId] = listLength;

    maxListLength_ = std::max(maxListLength_, listLength);

    if (stream != 0) {
        streamWait({stream}, {0});
    }
}

void
IVFSQ::addTrainedDataFromCpu(const uint8_t* trained,
                            size_t numData) {
    addTrainedDataFromCpu_(trained, numData);
}

std::vector<uint8_t>
IVFSQ::getListVectors(int listId) const {
  FAISS_ASSERT(listId < deviceListData_.size());
  auto& encVecs = *deviceListData_[listId];

  auto stream = resources_->getDefaultStreamCurrentDevice();

  size_t num = encVecs.size() / sizeof(uint8_t);

  Tensor<uint8_t, 1, true> dev((uint8_t*) encVecs.data(), {(int) num});

  std::vector<uint8_t> out(num);
  HostTensor<uint8_t, 1, true> host(out.data(), {(int) num});
  host.copyFrom(dev, stream);

  return out;
}

 void
 IVFSQ::query(Tensor<float, 2, true>& queries,
                int nprobe,
                int k,
                Tensor<float, 2, true>& outDistances,
                Tensor<long, 2, true>& outIndices) {
   auto& mem = resources_->getMemoryManagerCurrentDevice();
   auto stream = resources_->getDefaultStreamCurrentDevice();

   // These are caught at a higher level
   FAISS_ASSERT(nprobe <= GPU_MAX_SELECTION_K);
   FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
   nprobe = std::min(nprobe, quantizer_->getSize());

   FAISS_ASSERT(queries.getSize(1) == dim_);

   FAISS_ASSERT(outDistances.getSize(0) == queries.getSize(0));
   FAISS_ASSERT(outIndices.getSize(0) == queries.getSize(0));

   // Reserve space for the quantized information
   DeviceTensor<float, 2, true>
     coarseDistances(mem, {queries.getSize(0), nprobe}, stream);
   DeviceTensor<int, 2, true>
     coarseIndices(mem, {queries.getSize(0), nprobe}, stream);

   // Find the `nprobe` closest lists; we can use int indices both
   // internally and externally
   quantizer_->query(queries,
                     nprobe,
                     coarseDistances,
                     coarseIndices,
                     false);

   runIVFScalarQuantizerScan(queries,
                  coarseIndices,
                  deviceListDataPointers_,
                  deviceListIndexPointers_,
                  indicesOptions_,
                  deviceListLengths_,
                  maxListLength_,
                  k,
                  l2Distance_,
                  outDistances,
                  outIndices,
                  resources_,
                  gpu_scalar_quantizer_);

   // If the GPU isn't storing indices (they are on the CPU side), we
   // need to perform the re-mapping here
   // FIXME: we might ultimately be calling this function with inputs
   // from the CPU, these are unnecessary copies
   if (indicesOptions_ == INDICES_CPU) {
     HostTensor<long, 2, true> hostOutIndices(outIndices, stream);

     ivfOffsetToUserIndex(hostOutIndices.data(),
                          numLists_,
                          hostOutIndices.getSize(0),
                          hostOutIndices.getSize(1),
                          listOffsetToUserIndex_);

     // Copy back to GPU, since the input to this function is on the
     // GPU
     outIndices.copyFrom(hostOutIndices, stream);
   }
 }

 } } // namespace
