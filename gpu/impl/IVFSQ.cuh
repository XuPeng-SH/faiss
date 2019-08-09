#pragma once

#include "IVFBase.cuh"
#include "IVFSQScan.cuh"
 
namespace faiss { namespace gpu {
 
class IVFSQ : public IVFBase {
public:
    using QueryT = Tensor<float, 2, true>;
    using OutDistanceT = Tensor<float, 2, true>;
    using OutIndexT = Tensor<long, 2, true>;
    using CoarseQuantizerT = FlatIndex;
    using VecT = uint8_t;
   /// Construct from a quantizer that has elemen
   IVFSQ(GpuResources* resources,
           CoarseQuantizerT* quantizer,
           int code_size,
           bool l2Distance,
           IndicesOptions indicesOptions,
           MemorySpace space,
           float vdiff,
           float vmin
           );
 
   ~IVFSQ() override;
 
   void addCodeVectorsFromCpu(int listId, const VecT* vecs, const long* indices, size_t numVecs);
 
   void addTrainedDataFromCpu(const uint8_t* trained, size_t numData);

   std::vector<uint8_t> getListVectors(int listId) const;
 
   void query(Tensor<float, 2, true>& queries,
              int nprobe,
              int k,
              Tensor<float, 2, true>& outDistances,
              Tensor<long, 2, true>& outIndices);
 
  private:
   const bool l2Distance_;

   GpuScalarQuantizer gpu_scalar_quantizer_;


 };
 
 } } // namespace
 