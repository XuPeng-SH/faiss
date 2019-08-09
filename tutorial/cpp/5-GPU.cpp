/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <unistd.h>

#include <iostream>

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuAutoTune.h>


#include "faiss/FaissAssert.h"
#include "faiss/AuxIndexStructures.h"

#include "faiss/IndexFlat.h"
#include "faiss/VectorTransform.h"
#include "faiss/IndexLSH.h"
#include "faiss/IndexPQ.h"
#include "faiss/IndexIVF.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexIVFSpectralHash.h"
#include "faiss/MetaIndexes.h"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/IndexHNSW.h"
#include "faiss/OnDiskInvertedLists.h"
#include "faiss/IndexBinaryFlat.h"
#include "faiss/IndexBinaryFromFloat.h"
#include "faiss/IndexBinaryHNSW.h"
#include "faiss/IndexBinaryIVF.h"
#include "faiss/gpu/GpuIndexIVFSQ.h"

#if 0
static uint32_t fourcc (const char sx[4]) {
    assert(4 == strlen(sx));
    const unsigned char *x = (unsigned char*)sx;
    return x[0] | x[1] << 8 | x[2] << 16 | x[3] << 24;
}

#define WRITEANDCHECK(ptr, n) {                                 \
        size_t ret = (*f)(ptr, sizeof(*(ptr)), n);              \
        FAISS_THROW_IF_NOT_FMT(ret == (n),                      \
            "write error in %s: %ld != %ld (%s)",               \
            f->name.c_str(), ret, size_t(n), strerror(errno));  \
    }

#define READANDCHECK(ptr, n) {                                  \
        size_t ret = (*f)(ptr, sizeof(*(ptr)), n);              \
        FAISS_THROW_IF_NOT_FMT(ret == (n),                      \
            "read error in %s: %ld != %ld (%s)",                \
            f->name.c_str(), ret, size_t(n), strerror(errno));  \
    }

#define WRITE1(x) WRITEANDCHECK(&(x), 1)
#define READ1(x)  READANDCHECK(&(x), 1)

#define WRITEVECTOR(vec) {                      \
        size_t size = (vec).size ();            \
        WRITEANDCHECK (&size, 1);               \
        WRITEANDCHECK ((vec).data (), size);    \
    }

// will fail if we write 256G of data at once...
#define READVECTOR(vec) {                       \
        long size;                            \
        READANDCHECK (&size, 1);                \
        FAISS_THROW_IF_NOT (size >= 0 && size < (1L << 40));  \
        (vec).resize (size);                    \
        READANDCHECK ((vec).data (), size);     \
    }

struct ScopeFileCloser {
    FILE *f;
    ScopeFileCloser (FILE *f): f (f) {}
    ~ScopeFileCloser () {fclose (f); }
};

struct FileIOReader: faiss::IOReader {
    FILE *f = nullptr;
    bool need_close = false;

    FileIOReader(FILE *rf): f(rf) {}

    FileIOReader(const char * fname)
    {
        name = fname;
        f = fopen(fname, "rb");
        FAISS_THROW_IF_NOT_FMT (
             f, "could not open %s for reading: %s",
             fname, strerror(errno));
        need_close = true;
    }

    ~FileIOReader() override {
        if (need_close) {
            int ret = fclose(f);
            if (ret != 0) {// we cannot raise and exception in the destructor
                fprintf(stderr, "file %s close error: %s",
                        name.c_str(), strerror(errno));
            }
        }
    }

    size_t operator()(
            void *ptr, size_t size, size_t nitems) override {
        return fread(ptr, size, nitems, f);
    }

    int fileno() override {
        return ::fileno (f);
    }

};

struct FileIOWriter: faiss::IOWriter {
    FILE *f = nullptr;
    bool need_close = false;

    FileIOWriter(FILE *wf): f(wf) {}

    FileIOWriter(const char * fname)
    {
        name = fname;
        f = fopen(fname, "wb");
        FAISS_THROW_IF_NOT_FMT (
             f, "could not open %s for writing: %s",
             fname, strerror(errno));
        need_close = true;
    }

    ~FileIOWriter() override {
        if (need_close) {
            int ret = fclose(f);
            if (ret != 0) {
                // we cannot raise and exception in the destructor
                fprintf(stderr, "file %s close error: %s",
                        name.c_str(), strerror(errno));
            }
        }
    }

    size_t operator()(
            const void *ptr, size_t size, size_t nitems) override {
        return fwrite(ptr, size, nitems, f);
    }
    int fileno() override {
        return ::fileno (f);
    }

};

static void write_index_header (const faiss::Index *idx, faiss::IOWriter *f) {
    WRITE1 (idx->d);
    WRITE1 (idx->ntotal);
    faiss::Index::idx_t dummy = 1 << 20;
    WRITE1 (dummy);
    WRITE1 (dummy);
    WRITE1 (idx->is_trained);
    WRITE1 (idx->metric_type);
    if (idx->metric_type > 1) {
        WRITE1 (idx->metric_arg);
    }
}

static void write_ivf_header (const faiss::IndexIVF *ivf, faiss::IOWriter *f) {
    write_index_header (ivf, f);
    WRITE1 (ivf->nlist);
    WRITE1 (ivf->nprobe);
    write_index (ivf->quantizer, f);
    WRITE1 (ivf->maintain_direct_map);
    WRITEVECTOR (ivf->direct_map);
}

static void write_ScalarQuantizer (
        const faiss::ScalarQuantizer *ivsc, faiss::IOWriter *f) {
    WRITE1 (ivsc->qtype);
    WRITE1 (ivsc->rangestat);
    WRITE1 (ivsc->rangestat_arg);
    WRITE1 (ivsc->d);
    WRITE1 (ivsc->code_size);
    WRITEVECTOR (ivsc->trained);
}

faiss::Index *read_header (const char *fname, int io_flags = 0) {
    FileIOReader reader(fname);
    FileIOReader* f = &reader;
    uint32_t h;
    READ1 (h);
    if(h == fourcc ("IwFl")) {
        printf("This is a IVFFlat index\n");
    } else if(h == fourcc ("IwSQ") || h == fourcc ("IwSq")) {
        printf("This is a IVFFlatSQ index\n");
    }
    return nullptr;
}

faiss::Index *read_data (const char *fname, int io_flags = 0) {
    FileIOReader reader(fname);
    FileIOReader* f = &reader;
    uint32_t h;
    READ1 (h);
    if(h == fourcc ("IwFl")) {
        printf("This is a IVFFlat index\n");
    } else if(h == fourcc ("IwSQ") || h == fourcc ("IwSq")) {
        printf("This is a IVFFlatSQ index\n");
    }
    return nullptr;
}

void write_header(const faiss::Index *idx, const char *fname) {
    FileIOWriter writer(fname);
    FileIOWriter* f = &writer;

    if(const faiss::IndexIVFFlat * ivfl =
              dynamic_cast<const faiss::IndexIVFFlat *> (idx)) {
        printf("Write IndexIVFFlat index header\n");
        uint32_t h = fourcc ("IwFl");
        WRITE1 (h);
        write_ivf_header (ivfl, f);
        // write_InvertedLists (ivfl->invlists, f);
    } else if(const faiss::IndexIVFScalarQuantizer * ivsc =
              dynamic_cast<const faiss::IndexIVFScalarQuantizer *> (idx)) {
        printf("Write IndexIVFSQ index header\n");
        uint32_t h = fourcc ("IwSq");
        WRITE1 (h);
        write_ivf_header (ivsc, f);
        write_ScalarQuantizer (&ivsc->sq, f);
        WRITE1 (ivsc->code_size);
        WRITE1 (ivsc->by_residual);
        write_InvertedLists (ivsc->invlists, f);
    } else {
        FAISS_THROW_MSG ("don't know how to serialize this type of index");
    }
    return ;
}

void write_data(const faiss::Index* idx, const char *fname) {
    FileIOWriter writer(fname);
    FileIOWriter* f = &writer;

    if(const faiss::IndexIVFFlat * ivfl =
              dynamic_cast<const faiss::IndexIVFFlat *> (idx)) {
        uint32_t h = fourcc ("IwFl");
        WRITE1 (h);
        write_ivf_header (ivfl, f);
        write_InvertedLists (ivfl->invlists, f);
    } else if(const faiss::IndexIVFScalarQuantizer * ivsc =
              dynamic_cast<const faiss::IndexIVFScalarQuantizer *> (idx)) {
        uint32_t h = fourcc ("IwSq");
        WRITE1 (h);
        write_ivf_header (ivsc, f);
        write_ScalarQuantizer (&ivsc->sq, f);
        WRITE1 (ivsc->code_size);
        WRITE1 (ivsc->by_residual);
        write_InvertedLists (ivsc->invlists, f);
    } else {
        FAISS_THROW_MSG ("don't know how to serialize this type of index");
    }
    return ;
}

#endif

int main() {
    int d = 8;                            // dimension
    int nq = 4;                        // nb of queries
    float *xq = new float[d * nq];
    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++) {
            xq[d * i + j] = drand48();
            printf("%lf ", xq[d * i + j]);
        }
        xq[d * i] += i / 1000.;
        printf("\n");
    }

    faiss::gpu::StandardGpuResources res;

    int k = 4;
    std::shared_ptr<faiss::Index> gpu_index_ivf_ptr;

    const char* index_description = "IVF4,SQ8";
    // const char* index_description = "IVF3276,Flat";
//    Index *index_factory (int d, const char *description,
//                          MetricType metric = METRIC_L2);

    faiss::Index *cpu_index = nullptr;
    if((access("index.index",F_OK))==-1) {
        // create database
        int nb = 156;                       // database size
        printf("-----------------------\n");
        float *xb = new float[d * nb];
        for(int i = 0; i < nb; i++) {
            for(int j = 0; j < d; j++) {
                xb[d * i + j] = drand48();
                printf("%lf ", xb[d * i + j]);
            }
            xb[d * i] += i / 1000.;
            printf("\n");
        }
        // Using an IVF index
        // here we specify METRIC_L2, by default it performs inner-product search

        faiss::Index *ori_index = faiss::index_factory(d, index_description, faiss::METRIC_L2);
        auto device_index = faiss::gpu::index_cpu_to_gpu(&res, 0, ori_index);

        gpu_index_ivf_ptr = std::shared_ptr<faiss::Index>(device_index);

        assert(!device_index->is_trained);
        device_index->train(nb, xb);
        assert(device_index->is_trained);
        device_index->add(nb, xb);  // add vectors to the index

        printf("is_trained = %s\n", device_index->is_trained ? "true" : "false");
        printf("ntotal = %ld\n", device_index->ntotal);

        cpu_index = faiss::gpu::index_gpu_to_cpu ((device_index));
        faiss::write_index(cpu_index, "index.index");
        printf("index.index is stored successfully.\n");
        delete [] xb;
    } else {
        cpu_index = faiss::read_index("index.index");
    }

    {
        // cpu to gpu
        faiss::gpu::CpuToGpuClonerOptions option;
        option.readonly = true;
        faiss::Index* tmp_index = faiss::gpu::cpu_to_gpu(&res, 0, cpu_index, &option);

        gpu_index_ivf_ptr = std::shared_ptr<faiss::Index>(tmp_index);

        // Gpu index dump
        auto gpu_index_ivf_sq_ptr = dynamic_cast<faiss::gpu::GpuIndexIVFSQ*>(tmp_index);
        gpu_index_ivf_sq_ptr->dump();

        // Cpu index dump
        auto cpu_index_ivf_sq_ptr = dynamic_cast<faiss::IndexIVF*>(cpu_index);
        cpu_index_ivf_sq_ptr->dump();
    }


    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        gpu_index_ivf_ptr->search(nq, xq, k, D, I);

        // print results
        printf("I (4 first results)=\n");
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (4 last results)=\n");
        for(int i = nq - 4; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }
    printf("----------------------------------\n");
    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        cpu_index->search(nq, xq, k, D, I);

        // print results
        printf("I (4 first results)=\n");
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (4 last results)=\n");
        for(int i = nq - 4; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }


    delete [] xq;
    return 0;
}
