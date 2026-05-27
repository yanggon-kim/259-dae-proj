#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t status = (call);                                           \
        if (status != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(status));                          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

static constexpr int kBlockThreads = 1024;
static constexpr int kProducerThreads = 512;

struct Options {
    const char* mode = "baseline";
    int n = 67584;
    int iters = 8;
    int tiles = 64;
    int warmup = 0;
    int repeats = 1;
};

__host__ __device__ __forceinline__ int wrap_index(int value, int n) {
    int wrapped = value % n;
    return wrapped < 0 ? wrapped + n : wrapped;
}

__host__ __device__ __forceinline__ unsigned rotl32(unsigned x, unsigned s) {
    return (x << s) | (x >> (32 - s));
}

__host__ __device__ __forceinline__ float producer_stream_v7(
    const float* __restrict__ a, const float* __restrict__ b,
    const int* __restrict__ index, int i, int n, int iters, int tiles) {
    float out0 = 0.0f;
    float out1 = 0.0f;
    unsigned carry0 = static_cast<unsigned>(i) ^ 0x243f6a88u;
    unsigned carry1 = static_cast<unsigned>(i * 1664525u) ^ 0x85a308d3u;

    for (int t = 0; t < tiles; ++t) {
        int base = i + t * 911;
        int p0 = wrap_index(base, n);
        int p1 = wrap_index(base + 97, n);
        int p2 = wrap_index(base + 193, n);
        int p3 = wrap_index(base + 389, n);

        int j0 = index[p0];
        int j1 = index[p1];
        int j2 = index[p2];
        int j3 = index[p3];

        float l0 = a[j0];
        float l1 = b[j1];
        float l2 = a[j2];
        float l3 = b[j3];

        unsigned x0 = carry0 ^ static_cast<unsigned>(j0 << 1);
        unsigned x1 = carry1 + static_cast<unsigned>(j1);
        unsigned x2 = static_cast<unsigned>(base + 1013904223u) ^
                      static_cast<unsigned>(j2 << 2);
        unsigned x3 = static_cast<unsigned>((base << 4) + 17) ^
                      static_cast<unsigned>(j3);

#pragma unroll 4
        for (int r = 0; r < iters; ++r) {
            x0 = (x0 + 0x9e3779b9u + static_cast<unsigned>(r + t)) ^
                 rotl32(x0, 5);
            x1 = (x1 ^ 0x85ebca6bu) + rotl32(x1, 7) +
                 static_cast<unsigned>(r * 17 + t);
            x2 = (x2 + 0xc2b2ae35u) ^ rotl32(x2, 11) ^
                 static_cast<unsigned>(j0 + t);
            x3 = (x3 ^ (x3 >> 13)) + 0x27d4eb2du +
                 static_cast<unsigned>(j1 + r + t);
        }

        carry0 = x0 ^ x2;
        carry1 = x1 + x3;
        out0 += (l0 + l1) + static_cast<float>(carry0 & 1023u) * 0.000013f;
        out1 += (l2 + l3) + static_cast<float>(carry1 & 1023u) * 0.000017f;
    }

    return (out0 + out1) * (1.0f / static_cast<float>(tiles));
}

#define PC_PAIR_V7_FFMA_STEP()                                                 \
    do {                                                                       \
        a0 = fmaf(a0, 1.00013f, a8);                                           \
        a1 = fmaf(a1, 0.99991f, a9);                                           \
        a2 = fmaf(a2, 1.00031f, a10);                                          \
        a3 = fmaf(a3, 0.99973f, a11);                                          \
        a4 = fmaf(a4, 1.00017f, a12);                                          \
        a5 = fmaf(a5, 0.99983f, a13);                                          \
        a6 = fmaf(a6, 1.00029f, a14);                                          \
        a7 = fmaf(a7, 0.99961f, a15);                                          \
        a8 = fmaf(a8, 0.99987f, 0.03125f);                                     \
        a9 = fmaf(a9, 1.00007f, 0.06250f);                                     \
        a10 = fmaf(a10, 0.99961f, 0.09375f);                                   \
        a11 = fmaf(a11, 1.00019f, 0.12500f);                                   \
        a12 = fmaf(a12, 0.99977f, 0.15625f);                                   \
        a13 = fmaf(a13, 1.00023f, 0.18750f);                                   \
        a14 = fmaf(a14, 0.99969f, 0.21875f);                                   \
        a15 = fmaf(a15, 1.00011f, 0.25000f);                                   \
    } while (0)

#define PC_PAIR_V7_INIT_ACCUMULATORS()                                         \
    float a0 = 0.125000f;                                                      \
    float a1 = 0.250000f;                                                      \
    float a2 = 0.375000f;                                                      \
    float a3 = 0.500000f;                                                      \
    float a4 = 0.625000f;                                                      \
    float a5 = 0.750000f;                                                      \
    float a6 = 0.875000f;                                                      \
    float a7 = 1.000000f;                                                      \
    float a8 = 0.093750f;                                                      \
    float a9 = 0.187500f;                                                      \
    float a10 = 0.281250f;                                                     \
    float a11 = 0.343750f;                                                     \
    float a12 = 0.437500f;                                                     \
    float a13 = 0.531250f;                                                     \
    float a14 = 0.625000f;                                                     \
    float a15 = 0.718750f

__host__ __device__ __forceinline__ float reduce_accumulators_v7(
    float a0, float a1, float a2, float a3, float a4, float a5, float a6,
    float a7, float a8, float a9, float a10, float a11, float a12, float a13,
    float a14, float a15) {
    float s0 = (a0 + a1) + (a2 + a3);
    float s1 = (a4 + a5) + (a6 + a7);
    float s2 = (a8 + a9) + (a10 + a11);
    float s3 = (a12 + a13) + (a14 + a15);
    return (s0 + s1) + 0.125f * (s2 + s3);
}

#define PC_PAIR_V7_H200_TILE(iter_count)                                       \
    do {                                                                       \
        _Pragma("unroll 4")                                                    \
        for (int r = 0; r < (iter_count); ++r) {                               \
            PC_PAIR_V7_FFMA_STEP();                                            \
        }                                                                      \
    } while (0)

#define PC_PAIR_V7_DUAL_TILE()                                                 \
    do {                                                                       \
        PC_PAIR_V7_FFMA_STEP();                                                \
        PC_PAIR_V7_FFMA_STEP();                                                \
        PC_PAIR_V7_FFMA_STEP();                                                \
        PC_PAIR_V7_FFMA_STEP();                                                \
        PC_PAIR_V7_FFMA_STEP();                                                \
        PC_PAIR_V7_FFMA_STEP();                                                \
        PC_PAIR_V7_FFMA_STEP();                                                \
        PC_PAIR_V7_FFMA_STEP();                                                \
    } while (0)

__host__ __device__ __forceinline__ float consumer_stream_h200_v7(int iters,
                                                                  int tiles) {
    PC_PAIR_V7_INIT_ACCUMULATORS();
    for (int t = 0; t < tiles; ++t) {
        PC_PAIR_V7_H200_TILE(iters);
    }
    return reduce_accumulators_v7(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                                  a11, a12, a13, a14, a15);
}

__host__ __device__ __forceinline__ float consumer_stream_dual_v7(int iters,
                                                                  int tiles) {
    PC_PAIR_V7_INIT_ACCUMULATORS();
    for (int t = 0; t < tiles; ++t) {
        PC_PAIR_V7_DUAL_TILE();
    }
    return reduce_accumulators_v7(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                                  a11, a12, a13, a14, a15);
}

__host__ __device__ __forceinline__ float consumer_stream_dual32_v9() {
    PC_PAIR_V7_INIT_ACCUMULATORS();
#pragma unroll 1
    for (int t = 0; t < 32; ++t) {
        PC_PAIR_V7_DUAL_TILE();
    }
    return reduce_accumulators_v7(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
                                  a11, a12, a13, a14, a15);
}

#undef PC_PAIR_V7_INIT_ACCUMULATORS
#undef PC_PAIR_V7_H200_TILE
#undef PC_PAIR_V7_DUAL_TILE
#undef PC_PAIR_V7_FFMA_STEP

extern "C" __global__ void pc_pair_v9_baseline_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    const int* __restrict__ index, float* __restrict__ producer_out,
    float* __restrict__ consumer_out, int n, int iters, int tiles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    producer_out[i] = producer_stream_v7(a, b, index, i, n, iters, tiles);
    consumer_out[i] = consumer_stream_h200_v7(iters, tiles);
}

extern "C" __global__ __launch_bounds__(kBlockThreads, 1)
void pc_pair_v9_split_h200_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  const int* __restrict__ index,
                                  float* __restrict__ producer_out,
                                  float* __restrict__ consumer_out, int n,
                                  int iters, int tiles) {
    if (threadIdx.x < kProducerThreads) {
        int i = blockIdx.x * kProducerThreads + threadIdx.x;
        if (i >= n) return;
        producer_out[i] = producer_stream_v7(a, b, index, i, n, iters, tiles);
    } else {
        float value = consumer_stream_h200_v7(iters, tiles);
        unsigned zero_dep;
        asm volatile("and.b32 %0, %1, 0;" : "=r"(zero_dep)
                                             : "r"(__float_as_uint(value)));
        int role_tid = threadIdx.x - kProducerThreads +
                       static_cast<int>(zero_dep);
        int i = blockIdx.x * kProducerThreads + role_tid;
        if (i < n) consumer_out[i] = value;
    }
}

extern "C" __global__ __launch_bounds__(kBlockThreads, 1)
void pc_pair_v9_stream_dual_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   const int* __restrict__ index,
                                   float* __restrict__ producer_out,
                                   float* __restrict__ consumer_out, int n,
                                   int iters, int tiles) {
    if (threadIdx.x < kProducerThreads) {
        int i = blockIdx.x * kProducerThreads + threadIdx.x;
        if (i >= n) return;
        producer_out[i] = producer_stream_v7(a, b, index, i, n, iters, tiles);
    } else {
        float value = consumer_stream_dual32_v9();
        unsigned zero_dep;
        asm volatile("and.b32 %0, %1, 0;" : "=r"(zero_dep)
                                             : "r"(__float_as_uint(value)));
        int role_tid = threadIdx.x - kProducerThreads +
                       static_cast<int>(zero_dep);
        int i = blockIdx.x * kProducerThreads + role_tid;
        if (i < n) consumer_out[i] = value;
    }
}

int parse_int(const char* text, const char* name) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' ||
        value > std::numeric_limits<int>::max() || value < 0) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value after %s\n", name);
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        if (std::strcmp(argv[i], "--mode") == 0) {
            opt.mode = require_value("--mode");
        } else if (std::strcmp(argv[i], "--n") == 0) {
            opt.n = parse_int(require_value("--n"), "--n");
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opt.iters = parse_int(require_value("--iters"), "--iters");
        } else if (std::strcmp(argv[i], "--tiles") == 0) {
            opt.tiles = parse_int(require_value("--tiles"), "--tiles");
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            opt.warmup = parse_int(require_value("--warmup"), "--warmup");
        } else if (std::strcmp(argv[i], "--repeats") == 0) {
            opt.repeats = parse_int(require_value("--repeats"), "--repeats");
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (std::strcmp(opt.mode, "baseline") != 0 &&
        std::strcmp(opt.mode, "split_h200") != 0 &&
        std::strcmp(opt.mode, "stream_dual") != 0) {
        std::fprintf(stderr,
                     "--mode must be baseline, split_h200, or stream_dual\n");
        std::exit(EXIT_FAILURE);
    }
    if (opt.n <= 0 || opt.iters <= 0 || opt.tiles <= 0 || opt.repeats <= 0) {
        std::fprintf(stderr, "Require n>0, iters>0, tiles>0, and repeats>0\n");
        std::exit(EXIT_FAILURE);
    }
    if (std::strcmp(opt.mode, "stream_dual") == 0 &&
        (opt.iters != 8 || opt.tiles != 32)) {
        std::fprintf(stderr,
                     "stream_dual specializes the v9 FFMA stream for --iters "
                     "8 --tiles 32\n");
        std::exit(EXIT_FAILURE);
    }
    return opt;
}

void initialize(std::vector<float>& a, std::vector<float>& b,
                std::vector<int>& index) {
    int n = static_cast<int>(a.size());
    for (int i = 0; i < n; ++i) {
        a[i] = 0.125f + static_cast<float>(i % 2048) * 0.00031f;
        b[i] = -0.375f + static_cast<float>(i % 4096) * 0.00019f;
        index[i] = (i * 67 + 29) % n;
    }
}

void reference(const std::vector<float>& a, const std::vector<float>& b,
               const std::vector<int>& index, std::vector<float>& producer_ref,
               std::vector<float>& consumer_ref, int iters, int tiles) {
    int n = static_cast<int>(a.size());
    for (int i = 0; i < n; ++i) {
        producer_ref[i] = producer_stream_v7(a.data(), b.data(), index.data(),
                                             i, n, iters, tiles);
        consumer_ref[i] = consumer_stream_h200_v7(iters, tiles);
    }
}

double max_abs_error(const std::vector<float>& expected,
                     const std::vector<float>& got) {
    double max_err = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        max_err = std::max(max_err, std::abs(static_cast<double>(expected[i]) -
                                             static_cast<double>(got[i])));
    }
    return max_err;
}

int main(int argc, char** argv) {
    Options opt = parse_options(argc, argv);
    std::vector<float> a(opt.n), b(opt.n);
    std::vector<int> index(opt.n);
    std::vector<float> producer_ref(opt.n), consumer_ref(opt.n);
    std::vector<float> producer_out(opt.n, 0.0f), consumer_out(opt.n, 0.0f);
    initialize(a, b, index);
    reference(a, b, index, producer_ref, consumer_ref, opt.iters, opt.tiles);

    float *da, *db, *dproducer, *dconsumer;
    int* dindex;
    CHECK_CUDA(cudaMalloc(&da, opt.n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db, opt.n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dindex, opt.n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dproducer, opt.n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dconsumer, opt.n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(da, a.data(), opt.n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db, b.data(), opt.n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dindex, index.data(), opt.n * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dproducer, 0, opt.n * sizeof(float)));
    CHECK_CUDA(cudaMemset(dconsumer, 0, opt.n * sizeof(float)));

    bool split = std::strcmp(opt.mode, "baseline") != 0;
    bool stream_dual = std::strcmp(opt.mode, "stream_dual") == 0;
    int block = kBlockThreads;
    int grid = split ? (opt.n + kProducerThreads - 1) / kProducerThreads
                     : (opt.n + block - 1) / block;

    auto launch = [&]() {
        if (stream_dual) {
            pc_pair_v9_stream_dual_kernel<<<grid, block>>>(
                da, db, dindex, dproducer, dconsumer, opt.n, opt.iters,
                opt.tiles);
        } else if (split) {
            pc_pair_v9_split_h200_kernel<<<grid, block>>>(
                da, db, dindex, dproducer, dconsumer, opt.n, opt.iters,
                opt.tiles);
        } else {
            pc_pair_v9_baseline_kernel<<<grid, block>>>(
                da, db, dindex, dproducer, dconsumer, opt.n, opt.iters,
                opt.tiles);
        }
    };

    for (int i = 0; i < opt.warmup; ++i) {
        launch();
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < opt.repeats; ++i) {
        launch();
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(producer_out.data(), dproducer,
                          opt.n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(consumer_out.data(), dconsumer,
                          opt.n * sizeof(float), cudaMemcpyDeviceToHost));
    double producer_err = max_abs_error(producer_ref, producer_out);
    double consumer_err = max_abs_error(consumer_ref, consumer_out);
    bool passed = producer_err < 1.0e-4 && consumer_err < 5.0e-1;

    std::printf(
        "%s mode=%s n=%d iters=%d tiles=%d grid=%d block=%d avg_ms=%.6f "
        "producer_max_abs=%.6e consumer_max_abs=%.6e\n",
        passed ? "PASSED" : "FAILED", opt.mode, opt.n, opt.iters, opt.tiles,
        grid, block, elapsed_ms / static_cast<float>(opt.repeats),
        producer_err, consumer_err);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(da));
    CHECK_CUDA(cudaFree(db));
    CHECK_CUDA(cudaFree(dindex));
    CHECK_CUDA(cudaFree(dproducer));
    CHECK_CUDA(cudaFree(dconsumer));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
