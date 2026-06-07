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
    int warmup = 0;
    int repeats = 1;
};

__host__ __device__ __forceinline__ float producer_value(
    const float* __restrict__ a, const float* __restrict__ b,
    const int* __restrict__ index, int i) {
    int j = index[i];
    int acc = i ^ (j << 1);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        acc = (acc + (j ^ (r * 131))) ^ ((acc << 3) | (acc >> 5));
    }
    return a[j] + 0.5f * b[i] + static_cast<float>(acc & 255) * 0.001f;
}

__host__ __device__ __forceinline__ float consumer_value(
    const float* __restrict__ c, const float* __restrict__ d, int i,
    int iters) {
    float x = c[i];
    float y = d[i];
    float acc = 0.125f + x;
    for (int r = 0; r < iters; ++r) {
        acc = fmaf(acc, 1.0001f, x);
        x = fmaf(x, 0.9997f, y);
        y = fmaf(y, 1.0003f, acc);
    }
    return acc + x + y;
}

extern "C" __global__ void pc_pair_baseline_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ c, const float* __restrict__ d,
    const int* __restrict__ index, float* __restrict__ producer_out,
    float* __restrict__ consumer_out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    producer_out[i] = producer_value(a, b, index, i);
    consumer_out[i] = consumer_value(c, d, i, iters);
}

extern "C" __global__ __launch_bounds__(kBlockThreads, 1)
void pc_pair_split_kernel(const float* __restrict__ a,
                          const float* __restrict__ b,
                          const float* __restrict__ c,
                          const float* __restrict__ d,
                          const int* __restrict__ index,
                          float* __restrict__ producer_out,
                          float* __restrict__ consumer_out, int n,
                          int iters) {
    int warp_id = threadIdx.x / 32;
    bool producer = warp_id < 16;
    int role_tid = producer ? threadIdx.x : threadIdx.x - kProducerThreads;
    int i = blockIdx.x * kProducerThreads + role_tid;
    if (i >= n) return;

    if (producer) {
        producer_out[i] = producer_value(a, b, index, i);
    } else {
        consumer_out[i] = consumer_value(c, d, i, iters);
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
        std::strcmp(opt.mode, "split") != 0) {
        std::fprintf(stderr, "--mode must be baseline or split\n");
        std::exit(EXIT_FAILURE);
    }
    if (opt.n <= 0 || opt.repeats <= 0) {
        std::fprintf(stderr, "Require n>0 and repeats>0\n");
        std::exit(EXIT_FAILURE);
    }
    return opt;
}

void initialize(std::vector<float>& a, std::vector<float>& b,
                std::vector<float>& c, std::vector<float>& d,
                std::vector<int>& index) {
    int n = static_cast<int>(a.size());
    for (int i = 0; i < n; ++i) {
        a[i] = 0.25f + static_cast<float>(i % 1024) * 0.001f;
        b[i] = -0.125f + static_cast<float>(i % 2048) * 0.0004f;
        c[i] = 0.5f + static_cast<float>(i % 4096) * 0.0002f;
        d[i] = -0.75f + static_cast<float>(i % 8192) * 0.0001f;
        index[i] = (i * 17 + 13) % n;
    }
}

void reference(const std::vector<float>& a, const std::vector<float>& b,
               const std::vector<float>& c, const std::vector<float>& d,
               const std::vector<int>& index, std::vector<float>& producer_ref,
               std::vector<float>& consumer_ref, int iters) {
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
        producer_ref[i] = producer_value(a.data(), b.data(), index.data(), i);
        consumer_ref[i] = consumer_value(c.data(), d.data(), i, iters);
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
    std::vector<float> a(opt.n), b(opt.n), c(opt.n), d(opt.n);
    std::vector<int> index(opt.n);
    std::vector<float> producer_ref(opt.n), consumer_ref(opt.n);
    std::vector<float> producer_out(opt.n, 0.0f), consumer_out(opt.n, 0.0f);
    initialize(a, b, c, d, index);
    reference(a, b, c, d, index, producer_ref, consumer_ref, opt.iters);

    float *da, *db, *dc, *dd, *dproducer, *dconsumer;
    int* dindex;
    CHECK_CUDA(cudaMalloc(&da, opt.n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db, opt.n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dc, opt.n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dd, opt.n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dindex, opt.n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dproducer, opt.n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dconsumer, opt.n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(da, a.data(), opt.n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db, b.data(), opt.n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dc, c.data(), opt.n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dd, d.data(), opt.n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dindex, index.data(), opt.n * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dproducer, 0, opt.n * sizeof(float)));
    CHECK_CUDA(cudaMemset(dconsumer, 0, opt.n * sizeof(float)));

    bool split = std::strcmp(opt.mode, "split") == 0;
    int block = kBlockThreads;
    int grid = split ? (opt.n + kProducerThreads - 1) / kProducerThreads
                     : (opt.n + block - 1) / block;

    auto launch = [&]() {
        if (split) {
            pc_pair_split_kernel<<<grid, block>>>(da, db, dc, dd, dindex,
                                                  dproducer, dconsumer, opt.n,
                                                  opt.iters);
        } else {
            pc_pair_baseline_kernel<<<grid, block>>>(da, db, dc, dd, dindex,
                                                     dproducer, dconsumer,
                                                     opt.n, opt.iters);
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
    bool passed = producer_err < 1.0e-4 && consumer_err < 1.0e-4;

    std::printf(
        "%s mode=%s n=%d iters=%d grid=%d block=%d avg_ms=%.6f "
        "producer_max_abs=%.6e consumer_max_abs=%.6e\n",
        passed ? "PASSED" : "FAILED", opt.mode, opt.n, opt.iters, grid, block,
        elapsed_ms / static_cast<float>(opt.repeats), producer_err,
        consumer_err);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(da));
    CHECK_CUDA(cudaFree(db));
    CHECK_CUDA(cudaFree(dc));
    CHECK_CUDA(cudaFree(dd));
    CHECK_CUDA(cudaFree(dindex));
    CHECK_CUDA(cudaFree(dproducer));
    CHECK_CUDA(cudaFree(dconsumer));
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
