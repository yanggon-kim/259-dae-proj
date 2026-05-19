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

struct Options {
    size_t n = 1032192ull;
    int compute_iters = 16;
    int memory_iters = 4;
    int repeats = 20;
    int warmup = 5;
};

__host__ __device__ __forceinline__ float stream_fma_value(float acc, float x,
                                                           float y,
                                                           int compute_iters) {
    for (int r = 0; r < compute_iters; ++r) {
        acc = fmaf(acc, 1.0001f, x);
        x = fmaf(x, 0.9999f, y);
        y = fmaf(y, 1.0003f, acc);
    }
    return acc + x + y;
}

template <int MemoryIters>
__device__ __forceinline__ void stream_fma_v6_kernel_impl(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ c, float* __restrict__ out, size_t n,
    int compute_iters) {
    static_assert(MemoryIters == 1 || MemoryIters == 2 || MemoryIters == 4 ||
                      MemoryIters == 8,
                  "Unsupported MemoryIters");

    size_t block_base = static_cast<size_t>(blockIdx.x) * blockDim.x *
                        static_cast<size_t>(MemoryIters);
    size_t lane_index = static_cast<size_t>(threadIdx.x);
    size_t idx[MemoryIters];
    float acc[MemoryIters];
    float x[MemoryIters];
    float y[MemoryIters];

    // Full-tile benchmark contract:
    // n = k * blockDim.x * MemoryIters, where k is a positive integer.
    // The host enforces this before launch, so all idx[m] values are valid.
    // On the RTX 5080 experiments with 256 threads/CTA and n=1032192:
    //   MemoryIters=1: CTAs=4032, one element/thread
    //   MemoryIters=2: CTAs=2016, two elements/thread
    //   MemoryIters=4: CTAs=1008, four elements/thread
    //   MemoryIters=8: CTAs=504, eight elements/thread
#pragma unroll
    for (int m = 0; m < MemoryIters; ++m) {
        idx[m] = block_base + static_cast<size_t>(m) * blockDim.x + lane_index;
        acc[m] = a[idx[m]];
        x[m] = b[idx[m]];
        y[m] = c[idx[m]];
    }

    for (int r = 0; r < compute_iters; ++r) {
#pragma unroll
        for (int m = 0; m < MemoryIters; ++m) {
            acc[m] = fmaf(acc[m], 1.0001f, x[m]);
            x[m] = fmaf(x[m], 0.9999f, y[m]);
            y[m] = fmaf(y[m], 1.0003f, acc[m]);
        }
    }

#pragma unroll
    for (int m = 0; m < MemoryIters; ++m) {
        out[idx[m]] = acc[m] + x[m] + y[m];
    }
}

extern "C" __global__ void stream_fma_v6_m1_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ c, float* __restrict__ out, size_t n,
    int compute_iters) {
    stream_fma_v6_kernel_impl<1>(a, b, c, out, n, compute_iters);
}

extern "C" __global__ void stream_fma_v6_m2_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ c, float* __restrict__ out, size_t n,
    int compute_iters) {
    stream_fma_v6_kernel_impl<2>(a, b, c, out, n, compute_iters);
}

extern "C" __global__ void stream_fma_v6_m4_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ c, float* __restrict__ out, size_t n,
    int compute_iters) {
    stream_fma_v6_kernel_impl<4>(a, b, c, out, n, compute_iters);
}

extern "C" __global__ void stream_fma_v6_m8_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ c, float* __restrict__ out, size_t n,
    int compute_iters) {
    stream_fma_v6_kernel_impl<8>(a, b, c, out, n, compute_iters);
}

void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s [options]\n"
        "  --n <elements>          Number of elements (default: 1032192).\n"
        "                          Must be divisible by 256*memory_iters.\n"
        "  --iters <count>         FMA loop iterations (default: 16)\n"
        "  --memory-iters <count>  Elements/thread: 1,2,4,8 (default: 4)\n"
        "  --warmup <count>        Warmup kernel launches (default: 5)\n"
        "  --repeats <count>       Timed kernel launches (default: 20)\n"
        "  --help                  Show this message\n",
        argv0);
}

size_t parse_size(const char* text, const char* name) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        std::fprintf(stderr, "Invalid value for %s: %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<size_t>(value);
}

int parse_int(const char* text, const char* name) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' ||
        value > std::numeric_limits<int>::max() ||
        value < std::numeric_limits<int>::min()) {
        std::fprintf(stderr, "Invalid value for %s: %s\n", name, text);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(value);
}

bool is_supported_memory_iters(int value) {
    return value == 1 || value == 2 || value == 4 || value == 8;
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

        if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (std::strcmp(argv[i], "--n") == 0) {
            opt.n = parse_size(require_value("--n"), "--n");
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opt.compute_iters = parse_int(require_value("--iters"), "--iters");
        } else if (std::strcmp(argv[i], "--memory-iters") == 0 ||
                   std::strcmp(argv[i], "--mem-iters") == 0) {
            opt.memory_iters =
                parse_int(require_value("--memory-iters"), "--memory-iters");
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            opt.warmup = parse_int(require_value("--warmup"), "--warmup");
        } else if (std::strcmp(argv[i], "--repeats") == 0) {
            opt.repeats = parse_int(require_value("--repeats"), "--repeats");
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (opt.n == 0 || opt.compute_iters < 0 || opt.repeats <= 0 ||
        opt.warmup < 0 || !is_supported_memory_iters(opt.memory_iters)) {
        std::fprintf(stderr,
                     "Require n>0, repeats>0, warmup>=0, iters>=0, and "
                     "--memory-iters in {1,2,4,8}.\n");
        std::exit(EXIT_FAILURE);
    }
    return opt;
}

void initialize_inputs(std::vector<float>& a, std::vector<float>& b,
                       std::vector<float>& c) {
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = 0.1f + static_cast<float>(i % 1024) * 0.001f;
        b[i] = -0.2f + static_cast<float>(i % 2048) * 0.0005f;
        c[i] = 0.3f + static_cast<float>(i % 4096) * 0.00025f;
    }
}

void compute_reference(const std::vector<float>& a,
                       const std::vector<float>& b,
                       const std::vector<float>& c, std::vector<float>& ref,
                       int compute_iters) {
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] = stream_fma_value(a[i], b[i], c[i], compute_iters);
    }
}

struct ErrorStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
};

ErrorStats compare_outputs(const std::vector<float>& ref,
                           const std::vector<float>& got) {
    ErrorStats stats;
    for (size_t i = 0; i < ref.size(); ++i) {
        double abs_err = std::abs(static_cast<double>(ref[i]) - got[i]);
        double denom = std::max(1.0, std::abs(static_cast<double>(ref[i])));
        stats.max_abs = std::max(stats.max_abs, abs_err);
        stats.max_rel = std::max(stats.max_rel, abs_err / denom);
    }
    return stats;
}

template <typename Launcher>
float time_kernel_ms(Launcher launch, int warmup, int repeats) {
    for (int i = 0; i < warmup; ++i) {
        launch();
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        launch();
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(repeats);
}

void print_result(float ms, size_t n, int compute_iters,
                  const ErrorStats& err) {
    double seconds = static_cast<double>(ms) * 1.0e-3;
    double global_bytes = static_cast<double>(n) * 4.0 * sizeof(float);
    double flops = static_cast<double>(n) *
                   (6.0 * static_cast<double>(compute_iters) + 2.0);
    double gbps = global_bytes / seconds / 1.0e9;
    double gflops = flops / seconds / 1.0e9;
    double ai = flops / global_bytes;

    std::printf("stream_fma_v6 time_ms=%8.4f  global_GB/s=%8.2f  "
                "est_GFLOP/s=%8.2f  AI=%6.3f  max_abs=%9.3e  "
                "max_rel=%9.3e\n",
                ms, gbps, gflops, ai, err.max_abs, err.max_rel);
}

int main(int argc, char** argv) {
    Options opt = parse_options(argc, argv);

    constexpr int threads = 256;
    size_t elements_per_cta =
        static_cast<size_t>(threads) * static_cast<size_t>(opt.memory_iters);
    if (opt.n % elements_per_cta != 0) {
        std::fprintf(stderr,
                     "Unsafe n=%zu for full-tile stream_fma_v6. Require n to "
                     "be a multiple of threads*memory_iters=%zu "
                     "(threads=%d, memory_iters=%d).\n",
                     opt.n, elements_per_cta, threads, opt.memory_iters);
        return EXIT_FAILURE;
    }
    size_t blocks_size = opt.n / elements_per_cta;
    if (blocks_size == 0 ||
        blocks_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "Invalid CTA count: %zu\n", blocks_size);
        return EXIT_FAILURE;
    }
    int blocks = static_cast<int>(blocks_size);

    int device = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    float* d_out = nullptr;
    std::vector<float> h_a(opt.n);
    std::vector<float> h_b(opt.n);
    std::vector<float> h_c(opt.n);
    std::vector<float> h_ref(opt.n);
    std::vector<float> h_out(opt.n, 0.0f);
    initialize_inputs(h_a, h_b, h_c);
    compute_reference(h_a, h_b, h_c, h_ref, opt.compute_iters);

    size_t bytes = opt.n * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_out, 0, bytes));

    std::printf("device=%s  cc=%d.%d  n=%zu  iters=%d  memory_iters=%d  "
                "repeats=%d  warmup=%d\n",
                prop.name, prop.major, prop.minor, opt.n, opt.compute_iters,
                opt.memory_iters, opt.repeats, opt.warmup);
    std::printf("CTAs=%d  threads/CTA=%d  elements/CTA=%zu  "
                "elements/thread=%d  mapping=cta_tile_no_rounds\n",
                blocks, threads, elements_per_cta, opt.memory_iters);

    auto launch = [&]() {
        switch (opt.memory_iters) {
            case 1:
                stream_fma_v6_m1_kernel<<<blocks, threads>>>(
                    d_a, d_b, d_c, d_out, opt.n, opt.compute_iters);
                break;
            case 2:
                stream_fma_v6_m2_kernel<<<blocks, threads>>>(
                    d_a, d_b, d_c, d_out, opt.n, opt.compute_iters);
                break;
            case 4:
                stream_fma_v6_m4_kernel<<<blocks, threads>>>(
                    d_a, d_b, d_c, d_out, opt.n, opt.compute_iters);
                break;
            case 8:
                stream_fma_v6_m8_kernel<<<blocks, threads>>>(
                    d_a, d_b, d_c, d_out, opt.n, opt.compute_iters);
                break;
        }
    };

    float ms = time_kernel_ms(launch, opt.warmup, opt.repeats);

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    ErrorStats err = compare_outputs(h_ref, h_out);
    print_result(ms, opt.n, opt.compute_iters, err);

    bool ok = err.max_abs <= 1.0e-4 || err.max_rel <= 1.0e-4;
    std::printf("verification=%s\n", ok ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_out));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
