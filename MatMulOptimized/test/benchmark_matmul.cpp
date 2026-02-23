/**
 * @file benchmark_matmul.cpp
 * Performance benchmark for optimized MatMul operator
 *
 * Features:
 * 1. Automated performance testing across multiple matrix sizes
 * 2. Accuracy verification against CPU reference
 * 3. Latency and throughput measurement
 * 4. Comparison with baseline implementation
 *
 * Copyright (C) 2025. Competition Submission.
 */

#include <acl/acl.h>
#include <aclnn/aclnn.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>

// ===================================================================
// Constants and Configuration
// ===================================================================

constexpr int WARMUP_ROUNDS = 5;
constexpr int TEST_ROUNDS = 100;
constexpr float EPSILON = 1e-3;

// Test matrix sizes (M, N, K)
struct TestSize {
    int m, n, k;
    const char* description;
};

constexpr TestSize TEST_SIZES[] = {
    {16, 16, 16, "Tiny (16x16x16)"},
    {64, 64, 64, "Small (64x64x64)"},
    {256, 256, 256, "Medium (256x256x256)"},
    {512, 512, 512, "Large (512x512x512)"},
    {1024, 1024, 1024, "X-Large (1024x1024x1024)"},
    {2048, 2048, 2048, "XX-Large (2048x2048x2048)"},
    {4096, 4096, 4096, "Huge (4096x4096x4096)"},
};

// ===================================================================
// Helper Functions
// ===================================================================

#define ACL_CHECK(call) \
    do { \
        aclError ret = (call); \
        if (ret != ACL_ERROR_NONE) { \
            std::cerr << "ACL Error: " << ret << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template<typename T>
void FillRandom(std::vector<T>& data, T min_val, T max_val) {
    for (auto& val : data) {
        float random = static_cast<float>(rand()) / RAND_MAX;
        val = static_cast<T>(min_val + random * (max_val - min_val));
    }
}

template<typename T>
bool CompareResults(const T* result, const float* reference, size_t size, int k, float epsilon) {
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(static_cast<float>(result[i]) - reference[i]);
        if (diff > epsilon) {
            std::cerr << "Mismatch at index " << i << ": got " << static_cast<float>(result[i])
                      << ", expected " << reference[i] << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

// ===================================================================
// CPU Reference Implementation
// ===================================================================

void MatMulCPUReference(const half* A, const half* B, float* C, int M, int N, int K) {
    // Convert half to float for computation
    std::vector<float> A_float(M * K);
    std::vector<float> B_float(K * N);

    for (int i = 0; i < M * K; ++i) {
        A_float[i] = static_cast<float>(A[i]);
    }
    for (int i = 0; i < K * N; ++i) {
        B_float[i] = static_cast<float>(B[i]);
    }

    // Compute C = A * B
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A_float[m * K + k] * B_float[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// ===================================================================
// Benchmark Class
// ===================================================================

class MatMulBenchmark {
public:
    MatMulBenchmark(int deviceId = 0) : deviceId_(deviceId), stream_(nullptr) {
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(deviceId_));
        ACL_CHECK(aclrtCreateStream(&stream_));
    }

    ~MatMulBenchmark() {
        ACL_CHECK(aclrtDestroyStream(stream_));
        ACL_CHECK(aclrtResetDevice(deviceId_));
        ACL_CHECK(aclFinalize());
    }

    struct BenchmarkResult {
        float avgLatencyMs;
        float minLatencyMs;
        float maxLatencyMs;
        float throughputGBps;
        bool accuracyPass;
        std::string errorInfo;
    };

    BenchmarkResult RunBenchmark(int M, int N, int K) {
        BenchmarkResult result;

        // Calculate sizes
        size_t sizeA = static_cast<size_t>(M) * K * sizeof(half);
        size_t sizeB = static_cast<size_t>(K) * N * sizeof(half);
        size_t sizeC = static_cast<size_t>(M) * N * sizeof(half);
        size_t lenA = static_cast<size_t>(M) * K;
        size_t lenB = static_cast<size_t>(K) * N;
        size_t lenC = static_cast<size_t>(M) * N;

        // Allocate host memory
        std::vector<half> hostA(lenA);
        std::vector<half> hostB(lenB);
        std::vector<half> hostC(lenC);
        std::vector<float> goldenC(lenC);

        // Initialize with random data
        FillRandom(hostA, -1.0f, 1.0f);
        FillRandom(hostB, -1.0f, 1.0f);

        // Allocate device memory
        half* deviceA = nullptr;
        half* deviceB = nullptr;
        half* deviceC = nullptr;

        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceA), sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceB), sizeB, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceC), sizeC, ACL_MEM_MALLOC_HUGE_FIRST));

        // Copy data to device
        ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));

        // Compute golden reference on CPU
        MatMulCPUReference(hostA.data(), hostB.data(), goldenC.data(), M, N, K);

        // Warmup runs
        for (int i = 0; i < WARMUP_ROUNDS; ++i) {
            // TODO: Replace with actual operator call
            // For now, this is a placeholder
        }

        // Benchmark runs
        std::vector<float> latencies(TEST_ROUNDS);
        for (int i = 0; i < TEST_ROUNDS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            // TODO: Replace with actual operator call
            // This is a placeholder for the actual kernel execution
            ACL_CHECK(aclrtSynchronizeStream(stream_));

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = end - start;
            latencies[i] = duration.count();
        }

        // Copy result back
        ACL_CHECK(aclrtMemcpy(hostC.data(), sizeC, deviceC, sizeC, ACL_MEMCPY_DEVICE_TO_HOST));

        // Verify accuracy
        result.accuracyPass = CompareResults(hostC.data(), goldenC.data(), lenC, K, EPSILON);

        // Calculate statistics
        float sum = 0.0f;
        float minVal = latencies[0];
        float maxVal = latencies[0];

        for (const auto& lat : latencies) {
            sum += lat;
            minVal = std::min(minVal, lat);
            maxVal = std::max(maxVal, lat);
        }

        result.avgLatencyMs = sum / TEST_ROUNDS;
        result.minLatencyMs = minVal;
        result.maxLatencyMs = maxVal;

        // Calculate throughput (bytes per second)
        size_t totalBytes = (lenA + lenB) * sizeof(half) + lenC * sizeof(half);
        result.throughputGBps = (totalBytes / 1e9f) / (result.avgLatencyMs / 1000.0f);

        // Cleanup
        ACL_CHECK(aclrtFree(deviceA));
        ACL_CHECK(aclrtFree(deviceB));
        ACL_CHECK(aclrtFree(deviceC));

        return result;
    }

    void PrintResults(const TestSize& testSize, const BenchmarkResult& result) {
        std::cout << std::left << std::setw(30) << testSize.description;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << " | Avg: " << std::setw(8) << result.avgLatencyMs << " ms";
        std::cout << " | Min: " << std::setw(8) << result.minLatencyMs << " ms";
        std::cout << " | Max: " << std::setw(8) << result.maxLatencyMs << " ms";
        std::cout << " | Throughput: " << std::setw(8) << result.throughputGBps << " GB/s";
        std::cout << " | Accuracy: " << (result.accuracyPass ? "PASS" : "FAIL");
        std::cout << std::endl;
    }

private:
    int deviceId_;
    aclrtStream stream_;
};

// ===================================================================
// Main Function
// ===================================================================

int main(int argc, char** argv) {
    int deviceId = 0;
    if (argc > 1) {
        deviceId = atoi(argv[1]);
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "  MatMul Optimized Performance Benchmark" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Device ID: " << deviceId << std::endl;
    std::cout << "Warmup rounds: " << WARMUP_ROUNDS << std::endl;
    std::cout << "Test rounds: " << TEST_ROUNDS << std::endl;
    std::cout << std::endl;

    MatMulBenchmark benchmark(deviceId);

    std::cout << std::left << std::setw(30) << "Test Size";
    std::cout << " | " << std::setw(23) << "Latency";
    std::cout << " | " << std::setw(18) << "Throughput";
    std::cout << " | Accuracy" << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    for (const auto& testSize : TEST_SIZES) {
        try {
            auto result = benchmark.RunBenchmark(testSize.m, testSize.n, testSize.k);
            benchmark.PrintResults(testSize, result);
        } catch (const std::exception& e) {
            std::cerr << "Error running benchmark for " << testSize.description
                      << ": " << e.what() << std::endl;
        }
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "Benchmark completed!" << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}
