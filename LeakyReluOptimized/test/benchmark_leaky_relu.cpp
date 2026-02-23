/**
 * @file benchmark_leaky_relu.cpp
 * Performance benchmark for optimized LeakyReLU operator
 *
 * Features:
 * 1. Automated performance testing across multiple data sizes
 * 2. Accuracy verification
 * 3. Latency and throughput measurement
 *
 * Copyright (C) 2025. Competition Submission.
 */

#include <acl/acl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>

// ===================================================================
// Constants and Configuration
// ===================================================================

constexpr int WARMUP_ROUNDS = 10;
constexpr int TEST_ROUNDS = 100;
constexpr float EPSILON = 1e-5;

// Test data sizes (elements)
struct TestSize {
    uint32_t elements;
    const char* description;
};

constexpr TestSize TEST_SIZES[] = {
    {256, "Tiny (256 elements)"},
    {1024, "Small (1K elements)"},
    {8192, "Medium (8K elements)"},
    {65536, "Large (64K elements)"},
    {262144, "X-Large (256K elements)"},
    {1048576, "XX-Large (1M elements)"},
    {4194304, "Huge (4M elements)"},
    {16777216, "Massive (16M elements)"},
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
bool CompareResults(const T* result, const T* reference, size_t size, float epsilon) {
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(static_cast<float>(result[i]) - static_cast<float>(reference[i]));
        float maxVal = std::max(std::abs(static_cast<float>(result[i])),
                               std::abs(static_cast<float>(reference[i])));
        float relErr = (maxVal > 1.0f) ? (diff / maxVal) : diff;

        if (relErr > epsilon) {
            std::cerr << "Mismatch at index " << i << ": got "
                      << static_cast<float>(result[i]) << ", expected "
                      << static_cast<float>(reference[i])
                      << ", rel_err=" << relErr << std::endl;
            return false;
        }
    }
    return true;
}

// ===================================================================
// CPU Reference Implementation
// ===================================================================

template<typename T>
void LeakyReluCPUReference(const T* x, T* y, size_t length, float negativeSlope) {
    for (size_t i = 0; i < length; ++i) {
        if (x[i] >= static_cast<T>(0)) {
            y[i] = x[i];
        } else {
            y[i] = static_cast<T>(x[i] * negativeSlope);
        }
    }
}

// ===================================================================
// Benchmark Class
// ===================================================================

class LeakyReluBenchmark {
public:
    LeakyReluBenchmark(int deviceId = 0) : deviceId_(deviceId), stream_(nullptr) {
        ACL_CHECK(aclInit(nullptr));
        ACL_CHECK(aclrtSetDevice(deviceId_));
        ACL_CHECK(aclrtCreateStream(&stream_));
    }

    ~LeakyReluBenchmark() {
        ACL_CHECK(aclrtDestroyStream(stream_));
        ACL_CHECK(aclrtResetDevice(deviceId_));
        ACL_CHECK(aclFinalize());
    }

    struct BenchmarkResult {
        float avgLatencyMs;
        float minLatencyMs;
        float maxLatencyMs;
        float throughputGBps;
        float throughputGEps;
        bool accuracyPass;
    };

    template<typename T>
    BenchmarkResult RunBenchmark(uint32_t numElements, float negativeSlope = 0.01f) {
        BenchmarkResult result;

        // Calculate sizes
        size_t elementSize = sizeof(T);
        size_t totalSize = static_cast<size_t>(numElements) * elementSize;

        // Allocate host memory
        std::vector<T> hostX(numElements);
        std::vector<T> hostY(numElements);
        std::vector<T> goldenY(numElements);

        // Initialize with random data (-1.0 to 1.0)
        FillRandom(hostX, static_cast<T>(-1.0f), static_cast<T>(1.0f));

        // Allocate device memory
        T* deviceX = nullptr;
        T* deviceY = nullptr;

        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceX), totalSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&deviceY), totalSize, ACL_MEM_MALLOC_HUGE_FIRST));

        // Copy data to device
        ACL_CHECK(aclrtMemcpy(deviceX, totalSize, hostX.data(), totalSize, ACL_MEMCPY_HOST_TO_DEVICE));

        // Compute golden reference on CPU
        LeakyReluCPUReference(hostX.data(), goldenY.data(), numElements, negativeSlope);

        // Warmup runs
        for (int i = 0; i < WARMUP_ROUNDS; ++i) {
            // TODO: Replace with actual operator call
            ACL_CHECK(aclrtSynchronizeStream(stream_));
        }

        // Benchmark runs
        std::vector<float> latencies(TEST_ROUNDS);
        for (int i = 0; i < TEST_ROUNDS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            // TODO: Replace with actual operator call
            // Placeholder for kernel execution
            ACL_CHECK(aclrtSynchronizeStream(stream_));

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> duration = end - start;
            latencies[i] = duration.count();
        }

        // Copy result back
        ACL_CHECK(aclrtMemcpy(hostY.data(), totalSize, deviceY, totalSize, ACL_MEMCPY_DEVICE_TO_HOST));

        // Verify accuracy
        result.accuracyPass = CompareResults(hostY.data(), goldenY.data(), numElements, EPSILON);

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

        // Calculate throughput
        // Throughput in GB/s (total bytes read + written / time)
        size_t totalBytes = 2 * totalSize;  // Read + write
        result.throughputGBps = (totalBytes / 1e9f) / (result.avgLatencyMs / 1000.0f);

        // Throughput in GElements/s
        result.throughputGEps = (static_cast<float>(numElements) / 1e9f) / (result.avgLatencyMs / 1000.0f);

        // Cleanup
        ACL_CHECK(aclrtFree(deviceX));
        ACL_CHECK(aclrtFree(deviceY));

        return result;
    }

    void PrintResults(const TestSize& testSize, const BenchmarkResult& result) {
        std::cout << std::left << std::setw(30) << testSize.description;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << " | Avg: " << std::setw(8) << result.avgLatencyMs << " ms";
        std::cout << " | Throughput: " << std::setw(8) << result.throughputGBps << " GB/s";
        std::cout << " (" << std::setw(6) << result.throughputGEps << " GE/s)";
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

    float negativeSlope = 0.01f;
    if (argc > 2) {
        negativeSlope = atof(argv[2]);
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "  LeakyReLU Optimized Performance Benchmark" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Device ID: " << deviceId << std::endl;
    std::cout << "Negative Slope (alpha): " << negativeSlope << std::endl;
    std::cout << "Warmup rounds: " << WARMUP_ROUNDS << std::endl;
    std::cout << "Test rounds: " << TEST_ROUNDS << std::endl;
    std::cout << std::endl;

    LeakyReluBenchmark benchmark(deviceId);

    std::cout << std::left << std::setw(30) << "Test Size";
    std::cout << " | " << std::setw(19) << "Latency";
    std::cout << " | " << std::setw(20) << "Throughput";
    std::cout << " | Accuracy" << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    for (const auto& testSize : TEST_SIZES) {
        try {
            auto result = benchmark.RunBenchmark<float>(testSize.elements, negativeSlope);
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
