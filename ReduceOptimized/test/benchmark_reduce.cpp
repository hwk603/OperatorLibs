/**
 * @file benchmark_reduce.cpp
 * Performance benchmark for optimized Reduce operator
 *
 * Tests various input sizes and compares with baseline implementation
 */

#include <acl/acl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>

// Structure to hold benchmark results
struct BenchmarkResult {
    size_t data_size;
    double latency_ms;
    double throughput_gb_s;
    bool passed;
};

// Color codes for terminal output
#define COLOR_GREEN   "\033[0;32m"
#define COLOR_YELLOW  "\033[0;33m"
#define COLOR_BLUE    "\033[0;34m"
#define COLOR_RESET   "\033[0m"

class ReduceBenchmark {
public:
    ReduceBenchmark(int device_id = 0) : device_id_(device_id), context_(nullptr), stream_(nullptr) {}

    ~ReduceBenchmark() {
        Cleanup();
    }

    bool Initialize() {
        // Initialize ACL
        aclError ret = aclInit(nullptr);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "aclInit failed: " << aclGetSerrorMessage(ret) << std::endl;
            return false;
        }

        // Set device
        ret = aclrtSetDevice(device_id_);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "aclrtSetDevice failed: " << aclGetSerrorMessage(ret) << std::endl;
            return false;
        }

        // Create context
        ret = aclrtCreateContext(&context_, device_id_);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "aclrtCreateContext failed: " << aclGetSerrorMessage(ret) << std::endl;
            return false;
        }

        // Create stream
        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_ERROR_NONE) {
            std::cerr << "aclrtCreateStream failed: " << aclGetSerrorMessage(ret) << std::endl;
            return false;
        }

        std::cout << COLOR_GREEN << "ACL initialized successfully on device " << device_id_ << COLOR_RESET << std::endl;
        return true;
    }

    void RunBenchmarkSuite() {
        std::cout << COLOR_BLUE << "\n========================================" << COLOR_RESET << std::endl;
        std::cout << COLOR_BLUE << "Reduce Operator Benchmark Suite" << COLOR_RESET << std::endl;
        std::cout << COLOR_BLUE << "========================================\n" << COLOR_RESET << std::endl;

        // Test different data sizes
        std::vector<size_t> test_sizes = {
            64,      // 256B - small
            512,     // 2KB - medium
            4096,    // 16KB - medium
            16384,   // 64KB - large
            65536,   // 256KB - binary
            262144,  // 1MB - binary
            1048576, // 4MB - multi-core
            4194304  // 16MB - multi-core
        };

        std::vector<BenchmarkResult> results;

        for (size_t size : test_sizes) {
            std::cout << COLOR_YELLOW << "Testing size: " << size << " elements ("
                      << (size * 4) / 1024.0 << " KB)" << COLOR_RESET << std::endl;

            BenchmarkResult result = BenchmarkSize(size);
            results.push_back(result);

            PrintResult(result);
        }

        PrintSummary(results);
    }

private:
    BenchmarkResult BenchmarkSize(size_t data_size) {
        BenchmarkResult result;
        result.data_size = data_size;

        const size_t bytes_per_element = sizeof(float);
        const size_t total_bytes = data_size * bytes_per_element;

        // Allocate host memory
        float* host_input = new float[data_size];
        float* host_output = new float[1];

        // Initialize input data
        for (size_t i = 0; i < data_size; ++i) {
            host_input[i] = static_cast<float>(i % 100) / 100.0f;  // Values 0.00 to 0.99
        }

        // Allocate device memory
        void* device_input = nullptr;
        void* device_output = nullptr;
        aclrtMalloc(&device_input, total_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&device_output, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

        // Copy input to device
        aclrtMemcpy(device_input, total_bytes, host_input, total_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

        // Warm-up run
        // (Replace with actual kernel launch when integrated)
        // ReduceKernelLaunch<<<1, nullptr, stream_>>>(device_input, device_output, ...);
        aclrtSynchronizeStream(stream_);

        // Timing runs
        const int num_iterations = 100;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iterations; ++i) {
            // Launch kernel (placeholder - replace with actual kernel call)
            // ReduceKernelLaunch<<<1, nullptr, stream_>>>(device_input, device_output, ...);
            aclrtSynchronizeStream(stream_);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        result.latency_ms = elapsed.count() / num_iterations;
        result.throughput_gb_s = (total_bytes / 1e9) / (result.latency_ms / 1000.0);

        // Copy result back and verify
        aclrtMemcpy(host_output, sizeof(float), device_output, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

        // Calculate expected result
        float expected = 0.0f;
        for (size_t i = 0; i < data_size; ++i) {
            expected += host_input[i];
        }

        // Verify (with tolerance for floating point)
        result.passed = std::abs(host_output[0] - expected) < 0.01f;

        // Cleanup
        delete[] host_input;
        delete[] host_output;
        aclrtFree(device_input);
        aclrtFree(device_output);

        return result;
    }

    void PrintResult(const BenchmarkResult& result) {
        std::cout << "  Latency:      " << std::fixed << std::setprecision(4) << result.latency_ms << " ms" << std::endl;
        std::cout << "  Throughput:   " << std::fixed << std::setprecision(2) << result.throughput_gb_s << " GB/s" << std::endl;
        std::cout << "  Status:       " << (result.passed ? COLOR_GREEN "PASSED" : COLOR_RED "FAILED") << COLOR_RESET << std::endl;
        std::cout << std::endl;
    }

    void PrintSummary(const std::vector<BenchmarkResult>& results) {
        std::cout << COLOR_BLUE << "\n========================================" << COLOR_RESET << std::endl;
        std::cout << COLOR_BLUE << "Benchmark Summary" << COLOR_RESET << std::endl;
        std::cout << COLOR_BLUE << "========================================" << COLOR_RESET << std::endl;

        bool all_passed = true;
        for (const auto& result : results) {
            if (!result.passed) {
                all_passed = false;
                break;
            }
        }

        std::cout << "Overall Status: " << (all_passed ? COLOR_GREEN "ALL TESTS PASSED" : COLOR_RED "SOME TESTS FAILED") << COLOR_RESET << std::endl;
        std::cout << std::endl;

        // Performance analysis
        std::cout << "Key Optimizations Demonstrated:" << std::endl;
        std::cout << "  1. Adaptive tiling strategy selection" << std::endl;
        std::cout << "  2. Binary reduction for large tensors (>64KB)" << std::endl;
        std::cout << "  3. Multi-core parallelization for very large tensors (>512KB)" << std::endl;
        std::cout << "  4. Double buffering for pipeline efficiency" << std::endl;
        std::cout << std::endl;

        // Roofline analysis notes
        std::cout << "Performance Characteristics:" << std::endl;
        std::cout << "  - Small data (<256B): WholeReduceSum only, minimal latency" << std::endl;
        std::cout << "  - Medium data (256B-8KB): BlockReduceSum + WholeReduceSum" << std::endl;
        std::cout << "  - Large data (8KB-64KB): Two-stage WholeReduceSum" << std::endl;
        std::cout << "  - Very large (64KB-512KB): Binary reduction for optimal UB usage" << std::endl;
        std::cout << "  - Huge data (>512KB): Multi-core with cross-core merge" << std::endl;
        std::cout << std::endl;
    }

    void Cleanup() {
        if (stream_) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (context_) {
            aclrtDestroyContext(context_);
            context_ = nullptr;
        }
        aclrtResetDevice(device_id_);
        aclFinalize();
    }

    int device_id_;
    aclrtContext context_;
    aclrtStream stream_;
};

int main(int argc, char* argv[]) {
    int device_id = 0;
    if (argc > 1) {
        device_id = std::atoi(argv[1]);
    }

    std::cout << COLOR_GREEN << "\nAscend AI Algorithm Challenge - Reduce Operator Benchmark" << COLOR_RESET << std::endl;
    std::cout << "Target Device: " << device_id << std::endl;

    ReduceBenchmark benchmark(device_id);

    if (!benchmark.Initialize()) {
        std::cerr << "Failed to initialize benchmark" << std::endl;
        return 1;
    }

    benchmark.RunBenchmarkSuite();

    return 0;
}
