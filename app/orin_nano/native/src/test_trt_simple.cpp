/**
 * test_trt_simple.cpp - Simple TensorRT inference test
 *
 * Tests TensorRT engine loading and inference with random CUDA data
 * to verify the engine works before integrating with camera.
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

#include "tensorrt_engine.h"

int main(int argc, char** argv) {
    std::cout << "=== Simple TensorRT Inference Test ===" << std::endl;

    // Check for engine file argument
    std::string enginePath = "/mnt/nvme/mmoment/app/orin_nano/native/yolov8n-pose-native.engine";
    if (argc > 1) {
        enginePath = argv[1];
    }

    std::cout << "Using engine: " << enginePath << std::endl;

    // Load TensorRT engine
    mmoment::TensorRTEngine engine;
    if (!engine.loadEngine(enginePath)) {
        std::cerr << "Failed to load TensorRT engine" << std::endl;
        return 1;
    }

    // Get model dimensions
    auto inputDims = engine.getInputDims();
    int modelWidth = inputDims.d[3];   // NCHW format
    int modelHeight = inputDims.d[2];
    int modelChannels = inputDims.d[1];

    std::cout << "Model input: " << modelChannels << "x" << modelHeight << "x" << modelWidth << std::endl;

    // Allocate GPU memory for input
    size_t inputSize = modelChannels * modelHeight * modelWidth * sizeof(float);
    void* gpuInput = nullptr;
    cudaMalloc(&gpuInput, inputSize);

    // Fill with random test data
    std::vector<float> hostInput(modelChannels * modelHeight * modelWidth);
    for (auto& v : hostInput) {
        v = (float)rand() / RAND_MAX;  // Random 0-1
    }
    cudaMemcpy(gpuInput, hostInput.data(), inputSize, cudaMemcpyHostToDevice);

    std::cout << "Allocated and initialized input buffer: " << inputSize / 1024 << " KB" << std::endl;

    // Warm-up run
    std::cout << "Warming up..." << std::endl;
    std::vector<float> outputData;
    for (int i = 0; i < 5; i++) {
        if (!engine.infer(gpuInput, modelWidth, modelHeight, modelChannels, outputData)) {
            std::cerr << "Warm-up inference failed!" << std::endl;
            cudaFree(gpuInput);
            return 1;
        }
    }
    std::cout << "Warm-up complete. Output size: " << outputData.size() << " floats" << std::endl;

    // Benchmark
    std::cout << std::endl << "Running benchmark (100 iterations)..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    const int NUM_ITERS = 100;

    for (int i = 0; i < NUM_ITERS; i++) {
        if (!engine.infer(gpuInput, modelWidth, modelHeight, modelChannels, outputData)) {
            std::cerr << "Inference failed at iteration " << i << std::endl;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    double avgMs = totalMs / NUM_ITERS;
    double fps = 1000.0 / avgMs;

    std::cout << std::endl;
    std::cout << "=== Benchmark Results ===" << std::endl;
    std::cout << "Total time: " << totalMs << " ms" << std::endl;
    std::cout << "Average inference: " << avgMs << " ms" << std::endl;
    std::cout << "Throughput: " << fps << " FPS" << std::endl;

    // Test post-processing
    auto outputDims = engine.getOutputDims();
    int numDetections = outputDims.d[2];  // [1, 56, N]
    std::cout << std::endl;
    std::cout << "Output shape: [1, " << outputDims.d[1] << ", " << numDetections << "]" << std::endl;

    // Transpose and run post-processor
    std::vector<float> transposed(numDetections * 56);
    for (int i = 0; i < numDetections; i++) {
        for (int j = 0; j < 56; j++) {
            transposed[i * 56 + j] = outputData[j * numDetections + i];
        }
    }

    mmoment::YoloPosePostProcessor postProc(0.25f, 0.45f);
    auto detections = postProc.process(transposed, numDetections, 2.0f, 1280, 720);

    std::cout << "Detections (random data): " << detections.size() << std::endl;

    // Cleanup
    cudaFree(gpuInput);

    std::cout << std::endl << "=== Test Complete ===" << std::endl;
    return 0;
}
