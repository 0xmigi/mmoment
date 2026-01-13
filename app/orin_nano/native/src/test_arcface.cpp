/**
 * test_arcface.cpp - Simple ArcFace test
 * Tests just the ArcFace embedding model with dummy input
 */

#include <iostream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include "insightface_engine.h"

using namespace mmoment;

int main() {
    std::cout << "=== ArcFace Engine Test ===" << std::endl;

    // Load ArcFace engine
    ArcFaceEngine arcface;
    std::cout << "Loading ArcFace engine..." << std::endl;
    if (!arcface.loadEngine("arcface_r50.engine")) {
        std::cerr << "Failed to load ArcFace engine" << std::endl;
        return 1;
    }

    // Create dummy input (112x112x3 float)
    const int INPUT_SIZE = 112 * 112 * 3;
    void* gpuInput = nullptr;
    cudaMalloc(&gpuInput, INPUT_SIZE * sizeof(float));

    // Fill with zeros (normalized face-like input)
    cudaMemset(gpuInput, 0, INPUT_SIZE * sizeof(float));

    // Get embedding
    float embedding[512];

    std::cout << "Running inference..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        if (!arcface.getEmbedding(gpuInput, embedding)) {
            std::cerr << "Inference failed" << std::endl;
            return 1;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    float totalMs = std::chrono::duration<float, std::milli>(end - start).count();

    std::cout << "100 inferences in " << totalMs << "ms" << std::endl;
    std::cout << "Average: " << totalMs / 100 << "ms per inference" << std::endl;

    // Print first few embedding values
    std::cout << "Embedding[0:5]: [";
    for (int i = 0; i < 5; i++) {
        std::cout << embedding[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Check embedding norm (should be ~1.0 after L2 normalization)
    float norm = 0;
    for (int i = 0; i < 512; i++) {
        norm += embedding[i] * embedding[i];
    }
    std::cout << "Embedding L2 norm: " << sqrtf(norm) << " (should be ~1.0)" << std::endl;

    cudaFree(gpuInput);
    std::cout << "Done." << std::endl;
    return 0;
}
