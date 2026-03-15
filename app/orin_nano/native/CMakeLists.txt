cmake_minimum_required(VERSION 3.18)
project(mmoment_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# =============================================================================
# Find Dependencies
# =============================================================================

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0
)

# TensorRT (Jetson system install) - for later phases
set(TENSORRT_ROOT "/usr/lib/aarch64-linux-gnu")
find_library(TENSORRT_LIBRARY nvinfer HINTS ${TENSORRT_ROOT})
find_library(TENSORRT_ONNXPARSER nvonnxparser HINTS ${TENSORRT_ROOT})

message(STATUS "=== MMOMENT Native Build ===")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "TensorRT: ${TENSORRT_LIBRARY}")
message(STATUS "GStreamer: ${GSTREAMER_VERSION}")

# NVIDIA multimedia libraries (for NVMM buffer handling)
find_library(NVBUFSURFACE_LIBRARY nvbufsurface HINTS /usr/lib/aarch64-linux-gnu/tegra)
find_library(NVBUFSURFTRANSFORM_LIBRARY nvbufsurftransform HINTS /usr/lib/aarch64-linux-gnu/tegra)

message(STATUS "NvBufSurface: ${NVBUFSURFACE_LIBRARY}")

# =============================================================================
# Include Directories
# =============================================================================

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GSTREAMER_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    /usr/local/cuda/include
    /usr/src/jetson_multimedia_api/include  # NvBufSurface headers
)

link_directories(
    ${GSTREAMER_LIBRARY_DIRS}
    /usr/local/cuda/lib64
    /usr/lib/aarch64-linux-gnu/tegra
)

# =============================================================================
# Phase 1: Camera capture test
# =============================================================================

add_executable(test_camera
    src/test_camera.cpp
)

target_link_libraries(test_camera
    ${GSTREAMER_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NVBUFSURFACE_LIBRARY}
    cudart
)

# =============================================================================
# Phase 2: TensorRT inference library
# =============================================================================

# CUDA preprocessing kernels - build as SHARED for PIC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 --compiler-options -fPIC)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

cuda_add_library(preprocess_cuda SHARED
    src/preprocess.cu
)

# TensorRT engine wrapper
add_library(mmoment_trt STATIC
    src/tensorrt_engine.cpp
)

target_include_directories(mmoment_trt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/include/aarch64-linux-gnu
)

target_link_libraries(mmoment_trt
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNXPARSER}
    ${CUDA_LIBRARIES}
    preprocess_cuda
    cudart
)

# =============================================================================
# Phase 2: Inference test (camera + TensorRT)
# =============================================================================

add_executable(test_inference
    src/test_inference.cpp
)

target_include_directories(test_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_inference
    mmoment_trt
    ${GSTREAMER_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NVBUFSURFACE_LIBRARY}
    ${TENSORRT_LIBRARY}
    cudart
)

# Simple TensorRT test (no camera, just engine benchmark)
add_executable(test_trt_simple
    src/test_trt_simple.cpp
)

target_include_directories(test_trt_simple PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_trt_simple
    mmoment_trt
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
    cudart
)

# =============================================================================
# Phase 3: InsightFace (Face Detection + Recognition)
# =============================================================================

# InsightFace engine library
add_library(insightface_trt STATIC
    src/insightface_engine.cpp
)

target_include_directories(insightface_trt PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/include/aarch64-linux-gnu
)

target_link_libraries(insightface_trt
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
    preprocess_cuda
    cudart
)

# Full pipeline test (camera + YOLO + InsightFace)
add_executable(test_full_pipeline
    src/test_full_pipeline.cpp
)

target_include_directories(test_full_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_full_pipeline
    mmoment_trt
    insightface_trt
    ${GSTREAMER_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NVBUFSURFACE_LIBRARY}
    ${TENSORRT_LIBRARY}
    cudart
)

# Simple ArcFace test
add_executable(test_arcface
    src/test_arcface.cpp
)

target_include_directories(test_arcface PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_arcface
    insightface_trt
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
    cudart
)

# =============================================================================
# Phase 4: Shared Library for Python Integration
# =============================================================================

# Main shared library for Python ctypes
add_library(mmoment_native SHARED
    src/native_pipeline_api.cpp
    src/tensorrt_engine.cpp
    src/insightface_engine.cpp
    src/reid_engine.cpp
)

# CUDA kernels need to be linked
target_link_libraries(mmoment_native
    preprocess_cuda
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNXPARSER}
    ${CUDA_LIBRARIES}
    cudart
)

target_include_directories(mmoment_native PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/include/aarch64-linux-gnu
)

# Set output name without 'lib' prefix for easier Python import
set_target_properties(mmoment_native PROPERTIES
    OUTPUT_NAME "mmoment_native"
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
)

# Install target (optional)
install(TARGETS mmoment_native
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# =============================================================================
# Phase 5: Native Camera Server (standalone executable)
# =============================================================================

# Complete native server: camera + inference + socket server
add_executable(native_camera_server
    src/native_camera_server.cpp
    src/byte_tracker.cpp
    src/reid_engine.cpp
)

target_include_directories(native_camera_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/src/jetson_multimedia_api/include
)

target_link_libraries(native_camera_server
    mmoment_trt
    insightface_trt
    preprocess_cuda
    ${GSTREAMER_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${NVBUFSURFACE_LIBRARY}
    ${NVBUFSURFTRANSFORM_LIBRARY}
    ${TENSORRT_LIBRARY}
    cudart
    cuda        # CUDA driver API for EGL interop
    EGL         # EGL for NVMM-CUDA interop
    pthread
)
