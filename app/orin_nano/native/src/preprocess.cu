/**
 * preprocess.cu - CUDA kernels for image preprocessing
 *
 * Converts NVMM BGRx buffer to RGB float tensor for TensorRT inference.
 * Handles resize + pad (right/bottom) + normalize in a single kernel.
 */

#include <cuda_runtime.h>
#include <cstdio>

namespace mmoment {

// Bilinear interpolation sampling
__device__ float bilinearSample(const unsigned char* src, int srcW, int srcH, int srcPitch,
                                 float x, float y, int channel) {
    // Clamp coordinates
    x = fmaxf(0.0f, fminf(x, srcW - 1.0f));
    y = fmaxf(0.0f, fminf(y, srcH - 1.0f));

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = min(x0 + 1, srcW - 1);
    int y1 = min(y0 + 1, srcH - 1);

    float xFrac = x - x0;
    float yFrac = y - y0;

    // BGRx format: 4 bytes per pixel
    float v00 = src[y0 * srcPitch + x0 * 4 + channel];
    float v01 = src[y0 * srcPitch + x1 * 4 + channel];
    float v10 = src[y1 * srcPitch + x0 * 4 + channel];
    float v11 = src[y1 * srcPitch + x1 * 4 + channel];

    float v0 = v00 * (1 - xFrac) + v01 * xFrac;
    float v1 = v10 * (1 - xFrac) + v11 * xFrac;

    return v0 * (1 - yFrac) + v1 * yFrac;
}

/**
 * Preprocess BGRx NVMM buffer to RGB float tensor
 *
 * Input:  BGRx (4 channels, uint8) at srcW x srcH
 * Output: RGB (3 channels, float32, normalized 0-1) at dstW x dstH in CHW format
 *
 * - Resizes maintaining aspect ratio
 * - Pads right/bottom with gray (114/255)
 * - Converts BGR to RGB
 * - Normalizes to [0, 1]
 */
__global__ void preprocessBGRxToRGBFloat(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float scaleX, float scaleY,
    int newW, int newH,
    float padValue
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) return;

    float r, g, b;

    // Check if this pixel is in the valid (non-padded) region
    if (x < newW && y < newH) {
        // Map to source coordinates
        float srcX = x / scaleX;
        float srcY = y / scaleY;

        // Sample BGR from source (bilinear interpolation)
        b = bilinearSample(src, srcW, srcH, srcPitch, srcX, srcY, 0);
        g = bilinearSample(src, srcW, srcH, srcPitch, srcX, srcY, 1);
        r = bilinearSample(src, srcW, srcH, srcPitch, srcX, srcY, 2);
    } else {
        // Padding region - use gray
        r = g = b = padValue * 255.0f;
    }

    // Normalize to [0, 1]
    r /= 255.0f;
    g /= 255.0f;
    b /= 255.0f;

    // Write to CHW format: [C, H, W]
    int pixelIdx = y * dstW + x;
    dst[0 * dstH * dstW + pixelIdx] = r;  // R channel
    dst[1 * dstH * dstW + pixelIdx] = g;  // G channel
    dst[2 * dstH * dstW + pixelIdx] = b;  // B channel
}

/**
 * Host function to launch preprocessing kernel
 */
extern "C" void launchPreprocessBGRxToRGBFloat(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float* scaleRatio,
    cudaStream_t stream
) {
    // Calculate scale to maintain aspect ratio
    float scaleX = (float)dstW / srcW;
    float scaleY = (float)dstH / srcH;
    float scale = fminf(scaleX, scaleY);

    int newW = (int)(srcW * scale);
    int newH = (int)(srcH * scale);

    // Return scale ratio for coordinate transformation
    *scaleRatio = 1.0f / scale;

    // Pad value (114/255 ≈ 0.447)
    float padValue = 114.0f / 255.0f;

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

    preprocessBGRxToRGBFloat<<<grid, block, 0, stream>>>(
        (const unsigned char*)src,
        (float*)dst,
        srcW, srcH, srcPitch,
        dstW, dstH,
        scale, scale,
        newW, newH,
        padValue
    );
}

/**
 * Simple nearest-neighbor resize (faster but lower quality)
 */
__global__ void preprocessBGRxToRGBFloatNN(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float scaleX, float scaleY,
    int newW, int newH,
    float padValue
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) return;

    float r, g, b;

    if (x < newW && y < newH) {
        // Map to source coordinates (nearest neighbor)
        int srcX = (int)(x / scaleX);
        int srcY = (int)(y / scaleY);
        srcX = min(srcX, srcW - 1);
        srcY = min(srcY, srcH - 1);

        int srcIdx = srcY * srcPitch + srcX * 4;
        b = src[srcIdx + 0];
        g = src[srcIdx + 1];
        r = src[srcIdx + 2];
    } else {
        r = g = b = padValue * 255.0f;
    }

    // Normalize and write CHW
    int pixelIdx = y * dstW + x;
    dst[0 * dstH * dstW + pixelIdx] = r / 255.0f;
    dst[1 * dstH * dstW + pixelIdx] = g / 255.0f;
    dst[2 * dstH * dstW + pixelIdx] = b / 255.0f;
}

extern "C" void launchPreprocessBGRxToRGBFloatNN(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float* scaleRatio,
    cudaStream_t stream
) {
    float scaleX = (float)dstW / srcW;
    float scaleY = (float)dstH / srcH;
    float scale = fminf(scaleX, scaleY);

    int newW = (int)(srcW * scale);
    int newH = (int)(srcH * scale);
    *scaleRatio = 1.0f / scale;

    float padValue = 114.0f / 255.0f;

    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

    preprocessBGRxToRGBFloatNN<<<grid, block, 0, stream>>>(
        (const unsigned char*)src,
        (float*)dst,
        srcW, srcH, srcPitch,
        dstW, dstH,
        scale, scale,
        newW, newH,
        padValue
    );
}

/**
 * Preprocess with 90° CCW rotation for portrait camera orientation
 * Input: BGRx 1280x720 landscape (person appears sideways)
 * Output: RGB float 640x640 with person upright
 *
 * Rotation + resize + pad in single pass for maximum efficiency
 */
__global__ void preprocessBGRxToRGBFloatRotate90CCW(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int srcW, int srcH, int srcPitch,  // Original: 1280x720
    int dstW, int dstH,                 // Output: 640x640
    float scale, int newW, int newH,    // After rotation: 720x1280 scaled
    float padValue
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) return;

    float r, g, b;

    if (x < newW && y < newH) {
        // Map output (x, y) to rotated source coordinates
        // Rotated image would be 720x1280 (srcH x srcW)
        // For 90° CCW: rotated(x, y) = original(y, srcH - 1 - x)
        // But we're scaling the rotated image, so:
        float rotX = x / scale;  // x in rotated space (0-720)
        float rotY = y / scale;  // y in rotated space (0-1280)

        // Convert rotated coords to original coords (90° CCW inverse = 90° CW)
        // For 90° CCW rotation: output(x,y) samples from input(srcW - 1 - y, x)
        // where srcW is the original WIDTH (1280), not height
        int origX = (int)(srcW - 1 - rotY);  // Use srcW (1280), not srcH!
        int origY = (int)rotX;

        origX = max(0, min(origX, srcW - 1));
        origY = max(0, min(origY, srcH - 1));

        int srcIdx = origY * srcPitch + origX * 4;
        b = src[srcIdx + 0];
        g = src[srcIdx + 1];
        r = src[srcIdx + 2];
    } else {
        r = g = b = padValue * 255.0f;
    }

    // Normalize and write CHW
    int pixelIdx = y * dstW + x;
    dst[0 * dstH * dstW + pixelIdx] = r / 255.0f;
    dst[1 * dstH * dstW + pixelIdx] = g / 255.0f;
    dst[2 * dstH * dstW + pixelIdx] = b / 255.0f;
}

extern "C" void launchPreprocessBGRxToRGBFloatRotate90CCW(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float* scaleRatio,
    int* rotatedW, int* rotatedH,  // Output: dimensions after rotation
    cudaStream_t stream
) {
    // After 90° CCW rotation: 1280x720 -> 720x1280
    int rotW = srcH;   // 720
    int rotH = srcW;   // 1280

    *rotatedW = rotW;
    *rotatedH = rotH;

    // Scale rotated image to fit 640x640
    float scaleX = (float)dstW / rotW;   // 640/720 = 0.889
    float scaleY = (float)dstH / rotH;   // 640/1280 = 0.5
    float scale = fminf(scaleX, scaleY); // 0.5

    int newW = (int)(rotW * scale);  // 360
    int newH = (int)(rotH * scale);  // 640
    *scaleRatio = 1.0f / scale;      // 2.0

    float padValue = 114.0f / 255.0f;

    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

    preprocessBGRxToRGBFloatRotate90CCW<<<grid, block, 0, stream>>>(
        (const unsigned char*)src,
        (float*)dst,
        srcW, srcH, srcPitch,
        dstW, dstH,
        scale, newW, newH,
        padValue
    );
}

/**
 * Crop a region from BGRx frame and convert to RGB float CHW for face detection
 * Used to crop person bbox before feeding to RetinaFace
 *
 * IMPORTANT: RetinaFace expects [-1, 1] normalization (same as ArcFace)
 * Formula: (pixel - 127.5) / 127.5
 */
__global__ void cropBGRxToRGBFloat(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int srcW, int srcH, int srcPitch,
    int cropX, int cropY, int cropW, int cropH,
    int dstW, int dstH,
    float padValue
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) return;

    // Calculate scale to fit crop into destination while maintaining aspect ratio
    float scaleX = (float)dstW / cropW;
    float scaleY = (float)dstH / cropH;
    float scale = fminf(scaleX, scaleY);

    int newW = (int)(cropW * scale);
    int newH = (int)(cropH * scale);

    float r, g, b;

    if (x < newW && y < newH) {
        // Map to crop region coordinates
        float srcX = cropX + x / scale;
        float srcY = cropY + y / scale;

        // Clamp to source bounds
        srcX = fmaxf(0.0f, fminf(srcX, srcW - 1.0f));
        srcY = fmaxf(0.0f, fminf(srcY, srcH - 1.0f));

        int sx = (int)srcX;
        int sy = (int)srcY;
        int srcIdx = sy * srcPitch + sx * 4;

        b = src[srcIdx + 0];
        g = src[srcIdx + 1];
        r = src[srcIdx + 2];
    } else {
        // Padding: 127.5 maps to 0 in normalized [-1,1] space
        r = g = b = 127.5f;
    }

    // Normalize to [-1, 1] for RetinaFace (InsightFace standard normalization)
    int pixelIdx = y * dstW + x;
    dst[0 * dstH * dstW + pixelIdx] = (r - 127.5f) / 127.5f;
    dst[1 * dstH * dstW + pixelIdx] = (g - 127.5f) / 127.5f;
    dst[2 * dstH * dstW + pixelIdx] = (b - 127.5f) / 127.5f;
}

extern "C" void launchCropBGRxToRGBFloat(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int cropX, int cropY, int cropW, int cropH,
    int dstW, int dstH,
    float* scaleRatio,
    cudaStream_t stream
) {
    float scaleX = (float)dstW / cropW;
    float scaleY = (float)dstH / cropH;
    float scale = fminf(scaleX, scaleY);
    *scaleRatio = 1.0f / scale;

    float padValue = 114.0f / 255.0f;

    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

    cropBGRxToRGBFloat<<<grid, block, 0, stream>>>(
        (const unsigned char*)src,
        (float*)dst,
        srcW, srcH, srcPitch,
        cropX, cropY, cropW, cropH,
        dstW, dstH,
        padValue
    );
}

/**
 * Affine warp for face alignment using 5 landmarks
 * Warps a face region to 112x112 normalized coordinates for ArcFace
 *
 * Standard reference landmarks for 112x112 (from InsightFace):
 * Left eye:    (38.2946, 51.6963)
 * Right eye:   (73.5318, 51.5014)
 * Nose:        (56.0252, 71.7366)
 * Left mouth:  (41.5493, 92.3655)
 * Right mouth: (70.7299, 92.2041)
 *
 * NOTE: The actual transform computation is done on the CPU host using the
 * Umeyama algorithm (5-point SVD-based similarity transform) in the
 * launchWarpFaceToArcFace() function. This device function is kept for
 * reference but is not currently called.
 */

/**
 * Warp face region using affine transform for face recognition input
 * Input: BGRx face region, 5 landmarks
 * Output: 112x112 BGR float CHW normalized to [-1, 1] for InsightFace/AdaFace
 */
__global__ void warpFaceToArcFace(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int srcW, int srcH, int srcPitch,
    float invMatrix00, float invMatrix01, float invMatrix02,
    float invMatrix10, float invMatrix11, float invMatrix12
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int DST_SIZE = 112;
    if (x >= DST_SIZE || y >= DST_SIZE) return;

    // Apply inverse affine transform to find source coordinates
    float srcX = invMatrix00 * x + invMatrix01 * y + invMatrix02;
    float srcY = invMatrix10 * x + invMatrix11 * y + invMatrix12;

    float r, g, b;

    // Bilinear interpolation
    if (srcX >= 0 && srcX < srcW - 1 && srcY >= 0 && srcY < srcH - 1) {
        int x0 = (int)floorf(srcX);
        int y0 = (int)floorf(srcY);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float xFrac = srcX - x0;
        float yFrac = srcY - y0;

        // BGRx format
        float b00 = src[y0 * srcPitch + x0 * 4 + 0];
        float g00 = src[y0 * srcPitch + x0 * 4 + 1];
        float r00 = src[y0 * srcPitch + x0 * 4 + 2];

        float b01 = src[y0 * srcPitch + x1 * 4 + 0];
        float g01 = src[y0 * srcPitch + x1 * 4 + 1];
        float r01 = src[y0 * srcPitch + x1 * 4 + 2];

        float b10 = src[y1 * srcPitch + x0 * 4 + 0];
        float g10 = src[y1 * srcPitch + x0 * 4 + 1];
        float r10 = src[y1 * srcPitch + x0 * 4 + 2];

        float b11 = src[y1 * srcPitch + x1 * 4 + 0];
        float g11 = src[y1 * srcPitch + x1 * 4 + 1];
        float r11 = src[y1 * srcPitch + x1 * 4 + 2];

        b = (b00 * (1 - xFrac) + b01 * xFrac) * (1 - yFrac) +
            (b10 * (1 - xFrac) + b11 * xFrac) * yFrac;
        g = (g00 * (1 - xFrac) + g01 * xFrac) * (1 - yFrac) +
            (g10 * (1 - xFrac) + g11 * xFrac) * yFrac;
        r = (r00 * (1 - xFrac) + r01 * xFrac) * (1 - yFrac) +
            (r10 * (1 - xFrac) + r11 * xFrac) * yFrac;
    } else {
        r = g = b = 127.5f;  // Gray for out of bounds
    }

    // Normalize to [-1, 1] for face recognition (InsightFace/AdaFace normalization)
    r = (r - 127.5f) / 127.5f;
    g = (g - 127.5f) / 127.5f;
    b = (b - 127.5f) / 127.5f;

    // Write BGR CHW format for AdaFace (official preprocessing expects BGR)
    // See: https://github.com/mk-minchul/AdaFace inference.py to_input()
    int pixelIdx = y * DST_SIZE + x;
    dst[0 * DST_SIZE * DST_SIZE + pixelIdx] = b;  // Blue
    dst[1 * DST_SIZE * DST_SIZE + pixelIdx] = g;  // Green
    dst[2 * DST_SIZE * DST_SIZE + pixelIdx] = r;  // Red
}

/**
 * Compute similarity transform (rotation, uniform scale, translation) using
 * least-squares fitting from source to destination point correspondences.
 *
 * This is a simplified approach that avoids complex SVD:
 * 1. Compute centroids of both point sets
 * 2. Center the points
 * 3. Compute optimal rotation angle and scale using closed-form solution
 * 4. Compute translation
 *
 * For 2D similarity transform with uniform scale, the closed-form solution is:
 *   scale = sqrt(sum(dst_centered^2) / sum(src_centered^2))
 *   angle = atan2(cross_sum, dot_sum) where cross/dot are between centered points
 *
 * Reference: Procrustes analysis / Similarity transformation estimation
 */
static void computeSimilarityTransform(
    const float srcPts[5][2],
    const float dstPts[5][2],
    float matrix[2][3]
) {
    const int N = 5;

    // Step 1: Compute centroids
    float srcCx = 0.0f, srcCy = 0.0f;
    float dstCx = 0.0f, dstCy = 0.0f;

    for (int i = 0; i < N; i++) {
        srcCx += srcPts[i][0];
        srcCy += srcPts[i][1];
        dstCx += dstPts[i][0];
        dstCy += dstPts[i][1];
    }
    srcCx /= N;
    srcCy /= N;
    dstCx /= N;
    dstCy /= N;

    // Step 2: Compute sums for rotation and scale
    // Using the Procrustes solution:
    // dot_sum = sum(src_centered . dst_centered)  (dot product of centered vectors)
    // cross_sum = sum(src_centered x dst_centered) (2D cross product)
    // src_norm_sq = sum(|src_centered|^2)
    float dotSum = 0.0f;
    float crossSum = 0.0f;
    float srcNormSq = 0.0f;
    float dstNormSq = 0.0f;

    for (int i = 0; i < N; i++) {
        float sx = srcPts[i][0] - srcCx;
        float sy = srcPts[i][1] - srcCy;
        float dx = dstPts[i][0] - dstCx;
        float dy = dstPts[i][1] - dstCy;

        dotSum += sx * dx + sy * dy;
        crossSum += sx * dy - sy * dx;
        srcNormSq += sx * sx + sy * sy;
        dstNormSq += dx * dx + dy * dy;
    }

    // Step 3: Compute rotation angle
    // The optimal rotation angle is: theta = atan2(cross_sum, dot_sum)
    float angle = atan2f(crossSum, dotSum);
    float cosA = cosf(angle);
    float sinA = sinf(angle);

    // Step 4: Compute scale
    // scale = sqrt(dst_norm_sq / src_norm_sq) gives isotropic scale
    // But for better fit, use: scale = (dot_sum * cos + cross_sum * sin) / src_norm_sq
    // which is equivalent to: scale = sqrt(dot_sum^2 + cross_sum^2) / src_norm_sq
    float scale = 1.0f;
    if (srcNormSq > 1e-10f) {
        scale = sqrtf(dotSum * dotSum + crossSum * crossSum) / srcNormSq;
    }

    // Step 5: Build transformation matrix
    // [x']   [s*cos  -s*sin  tx] [x]
    // [y'] = [s*sin   s*cos  ty] [y]
    // [1 ]   [0       0      1 ] [1]
    //
    // Translation: dst_centroid = s*R*src_centroid + t
    // So: t = dst_centroid - s*R*src_centroid

    matrix[0][0] = scale * cosA;
    matrix[0][1] = -scale * sinA;
    matrix[1][0] = scale * sinA;
    matrix[1][1] = scale * cosA;

    matrix[0][2] = dstCx - (matrix[0][0] * srcCx + matrix[0][1] * srcCy);
    matrix[1][2] = dstCy - (matrix[1][0] * srcCx + matrix[1][1] * srcCy);
}

// Host function to compute inverse affine matrix and launch warp kernel
extern "C" void launchWarpFaceToArcFace(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    const float landmarks[5][2],  // 5 landmarks in source image coordinates
    cudaStream_t stream
) {
    // Reference landmarks for 112x112 - MTCNN alignment (used by AdaFace training)
    // These differ from InsightFace/ArcFace landmarks by ~3px horizontally for eyes
    // Using MTCNN landmarks since AdaFace was trained with this alignment
    const float dstPts[5][2] = {
        {35.3437f, 51.6963f},   // Left eye  (MTCNN: closer to center than ArcFace)
        {76.4538f, 51.5014f},   // Right eye (MTCNN: closer to center than ArcFace)
        {56.0294f, 71.7366f},   // Nose
        {39.1409f, 92.3655f},   // Left mouth
        {73.1849f, 92.2041f}    // Right mouth
    };

    // Compute similarity transform using closed-form Procrustes solution (5-point)
    float matrix[2][3];
    computeSimilarityTransform(landmarks, dstPts, matrix);

    // Compute inverse matrix for backward mapping (GPU kernel needs inverse)
    float det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    float invDet = 1.0f / (det + 1e-10f);

    float invMatrix[2][3];
    invMatrix[0][0] = matrix[1][1] * invDet;
    invMatrix[0][1] = -matrix[0][1] * invDet;
    invMatrix[1][0] = -matrix[1][0] * invDet;
    invMatrix[1][1] = matrix[0][0] * invDet;
    invMatrix[0][2] = -(invMatrix[0][0] * matrix[0][2] + invMatrix[0][1] * matrix[1][2]);
    invMatrix[1][2] = -(invMatrix[1][0] * matrix[0][2] + invMatrix[1][1] * matrix[1][2]);

    // Launch kernel
    const int DST_SIZE = 112;
    dim3 block(16, 16);
    dim3 grid((DST_SIZE + block.x - 1) / block.x, (DST_SIZE + block.y - 1) / block.y);

    warpFaceToArcFace<<<grid, block, 0, stream>>>(
        (const unsigned char*)src,
        (float*)dst,
        srcW, srcH, srcPitch,
        invMatrix[0][0], invMatrix[0][1], invMatrix[0][2],
        invMatrix[1][0], invMatrix[1][1], invMatrix[1][2]
    );
}

/**
 * Preprocess BGR (3-channel) buffer to RGB float tensor
 * Same as BGRx version but for 3-channel input (e.g., OpenCV BGR)
 */
__global__ void preprocessBGRToRGBFloatNN(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float scaleRatio, int newW, int newH, float padValue
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) return;

    float r, g, b;

    if (x < newW && y < newH) {
        float srcX = x / scaleRatio;
        float srcY = y / scaleRatio;

        srcX = fmaxf(0.0f, fminf(srcX, srcW - 1.0f));
        srcY = fmaxf(0.0f, fminf(srcY, srcH - 1.0f));

        int sx = (int)srcX;
        int sy = (int)srcY;

        // BGR format: 3 bytes per pixel
        int srcIdx = sy * srcPitch + sx * 3;
        b = src[srcIdx + 0];
        g = src[srcIdx + 1];
        r = src[srcIdx + 2];
    } else {
        r = g = b = padValue * 255.0f;
    }

    int pixelIdx = y * dstW + x;
    dst[0 * dstH * dstW + pixelIdx] = r / 255.0f;
    dst[1 * dstH * dstW + pixelIdx] = g / 255.0f;
    dst[2 * dstH * dstW + pixelIdx] = b / 255.0f;
}

extern "C" void launchPreprocessBGRToRGBFloatNN(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int dstW, int dstH,
    float* scaleRatio,
    cudaStream_t stream
) {
    float scaleX = (float)dstW / srcW;
    float scaleY = (float)dstH / srcH;
    float scale = fminf(scaleX, scaleY);

    int newW = (int)(srcW * scale);
    int newH = (int)(srcH * scale);

    *scaleRatio = 1.0f / scale;

    float padValue = 114.0f / 255.0f;

    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

    preprocessBGRToRGBFloatNN<<<grid, block, 0, stream>>>(
        (const unsigned char*)src,
        (float*)dst,
        srcW, srcH, srcPitch,
        dstW, dstH,
        scale, newW, newH, padValue
    );
}

// =============================================================================
// GPU Transpose kernel for YOLO output [1, 56, 8400] -> [8400, 56]
// =============================================================================
__global__ void transposeYoloOutput(
    const float* __restrict__ src,  // [56, 8400] in row-major
    float* __restrict__ dst,         // [8400, 56] in row-major
    int rows,  // 56
    int cols   // 8400
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0..8399
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0..55

    if (col < cols && row < rows) {
        // src[row][col] = src[row * cols + col]
        // dst[col][row] = dst[col * rows + row]
        dst[col * rows + row] = src[row * cols + col];
    }
}

extern "C" void launchTransposeYoloOutput(
    const float* src, float* dst,
    int rows, int cols,
    cudaStream_t stream
) {
    dim3 block(32, 8);  // 256 threads per block
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    transposeYoloOutput<<<grid, block, 0, stream>>>(src, dst, rows, cols);
}

// =============================================================================
// 90° CCW rotation for BGRx frame (for InsightFace to see upright faces)
// Input: 1280x720 landscape BGRx (people appear sideways)
// Output: 720x1280 portrait BGRx (people appear upright)
// =============================================================================
__global__ void rotateBGRx90CCW(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int srcW, int srcH, int srcPitch,   // 1280, 720
    int dstW, int dstH                   // 720, 1280
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) return;

    // 90° CCW: dst(x,y) = src(srcW - 1 - y, x)
    int srcX = srcW - 1 - y;
    int srcY = x;

    int srcIdx = srcY * srcPitch + srcX * 4;
    int dstIdx = y * dstW * 4 + x * 4;

    dst[dstIdx + 0] = src[srcIdx + 0];  // B
    dst[dstIdx + 1] = src[srcIdx + 1];  // G
    dst[dstIdx + 2] = src[srcIdx + 2];  // R
    dst[dstIdx + 3] = src[srcIdx + 3];  // x
}

extern "C" void launchRotateBGRx90CCW(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    cudaStream_t stream
) {
    int dstW = srcH;  // 720
    int dstH = srcW;  // 1280

    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

    rotateBGRx90CCW<<<grid, block, 0, stream>>>(
        (const unsigned char*)src, (unsigned char*)dst,
        srcW, srcH, srcPitch, dstW, dstH
    );
}

// =============================================================================
// GPU BGRx to BGR conversion kernel
// =============================================================================
__global__ void convertBGRxToBGR(
    const unsigned char* __restrict__ src,  // BGRx (4 channels)
    unsigned char* __restrict__ dst,         // BGR (3 channels)
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int srcIdx = (y * width + x) * 4;
        int dstIdx = (y * width + x) * 3;
        dst[dstIdx + 0] = src[srcIdx + 0];  // B
        dst[dstIdx + 1] = src[srcIdx + 1];  // G
        dst[dstIdx + 2] = src[srcIdx + 2];  // R
    }
}

extern "C" void launchConvertBGRxToBGR(
    const void* src, void* dst,
    int width, int height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    convertBGRxToBGR<<<grid, block, 0, stream>>>(
        (const unsigned char*)src, (unsigned char*)dst, width, height
    );
}

// =============================================================================
// GPU BGR to BGRx conversion kernel (for process_image - phone selfie input)
// =============================================================================
__global__ void convertBGRToBGRx(
    const unsigned char* __restrict__ src,  // BGR (3 channels)
    unsigned char* __restrict__ dst,         // BGRx (4 channels)
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int srcIdx = (y * width + x) * 3;
        int dstIdx = (y * width + x) * 4;
        dst[dstIdx + 0] = src[srcIdx + 0];  // B
        dst[dstIdx + 1] = src[srcIdx + 1];  // G
        dst[dstIdx + 2] = src[srcIdx + 2];  // R
        dst[dstIdx + 3] = 255;               // X (padding)
    }
}

extern "C" void launchConvertBGRToBGRx(
    const void* src, void* dst,
    int width, int height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    convertBGRToBGRx<<<grid, block, 0, stream>>>(
        (const unsigned char*)src, (unsigned char*)dst, width, height
    );
}

// =============================================================================
// ReID (OSNet) preprocessing kernel
// Crops person bbox, resizes to 256x128, converts to RGB float CHW with ImageNet normalization
// =============================================================================

/**
 * Preprocess person crop for OSNet ReID
 * Input: BGRx frame with person bbox
 * Output: 256x128 RGB float CHW with ImageNet normalization
 *
 * ImageNet normalization:
 *   mean = [0.485, 0.456, 0.406] (RGB)
 *   std  = [0.229, 0.224, 0.225] (RGB)
 *   output = (pixel/255 - mean) / std
 */
__global__ void cropPersonToReID(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int srcW, int srcH, int srcPitch,
    int cropX, int cropY, int cropW, int cropH
) {
    // OSNet input: 256 height x 128 width
    const int DST_H = 256;
    const int DST_W = 128;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DST_W || y >= DST_H) return;

    // ImageNet normalization constants
    const float mean_r = 0.485f, mean_g = 0.456f, mean_b = 0.406f;
    const float std_r = 0.229f, std_g = 0.224f, std_b = 0.225f;

    // Map destination to source (simple resize, no padding - person fills frame)
    float srcX = cropX + (x + 0.5f) * cropW / DST_W;
    float srcY = cropY + (y + 0.5f) * cropH / DST_H;

    // Clamp to source bounds
    srcX = fmaxf(0.0f, fminf(srcX, srcW - 1.0f));
    srcY = fmaxf(0.0f, fminf(srcY, srcH - 1.0f));

    // Bilinear interpolation
    int x0 = (int)floorf(srcX);
    int y0 = (int)floorf(srcY);
    int x1 = min(x0 + 1, srcW - 1);
    int y1 = min(y0 + 1, srcH - 1);

    float xFrac = srcX - x0;
    float yFrac = srcY - y0;

    // Sample BGRx
    float b00 = src[y0 * srcPitch + x0 * 4 + 0];
    float g00 = src[y0 * srcPitch + x0 * 4 + 1];
    float r00 = src[y0 * srcPitch + x0 * 4 + 2];

    float b01 = src[y0 * srcPitch + x1 * 4 + 0];
    float g01 = src[y0 * srcPitch + x1 * 4 + 1];
    float r01 = src[y0 * srcPitch + x1 * 4 + 2];

    float b10 = src[y1 * srcPitch + x0 * 4 + 0];
    float g10 = src[y1 * srcPitch + x0 * 4 + 1];
    float r10 = src[y1 * srcPitch + x0 * 4 + 2];

    float b11 = src[y1 * srcPitch + x1 * 4 + 0];
    float g11 = src[y1 * srcPitch + x1 * 4 + 1];
    float r11 = src[y1 * srcPitch + x1 * 4 + 2];

    // Bilinear blend
    float b = (b00 * (1 - xFrac) + b01 * xFrac) * (1 - yFrac) +
              (b10 * (1 - xFrac) + b11 * xFrac) * yFrac;
    float g = (g00 * (1 - xFrac) + g01 * xFrac) * (1 - yFrac) +
              (g10 * (1 - xFrac) + g11 * xFrac) * yFrac;
    float r = (r00 * (1 - xFrac) + r01 * xFrac) * (1 - yFrac) +
              (r10 * (1 - xFrac) + r11 * xFrac) * yFrac;

    // Normalize to [0, 1] then apply ImageNet normalization
    r = ((r / 255.0f) - mean_r) / std_r;
    g = ((g / 255.0f) - mean_g) / std_g;
    b = ((b / 255.0f) - mean_b) / std_b;

    // Write CHW format (channels first)
    int pixelIdx = y * DST_W + x;
    dst[0 * DST_H * DST_W + pixelIdx] = r;  // R channel
    dst[1 * DST_H * DST_W + pixelIdx] = g;  // G channel
    dst[2 * DST_H * DST_W + pixelIdx] = b;  // B channel
}

extern "C" void launchCropPersonToReID(
    const void* src, void* dst,
    int srcW, int srcH, int srcPitch,
    int cropX, int cropY, int cropW, int cropH,
    cudaStream_t stream
) {
    const int DST_H = 256;
    const int DST_W = 128;

    dim3 block(16, 16);
    dim3 grid((DST_W + block.x - 1) / block.x, (DST_H + block.y - 1) / block.y);

    cropPersonToReID<<<grid, block, 0, stream>>>(
        (const unsigned char*)src,
        (float*)dst,
        srcW, srcH, srcPitch,
        cropX, cropY, cropW, cropH
    );
}

// =============================================================================
// Convert 112x112 BGR uint8 to RGB float CHW with ArcFace normalization
// For processing pre-aligned faces from Python InsightFace norm_crop
// =============================================================================
__global__ void convertAlignedBGRToArcFaceInput(
    const unsigned char* __restrict__ src,  // 112x112 BGR (HWC)
    float* __restrict__ dst                  // 112x112 RGB float (CHW)
) {
    const int SIZE = 112;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= SIZE || y >= SIZE) return;

    // Read BGR
    int srcIdx = (y * SIZE + x) * 3;
    float b = src[srcIdx + 0];
    float g = src[srcIdx + 1];
    float r = src[srcIdx + 2];

    // Normalize to [-1, 1] for ArcFace (same as InsightFace)
    r = (r - 127.5f) / 127.5f;
    g = (g - 127.5f) / 127.5f;
    b = (b - 127.5f) / 127.5f;

    // Write CHW format (channels first)
    int pixelIdx = y * SIZE + x;
    dst[0 * SIZE * SIZE + pixelIdx] = r;  // R channel
    dst[1 * SIZE * SIZE + pixelIdx] = g;  // G channel
    dst[2 * SIZE * SIZE + pixelIdx] = b;  // B channel
}

extern "C" void launchConvertAlignedBGRToArcFaceInput(
    const void* src, void* dst,
    cudaStream_t stream
) {
    const int SIZE = 112;
    dim3 block(16, 16);
    dim3 grid((SIZE + block.x - 1) / block.x, (SIZE + block.y - 1) / block.y);
    convertAlignedBGRToArcFaceInput<<<grid, block, 0, stream>>>(
        (const unsigned char*)src, (float*)dst
    );
}

} // namespace mmoment
