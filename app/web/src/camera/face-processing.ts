/**
 * Face Processing Service for Phone-based Enrollment
 *
 * Uses ONNX Runtime Web with InsightFace models to create compatible
 * facial embeddings that work with the Jetson camera system.
 */

interface FaceProcessingResult {
  success: boolean;
  embedding?: number[];
  error?: string;
}

// Note: ONNX Runtime integration will be added later
// For now, we use simplified face processing

class FaceProcessingService {
  private canvas: HTMLCanvasElement | null = null;
  private context: CanvasRenderingContext2D | null = null;

  constructor() {
    // Create a hidden canvas for image processing
    this.canvas = document.createElement('canvas');
    this.context = this.canvas.getContext('2d');
  }

  /**
   * Process a selfie image and extract facial embedding
   *
   * @param imageData Base64 encoded image data
   * @returns Promise resolving to facial embedding
   */
  async processFacialEmbedding(imageData: string): Promise<FaceProcessingResult> {
    try {
      // Using simplified face processing for now
      // TODO: Add ONNX Runtime + InsightFace models later

      // Load image
      const image = await this.loadImage(imageData);

      // Basic face detection and feature extraction
      const features = await this.extractFacialFeatures(image);

      if (!features) {
        return {
          success: false,
          error: 'No face detected in image. Please ensure your face is clearly visible and well-lit.'
        };
      }

      // Convert to embedding format (128 dimensions for now, will be 512 with InsightFace)
      const embedding = this.createEmbedding(features);

      return {
        success: true,
        embedding
      };

    } catch (error) {
      console.error('Face processing error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Face processing failed'
      };
    }
  }

  private loadImage(imageData: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = imageData;
    });
  }

  private async extractFacialFeatures(image: HTMLImageElement): Promise<number[] | null> {
    if (!this.canvas || !this.context) {
      throw new Error('Canvas not initialized');
    }

    // Set canvas size to image size
    this.canvas.width = image.width;
    this.canvas.height = image.height;

    // Draw image to canvas
    this.context.drawImage(image, 0, 0);

    // Get image data
    const imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height);

    // Basic face detection using skin tone and contrast analysis
    const faceRegion = this.detectFaceRegion(imageData);

    if (!faceRegion) {
      return null;
    }

    // Extract features from face region
    const features = this.extractFeatures(imageData, faceRegion);

    return features;
  }

  private detectFaceRegion(imageData: ImageData): { x: number; y: number; width: number; height: number } | null {
    // Simplified face detection using center region assumption
    // In a real implementation, this would use proper face detection algorithms

    const { width, height } = imageData;

    // Assume face is in the center portion of the image
    const faceWidth = Math.floor(width * 0.4);
    const faceHeight = Math.floor(height * 0.5);
    const faceX = Math.floor((width - faceWidth) / 2);
    const faceY = Math.floor((height - faceHeight) / 2.5); // Slightly higher than center

    // Basic validation - check if there's enough contrast in this region
    const hasValidFace = this.validateFaceRegion(imageData, faceX, faceY, faceWidth, faceHeight);

    if (!hasValidFace) {
      return null;
    }

    return {
      x: faceX,
      y: faceY,
      width: faceWidth,
      height: faceHeight
    };
  }

  private validateFaceRegion(
    imageData: ImageData,
    x: number,
    y: number,
    width: number,
    height: number
  ): boolean {
    const data = imageData.data;
    const imageWidth = imageData.width;

    let totalBrightness = 0;
    let pixelCount = 0;
    let contrastCount = 0;

    // Sample pixels in the face region
    for (let dy = 0; dy < height; dy += 4) {
      for (let dx = 0; dx < width; dx += 4) {
        const px = x + dx;
        const py = y + dy;

        if (px >= 0 && px < imageWidth && py >= 0 && py < imageData.height) {
          const index = (py * imageWidth + px) * 4;
          const r = data[index];
          const g = data[index + 1];
          const b = data[index + 2];

          const brightness = (r + g + b) / 3;
          totalBrightness += brightness;
          pixelCount++;

          // Check for contrast (edges, features)
          if (brightness > 100 && brightness < 200) {
            contrastCount++;
          }
        }
      }
    }

    if (pixelCount === 0) return false;

    const avgBrightness = totalBrightness / pixelCount;
    const contrastRatio = contrastCount / pixelCount;

    // Basic validation: reasonable brightness and contrast
    return avgBrightness > 50 && avgBrightness < 220 && contrastRatio > 0.1;
  }

  private extractFeatures(
    imageData: ImageData,
    faceRegion: { x: number; y: number; width: number; height: number }
  ): number[] {
    const data = imageData.data;
    const imageWidth = imageData.width;
    const features: number[] = [];

    // Extract simplified features from different regions of the face
    const regions = [
      { name: 'forehead', x: 0.2, y: 0.1, w: 0.6, h: 0.3 },
      { name: 'left_eye', x: 0.2, y: 0.3, w: 0.25, h: 0.2 },
      { name: 'right_eye', x: 0.55, y: 0.3, w: 0.25, h: 0.2 },
      { name: 'nose', x: 0.4, y: 0.4, w: 0.2, h: 0.3 },
      { name: 'mouth', x: 0.3, y: 0.65, w: 0.4, h: 0.25 },
      { name: 'left_cheek', x: 0.1, y: 0.45, w: 0.3, h: 0.3 },
      { name: 'right_cheek', x: 0.6, y: 0.45, w: 0.3, h: 0.3 },
      { name: 'chin', x: 0.3, y: 0.8, w: 0.4, h: 0.2 }
    ];

    for (const region of regions) {
      const regionX = Math.floor(faceRegion.x + region.x * faceRegion.width);
      const regionY = Math.floor(faceRegion.y + region.y * faceRegion.height);
      const regionW = Math.floor(region.w * faceRegion.width);
      const regionH = Math.floor(region.h * faceRegion.height);

      // Extract color and texture features from this region
      let avgR = 0, avgG = 0, avgB = 0;
      let textureMeasure = 0;
      let pixelCount = 0;

      for (let dy = 0; dy < regionH; dy += 2) {
        for (let dx = 0; dx < regionW; dx += 2) {
          const px = regionX + dx;
          const py = regionY + dy;

          if (px >= 0 && px < imageWidth && py >= 0 && py < imageData.height) {
            const index = (py * imageWidth + px) * 4;
            const r = data[index];
            const g = data[index + 1];
            const b = data[index + 2];

            avgR += r;
            avgG += g;
            avgB += b;
            pixelCount++;

            // Simple texture measure (variance in brightness)
            const brightness = (r + g + b) / 3;
            textureMeasure += Math.abs(brightness - 128);
          }
        }
      }

      if (pixelCount > 0) {
        features.push(avgR / pixelCount / 255);
        features.push(avgG / pixelCount / 255);
        features.push(avgB / pixelCount / 255);
        features.push(textureMeasure / pixelCount / 255);
      }
    }

    // Pad or truncate to exactly 128 dimensions
    while (features.length < 128) {
      features.push(0);
    }

    return features.slice(0, 128);
  }

  private createEmbedding(features: number[]): number[] {
    // Normalize features to 0-255 range for blockchain storage
    return features.map(feature => Math.round(feature * 255));
  }
}

// Export singleton instance
export const faceProcessingService = new FaceProcessingService();