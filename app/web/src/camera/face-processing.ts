/**
 * Face Processing Service for Phone-based Enrollment
 *
 * This service handles face image capture and sends it to a connected
 * Jetson camera for processing with InsightFace to generate compatible
 * 512-dimensional embeddings.
 */

export interface FaceProcessingResult {
  success: boolean;
  embedding?: number[];
  error?: string;
  quality?: FaceQualityMetrics;
}

export interface FaceQualityMetrics {
  hasFace: boolean;
  faceCount: number;
  brightness: number; // Average brightness (0-255)
  contrast: number; // Contrast ratio (0-100)
  faceSize: number; // Face area relative to image (0-1)
  issues: string[]; // List of quality issues detected
}

class FaceProcessingService {
  private canvas: HTMLCanvasElement | null = null;
  private context: CanvasRenderingContext2D | null = null;

  constructor() {
    // Create a hidden canvas for image processing
    this.canvas = document.createElement("canvas");
    this.context = this.canvas.getContext("2d", { willReadFrequently: true });
  }

  /**
   * Process a selfie image by sending it to the Jetson camera
   *
   * @param imageData Base64 encoded image data or blob URL
   * @param cameraUrl URL of the connected Jetson camera
   * @returns Promise resolving to facial embedding from Jetson
   */
  async processFacialEmbedding(
    imageData: string,
    cameraUrl: string
  ): Promise<FaceProcessingResult> {
    try {
      // First perform quality checks on the image
      const qualityMetrics = await this.analyzeImageQuality(imageData);

      // Check for critical quality issues
      if (qualityMetrics.issues.length > 0) {
        const criticalIssues = qualityMetrics.issues.filter(
          (issue) =>
            issue.includes("No face") ||
            issue.includes("Multiple faces") ||
            issue.includes("too dark") ||
            issue.includes("too bright")
        );

        if (criticalIssues.length > 0) {
          return {
            success: false,
            error: criticalIssues.join(". "),
            quality: qualityMetrics,
          };
        }
      }

      // Send image to Jetson camera for processing
      const response = await fetch(`${cameraUrl}/api/face/extract-embedding`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: imageData,
          // Include session info if needed
        }),
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`Jetson processing failed: ${error}`);
      }

      const result = await response.json();

      if (!result.success) {
        return {
          success: false,
          error: result.error || "Failed to extract facial embedding",
          quality: qualityMetrics,
        };
      }

      // Validate the embedding
      if (
        !result.embedding ||
        !Array.isArray(result.embedding) ||
        result.embedding.length !== 512
      ) {
        return {
          success: false,
          error: "Invalid embedding received from camera",
          quality: qualityMetrics,
        };
      }

      return {
        success: true,
        embedding: result.embedding,
        quality: qualityMetrics,
      };
    } catch (error) {
      console.error("Face processing error:", error);
      return {
        success: false,
        error:
          error instanceof Error ? error.message : "Face processing failed",
      };
    }
  }

  /**
   * Analyze image quality and detect basic face presence
   * This is for UI feedback only - actual face processing happens on Jetson
   */
  async analyzeImageQuality(imageData: string): Promise<FaceQualityMetrics> {
    if (!this.canvas || !this.context) {
      return {
        hasFace: false,
        faceCount: 0,
        brightness: 0,
        contrast: 0,
        faceSize: 0,
        issues: ["Image processing not available"],
      };
    }

    try {
      // Load image
      const image = await this.loadImage(imageData);

      // Set canvas size to image size
      this.canvas.width = image.width;
      this.canvas.height = image.height;

      // Draw image to canvas
      this.context.drawImage(image, 0, 0);

      // Get image data for analysis
      const imageDataObj = this.context.getImageData(
        0,
        0,
        image.width,
        image.height
      );
      const data = imageDataObj.data;

      // Calculate brightness and contrast
      let brightness = 0;
      let min = 255;
      let max = 0;

      for (let i = 0; i < data.length; i += 4) {
        const gray =
          0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        brightness += gray;
        min = Math.min(min, gray);
        max = Math.max(max, gray);
      }

      brightness = brightness / (data.length / 4);
      const contrast = ((max - min) / 255) * 100;

      // Check for face-like region in center (simplified check)
      const centerX = image.width / 2;
      const centerY = image.height / 2;
      const checkRadius = Math.min(image.width, image.height) * 0.3;

      // Sample center region for skin-tone colors
      let skinPixels = 0;
      let totalPixels = 0;

      for (let y = centerY - checkRadius; y < centerY + checkRadius; y += 5) {
        for (let x = centerX - checkRadius; x < centerX + checkRadius; x += 5) {
          if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
            const idx = (y * image.width + x) * 4;
            const r = data[idx];
            const g = data[idx + 1];
            const b = data[idx + 2];

            // Basic skin tone detection
            if (
              r > 95 &&
              g > 40 &&
              b > 20 &&
              r > g &&
              r > b &&
              Math.abs(r - g) > 15 &&
              Math.max(r, g, b) - Math.min(r, g, b) > 15
            ) {
              skinPixels++;
            }
            totalPixels++;
          }
        }
      }

      const skinRatio = totalPixels > 0 ? skinPixels / totalPixels : 0;
      const hasFace = skinRatio > 0.2; // At least 20% skin-colored pixels

      // Estimate face size
      const faceSize = hasFace ? skinRatio : 0;

      // Collect issues
      const issues: string[] = [];

      if (!hasFace) {
        issues.push("No face detected in image");
      }

      if (brightness < 50) {
        issues.push("Image too dark");
      } else if (brightness > 200) {
        issues.push("Image too bright");
      }

      if (contrast < 20) {
        issues.push("Low contrast - improve lighting");
      }

      if (hasFace && faceSize < 0.15) {
        issues.push("Face too small - move closer to camera");
      }

      return {
        hasFace,
        faceCount: hasFace ? 1 : 0, // Simplified - can't detect multiple faces without ML
        brightness,
        contrast,
        faceSize,
        issues,
      };
    } catch (error) {
      console.error("Quality analysis error:", error);
      return {
        hasFace: false,
        faceCount: 0,
        brightness: 0,
        contrast: 0,
        faceSize: 0,
        issues: ["Failed to analyze image quality"],
      };
    }
  }

  /**
   * Load image from base64 or blob URL
   */
  private loadImage(imageData: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Failed to load image"));
      img.src = imageData;
    });
  }

  /**
   * Clean up resources
   */
  cleanup() {
    this.canvas = null;
    this.context = null;
  }
}

// Export singleton instance
export const faceProcessingService = new FaceProcessingService();
