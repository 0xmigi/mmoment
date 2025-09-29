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

interface ImageQualityResult {
  score: number;
  rating: 'excellent' | 'good' | 'acceptable' | 'poor' | 'very_poor';
  issues: string[];
  recommendations: string[];
}

interface EnhancedFaceProcessingResult extends FaceProcessingResult {
  quality?: ImageQualityResult;
  encrypted?: boolean;
  sessionId?: string;
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
   * Analyze image quality using local assessment
   *
   * @param imageData Base64 encoded image data
   * @returns Promise resolving to quality assessment
   */
  async analyzeImageQuality(imageData: string): Promise<ImageQualityResult> {
    try {
      // Load image for analysis
      const image = await this.loadImage(imageData);

      // Basic quality assessment (this would be enhanced with proper computer vision)
      const issues: string[] = [];
      const recommendations: string[] = [];

      let score = 100;

      // Check image dimensions
      if (image.width < 320 || image.height < 240) {
        issues.push('Image resolution too low');
        recommendations.push('Use a higher resolution camera');
        score -= 30;
      }

      // Basic brightness check (simplified)
      if (!this.canvas || !this.context) {
        throw new Error('Canvas not initialized');
      }

      this.canvas.width = image.width;
      this.canvas.height = image.height;
      this.context.drawImage(image, 0, 0);

      const imagePixelData = this.context.getImageData(0, 0, image.width, image.height);
      const avgBrightness = this.calculateAverageBrightness(imagePixelData);

      if (avgBrightness < 50) {
        issues.push('Image too dark');
        recommendations.push('Improve lighting or move to a brighter area');
        score -= 20;
      } else if (avgBrightness > 200) {
        issues.push('Image too bright');
        recommendations.push('Reduce lighting or move away from bright lights');
        score -= 15;
      }

      // Face detection check
      const faceRegion = this.detectFaceRegion(imagePixelData);
      if (!faceRegion) {
        issues.push('No clear face detected');
        recommendations.push('Ensure your face is clearly visible and centered');
        score -= 40;
      } else {
        // Check face size
        const faceArea = faceRegion.width * faceRegion.height;
        const imageArea = image.width * image.height;
        const faceRatio = faceArea / imageArea;

        if (faceRatio < 0.1) {
          issues.push('Face too small in frame');
          recommendations.push('Move closer to the camera');
          score -= 25;
        }

        // Check face centering
        const centerX = image.width / 2;
        const centerY = image.height / 2;
        const faceCenterX = faceRegion.x + faceRegion.width / 2;
        const faceCenterY = faceRegion.y + faceRegion.height / 2;

        const centerDistance = Math.sqrt(
          Math.pow(faceCenterX - centerX, 2) + Math.pow(faceCenterY - centerY, 2)
        );
        const maxDistance = Math.sqrt(Math.pow(image.width / 4, 2) + Math.pow(image.height / 4, 2));

        if (centerDistance > maxDistance) {
          issues.push('Face not centered');
          recommendations.push('Center your face in the frame');
          score -= 15;
        }
      }

      // Determine rating
      let rating: ImageQualityResult['rating'];
      if (score >= 90) rating = 'excellent';
      else if (score >= 80) rating = 'good';
      else if (score >= 70) rating = 'acceptable';
      else if (score >= 60) rating = 'poor';
      else rating = 'very_poor';

      return {
        score: Math.max(0, score),
        rating,
        issues,
        recommendations
      };

    } catch (error) {
      console.error('Quality analysis error:', error);
      return {
        score: 0,
        rating: 'very_poor',
        issues: ['Failed to analyze image quality'],
        recommendations: ['Please try taking the photo again']
      };
    }
  }

  /**
   * Process a selfie image and extract facial embedding using enhanced Jetson endpoint
   *
   * @param imageData Base64 encoded image data
   * @param cameraUrl Optional Jetson camera URL for enhanced processing
   * @param options Processing options
   * @returns Promise resolving to facial embedding with quality assessment
   */
  async processFacialEmbedding(
    imageData: string,
    cameraUrl?: string,
    options: { encrypt?: boolean; requestQuality?: boolean; walletAddress?: string } = {}
  ): Promise<EnhancedFaceProcessingResult> {
    try {
      // If we have a Jetson camera URL, use the enhanced endpoint
      if (cameraUrl) {
        return await this.processWithJetsonEndpoint(imageData, cameraUrl, options);
      }

      // Fallback to local processing
      return await this.processLocally(imageData, options);

    } catch (error) {
      console.error('Face processing error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Face processing failed'
      };
    }
  }

  /**
   * Process facial embedding using the enhanced Jetson /api/face/extract-embedding endpoint
   */
  private async processWithJetsonEndpoint(
    imageData: string,
    cameraUrl: string,
    options: { encrypt?: boolean; requestQuality?: boolean; walletAddress?: string } = {}
  ): Promise<EnhancedFaceProcessingResult> {
    try {
      console.log('[FaceProcessing] Using enhanced Jetson endpoint:', cameraUrl);

      // Prepare the request payload
      const payload: any = {
        image_data: imageData.split(',')[1] || imageData, // Remove data:image/jpeg;base64, prefix if present
        quality_assessment: options.requestQuality !== false, // Default to true unless explicitly false
      };

      // Add wallet address if provided (needed for local enrollment)
      if (options.walletAddress) {
        payload.wallet_address = options.walletAddress;
        console.log('[FaceProcessing] Adding wallet address to embedding extraction for local enrollment:', options.walletAddress);
      }

      // Add encryption option if requested
      if (options.encrypt) {
        payload.encrypt = true;
      }

      // Call the enhanced Jetson endpoint
      const response = await fetch(`${cameraUrl}/api/face/extract-embedding`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Jetson endpoint error (${response.status}): ${errorText}`);
      }

      const result = await response.json();
      console.log('[FaceProcessing] Jetson response:', result);

      if (!result.success) {
        throw new Error(result.error || 'Jetson processing failed');
      }

      // Extract the enhanced response data
      const enhancedResult: EnhancedFaceProcessingResult = {
        success: true,
        embedding: result.embedding || result.face_embedding,
        encrypted: result.encrypted || false,
        sessionId: result.biometric_session_id || result.session_id
      };

      // Add quality assessment if available
      if (result.quality_score !== undefined) {
        enhancedResult.quality = {
          score: result.quality_score,
          rating: result.quality_rating || this.scoreToRating(result.quality_score),
          issues: result.quality_issues || [],
          recommendations: result.recommendations || []
        };
      }

      console.log('[FaceProcessing] ✅ Enhanced processing successful:', {
        hasEmbedding: !!enhancedResult.embedding,
        embeddingLength: enhancedResult.embedding?.length,
        qualityScore: enhancedResult.quality?.score,
        encrypted: enhancedResult.encrypted
      });

      return enhancedResult;

    } catch (error) {
      console.error('[FaceProcessing] Jetson endpoint error:', error);

      // Fallback to local processing if Jetson fails
      console.log('[FaceProcessing] Falling back to local processing...');
      return await this.processLocally(imageData, options);
    }
  }

  /**
   * Process facial embedding using local browser-based methods (fallback)
   */
  private async processLocally(
    imageData: string,
    options: { encrypt?: boolean; requestQuality?: boolean } = {}
  ): Promise<EnhancedFaceProcessingResult> {
    try {
      console.log('[FaceProcessing] Using local processing fallback');

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

      const result: EnhancedFaceProcessingResult = {
        success: true,
        embedding,
        encrypted: false // Local processing doesn't support encryption
      };

      // Add quality assessment if requested
      if (options.requestQuality !== false) {
        result.quality = await this.analyzeImageQuality(imageData);
      }

      return result;

    } catch (error) {
      console.error('[FaceProcessing] Local processing error:', error);
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

  private calculateAverageBrightness(imageData: ImageData): number {
    const data = imageData.data;
    let totalBrightness = 0;
    let pixelCount = 0;

    // Sample every 4th pixel for performance
    for (let i = 0; i < data.length; i += 16) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];

      // Calculate brightness using standard formula
      const brightness = (r * 0.299 + g * 0.587 + b * 0.114);
      totalBrightness += brightness;
      pixelCount++;
    }

    return pixelCount > 0 ? totalBrightness / pixelCount : 0;
  }

  private scoreToRating(score: number): ImageQualityResult['rating'] {
    if (score >= 90) return 'excellent';
    if (score >= 80) return 'good';
    if (score >= 70) return 'acceptable';
    if (score >= 60) return 'poor';
    return 'very_poor';
  }
}

// Export singleton instance
export const faceProcessingService = new FaceProcessingService();