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
  transactionBuffer?: string; // Base64 encoded transaction from Jetson
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
    options: { encrypt?: boolean; requestQuality?: boolean; walletAddress?: string; buildTransaction?: boolean } = {}
  ): Promise<EnhancedFaceProcessingResult> {
    try {
      // MUST have a Jetson camera URL - no fake local processing
      if (!cameraUrl) {
        return {
          success: false,
          error: 'No camera URL provided - face processing requires Jetson camera connection'
        };
      }

      return await this.processWithJetsonEndpoint(imageData, cameraUrl, options);

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
    options: { encrypt?: boolean; requestQuality?: boolean; walletAddress?: string; buildTransaction?: boolean } = {}
  ): Promise<EnhancedFaceProcessingResult> {
    try {
      console.log('[FaceProcessing] Using enhanced Jetson endpoint:', cameraUrl);
      console.log('[FaceProcessing] Raw image data length:', imageData.length);
      console.log('[FaceProcessing] Image data prefix:', imageData.substring(0, 50));

      // The Jetson expects raw base64 without data:image prefix
      let processedImageData = imageData;

      // Remove data:image/jpeg;base64, prefix if present
      if (imageData.includes(',')) {
        processedImageData = imageData.split(',')[1];
      }

      // Validate we have actual image data
      if (!processedImageData || processedImageData.length < 100) {
        throw new Error('Invalid image data - too short or empty');
      }

      console.log('[FaceProcessing] Image data length after processing:', processedImageData.length);
      console.log('[FaceProcessing] Image data starts with:', processedImageData.substring(0, 20));

      const payload: any = {
        image_data: processedImageData,
        quality_assessment: options.requestQuality !== false,
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

      // Add transaction building option if requested
      if (options.buildTransaction) {
        payload.build_transaction = true;
      }

      // Call the enhanced Jetson endpoint
      const endpoint = `${cameraUrl}/api/face/extract-embedding`;
      console.log('[FaceProcessing] Making request to:', endpoint);
      console.log('[FaceProcessing] Request payload size:', JSON.stringify(payload).length, 'bytes');

      let response;
      try {
        response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          mode: 'cors',
          credentials: 'omit',
          body: JSON.stringify(payload)
        });
      } catch (fetchError) {
        console.error('[FaceProcessing] Network request failed:', fetchError);

        // Check if it's a CORS error
        if (fetchError instanceof TypeError && fetchError.message.includes('CORS')) {
          throw new Error(`CORS error - camera may not allow cross-origin requests`);
        }

        throw new Error(`Network error: ${fetchError instanceof Error ? fetchError.message : 'Failed to connect to Jetson'}`);
      }

      console.log('[FaceProcessing] Response status:', response.status);
      console.log('[FaceProcessing] Response headers:', [...response.headers.entries()]);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[FaceProcessing] Jetson error response:', {
          status: response.status,
          statusText: response.statusText,
          errorText,
          url: endpoint
        });

        // Provide specific error messages based on status
        if (response.status === 400) {
          throw new Error(`Image validation failed: ${errorText}`);
        } else if (response.status === 500) {
          throw new Error(`Jetson processing error: ${errorText}`);
        } else if (response.status === 404) {
          throw new Error(`Jetson endpoint not found - camera may be offline`);
        } else {
          throw new Error(`Jetson communication failed (${response.status}): ${errorText}`);
        }
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
        sessionId: result.biometric_session_id || result.session_id,
        transactionBuffer: result.transaction_buffer // Base64 encoded transaction from Jetson
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

      // NO FALLBACK - if Jetson fails, the enrollment fails
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to connect to Jetson camera'
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