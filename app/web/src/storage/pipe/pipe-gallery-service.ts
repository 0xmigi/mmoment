
export interface PipeFile {
  id: string;
  name: string;
  size: number;
  contentType: string;
  uploadedAt: string;
  url: string;
  metadata?: any;
}

export interface PipeGalleryItem {
  id: string; // Use cid as id for compatibility
  cid: string;
  name: string;
  url: string;
  type: 'image' | 'video';
  mimeType: string; // Make required to match IPFSMedia
  timestamp: number;
  backupUrls?: string[];
  walletAddress?: string;
  provider?: string;
  transactionId?: string;
  cameraId?: string;
  metadata?: {
    camera?: string;
    location?: string;
    faces?: number;
  };
}

class PipeGalleryService {
  private backendUrl: string;

  constructor() {
    // Use the same backend URL that's running on the local network
    this.backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3001';
  }

  /**
   * Fetch user's files from Pipe storage
   */
  async getUserFiles(walletAddress: string): Promise<PipeGalleryItem[]> {
    try {
      const response = await fetch(`${this.backendUrl}/api/pipe/files/${walletAddress}`);

      if (!response.ok) {
        console.error('Failed to fetch Pipe files:', response.statusText);
        return [];
      }

      const data = await response.json();

      // Transform Pipe files into gallery items
      return data.files.map((file: PipeFile) => {
        // Determine file type from contentType or name
        const isVideo = file.contentType?.startsWith('video/') ||
                        file.name.match(/\.(mp4|webm|ogg|mov)$/i);

        return {
          id: file.id, // Use Pipe file ID as id for compatibility
          cid: file.id, // Use Pipe file ID as CID equivalent
          name: file.name,
          url: file.url,
          type: isVideo ? 'video' : 'image' as 'image' | 'video',
          mimeType: file.contentType || 'application/octet-stream',
          timestamp: new Date(file.uploadedAt).getTime(),
          backupUrls: [],
          walletAddress: undefined,
          provider: 'pipe',
          transactionId: undefined,
          cameraId: file.metadata?.camera,
          metadata: file.metadata
        };
      });
    } catch (error) {
      console.error('Error fetching Pipe files:', error);
      return [];
    }
  }

  /**
   * Get a specific file from Pipe storage
   */
  async getFile(walletAddress: string, fileId: string): Promise<PipeGalleryItem | null> {
    try {
      const response = await fetch(`${this.backendUrl}/api/pipe/file/${walletAddress}/${fileId}`);

      if (!response.ok) {
        console.error('Failed to fetch Pipe file:', response.statusText);
        return null;
      }

      const file: PipeFile = await response.json();

      // Determine file type from contentType or name
      const isVideo = file.contentType?.startsWith('video/') ||
                      file.name.match(/\.(mp4|webm|ogg|mov)$/i);

      return {
        id: file.id,
        cid: file.id,
        name: file.name,
        url: file.url,
        type: isVideo ? 'video' : 'image' as 'image' | 'video',
        mimeType: file.contentType || 'application/octet-stream',
        timestamp: new Date(file.uploadedAt).getTime(),
        backupUrls: [],
        walletAddress: undefined,
        provider: 'pipe',
        transactionId: undefined,
        cameraId: file.metadata?.camera,
        metadata: file.metadata
      };
    } catch (error) {
      console.error('Error fetching Pipe file:', error);
      return null;
    }
  }

  /**
   * Search files by metadata or name
   */
  async searchFiles(walletAddress: string, query: string): Promise<PipeGalleryItem[]> {
    try {
      const allFiles = await this.getUserFiles(walletAddress);

      // Filter files based on query
      return allFiles.filter(file => {
        const nameMatch = file.name.toLowerCase().includes(query.toLowerCase());
        const cameraMatch = file.metadata?.camera?.toLowerCase().includes(query.toLowerCase());
        const locationMatch = file.metadata?.location?.toLowerCase().includes(query.toLowerCase());

        return nameMatch || cameraMatch || locationMatch;
      });
    } catch (error) {
      console.error('Error searching Pipe files:', error);
      return [];
    }
  }

  /**
   * Get recent files with pagination
   */
  async getRecentFiles(
    walletAddress: string,
    limit: number = 20,
    offset: number = 0
  ): Promise<PipeGalleryItem[]> {
    try {
      const allFiles = await this.getUserFiles(walletAddress);

      // Sort by timestamp descending and paginate
      return allFiles
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(offset, offset + limit);
    } catch (error) {
      console.error('Error fetching recent Pipe files:', error);
      return [];
    }
  }

  /**
   * Check if Pipe storage is enabled
   */
  isPipeStorageEnabled(): boolean {
    return localStorage.getItem('mmoment_storage_type') === 'pipe';
  }

  /**
   * Get the storage type preference
   */
  getStorageType(): 'pipe' | 'pinata' {
    const stored = localStorage.getItem('mmoment_storage_type');
    return stored === 'pipe' ? 'pipe' : 'pinata';
  }
}

export const pipeGalleryService = new PipeGalleryService();