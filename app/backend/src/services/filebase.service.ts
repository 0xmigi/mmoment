import { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand, ListObjectsCommand } from '@aws-sdk/client-s3';

export interface UploadResult {
  url: string;
  cid: string;
}

export class FilebaseService {
  private readonly gateway = 'https://ipfs.filebase.io';
  private readonly s3Client: S3Client;
  private readonly FILEBASE_KEY: string;
  private readonly FILEBASE_SECRET: string;
  private readonly FILEBASE_BUCKET: string;
  private static instance: FilebaseService;

  private constructor() {
    this.FILEBASE_KEY = process.env.FILEBASE_KEY || '';
    this.FILEBASE_SECRET = process.env.FILEBASE_SECRET || '';
    this.FILEBASE_BUCKET = process.env.FILEBASE_BUCKET || 'mmoment';

    if (!this.FILEBASE_KEY || !this.FILEBASE_SECRET) {
      throw new Error('Filebase credentials are not configured');
    }

    this.s3Client = new S3Client({
      endpoint: 'https://s3.filebase.com',
      region: 'us-east-1',
      credentials: {
        accessKeyId: this.FILEBASE_KEY,
        secretAccessKey: this.FILEBASE_SECRET
      },
      forcePathStyle: true
    });
  }

  public static getInstance(): FilebaseService {
    if (!FilebaseService.instance) {
      FilebaseService.instance = new FilebaseService();
    }
    return FilebaseService.instance;
  }

  async uploadFile(file: Express.Multer.File, walletAddress: string, type: 'image' | 'video'): Promise<UploadResult> {
    try {
      const fileName = `${walletAddress}_${Date.now()}.${type === 'video' ? 'mp4' : 'jpg'}`;
      console.log('Uploading file:', {
        fileName,
        size: file.size,
        mimeType: file.mimetype,
        walletAddress
      });

      const command = new PutObjectCommand({
        Bucket: this.FILEBASE_BUCKET,
        Key: fileName,
        Body: file.buffer,
        ContentType: type === 'video' ? 'video/mp4' : 'image/jpeg',
        ACL: 'public-read',
        Metadata: {
          'wallet-address': walletAddress,
          'timestamp': Date.now().toString(),
          'type': type
        }
      });

      const response = await this.s3Client.send(command);
      
      if (!response.ETag) {
        throw new Error('Upload failed - no ETag received');
      }

      const cid = response.ETag.replace(/"/g, '');
      const url = `${this.gateway}/ipfs/${cid}`;

      console.log('Upload successful:', { fileName, cid, url });
      return { url, cid };
    } catch (error) {
      console.error('Upload failed:', error);
      throw error;
    }
  }

  async getMediaForWallet(walletAddress: string) {
    try {
      console.log('Fetching media for wallet:', walletAddress);
      
      const command = new ListObjectsCommand({
        Bucket: this.FILEBASE_BUCKET,
        Prefix: walletAddress
      });

      const response = await this.s3Client.send(command);
      
      if (!response.Contents) {
        return [];
      }

      return response.Contents.map(item => ({
        id: item.Key || '',
        url: `${this.gateway}/ipfs/${item.ETag?.replace(/"/g, '')}`,
        type: item.Key?.endsWith('.mp4') ? 'video' : 'image',
        mimeType: item.Key?.endsWith('.mp4') ? 'video/mp4' : 'image/jpeg',
        walletAddress,
        timestamp: item.LastModified?.toISOString() || new Date().toISOString(),
        provider: 'Filebase'
      }));
    } catch (error) {
      console.error('Failed to fetch media:', error);
      throw error;
    }
  }

  async deleteMedia(ipfsHash: string) {
    try {
      console.log('Deleting media:', ipfsHash);
      
      const command = new DeleteObjectCommand({
        Bucket: this.FILEBASE_BUCKET,
        Key: ipfsHash
      });

      await this.s3Client.send(command);
      return true;
    } catch (error) {
      console.error('Failed to delete media:', error);
      throw error;
    }
  }

  async checkPinStatus(ipfsHash: string): Promise<boolean> {
    try {
      console.log('Checking pin status for:', ipfsHash);
      
      const command = new GetObjectCommand({
        Bucket: this.FILEBASE_BUCKET,
        Key: ipfsHash
      });

      await this.s3Client.send(command);
      return true;
    } catch (error) {
      console.error('Failed to check pin status:', error);
      return false;
    }
  }
}

export const getFilebaseService = () => FilebaseService.getInstance(); 