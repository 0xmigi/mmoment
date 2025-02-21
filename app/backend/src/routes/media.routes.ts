import { Router, Response, Request } from 'express';
import multer from 'multer';
import { getFilebaseService } from '../services/filebase.service';

const router = Router();

// Configure multer for memory storage
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  }
});

// Upload media file
router.post('/media/upload', upload.single('file'), async (req: Request, res: Response): Promise<void> => {
  try {
    console.log('Upload request received:', {
      hasFile: !!req.file,
      fileSize: req.file?.size,
      mimeType: req.file?.mimetype,
      walletAddress: req.body?.walletAddress,
      type: req.body?.type
    });

    if (!req.file) {
      res.status(400).json({ error: 'No file provided' });
      return;
    }

    const walletAddress = req.body.walletAddress;
    if (!walletAddress) {
      res.status(400).json({ error: 'Wallet address is required' });
      return;
    }

    const type = req.body.type as 'image' | 'video';
    if (!type || !['image', 'video'].includes(type)) {
      res.status(400).json({ error: 'Invalid media type' });
      return;
    }

    const result = await getFilebaseService().uploadFile(req.file, walletAddress, type);
    res.json(result);
  } catch (error) {
    console.error('Upload failed:', error);
    res.status(500).json({ 
      error: 'Failed to upload file',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Get media for a wallet
router.get('/media/:walletAddress', async (req: Request, res: Response): Promise<void> => {
  try {
    const { walletAddress } = req.params;
    const media = await getFilebaseService().getMediaForWallet(walletAddress);
    res.json({ media });
  } catch (error) {
    console.error('Failed to fetch media:', error);
    res.status(500).json({ 
      error: 'Failed to fetch media',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Delete media
router.delete('/media/:ipfsHash', async (req: Request, res: Response): Promise<void> => {
  try {
    const { ipfsHash } = req.params;
    const success = await getFilebaseService().deleteMedia(ipfsHash);
    res.json({ success });
  } catch (error) {
    console.error('Failed to delete media:', error);
    res.status(500).json({ 
      error: 'Failed to delete media',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Check pin status
router.get('/media/:ipfsHash/status', async (req: Request, res: Response): Promise<void> => {
  try {
    const { ipfsHash } = req.params;
    const isPinned = await getFilebaseService().checkPinStatus(ipfsHash);
    res.json({ isPinned });
  } catch (error) {
    console.error('Failed to check pin status:', error);
    res.status(500).json({ 
      error: 'Failed to check pin status',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router; 