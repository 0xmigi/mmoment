import { IPFSService } from './ipfs-service';
import { pinataService } from './pinata-service';
import { filebaseService } from './filebase-service';

// Create and configure the unified IPFS service
const unifiedIpfsService = new IPFSService();

// Add providers in priority order (Pinata first, Filebase as backup)
unifiedIpfsService.setPrimaryProvider(pinataService);
unifiedIpfsService.setBackupProvider(filebaseService);

// Set up periodic pin checking (every 6 hours)
const CHECK_INTERVAL = 6 * 60 * 60 * 1000; // 6 hours in milliseconds

async function checkAndRepinAllMedia() {
  try {
    // Get all media from primary provider
    const allMedia = await unifiedIpfsService.getMediaForWallet('all');
    
    // Check and repin each item
    for (const media of allMedia) {
      await unifiedIpfsService.checkAndRepinMedia(media.id);
    }
  } catch (error) {
    console.error('Failed to check and repin media:', error);
  }
}

// Start periodic checking if we're in a browser environment
if (typeof window !== 'undefined') {
  // Initial check after 5 minutes (to allow the app to fully load)
  setTimeout(checkAndRepinAllMedia, 5 * 60 * 1000);
  
  // Then check periodically
  setInterval(checkAndRepinAllMedia, CHECK_INTERVAL);
}

export { unifiedIpfsService }; 