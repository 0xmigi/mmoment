import { IPFSService } from './ipfs-service';
import { pinataService } from './pinata-service';
// Remove filebase import as it's not used anymore
// import { filebaseService } from './filebase-service';

// Create and configure the unified IPFS service
const unifiedIpfsService = new IPFSService();

// Log Pinata configuration for debugging
console.log('Configuring IPFS with Pinata provider');

// Add Pinata as the only provider - Filebase is no longer used
unifiedIpfsService.setPrimaryProvider(pinataService);
// unifiedIpfsService.setBackupProvider(filebaseService);

// Log when the IPFS service is ready
console.log('IPFS service initialized with primary provider:', pinataService.name);

// Set up periodic pin checking (every 6 hours)
const CHECK_INTERVAL = 6 * 60 * 60 * 1000; // 6 hours in milliseconds

async function checkAndRepinAllMedia() {
  try {
    // Get all media from primary provider
    const allMedia = await unifiedIpfsService.getMediaForWallet('all');
    console.log(`Found ${allMedia.length} media items to check for repinning`);
    
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
  setTimeout(() => {
    console.log('Starting initial IPFS media check');
    checkAndRepinAllMedia();
  }, 5 * 60 * 1000);
  
  // Then check periodically
  setInterval(() => {
    console.log('Running periodic IPFS media check');
    checkAndRepinAllMedia();
  }, CHECK_INTERVAL);
}

export { unifiedIpfsService }; 