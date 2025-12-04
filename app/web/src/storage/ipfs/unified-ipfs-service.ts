import { IPFSService } from './ipfs-service';
import { pinataService } from './pinata-service';
// Remove filebase import as it's not used anymore
// import { filebaseService } from './filebase-service';

// IPFS is no longer the default storage provider - Pipe Network is now used
// This service is kept for backwards compatibility but background processes are disabled

// Flag to control IPFS background processes (disabled since Pipe is now default)
const IPFS_BACKGROUND_ENABLED = false;

// Create and configure the unified IPFS service
const unifiedIpfsService = new IPFSService();

// Only configure if IPFS background is enabled
if (IPFS_BACKGROUND_ENABLED) {
  console.log('Configuring IPFS with Pinata provider');
  unifiedIpfsService.setPrimaryProvider(pinataService);
  console.log('IPFS service initialized with primary provider:', pinataService.name);
} else {
  console.log('IPFS service created (background processes disabled - using Pipe Network)');
  // Still set provider for on-demand use, but skip background tasks
  unifiedIpfsService.setPrimaryProvider(pinataService);
}

// Set up periodic pin checking (every 6 hours) - DISABLED when using Pipe
const CHECK_INTERVAL = 6 * 60 * 60 * 1000; // 6 hours in milliseconds

async function checkAndRepinAllMedia() {
  if (!IPFS_BACKGROUND_ENABLED) return;

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

// Start periodic checking only if IPFS background is enabled
if (typeof window !== 'undefined' && IPFS_BACKGROUND_ENABLED) {
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