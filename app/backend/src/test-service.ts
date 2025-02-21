import { getFilebaseService } from './services/filebase.service';
import { config } from 'dotenv';
import * as path from 'path';

// Load environment variables
config({ path: path.resolve(__dirname, '../.env') });

async function testService() {
  console.log('Testing FilebaseService...');

  // Create a mock file
  const mockFile: Express.Multer.File = {
    fieldname: 'file',
    originalname: 'test-image.jpg',
    encoding: '7bit',
    mimetype: 'image/jpeg',
    buffer: Buffer.from('Mock image content'),
    size: 123,
    stream: null as any,
    destination: '',
    filename: '',
    path: ''
  };

  const walletAddress = 'test-wallet-' + Date.now();

  try {
    const service = getFilebaseService();

    // Test upload
    console.log('\nTesting upload...');
    const uploadResult = await service.uploadFile(mockFile, walletAddress, 'image');
    console.log('Upload result:', uploadResult);

    // Test fetching media
    console.log('\nTesting getMediaForWallet...');
    const media = await service.getMediaForWallet(walletAddress);
    console.log('Media for wallet:', media);

    // Test pin status
    console.log('\nTesting checkPinStatus...');
    const isPinned = await service.checkPinStatus(media[0]?.id);
    console.log('Pin status:', isPinned);

    // Test delete
    if (media.length > 0) {
      console.log('\nTesting delete...');
      const deleteResult = await service.deleteMedia(media[0].id);
      console.log('Delete result:', deleteResult);
    }

    console.log('\nAll tests completed successfully!');
  } catch (error) {
    console.error('Test failed:', error);
    process.exit(1);
  }
}

// Run the test
testService(); 