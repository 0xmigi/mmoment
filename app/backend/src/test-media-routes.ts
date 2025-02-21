import axios, { AxiosError } from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import path from 'path';

const API_URL = 'http://localhost:3001/api';
const TEST_WALLET = `test-wallet-${Date.now()}`;

async function testMediaRoutes() {
  try {
    console.log('Testing media routes with wallet:', TEST_WALLET);

    // Create a test image file
    const testImagePath = path.join(__dirname, 'test-image.jpg');
    const imageData = Buffer.alloc(1024, 'test image content');
    fs.writeFileSync(testImagePath, imageData);

    // Test file upload
    console.log('\n1. Testing file upload...');
    const formData = new FormData();
    formData.append('file', fs.createReadStream(testImagePath));
    formData.append('walletAddress', TEST_WALLET);
    formData.append('type', 'image');

    const uploadResponse = await axios.post(`${API_URL}/media/upload`, formData, {
      headers: formData.getHeaders()
    });
    console.log('Upload response:', uploadResponse.data);

    const ipfsHash = uploadResponse.data.cid;

    // Test get media for wallet
    console.log('\n2. Testing get media for wallet...');
    const mediaResponse = await axios.get(`${API_URL}/media/${TEST_WALLET}`);
    console.log('Media response:', mediaResponse.data);

    // Test pin status
    console.log('\n3. Testing pin status...');
    const statusResponse = await axios.get(`${API_URL}/media/${ipfsHash}/status`);
    console.log('Status response:', statusResponse.data);

    // Test delete media
    console.log('\n4. Testing delete media...');
    const deleteResponse = await axios.delete(`${API_URL}/media/${ipfsHash}`);
    console.log('Delete response:', deleteResponse.data);

    // Cleanup
    fs.unlinkSync(testImagePath);
    console.log('\nAll tests completed successfully!');

  } catch (error) {
    if (error instanceof AxiosError) {
      console.error('Test failed:', error.response?.data || error.message);
    } else if (error instanceof Error) {
      console.error('Test failed:', error.message);
    } else {
      console.error('Test failed with unknown error');
    }
    process.exit(1);
  }
}

testMediaRoutes(); 