import { S3Client, PutObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
import { config } from 'dotenv';
import * as path from 'path';

// Load environment variables
config({ path: path.resolve(__dirname, '../.env') });

async function testFilebaseUpload() {
  console.log('Testing Filebase file upload...');
  
  const key = process.env.FILEBASE_KEY || '';
  const secret = process.env.FILEBASE_SECRET || '';
  const bucket = process.env.FILEBASE_BUCKET || 'mmoment';
  
  const client = new S3Client({
    endpoint: 'https://s3.filebase.com',
    region: 'us-east-1',
    credentials: {
      accessKeyId: key,
      secretAccessKey: secret
    },
    forcePathStyle: true
  });

  try {
    // Create a test file content
    const testContent = 'Hello from Filebase! ' + new Date().toISOString();
    const fileName = `test-${Date.now()}.txt`;

    console.log('Uploading test file:', fileName);
    
    // Upload the file
    const uploadCommand = new PutObjectCommand({
      Bucket: bucket,
      Key: fileName,
      Body: testContent,
      ContentType: 'text/plain',
      ACL: 'public-read'
    });

    const uploadResponse = await client.send(uploadCommand);
    console.log('Upload successful!');
    console.log('ETag (IPFS CID):', uploadResponse.ETag?.replace(/"/g, ''));

    // Get the IPFS URL
    const ipfsCid = uploadResponse.ETag?.replace(/"/g, '');
    const ipfsUrl = `https://ipfs.filebase.io/ipfs/${ipfsCid}`;
    
    console.log('\nFile is now available at:');
    console.log('IPFS URL:', ipfsUrl);
    
    // Try to read back the file to verify
    console.log('\nVerifying file can be read back...');
    const getCommand = new GetObjectCommand({
      Bucket: bucket,
      Key: fileName
    });
    
    const getResponse = await client.send(getCommand);
    const content = await getResponse.Body?.transformToString();
    
    if (content === testContent) {
      console.log('✅ File content verified successfully!');
    } else {
      console.log('❌ File content verification failed!');
      console.log('Expected:', testContent);
      console.log('Got:', content);
    }

    console.log('\nTest completed successfully!');
    console.log('You can verify the file in your browser at:', ipfsUrl);
    
  } catch (error) {
    console.error('Failed to upload file to Filebase:', error);
    if (error instanceof Error) {
      console.error('Error details:', {
        name: error.name,
        message: error.message,
        code: (error as any).Code,
        requestId: (error as any).RequestId
      });
    }
    throw error;
  }
}

// Run the test
testFilebaseUpload(); 