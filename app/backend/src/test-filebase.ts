import { S3Client, ListObjectsCommand, CreateBucketCommand, PutBucketCorsCommand } from '@aws-sdk/client-s3';
import { config } from 'dotenv';
import * as path from 'path';

// Load environment variables
config({ path: path.resolve(__dirname, '../.env') });

async function testFilebaseConnection() {
  console.log('Testing Filebase connection...');
  
  // Log credentials (partially masked)
  const key = process.env.FILEBASE_KEY || '';
  const secret = process.env.FILEBASE_SECRET || '';
  const bucket = process.env.FILEBASE_BUCKET || 'mmoment';
  
  console.log('Credentials:', {
    FILEBASE_KEY: key.substring(0, 4) + '...' + key.substring(key.length - 4),
    FILEBASE_SECRET: secret.substring(0, 4) + '...' + secret.substring(secret.length - 4),
    FILEBASE_BUCKET: bucket
  });

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
    // Step 1: Try to create the bucket (will fail if it exists, which is fine)
    console.log('Attempting to create bucket:', bucket);
    try {
      const createBucketCommand = new CreateBucketCommand({
        Bucket: bucket
      });
      await client.send(createBucketCommand);
      console.log('Bucket created successfully');
    } catch (error: any) {
      if (error.Code === 'BucketAlreadyExists') {
        console.log('Bucket already exists, continuing...');
      } else {
        throw error;
      }
    }

    // Step 2: Configure CORS for the bucket
    console.log('Configuring CORS for bucket...');
    const corsConfig = {
      CORSRules: [
        {
          AllowedHeaders: ["*"],
          AllowedMethods: ["PUT", "POST", "DELETE", "GET"],
          AllowedOrigins: ["*"],
          ExposeHeaders: ["ETag"],
          MaxAgeSeconds: 3000
        }
      ]
    };

    const putCorsCommand = new PutBucketCorsCommand({
      Bucket: bucket,
      CORSConfiguration: corsConfig
    });
    
    await client.send(putCorsCommand);
    console.log('CORS configuration applied successfully');

    // Step 3: List objects in the bucket
    console.log('Listing objects in bucket...');
    const listCommand = new ListObjectsCommand({
      Bucket: bucket,
      MaxKeys: 10
    });
    
    const response = await client.send(listCommand);
    console.log('Success! Objects in bucket:', response.Contents?.length || 0);
    
    if (response.Contents && response.Contents.length > 0) {
      console.log('First few objects:', response.Contents.slice(0, 3).map(obj => ({
        Key: obj.Key,
        Size: obj.Size,
        LastModified: obj.LastModified
      })));
    } else {
      console.log('Bucket is empty');
    }

    console.log('\nFilebase setup completed successfully!');
    console.log('IPFS Gateway URL format: https://ipfs.filebase.io/ipfs/{CID}');
    
  } catch (error) {
    console.error('Failed to setup Filebase:', error);
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
testFilebaseConnection(); 