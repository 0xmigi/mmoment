# Firestarter SDK

⚠️ **Work in Progress** - This SDK is currently under development and not yet published to npm.

A TypeScript SDK for integrating Firestarter (Pipe Network's decentralized storage system) into your backend applications, giving your users permissionless, censorship-resistant file storage.

**Development Context**: This SDK is being developed alongside the [MMOMENT project](https://github.com/0xmigi/mmoment) and draws its primary design influence from real-world DePIN camera network requirements. However, it is designed for general use across any application needing decentralized storage capabilities.

## What is this SDK for?

This SDK is designed for **backend developers** who want to offer their users decentralized storage capabilities without the complexity of directly interfacing with blockchain infrastructure. Instead of storing user files on traditional cloud services (AWS S3, Google Cloud, etc.), you can store them on Firestarter - Pipe Network's decentralized storage system built on Solana.

### Key Use Cases

**🎯 Target Audience: Backend Developers & DApp Builders**

- **Web2 to Web3 Bridge**: Easily add decentralized storage to existing web applications
- **DApp Backend Services**: Provide decentralized file storage for your decentralized applications
- **Multi-tenant Applications**: Manage file storage for multiple users with session-based tracking
- **Censorship-Resistant Storage**: Files stored on Pipe Network cannot be taken down by centralized authorities
- **Cost-Effective Alternative**: Potentially lower costs compared to traditional cloud storage

### Architecture & Deployment

**Where it runs:**
- ✅ **Backend/Server-side**: Node.js applications, Express servers, Next.js API routes
- ✅ **Self-hosted**: Your own infrastructure with full control
- ✅ **Cloud Functions**: AWS Lambda, Vercel Functions, Netlify Functions
- ❌ **Frontend/Browser**: Not designed for direct browser use (requires server environment)

**Why backend-only?**
- Handles private keys and user credentials securely
- Manages complex session state across multiple users
- Provides a clean abstraction layer between your app and blockchain complexity
- Enables you to implement your own authentication and authorization

## What You Get

- 🔐 **Secure Key Management** - Handle user credentials without exposing private keys to frontend
- 👥 **Multi-User Sessions** - Track uploads and files for different users in your application
- 📊 **File Management** - List, search, and organize files with metadata
- 💰 **Token Abstraction** - Handle SOL/PIPE token operations behind the scenes
- 🔗 **Public Sharing** - Generate shareable links for files
- 📝 **Upload History** - Track all uploads with persistent local storage
- ⚡ **TypeScript Ready** - Full type safety and IDE autocomplete

## Installation

⚠️ **Not yet available on npm** - This package is still in development.

For now, you can install directly from the GitHub repository:

```bash
# Install from GitHub (development version)
npm install git+https://github.com/0xmigi/firestarter-sdk.git
# or
yarn add git+https://github.com/0xmigi/firestarter-sdk.git
```

Once published to npm, it will be available as:
```bash
npm install firestarter-sdk  # Coming soon
```

## How It Works

### Traditional vs Decentralized Storage

**Traditional Approach:**
```
Your App → AWS S3/Google Cloud → Centralized servers
❌ Single point of failure
❌ Censorship risk
❌ Vendor lock-in
❌ Monthly storage fees
```

**With Firestarter SDK:**
```
Your App → Firestarter SDK → Firestarter (Pipe Network) → Distributed storage
✅ Decentralized & resilient
✅ Censorship resistant
✅ No vendor lock-in
✅ Pay-per-use model
```

### Typical Implementation Pattern

1. **User uploads file** to your application
2. **Your backend** receives the file and uses Firestarter SDK
3. **SDK handles** Firestarter interaction, key management, and token operations
4. **File is stored** across Firestarter's distributed nodes
5. **Your app** receives file ID and metadata for database storage
6. **Users can access** files via generated public links or through your app

## When to Use This SDK

### ✅ Good For:
- **Content platforms** (blogs, social media, file sharing)
- **DApps needing storage** (NFT metadata, user content)
- **Applications requiring censorship resistance**
- **Multi-user platforms** with file management needs
- **Developers wanting Web3 storage without complexity**

### ❌ Not Ideal For:
- **Frontend-only applications** (use Firestarter directly)
- **Single-user desktop apps** (consider Pipe CLI)
- **Applications needing immediate consistency** (blockchain has latency)
- **Very large files** (check Firestarter limits)

## Example Usage Scenarios

### Social Media Platform
```typescript
// In your Express.js API endpoint
app.post('/api/upload-post-image', async (req, res) => {
  const { userId, imageFile } = req.body;

  // Store user's post image on decentralized storage
  const result = await sessionManager.uploadForUser(
    userId,
    imageFile,
    'post-image.jpg'
  );

  // Save file reference in your database
  await db.posts.create({
    userId,
    imageUrl: result.publicUrl,
    fileId: result.fileId
  });
});
```

### NFT Marketplace Backend
```typescript
// Store NFT metadata and assets decentrally
app.post('/api/mint-nft', async (req, res) => {
  const { creatorWallet, metadata, imageFile } = req.body;

  // Upload image to decentralized storage
  const imageResult = await sessionManager.uploadForUser(
    creatorWallet,
    imageFile,
    'nft-asset.png'
  );

  // Upload metadata JSON
  const metadataResult = await sessionManager.uploadForUser(
    creatorWallet,
    JSON.stringify(metadata),
    'metadata.json'
  );

  // Now mint NFT with decentralized URLs
  // metadata.image = imageResult.publicUrl
});
```

### Content Management System
```typescript
// Multi-tenant CMS with decentralized file storage
class CMSFileManager {
  async uploadUserContent(tenantId: string, file: Buffer, filename: string) {
    // Each tenant gets isolated file storage
    const result = await sessionManager.uploadForUser(
      `tenant_${tenantId}`,
      file,
      filename
    );

    // Files are censorship-resistant and always accessible
    return {
      url: result.publicUrl,
      fileId: result.fileId,
      hash: result.blake3Hash
    };
  }

  async getUserFiles(tenantId: string) {
    // Get all files for this tenant
    return await sessionManager.listUserFiles(`tenant_${tenantId}`);
  }
}

## Getting Started

### Prerequisites
- Node.js 16+ environment
- Basic understanding of async/await JavaScript
- A backend application (Express, Next.js API, etc.)

### Basic Setup
```typescript
import { SessionManager } from 'firestarter-sdk';

// Initialize the SDK
const sessionManager = new SessionManager({
  baseUrl: 'https://us-west-00-firestarter.pipenetwork.com',
  timeout: 30000
});

// Start using decentralized storage in your API endpoints
app.post('/upload', async (req, res) => {
  const result = await sessionManager.uploadForUser(
    req.user.id,
    req.file.buffer,
    req.file.originalname
  );

  res.json({ fileUrl: result.publicUrl });
});
```

### Configuration Options
- **baseUrl**: Firestarter API endpoint
- **timeout**: Request timeout in milliseconds
- **uploadHistoryPath**: Custom path for storing upload history

## API Reference

### SessionManager

The main class for managing multiple users and file tracking.

```typescript
class SessionManager {
  // Upload files
  uploadForUser(userId: string, data: Buffer, fileName: string, options?: UploadOptions): Promise<UploadResult>
  uploadCameraCapture(userId: string, imageData: Buffer, captureType?: string): Promise<UploadResult>

  // File management
  listUserFiles(userId: string): Promise<FileRecord[]>
  searchUserFiles(userId: string, pattern: string): Promise<FileRecord[]>
  getRecentUserFiles(userId: string, limit: number): Promise<FileRecord[]>

  // User operations
  getUserBalance(userId: string): Promise<{sol: number, pipe: number, publicKey: string}>
  exchangeSolForPipe(userId: string, amountSol: number): Promise<number>
  createPublicLink(userId: string, fileName: string): Promise<string>

  // Monitoring
  getUploadStats(): Promise<Stats>
  getActiveSessionCount(): number
  cleanupInactiveSessions(): number
}
```

### PipeClient

Low-level client for direct Pipe Network API access.

```typescript
class PipeClient {
  createUser(username: string): Promise<PipeUser>
  upload(user: PipeUser, data: Buffer, fileName: string, options?: UploadOptions): Promise<UploadResult>
  download(user: PipeUser, fileName: string, priority?: boolean): Promise<Buffer>
  createPublicLink(user: PipeUser, fileName: string): Promise<string>
  checkSolBalance(user: PipeUser): Promise<WalletBalance>
  checkPipeBalance(user: PipeUser): Promise<TokenBalance>
  exchangeSolForPipe(user: PipeUser, amountSol: number): Promise<number>
}
```

### Types

```typescript
interface FileRecord {
  fileId: string;
  originalFileName: string;
  storedFileName: string;
  userId: string;
  uploadedAt: Date;
  size: number;
  mimeType?: string;
  blake3Hash?: string;
  metadata: Record<string, any>;
}

interface UploadOptions {
  priority?: boolean;
  fileName?: string;
  metadata?: Record<string, any>;
}

interface PipeUser {
  userId: string;
  userAppKey: string;
  username?: string;
  solanaPubkey?: string;
}
```

## Configuration

```typescript
const config = {
  baseUrl: 'https://us-west-00-firestarter.pipenetwork.com', // Default
  timeout: 30000, // 30 seconds
};

const sessionManager = new SessionManager(config, './uploads.json');
```

## Error Handling

```typescript
import { PipeApiError, PipeValidationError } from 'firestarter-sdk';

try {
  await sessionManager.uploadForUser(userId, data, fileName);
} catch (error) {
  if (error instanceof PipeApiError) {
    console.log('API Error:', error.message, 'Status:', error.status);
  } else if (error instanceof PipeValidationError) {
    console.log('Validation Error:', error.message);
  }
}
```

## Local Storage

Upload history is stored locally in JSON format (like the Pipe CLI). Default location: `~/.firestarter/uploads.json`

You can specify a custom path:

```typescript
const manager = new SessionManager(config, '/custom/path/uploads.json');
```

## Testing

```bash
npm test
```

## License

MIT

## Development Status

🚧 **Alpha Version** - This SDK is in active development. APIs may change without notice.

### Current Status:
- ✅ Core client functionality
- ✅ Session management
- ✅ Upload tracking
- 🚧 Testing coverage (in progress)
- 🚧 Documentation (in progress)
- ❌ Published to npm (coming soon)

## Contributing

This project is currently in early development. Contributions are welcome!

1. Fork the repository
2. Create your feature branch
3. Add tests for new functionality
4. Run tests and ensure they pass
5. Submit a pull request

## Support

- GitHub Issues: [Report bugs](https://github.com/0xmigi/firestarter-sdk/issues)
- Documentation: [Full API docs](https://github.com/0xmigi/firestarter-sdk)