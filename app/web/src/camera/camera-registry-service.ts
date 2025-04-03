import { BN, Program, Idl } from '@coral-xyz/anchor';
import { Connection, PublicKey, SystemProgram, Transaction, TransactionInstruction } from '@solana/web3.js';
import { CONFIG } from '../core/config';

// Simple request management
const THROTTLE_MS = 1000; // Reduced from 5000ms to 1000ms
let lastRequestTime = 0;
let pendingRequests = 0;
const MAX_CONCURRENT_REQUESTS = 3;

// Anchor instruction discriminators (first 8 bytes of sha256 hash of method name)
const INSTRUCTION_DISCRIMINATORS = {
  initialize: [175, 175, 109, 31, 13, 152, 155, 237],
  registerCamera: [57, 157, 128, 170, 224, 240, 118, 131],
  setCameraActive: [183, 18, 70, 156, 148, 109, 161, 34],
  recordActivity: [44, 126, 192, 118, 189, 208, 175, 166]
};

async function throttleRequest() {
  // If we have too many concurrent requests, wait for some to finish
  if (pendingRequests >= MAX_CONCURRENT_REQUESTS) {
    await new Promise(resolve => setTimeout(resolve, THROTTLE_MS));
  }
  
  const now = Date.now();
  const timeSinceLastRequest = now - lastRequestTime;
  if (timeSinceLastRequest < THROTTLE_MS) {
    await new Promise(resolve => setTimeout(resolve, THROTTLE_MS - timeSinceLastRequest));
  }
  lastRequestTime = Date.now();
  pendingRequests++;
}

async function withRetry<T>(
  operation: () => Promise<T>
): Promise<T | null> {
  try {
    await throttleRequest();
    const result = await operation();
    pendingRequests = Math.max(0, pendingRequests - 1);
    return result;
  } catch (error) {
    pendingRequests = Math.max(0, pendingRequests - 1);
    if (error instanceof Error &&
        (error.message.includes('429') || error.message.includes('Too many requests'))) {
      CONFIG.rpcEndpoint = CONFIG.getNextEndpoint();
      await new Promise(resolve => setTimeout(resolve, 5000));
      await throttleRequest();
      try {
        const result = await operation();
        pendingRequests = Math.max(0, pendingRequests - 1);
        return result;
      } catch {
        return null;
      }
    }
    return null;
  }
}

// Define the types based on your camera-activation program
export type ActivityType = 
  | { photoCapture: {} }
  | { videoRecord: {} }
  | { liveStream: {} }
  | { custom: {} };

export interface CameraMetadata {
  name: string;
  location?: [number, number];
  model: string;
  registrationDate: number;
  lastActivity: number;
}

export interface RegisterCameraArgs {
  cameraId: string;
  name: string;
  model: string;
  location?: [number, number];
  fee: number;
}

export interface UpdateCameraArgs {
  name?: string;
  location?: [number, number];
  model?: string;
}

export interface RecordActivityArgs {
  activityType: ActivityType;
  metadata: string;
}

export interface SetCameraActiveArgs {
  isActive: boolean;
}

export interface CameraAccount {
  owner: PublicKey;
  cameraId: string;
  isActive: boolean;
  metadata: CameraMetadata;
  bump: number;
}

export interface CameraRegistry {
  authority: PublicKey;
  cameraCount: BN;
  bump: number;
}

export interface Activity {
  camera: PublicKey;
  activityType: ActivityType;
  timestamp: BN;
  metadata: string;
  bump: number;
}

export class CameraRegistryService {
  private static instance: CameraRegistryService | null = null;
  private program: Program<Idl> | null = null;
  private programId = new PublicKey('7BYuxsNyaxxsxwzcRzFd6UJGnUctN6V1vDQxjGPaK2L4');
  private initialized = false;
  
  // Cache for registry initialization status
  private registryInitializedCache: boolean | null = null;
  private lastRegistryCheck = 0;
  private readonly REGISTRY_CACHE_TTL = 30000; // 30 seconds cache TTL

  private constructor() {}

  public static getInstance(): CameraRegistryService {
    if (!CameraRegistryService.instance) {
      CameraRegistryService.instance = new CameraRegistryService();
    }
    return CameraRegistryService.instance;
  }

  public useExistingProgram(program: Program<Idl>) {
    if (!program) {
      console.error('Attempted to use null program in CameraRegistryService');
      return this;
    }
    
    console.log('Initializing CameraRegistryService with program ID:', program.programId.toString());
    this.program = program;
    this.programId = program.programId;
    this.initialized = true;
    
    // Reset cache when program changes
    this.registryInitializedCache = null;
    
    return this;
  }

  // Add a method to use a minimal program with just the program ID and provider
  public useMinimalProgram(programId: PublicKey, provider: any) {
    console.log('Initializing CameraRegistryService with minimal program, ID:', programId.toString());
    this.programId = programId;
    
    // Create a minimal program object with just what we need
    this.program = {
      programId,
      provider: {
        connection: provider.connection,
        publicKey: provider.publicKey,
        signTransaction: provider.signTransaction,
        signAllTransactions: provider.signAllTransactions
      }
    } as any;
    
    this.initialized = true;
    
    // Reset cache when program changes
    this.registryInitializedCache = null;
    
    return this;
  }

  private ensureInitialized(): boolean {
    if (!this.initialized) {
      console.error('CameraRegistryService not initialized');
      return false;
    }
    
    if (!this.program) {
      console.error('Program not available in CameraRegistryService');
      return false;
    }
    
    return true;
  }

  private async findRegistryAddress(): Promise<[PublicKey, number]> {
    return PublicKey.findProgramAddress(
      [Buffer.from('camera-registry')],
      this.programId
    );
  }

  // Add a method to check if the registry is initialized
  public async isRegistryInitialized(): Promise<boolean> {
    try {
      // Return cached value if available and not expired
      const now = Date.now();
      if (this.registryInitializedCache !== null && (now - this.lastRegistryCheck < this.REGISTRY_CACHE_TTL)) {
        console.log('Using cached registry initialization status:', this.registryInitializedCache);
        return this.registryInitializedCache;
      }

      // We don't need a fully initialized program to check if the registry exists
      // We just need the program ID and a connection
      let connection;
      
      if (this.program?.provider?.connection) {
        connection = this.program.provider.connection;
      } else {
        // Create a new connection to devnet
        connection = new Connection('https://api.devnet.solana.com', 'confirmed');
      }

      const [registryAddress] = await this.findRegistryAddress();
      console.log('Checking registry at address:', registryAddress.toString());

      try {
        // Use a direct getAccountInfo call instead of going through Anchor
        const registryAccountInfo = await connection.getAccountInfo(registryAddress);
        
        // If we got an account back with data, the registry is initialized
        const isInitialized = !!registryAccountInfo && registryAccountInfo.data.length > 0;
        
        // Update cache
        this.registryInitializedCache = isInitialized;
        this.lastRegistryCheck = now;
        
        console.log('Registry initialized check result:', isInitialized);
        return isInitialized;
      } catch (error) {
        console.error('Error fetching registry account:', error);
        return false;
      }
    } catch (error) {
      console.error('Error checking if registry is initialized:', error);
      return false;
    }
  }

  private async findCameraAddress(cameraId: string, owner: PublicKey): Promise<[PublicKey, number]> {
    return PublicKey.findProgramAddress(
      [
        Buffer.from('camera'),
        Buffer.from(cameraId),
        owner.toBuffer()
      ],
      this.programId
    );
  }

  public async getCameraAccount(cameraId: string, owner: PublicKey): Promise<CameraAccount | null> {
    if (!this.ensureInitialized()) {
      return null;
    }

    try {
      const [cameraAddress] = await this.findCameraAddress(cameraId, owner);
      
      return await withRetry(async () => {
        try {
          // @ts-ignore - We know this exists at runtime
          const account = await this.program?.account.cameraAccount.fetch(cameraAddress);
          if (!account) return null;

          return {
            owner: account.owner as PublicKey,
            cameraId: account.cameraId as string || cameraId,
            isActive: !!account.isActive,
            metadata: {
              name: (account.metadata as any)?.name || '',
              model: (account.metadata as any)?.model || '',
              registrationDate: (account.metadata as any)?.registrationDate?.toNumber() || 0,
              lastActivity: (account.metadata as any)?.lastActivity?.toNumber() || 0,
              location: (account.metadata as any)?.location || undefined
            },
            bump: account.bump as number || 0
          };
        } catch (e) {
          return null;
        }
      });
    } catch (error) {
      return null;
    }
  }

  public async registerCamera(
    owner: PublicKey,
    cameraId: string,
    name: string,
    model: string,
    location?: [number, number],
    fee: number = 100
  ): Promise<string | null> {
    if (!this.ensureInitialized()) {
      return null;
    }

    try {
      const [registryAddress] = await this.findRegistryAddress();
      const [cameraAddress] = await this.findCameraAddress(cameraId, owner);

      const locationBN = location?.map(coord => new BN(coord)) || null;
      const feeBN = new BN(fee);

      return await withRetry(async () => {
        // @ts-ignore
        const tx = await this.program?.methods
          .registerCamera({
            cameraId,
            name,
            model,
            location: locationBN,
            fee: feeBN
          })
          .accounts({
            owner,
            registry: registryAddress,
            camera: cameraAddress,
            systemProgram: SystemProgram.programId,
          })
          .rpc();

        return tx || null;
      });
    } catch {
      return null;
    }
  }

  public async setCameraActive(
    owner: PublicKey,
    cameraId: string,
    isActive: boolean
  ): Promise<string | null> {
    if (!this.ensureInitialized()) {
      return null;
    }

    try {
      const [cameraAddress] = await this.findCameraAddress(cameraId, owner);

      return await withRetry(async () => {
        // @ts-ignore
        const tx = await this.program?.methods
          .setCameraActive({ isActive })
          .accounts({
            owner,
            camera: cameraAddress,
          })
          .rpc();

        return tx || null;
      });
    } catch {
      return null;
    }
  }

  public async recordActivity(
    owner: PublicKey,
    cameraId: string,
    activityType: ActivityType,
    metadata: string
  ): Promise<string | null> {
    if (!this.ensureInitialized()) {
      return null;
    }

    try {
      const [cameraAddress] = await this.findCameraAddress(cameraId, owner);
      const timestamp = new BN(Math.floor(Date.now() / 1000));
      const timestampBytes = timestamp.toArrayLike(Buffer, 'le', 8);
      
      const [activityAddress] = await PublicKey.findProgramAddress(
        [
          Buffer.from('activity'),
          cameraAddress.toBuffer(),
          timestampBytes
        ],
        this.programId
      );

      return await withRetry(async () => {
        // @ts-ignore
        const tx = await this.program?.methods
          .recordActivity({
            activityType,
            metadata
          })
          .accounts({
            owner,
            camera: cameraAddress,
            activity: activityAddress,
            systemProgram: SystemProgram.programId
          })
          .rpc();

        return tx || null;
      });
    } catch {
      return null;
    }
  }

  // Add a direct method to register a camera without using Anchor Program
  public async directRegisterCamera(
    owner: PublicKey,
    cameraId: string,
    name: string,
    model: string,
    location?: [number, number]  ): Promise<string | null> {
    try {
      console.log('Direct register camera:', { owner: owner.toString(), cameraId, name, model });
      
      // Use provided connection or create a new one
      
      // Find the camera PDA
      const [cameraAddress] = await this.findCameraAddress(cameraId, owner);
      console.log('Camera address:', cameraAddress.toString());
      
      // Find the registry PDA
      const [registryAddress] = await this.findRegistryAddress();
      console.log('Registry address:', registryAddress.toString());
      
      // Create the instruction data
      const instructionData = Buffer.from([
        ...INSTRUCTION_DISCRIMINATORS.registerCamera, // Instruction discriminator
        
        // Camera ID (string) - encode length and then the string bytes
        ...new Uint8Array([cameraId.length]), // Length as a single byte
        ...Buffer.from(cameraId), // The string bytes
        
        // Name (string)
        ...new Uint8Array([name.length]),
        ...Buffer.from(name),
        
        // Model (string)
        ...new Uint8Array([model.length]),
        ...Buffer.from(model),
        
        // Location (optional)
        ...(location ? [1, ...this.encodeFloat(location[0]), ...this.encodeFloat(location[1])] : [0]),
      ]);
      
      // Create the instruction
      const instruction = new TransactionInstruction({
        keys: [
          { pubkey: owner, isSigner: true, isWritable: true },
          { pubkey: cameraAddress, isSigner: false, isWritable: true },
          { pubkey: registryAddress, isSigner: false, isWritable: true },
          { pubkey: SystemProgram.programId, isSigner: false, isWritable: false }
        ],
        programId: this.programId,
        data: instructionData
      });
      
      // Create and send the transaction
      const transaction = new Transaction().add(instruction);
      
      // This is where we would normally sign and send the transaction
      // But since we can't do that directly in the browser without the wallet adapter,
      // we'll return the serialized transaction for the caller to sign
      
      const serializedTransaction = transaction.serialize({
        requireAllSignatures: false,
        verifySignatures: false
      }).toString('base64');
      
      return serializedTransaction;
    } catch (error) {
      console.error('Error in directRegisterCamera:', error);
      return null;
    }
  }
  
  // Helper method to encode a float as bytes
  private encodeFloat(value: number): Uint8Array {
    const buffer = new ArrayBuffer(4);
    const view = new DataView(buffer);
    view.setFloat32(0, value, true); // true for little-endian
    return new Uint8Array(buffer);
  }
  
  // Add a direct method to set camera active state
  public async directSetCameraActive(
    owner: PublicKey,
    cameraId: string,
    isActive: boolean  ): Promise<string | null> {
    try {
      console.log('Direct set camera active:', { owner: owner.toString(), cameraId, isActive });
      
      // Use provided connection or create a new one
      
      // Find the camera PDA
      const [cameraAddress] = await this.findCameraAddress(cameraId, owner);
      console.log('Camera address:', cameraAddress.toString());
      
      // Create the instruction data
      const instructionData = Buffer.from([
        ...INSTRUCTION_DISCRIMINATORS.setCameraActive, // Instruction discriminator
        isActive ? 1 : 0 // Boolean as a single byte
      ]);
      
      // Create the instruction
      const instruction = new TransactionInstruction({
        keys: [
          { pubkey: owner, isSigner: true, isWritable: false },
          { pubkey: cameraAddress, isSigner: false, isWritable: true }
        ],
        programId: this.programId,
        data: instructionData
      });
      
      // Create and send the transaction
      const transaction = new Transaction().add(instruction);
      
      // Return the serialized transaction for the caller to sign
      const serializedTransaction = transaction.serialize({
        requireAllSignatures: false,
        verifySignatures: false
      }).toString('base64');
      
      return serializedTransaction;
    } catch (error) {
      console.error('Error in directSetCameraActive:', error);
      return null;
    }
  }
  
  // Add a direct method to record activity
  public async directRecordActivity(
    owner: PublicKey,
    cameraId: string,
    activityType: ActivityType,
    metadata: string  ): Promise<string | null> {
    try {
      console.log('Direct record activity:', { owner: owner.toString(), cameraId, activityType, metadata });
      
      // Use provided connection or create a new one
      
      // Find the camera PDA
      const [cameraAddress] = await this.findCameraAddress(cameraId, owner);
      console.log('Camera address:', cameraAddress.toString());
      
      // Generate a unique activity ID based on timestamp
      const activityId = `activity_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
      
      // Find the activity PDA
      const [activityAddress] = await PublicKey.findProgramAddress(
        [
          Buffer.from('activity'),
          Buffer.from(activityId),
          cameraAddress.toBuffer()
        ],
        this.programId
      );
      console.log('Activity address:', activityAddress.toString());
      
      // Determine activity type byte
      let activityTypeByte = 0;
      if ('photoCapture' in activityType) activityTypeByte = 0;
      else if ('videoRecord' in activityType) activityTypeByte = 1;
      else if ('liveStream' in activityType) activityTypeByte = 2;
      else if ('custom' in activityType) activityTypeByte = 3;
      
      // Create the instruction data
      const instructionData = Buffer.from([
        ...INSTRUCTION_DISCRIMINATORS.recordActivity, // Instruction discriminator
        
        // Activity ID (string)
        ...new Uint8Array([activityId.length]),
        ...Buffer.from(activityId),
        
        // Activity type
        activityTypeByte,
        
        // Metadata (string)
        ...new Uint8Array([metadata.length]),
        ...Buffer.from(metadata)
      ]);
      
      // Create the instruction
      const instruction = new TransactionInstruction({
        keys: [
          { pubkey: owner, isSigner: true, isWritable: true },
          { pubkey: cameraAddress, isSigner: false, isWritable: true },
          { pubkey: activityAddress, isSigner: false, isWritable: true },
          { pubkey: SystemProgram.programId, isSigner: false, isWritable: false }
        ],
        programId: this.programId,
        data: instructionData
      });
      
      // Create and send the transaction
      const transaction = new Transaction().add(instruction);
      
      // Return the serialized transaction for the caller to sign
      const serializedTransaction = transaction.serialize({
        requireAllSignatures: false,
        verifySignatures: false
      }).toString('base64');
      
      return serializedTransaction;
    } catch (error) {
      console.error('Error in directRecordActivity:', error);
      return null;
    }
  }

  // Add the getCamerasByOwner method to the CameraRegistryService class
  public async getCamerasByOwner(owner: PublicKey): Promise<CameraAccount[]> {
    if (!this.ensureInitialized()) {
      return [];
    }

    try {
      return await withRetry(async () => {
        // @ts-ignore
        const accounts = await this.program?.account.cameraAccount.all([
          {
            memcmp: {
              offset: 8, // After discriminator
              bytes: owner.toBase58()
            }
          }
        ]);

        if (!accounts) return [];

        return accounts.map(a => ({
          owner: a.account.owner as PublicKey,
          cameraId: a.account.cameraId as string || '',
          isActive: !!a.account.isActive,
          metadata: {
            name: (a.account.metadata as any)?.name || '',
            model: (a.account.metadata as any)?.model || '',
            registrationDate: (a.account.metadata as any)?.registrationDate?.toNumber() || 0,
            lastActivity: (a.account.metadata as any)?.lastActivity?.toNumber() || 0,
            location: (a.account.metadata as any)?.location || undefined
          },
          bump: a.account.bump as number || 0
        }));
      }) || [];
    } catch (error) {
      console.error('Error fetching cameras by owner:', error);
      return [];
    }
  }
}

export const cameraRegistryService = CameraRegistryService.getInstance();