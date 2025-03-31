import * as anchor from '@coral-xyz/anchor';
import { Program } from '@coral-xyz/anchor';
import { 
  PublicKey, 
  Connection, 
  Keypair, 
  SystemProgram,
  Transaction
} from '@solana/web3.js';
import { CameraActivation } from '../target/types/camera_activation';
import { BN } from 'bn.js';

export class CameraClient {
  program: Program<CameraActivation>;
  provider: anchor.AnchorProvider;
  
  constructor(
    connection: Connection,
    wallet: anchor.Wallet,
    programId: PublicKey
  ) {
    this.provider = new anchor.AnchorProvider(
      connection,
      wallet,
      { commitment: 'confirmed' }
    );
    
    // @ts-ignore
    this.program = new anchor.Program(
      require('../target/idl/camera_activation.json'),
      programId,
      this.provider
    );
  }
  
  // Find PDA for the central registry
  async findRegistryAddress(): Promise<[PublicKey, number]> {
    return PublicKey.findProgramAddressSync(
      [Buffer.from('camera-registry')],
      this.program.programId
    );
  }
  
  // Find PDA for a camera account
  async findCameraAddress(cameraId: string, owner: PublicKey): Promise<[PublicKey, number]> {
    return PublicKey.findProgramAddressSync(
      [
        Buffer.from('camera'),
        Buffer.from(cameraId),
        owner.toBuffer()
      ],
      this.program.programId
    );
  }
  
  // Initialize the camera registry
  async initialize(): Promise<string> {
    const [registryAddress] = await this.findRegistryAddress();
    
    const tx = await this.program.methods
      .initialize()
      .accounts({
        authority: this.provider.wallet.publicKey,
        registry: registryAddress,
        systemProgram: SystemProgram.programId,
      })
      .rpc();
      
    return tx;
  }
  
  // Register a new camera
  async registerCamera(
    cameraId: string,
    name: string,
    model: string,
    location?: [number, number],
    fee: number = 100
  ): Promise<string> {
    const [registryAddress] = await this.findRegistryAddress();
    const [cameraAddress] = await this.findCameraAddress(
      cameraId,
      this.provider.wallet.publicKey
    );
    
    const locationBN = location ? location.map(coord => new BN(coord)) : null;
    
    const tx = await this.program.methods
      .registerCamera({
        cameraId,
        name,
        model,
        location: locationBN,
        fee: new BN(fee)
      })
      .accounts({
        owner: this.provider.wallet.publicKey,
        registry: registryAddress,
        camera: cameraAddress,
        systemProgram: SystemProgram.programId,
      })
      .rpc();
      
    return tx;
  }
  
  // Update a camera's info
  async updateCamera(
    cameraId: string,
    name?: string,
    model?: string,
    location?: [number, number]
  ): Promise<string> {
    const [cameraAddress] = await this.findCameraAddress(
      cameraId,
      this.provider.wallet.publicKey
    );
    
    const locationBN = location ? location.map(coord => new BN(coord)) : null;
    
    const tx = await this.program.methods
      .updateCamera({
        name,
        model,
        location: locationBN,
      })
      .accounts({
        owner: this.provider.wallet.publicKey,
        camera: cameraAddress,
      })
      .rpc();
      
    return tx;
  }
  
  // Set a camera's active status
  async setCameraActive(cameraId: string, isActive: boolean): Promise<string> {
    const [cameraAddress] = await this.findCameraAddress(
      cameraId,
      this.provider.wallet.publicKey
    );
    
    const tx = await this.program.methods
      .setCameraActive({
        isActive,
      })
      .accounts({
        owner: this.provider.wallet.publicKey,
        camera: cameraAddress,
      })
      .rpc();
      
    return tx;
  }
  
  // Record a camera activity
  async recordActivity(
    cameraId: string,
    activityType: 'PhotoCapture' | 'VideoRecord' | 'LiveStream' | 'Custom',
    metadata: string
  ): Promise<string> {
    const [cameraAddress] = await this.findCameraAddress(
      cameraId,
      this.provider.wallet.publicKey
    );
    
    // Current timestamp in seconds as bytes for the PDA seed
    const timestamp = Math.floor(Date.now() / 1000);
    const timestampBytes = new BN(timestamp).toArrayLike(Buffer, 'le', 8);
    
    const [activityAddress] = PublicKey.findProgramAddressSync(
      [
        Buffer.from('activity'),
        cameraAddress.toBuffer(),
        timestampBytes
      ],
      this.program.programId
    );
    
    const tx = await this.program.methods
      .recordActivity({
        activityType: { [activityType]: {} },
        metadata,
      })
      .accounts({
        owner: this.provider.wallet.publicKey,
        camera: cameraAddress,
        activity: activityAddress,
        systemProgram: SystemProgram.programId,
      })
      .rpc();
      
    return tx;
  }
  
  // Fetch a camera's details
  async getCamera(cameraId: string, owner: PublicKey): Promise<any> {
    const [cameraAddress] = await this.findCameraAddress(cameraId, owner);
    return this.program.account.cameraAccount.fetch(cameraAddress);
  }
  
  // Fetch all cameras in the registry
  async getAllCameras(): Promise<any[]> {
    const cameras = await this.program.account.cameraAccount.all();
    return cameras.map(c => ({
      pubkey: c.publicKey.toBase58(),
      account: c.account
    }));
  }
  
  // Fetch all active cameras
  async getActiveCameras(): Promise<any[]> {
    const cameras = await this.getAllCameras();
    return cameras.filter(c => c.account.isActive);
  }
  
  // Fetch all cameras owned by a specific wallet
  async getCamerasByOwner(owner: PublicKey): Promise<any[]> {
    const cameras = await this.program.account.cameraAccount.all([
      {
        memcmp: {
          offset: 8, // After discriminator
          bytes: owner.toBase58()
        }
      }
    ]);
    
    return cameras.map(c => ({
      pubkey: c.publicKey.toBase58(),
      account: c.account
    }));
  }
  
  // Fetch recent activities for a camera
  async getCameraActivities(cameraAddress: PublicKey): Promise<any[]> {
    const activities = await this.program.account.activity.all([
      {
        memcmp: {
          offset: 8, // After discriminator
          bytes: cameraAddress.toBase58()
        }
      }
    ]);
    
    return activities.map(a => ({
      pubkey: a.publicKey.toBase58(),
      account: a.account
    }));
  }
}