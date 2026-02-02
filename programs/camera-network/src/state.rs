use anchor_lang::prelude::*;

// The global camera registry that tracks all cameras in the network
#[account]
pub struct CameraRegistry {
    pub authority: Pubkey,       // The admin who can manage the registry
    pub camera_count: u64,       // Total number of cameras in the registry
    pub fee_account: Pubkey,     // Account where fees are collected (optional)
    pub bump: u8,                // PDA bump
}

// Individual camera account
#[account]
pub struct CameraAccount {
    pub owner: Pubkey,           // The owner of the camera
    pub metadata: CameraMetadata, // Camera metadata
    pub is_active: bool,         // Whether the camera is active and available for check-ins
    pub activity_counter: u64,   // Counter for camera activities
    pub last_activity_at: i64,   // Timestamp of last activity
    pub last_activity_type: u8,  // Type code of the last activity
    pub access_count: u64,       // Number of users who have accessed this camera
    pub features: CameraFeatures, // Features enabled on this camera
    pub bump: u8,                // PDA bump
    pub device_pubkey: Option<Pubkey>, // Device signing key for DePIN authentication (added at end for upgrade safety)
}

// Camera metadata structure
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Default)]
pub struct CameraMetadata {
    pub name: String,            // User-friendly name (unique identifier)
    pub model: String,           // Camera hardware model
    pub location: Option<[i64; 2]>, // Optional location [latitude, longitude]
    pub registration_date: i64,  // When the camera was registered
    pub description: String,     // User description of the camera purpose
}

// Camera features
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Default)]
pub struct CameraFeatures {
    pub face_recognition: bool,  // Whether face recognition is enabled
    pub gesture_control: bool,   // Whether gesture control is enabled
    pub video_recording: bool,   // Whether video recording is available
    pub live_streaming: bool,    // Whether live streaming is available
    pub messaging: bool,         // Whether messaging is available
}

// NOTE: UserSession removed - sessions are now managed off-chain by Jetson
// Check-in is an ed25519 handshake, checkout writes to CameraTimeline + UserSessionChain
// This breaks the visible on-chain user↔camera link

// Recognition token - stores encrypted facial embedding for user authentication
#[account]
pub struct RecognitionToken {
    pub user: Pubkey,                  // 32 bytes - owner
    pub encrypted_embedding: Vec<u8>,  // ~2800-3000 bytes - Fernet-encrypted embedding (512 floats + Fernet overhead)
    pub created_at: i64,               // 8 bytes - creation timestamp
    pub version: u8,                   // 1 byte - token version (incremented on regeneration)
    pub bump: u8,                      // 1 byte - PDA bump
    pub display_name: Option<String>,  // ~32 bytes - user-friendly label (e.g., "Phone Selfie 2025")
    pub source: u8,                    // 1 byte - 0=phone_selfie, 1=jetson_capture, 2=imported
}

// Space calculation: 8 + 32 + 4 + 3200 + 8 + 1 + 1 + 4 + 64 + 1 = 3323 bytes
// Rent: ~0.024 SOL (one-time, reclaimable)

// User gesture configuration
#[account]
pub struct GestureConfig {
    pub user: Pubkey,            // The user who defined this gesture
    pub gesture_type: u8,        // Type of gesture (1=wave, 2=thumbs up, etc.)
    pub data_hash: [u8; 32],     // Hash of the gesture data (actual data stored off-chain)
    pub created_at: i64,         // When the gesture was created
    pub bump: u8,                // PDA bump
}

// Message posted through a camera
#[account]
pub struct CameraMessage {
    pub user: Pubkey,            // The user who posted the message
    pub camera: Pubkey,          // The camera through which the message was posted
    pub message: String,         // The message content
    pub timestamp: i64,          // When the message was posted
    pub bump: u8,                // PDA bump
}

// Access grant for temporary camera access
#[account]
pub struct AccessGrant {
    pub camera: Pubkey,          // The camera being granted access to
    pub grantor: Pubkey,         // The user granting access (typically camera owner)
    pub grantee: Pubkey,         // The user being granted access
    pub expires_at: i64,         // When the access expires
    pub bump: u8,                // PDA bump
}

// Camera timeline - stores encrypted activity history for a camera
// Created lazily on first checkout with activities
#[account]
pub struct CameraTimeline {
    pub camera: Pubkey,                         // Link back to camera account
    pub encrypted_activities: Vec<EncryptedActivity>, // Growing list of encrypted activities
    pub activity_count: u64,                    // Total activities (public stat)
    pub bump: u8,                               // PDA bump
}

// Space: 8 (discriminator) + 32 (camera) + 4 (vec length) + (N * activity_size) + 8 (count) + 1 (bump)
// Initial: ~53 bytes + dynamic activity data
// Seeds: ["camera-timeline", camera.key()]

// Single encrypted activity entry
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct EncryptedActivity {
    pub timestamp: i64,              // When activity occurred (public for overlap queries)
    pub activity_type: u8,           // Type of activity (photo, video, etc.) - public
    pub encrypted_content: Vec<u8>,  // AES-256-GCM encrypted activity data
    pub nonce: [u8; 12],             // AES-GCM nonce for decryption
    pub access_grants: Vec<Vec<u8>>, // Encrypted activity key for each user present
}

// Typical size per activity: 8 + 1 + 4 + ~150 (content) + 12 + 4 + (N_users * 64) bytes
// Example: 2 users present = ~300 bytes per activity

// User session chain - stores encrypted access keys to camera session history
// This is the user's "keychain" for accessing their sessions across cameras
// Seeds: ["user-session-chain", user.key()]
#[account]
pub struct UserSessionChain {
    pub user: Pubkey,                               // 32 bytes - owner of this chain
    pub authority: Pubkey,                          // 32 bytes - mmoment cron bot (can write as fallback)
    pub encrypted_keys: Vec<EncryptedSessionKey>,   // Dynamic - encrypted access keys to sessions
    pub session_count: u64,                         // 8 bytes - total sessions recorded
    pub bump: u8,                                   // 1 byte - PDA bump
}

// Space: 8 (discriminator) + 32 (user) + 32 (authority) + 4 (vec length) + (N * key_size) + 8 (count) + 1 (bump)
// Initial: ~85 bytes + dynamic key data
// Writable by: user OR authority (cron bot)

// Single encrypted session access key
// Each key unlocks a portion of a camera's encrypted timeline
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct EncryptedSessionKey {
    pub key_ciphertext: Vec<u8>,    // Encrypted AES key that decrypts activities
    pub nonce: [u8; 12],            // Nonce used for key encryption
    pub timestamp: i64,             // When this session occurred (for ordering)
}

// Typical size per key: 4 + ~48 (ciphertext) + 12 + 8 = ~72 bytes

// Activity types enum (used to interpret last_activity_type)
// Values 0-49: Core camera activities
// Values 50-99: CV app results (pushup competition, basketball score, climbing, etc.)
// Values 100-254: Reserved for future custom CV apps (dynamically mapped)
// Value 255: Other/Unknown
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Copy)]
pub enum ActivityType {
    // Core camera activities (0-49)
    CheckIn = 0,
    CheckOut = 1,
    PhotoCapture = 2,
    VideoRecord = 3,
    LiveStream = 4,
    FaceRecognition = 5,  // User authentication via face recognition

    // CV app activity results (50-99)
    // Generic type for CV apps - specific app and result data goes in encrypted_content
    CVAppActivity = 50,

    // Reserved range (100-254): Custom CV app codes
    // Frontend/Jetson can dynamically map app-specific codes to app names
    // e.g., 100 = "pushup_competition", 101 = "basketball_2x2", 102 = "bouldering_scorecard"
    // encrypted_content contains: {app_name, participants[], result, metadata}

    Other = 255,
}

// Event emitted when a camera is registered
#[event]
pub struct CameraRegistered {
    pub camera: Pubkey,          // The camera account public key
    pub owner: Pubkey,           // The camera owner
    pub name: String,            // Camera name
    pub model: String,           // Camera model
    pub timestamp: i64,          // When the camera was registered
}

// NOTE: UserCheckedIn, UserCheckedOut, ActivityRecorded events removed
// These exposed user↔camera links. Timeline updates are now anonymous via TimelineUpdated event.

// Event emitted when a recognition token is created or regenerated
#[event]
pub struct RecognitionTokenCreated {
    pub user: Pubkey,            // The token owner
    pub token: Pubkey,           // The token account public key
    pub version: u8,             // Token version (increments on regeneration)
    pub source: u8,              // Token source (0=phone_selfie, 1=jetson_capture, 2=imported)
    pub display_name: Option<String>, // User-friendly label
    pub timestamp: i64,          // When the token was created
}

// Event emitted when a recognition token is deleted
#[event]
pub struct RecognitionTokenDeleted {
    pub user: Pubkey,            // The token owner
    pub token: Pubkey,           // The token account public key
    pub timestamp: i64,          // When the token was deleted
}

// NOTE: SessionAutoCheckout event removed - exposed user↔camera links

// Remove unused seed helper functions, keep only the essential ones
pub fn camera_registry_seeds() -> &'static [u8] {
    b"camera-registry"
}

// Registration arguments
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RegisterCameraArgs {
    pub name: String,
    pub model: String,
    pub location: Option<[i64; 2]>,
    pub description: String,
    pub features: CameraFeatures,
    pub device_pubkey: Option<Pubkey>, // Optional device key for backwards compatibility
}

// Update camera arguments
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct UpdateCameraArgs {
    pub name: Option<String>,
    pub location: Option<[i64; 2]>,
    pub description: Option<String>,
    pub features: Option<CameraFeatures>,
    pub device_pubkey: Option<Pubkey>,
}

// Activity data structure for timeline writes
// This is passed from Jetson (already encrypted) to the checkout instruction
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ActivityData {
    pub timestamp: i64,              // When the activity occurred (public)
    pub activity_type: u8,           // Type (photo, video, etc.) - public
    pub encrypted_content: Vec<u8>,  // AES-256-GCM encrypted by Jetson
    pub nonce: [u8; 12],             // AES-GCM nonce
    pub access_grants: Vec<Vec<u8>>, // Encrypted AES keys for each user present
} 
