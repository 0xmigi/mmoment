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

// User session account for check-in/check-out
#[account]
pub struct UserSession {
    pub user: Pubkey,            // The user who is checked in
    pub camera: Pubkey,          // The camera they're checked into
    pub check_in_time: i64,      // When the session started
    pub last_activity: i64,      // Last activity timestamp
    pub auto_checkout_at: i64,   // Unix timestamp when session expires (2 hour fallback)
    pub enabled_features: SessionFeatures, // What features are enabled for this session
    pub bump: u8,                // PDA bump
}

// Updated space: 8 + 32 + 32 + 8 + 8 + 8 + 5 + 1 = 102 bytes

// Features that can be enabled during a session
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Default)]
pub struct SessionFeatures {
    pub face_recognition: bool,  // Whether face recognition is being used
    pub gesture_control: bool,   // Whether gesture control is being used
    pub video_recording: bool,   // Whether video recording is enabled
    pub live_streaming: bool,    // Whether live streaming is enabled
    pub messaging: bool,         // Whether messaging is enabled
}

// Recognition token - stores encrypted facial embedding for user authentication
#[account]
pub struct RecognitionToken {
    pub user: Pubkey,                  // 32 bytes - owner
    pub encrypted_embedding: Vec<u8>,  // ~600-900 bytes - ACTUAL encrypted data
    pub created_at: i64,               // 8 bytes - creation timestamp
    pub version: u8,                   // 1 byte - token version (incremented on regeneration)
    pub bump: u8,                      // 1 byte - PDA bump
    pub display_name: Option<String>,  // ~32 bytes - user-friendly label (e.g., "Phone Selfie 2025")
    pub source: u8,                    // 1 byte - 0=phone_selfie, 1=jetson_capture, 2=imported
}

// Space calculation: 8 + 32 + 4 + 1024 + 8 + 1 + 1 + 4 + 64 + 1 = 1147 bytes
// Rent: ~0.009 SOL (one-time, reclaimable)

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

// Activity types enum (used to interpret last_activity_type)
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Copy)]
pub enum ActivityType {
    CheckIn = 0,
    CheckOut = 1,
    PhotoCapture = 2,
    VideoRecord = 3,
    LiveStream = 4,
    FaceRecognition = 5,
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

// Event emitted when a user checks in
#[event]
pub struct UserCheckedIn {
    pub user: Pubkey,            // The user public key
    pub camera: Pubkey,          // The camera public key
    pub session: Pubkey,         // The session public key
    pub timestamp: i64,          // When the check-in occurred
}

// Event emitted when a user checks out
#[event]
pub struct UserCheckedOut {
    pub user: Pubkey,            // The user public key
    pub camera: Pubkey,          // The camera public key
    pub session: Pubkey,         // The session public key
    pub duration: i64,           // Session duration in seconds
    pub timestamp: i64,          // When the check-out occurred
}

// Event for camera activity
#[event]
pub struct ActivityRecorded {
    pub camera: Pubkey,          // The camera public key
    pub user: Pubkey,            // The user public key (if applicable)
    pub activity_type: u8,       // Type of activity
    pub timestamp: i64,          // When the activity occurred
}

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

// Event emitted when a session auto-checks out
#[event]
pub struct SessionAutoCheckout {
    pub user: Pubkey,            // The user who was checked out
    pub camera: Pubkey,          // The camera public key
    pub session: Pubkey,         // The session public key
    pub reason: String,          // Reason: "expired" or "face_timeout"
    pub timestamp: i64,          // When the auto checkout occurred
}

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
}

// Camera activity arguments
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ActivityArgs {
    pub activity_type: u8,
    pub metadata: String,
} 
