use anchor_lang::prelude::*;

// The central registry that tracks all cameras
#[account]
pub struct CameraRegistry {
    pub authority: Pubkey,       // The admin who can manage the registry
    pub camera_count: u64,       // Total number of cameras in the registry
    pub bump: u8,                // PDA bump
}

// Individual camera account
#[account]
pub struct CameraAccount {
    pub owner: Pubkey,           // The owner of the camera
    pub is_active: bool,         // Whether the camera is active
    pub activity_counter: u64,   // Counter for camera activities
    pub last_activity_type: Option<ActivityType>, // Type of the last activity
    pub metadata: CameraMetadata, // Additional metadata
    pub bump: u8,                // PDA bump
}

// Camera metadata structure
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Default)]
pub struct CameraMetadata {
    pub name: String,            // User-friendly name (acts as the identifier)
    pub location: Option<[i64; 2]>, // Optional location data [latitude, longitude]
    pub model: String,           // Camera model
    pub registration_date: i64,  // When the camera was registered
    pub last_activity: i64,      // Last recorded activity
}

// Types of camera activities
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Copy)]
pub enum ActivityType {
    PhotoCapture,
    VideoRecord,
    LiveStream,
    Custom,
}

// Event for recording activities
#[event]
pub struct ActivityRecorded {
    pub camera: Pubkey,          // The camera public key
    pub name: String,            // Camera name (identifier) 
    pub activity_number: u64,    // Sequential activity number
    pub activity_type: ActivityType, // Type of activity
    pub timestamp: i64,          // When the activity occurred
    pub metadata: String,        // Additional metadata about the activity
}

// Helper functions to derive PDAs
pub fn camera_registry_seeds() -> &'static [u8] {
    b"camera-registry"
}

// Updated camera account seeds - now using name instead of camera_id
pub fn camera_account_seeds<'a>(name: &'a str, owner: &'a Pubkey) -> [&'a [u8]; 3] {
    [b"camera", name.as_bytes(), owner.as_ref()]
}