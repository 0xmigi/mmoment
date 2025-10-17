use anchor_lang::prelude::*;

#[error_code]
pub enum CameraNetworkError {
    #[msg("You are not authorized to perform this action")]
    Unauthorized,
    
    #[msg("Camera is currently inactive")]
    CameraInactive,
    
    #[msg("Camera not found in registry")]
    CameraNotFound,
    
    #[msg("Camera name already exists")]
    CameraNameExists,
    
    #[msg("Invalid camera data provided")]
    InvalidCameraData,
    
    #[msg("No active session found")]
    NoActiveSession,
    
    #[msg("Session already exists")]
    SessionExists,
    
    #[msg("Access denied to this camera")]
    AccessDenied,
    
    #[msg("Face data invalid or not properly formatted")]
    InvalidFaceData,
    
    #[msg("Face data already registered for this user")]
    FaceDataExists,
    
    #[msg("Gesture data invalid or improperly formatted")]
    InvalidGestureData,
    
    #[msg("Camera is already authorized for face recognition")]
    CameraAlreadyAuthorized,
    
    #[msg("Invalid temporary access duration")]
    InvalidAccessDuration,
    
    #[msg("Access grant has expired")]
    AccessGrantExpired,

    #[msg("Session has expired")]
    SessionExpired,

    #[msg("No recognition token found - please create one first")]
    NoRecognitionToken,

    #[msg("Feature not available on this camera")]
    FeatureNotAvailable,

    #[msg("Recognition token data too large (max 1024 bytes)")]
    RecognitionTokenTooLarge,
} 