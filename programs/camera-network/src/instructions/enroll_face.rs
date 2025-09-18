use anchor_lang::prelude::*;
use crate::state::{
    FaceData
};
use crate::error::CameraNetworkError;
use solana_program::keccak;

#[derive(Accounts)]
pub struct EnrollFace<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    
    // This account is owned by the user directly, not the program
    // It's important to note that this is "owned" by the user, but still a PDA
    // (which is necessary for deterministic derivation)
    #[account(
        init_if_needed,
        payer = user,
        space = 8 + 32 + 32 + 4 + 32 * 10 + 8 + 8 + 1, // Discriminator + user + hash + camera vec + timestamps + bump
        seeds = [
            b"face-nft",
            user.key().as_ref()
        ],
        bump
    )]
    pub face_nft: Account<'info, FaceData>,
    
    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<EnrollFace>, encrypted_embedding: Vec<u8>) -> Result<()> {
    let user = &ctx.accounts.user;
    let face_nft = &mut ctx.accounts.face_nft;
    
    // Validate face data
    if encrypted_embedding.is_empty() || encrypted_embedding.len() > 1024 {
        return err!(CameraNetworkError::InvalidFaceData);
    }
    
    // Compute hash of the face data (for verification, not storing raw data)
    let mut face_hash = [0u8; 32];
    face_hash.copy_from_slice(&keccak::hash(&encrypted_embedding).to_bytes()[0..32]);
    
    // Set face data account
    face_nft.user = user.key();
    face_nft.data_hash = face_hash;
    
    // If this is a new account, initialize the rest of the fields
    if face_nft.creation_date == 0 {
        face_nft.authorized_cameras = Vec::new();
        face_nft.creation_date = Clock::get()?.unix_timestamp;
        face_nft.bump = ctx.bumps.face_nft;
    }
    
    face_nft.last_used = Clock::get()?.unix_timestamp;
    
    msg!("Face ID NFT created for user {}", user.key());
    msg!("The face data is encrypted and only usable by the user who created it");
    
    Ok(())
} 