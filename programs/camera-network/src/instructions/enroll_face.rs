use anchor_lang::prelude::*;
use crate::state::{RecognitionToken, RecognitionTokenCreated};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
pub struct UpsertRecognitionToken<'info> {
    #[account(mut)]
    pub user: Signer<'info>,

    #[account(
        init_if_needed,
        payer = user,
        space = 8 + 32 + 4 + 1024 + 8 + 1 + 1 + 4 + 64 + 1, // 1147 bytes - standard size for single transaction
        seeds = [b"recognition-token", user.key().as_ref()],
        bump
    )]
    pub recognition_token: Account<'info, RecognitionToken>,

    pub system_program: Program<'info, System>,
}

pub fn handler(
    ctx: Context<UpsertRecognitionToken>,
    encrypted_embedding: Vec<u8>,
    display_name: Option<String>,
    source: u8,
) -> Result<()> {
    let user = &ctx.accounts.user;
    let token = &mut ctx.accounts.recognition_token;

    // Validate embedding size (1-1024 bytes for single transaction)
    require!(
        !encrypted_embedding.is_empty() && encrypted_embedding.len() <= 1024,
        CameraNetworkError::RecognitionTokenTooLarge
    );

    // Validate source type (0=phone_selfie, 1=jetson_capture, 2=imported)
    require!(source <= 2, CameraNetworkError::InvalidFaceData);

    let is_new = token.created_at == 0;
    let now = Clock::get()?.unix_timestamp;

    // Store full encrypted embedding
    token.user = user.key();
    token.encrypted_embedding = encrypted_embedding;
    token.created_at = now;
    token.bump = ctx.bumps.recognition_token;
    token.display_name = display_name.clone();
    token.source = source;

    // Increment version on regeneration
    if is_new {
        token.version = 1;
    } else {
        token.version = token.version.saturating_add(1);
    }

    // Emit event
    emit!(RecognitionTokenCreated {
        user: user.key(),
        token: token.key(),
        version: token.version,
        source,
        display_name,
        timestamp: now,
    });

    msg!("Recognition token {} for user {}",
        if is_new { "created" } else { "regenerated" },
        user.key()
    );

    Ok(())
} 