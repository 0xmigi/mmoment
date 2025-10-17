use anchor_lang::prelude::*;
use crate::state::{RecognitionToken, RecognitionTokenDeleted};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
pub struct DeleteRecognitionToken<'info> {
    #[account(mut)]
    pub user: Signer<'info>,

    #[account(
        mut,
        close = user,  // Rent goes back to user
        seeds = [b"recognition-token", user.key().as_ref()],
        bump = recognition_token.bump,
        constraint = recognition_token.user == user.key() @ CameraNetworkError::Unauthorized
    )]
    pub recognition_token: Account<'info, RecognitionToken>,
}

pub fn handler(ctx: Context<DeleteRecognitionToken>) -> Result<()> {
    let user = &ctx.accounts.user;
    let token = &ctx.accounts.recognition_token;
    let token_key = token.key();

    // Emit event before account is closed
    emit!(RecognitionTokenDeleted {
        user: user.key(),
        token: token_key,
        timestamp: Clock::get()?.unix_timestamp,
    });

    msg!("Recognition token deleted for user {}", user.key());
    msg!("Rent reclaimed: ~0.009 SOL");

    Ok(())
}
