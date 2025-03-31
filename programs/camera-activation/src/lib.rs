use anchor_lang::prelude::*;

declare_id!("5HFxUPQ7aZPZF8UqRSSHsoamq3X4VU1K9euMXuuGcUfj");

#[event]
pub struct CameraActivated {
    // Keep timestamp for transparency
    pub timestamp: i64,
    // Keep camera_account for verification
    pub camera_account: Pubkey,
    // Add session_id for tracking without exposing user
    pub session_id: String,
    // Remove direct user pubkey to reduce traceability
}

#[program]
pub mod camera_activation {
    use super::*;

    // Keep initialize exactly as is
    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        let camera_account = &mut ctx.accounts.camera_account;
        camera_account.owner = *ctx.accounts.user.key;
        camera_account.is_active = false;
        Ok(())
    }

    pub fn activate_camera(ctx: Context<ActivateCamera>, fee: u64) -> Result<()> {
        let camera_account = &mut ctx.accounts.camera_account;
        let user = &mut ctx.accounts.user;

        if fee < 100 {
            return Err(ErrorCode::InsufficientFee.into());
        }

        // Keep payment logic
        let ix = anchor_lang::solana_program::system_instruction::transfer(
            &user.key(),
            &camera_account.key(),
            fee,
        );
        anchor_lang::solana_program::program::invoke(
            &ix,
            &[
                user.to_account_info(),
                camera_account.to_account_info(),
            ],
        )?;

        camera_account.is_active = true;

        // Modified event emission
        let timestamp = Clock::get()?.unix_timestamp;
        emit!(CameraActivated {
            timestamp,
            camera_account: camera_account.key(),
            session_id: format!("sess_{}", timestamp),
        });

        Ok(())
    }
}

// Keep all these structs exactly as they are
#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + // discriminator
               32 + // Pubkey (owner)
               1 + // bool (is_active)
               32 + // padding for alignment
               8 // extra space for future updates
    )]
    pub camera_account: Account<'info, CameraAccount>,
    #[account(mut)]
    pub user: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ActivateCamera<'info> {
    #[account(mut)]
    pub camera_account: Account<'info, CameraAccount>,
    #[account(mut)]
    pub user: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[account]
pub struct CameraAccount {
    pub owner: Pubkey,
    pub is_active: bool,
}

#[error_code]
pub enum ErrorCode {
    #[msg("The fee is insufficient")]
    InsufficientFee,
}