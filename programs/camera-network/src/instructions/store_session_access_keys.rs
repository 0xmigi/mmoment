use anchor_lang::prelude::*;
use anchor_lang::system_program;
use crate::state::{UserSessionChain, EncryptedSessionKey};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
pub struct StoreSessionAccessKeys<'info> {
    /// The signer - must be either the user or the authority
    #[account(mut)]
    pub signer: Signer<'info>,

    /// The user whose session chain is being updated
    /// CHECK: Validated against the session chain's user field
    pub user: UncheckedAccount<'info>,

    #[account(
        mut,
        seeds = [b"user-session-chain", user.key().as_ref()],
        bump,
    )]
    pub user_session_chain: Account<'info, UserSessionChain>,

    pub system_program: Program<'info, System>,
}

pub fn handler(
    ctx: Context<StoreSessionAccessKeys>,
    keys: Vec<EncryptedSessionKey>,
) -> Result<()> {
    let signer = &ctx.accounts.signer;
    let user = &ctx.accounts.user;
    let chain = &mut ctx.accounts.user_session_chain;

    // Verify signer is either the user or the authority
    require!(
        signer.key() == chain.user || signer.key() == chain.authority,
        CameraNetworkError::Unauthorized
    );

    // Verify the user account matches the chain's user
    require!(
        user.key() == chain.user,
        CameraNetworkError::Unauthorized
    );

    // Calculate new size needed
    // Each EncryptedSessionKey: 4 (vec len for ciphertext) + ~48 (ciphertext) + 12 (nonce) + 8 (timestamp) = ~72 bytes
    let new_keys_count = chain.encrypted_keys.len() + keys.len();
    let new_size = 8 + 32 + 32 + 4 + (new_keys_count * 72) + 8 + 1;

    // Realloc if needed
    let account_info = chain.to_account_info();
    let current_size = account_info.data_len();

    if new_size > current_size {
        let rent = Rent::get()?;
        let new_minimum_balance = rent.minimum_balance(new_size);
        let current_balance = account_info.lamports();

        if current_balance < new_minimum_balance {
            let lamports_diff = new_minimum_balance.saturating_sub(current_balance);
            system_program::transfer(
                CpiContext::new(
                    ctx.accounts.system_program.to_account_info(),
                    system_program::Transfer {
                        from: ctx.accounts.signer.to_account_info(),
                        to: account_info.clone(),
                    },
                ),
                lamports_diff,
            )?;
        }

        account_info.realloc(new_size, false)?;
    }

    // Add the new keys
    let keys_added = keys.len();
    for key in keys {
        chain.encrypted_keys.push(key);
    }

    // Update session count
    chain.session_count = chain.session_count.saturating_add(keys_added as u64);

    msg!(
        "Stored {} session access key(s) for user {} (total: {})",
        keys_added,
        chain.user,
        chain.session_count
    );

    Ok(())
}
