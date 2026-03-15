use anchor_lang::prelude::*;

use crate::error::CompetitionError;
use crate::state::*;

#[derive(Accounts)]
pub struct DeclineCompetition<'info> {
    /// The participant declining the invite
    pub participant: Signer<'info>,

    /// The escrow account
    #[account(
        mut,
        seeds = [
            b"competition",
            escrow.camera.as_ref(),
            &escrow.created_at.to_le_bytes()
        ],
        bump = escrow.bump
    )]
    pub escrow: Account<'info, CompetitionEscrow>,
}

pub fn handler(ctx: Context<DeclineCompetition>) -> Result<()> {
    let escrow = &mut ctx.accounts.escrow;
    let participant = &ctx.accounts.participant;

    // Validate status
    require!(
        escrow.status == CompetitionStatus::Pending,
        CompetitionError::InvalidStatus
    );

    // Check participant is invited
    require!(
        escrow.is_invited(&participant.key()),
        CompetitionError::NotInvited
    );

    // Remove from pending invites
    escrow.pending_invites.retain(|p| p != &participant.key());

    emit!(ParticipantDeclined {
        escrow: escrow.key(),
        participant: participant.key(),
    });

    Ok(())
}
