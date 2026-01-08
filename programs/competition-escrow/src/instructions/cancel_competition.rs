use anchor_lang::prelude::*;

use crate::error::CompetitionError;
use crate::state::*;

/// Cancel the competition and refund all participants
/// Only the initiator can cancel, and only before settlement
#[derive(Accounts)]
pub struct CancelCompetition<'info> {
    /// The initiator cancelling the competition
    #[account(mut)]
    pub initiator: Signer<'info>,

    /// The escrow account
    #[account(
        mut,
        seeds = [
            b"competition",
            escrow.camera.as_ref(),
            &escrow.created_at.to_le_bytes()
        ],
        bump = escrow.bump,
        constraint = initiator.key() == escrow.initiator @ CompetitionError::UnauthorizedInitiator
    )]
    pub escrow: Account<'info, CompetitionEscrow>,

    pub system_program: Program<'info, System>,
    // Participant accounts for refunds are passed as remaining_accounts
}

pub fn handler<'info>(
    ctx: Context<'_, '_, '_, 'info, CancelCompetition<'info>>,
    reason: String,
) -> Result<()> {
    let escrow = &mut ctx.accounts.escrow;

    // Can only cancel if Pending or Active (not Settled or already Cancelled)
    require!(
        escrow.status == CompetitionStatus::Pending || escrow.status == CompetitionStatus::Active,
        CompetitionError::CannotCancel
    );

    // Get refund amount per participant
    let stake = escrow.stake_per_person;
    let participants = escrow.participants.clone();

    // Refund each participant
    let remaining_accounts = &ctx.remaining_accounts;
    require!(
        remaining_accounts.len() >= participants.len(),
        CompetitionError::IncompleteResults
    );

    for (i, participant) in participants.iter().enumerate() {
        let participant_account = &remaining_accounts[i];

        // Verify the account matches
        require!(
            participant_account.key() == *participant,
            CompetitionError::ParticipantNotInResults
        );

        // Transfer refund from escrow to participant
        **escrow.to_account_info().try_borrow_mut_lamports()? -= stake;
        **participant_account.try_borrow_mut_lamports()? += stake;
    }

    // Update escrow state
    escrow.status = CompetitionStatus::Cancelled;
    escrow.total_pool = 0;

    emit!(CompetitionCancelled {
        escrow: escrow.key(),
        reason,
        refunded_to: participants,
    });

    Ok(())
}
