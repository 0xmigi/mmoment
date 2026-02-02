use anchor_lang::prelude::*;

use crate::error::CompetitionError;
use crate::state::*;

/// Start the competition - can be called by initiator or camera
/// Called when all invites are resolved OR timeout has passed
#[derive(Accounts)]
pub struct StartCompetition<'info> {
    /// Either the initiator or the camera can start
    pub authority: Signer<'info>,

    /// The escrow account
    #[account(
        mut,
        seeds = [
            b"competition",
            escrow.camera.as_ref(),
            &escrow.created_at.to_le_bytes()
        ],
        bump = escrow.bump,
        constraint = authority.key() == escrow.initiator || authority.key() == escrow.camera
            @ CompetitionError::UnauthorizedInitiator
    )]
    pub escrow: Account<'info, CompetitionEscrow>,
}

pub fn handler(ctx: Context<StartCompetition>) -> Result<()> {
    let escrow = &mut ctx.accounts.escrow;
    let clock = Clock::get()?;

    // Validate status
    require!(
        escrow.status == CompetitionStatus::Pending,
        CompetitionError::InvalidStatus
    );

    // For ThresholdSplit (prize mode): auto-promote pending invites to participants
    // They don't need to deposit - only the initiator put up the prize
    // For WinnerTakesAll (bet mode): require invites resolved or timeout
    match &escrow.payout_rule {
        PayoutRule::ThresholdSplit { .. } => {
            // Prize mode: auto-add all pending invites as participants (no deposit needed)
            for invitee in escrow.pending_invites.clone() {
                if !escrow.participants.contains(&invitee) {
                    escrow.participants.push(invitee);
                }
            }
            escrow.pending_invites.clear();
        }
        PayoutRule::WinnerTakesAll => {
            // Bet mode: require all invites resolved OR timeout passed
            let invites_resolved = escrow.pending_invites.is_empty();
            let timeout_passed = escrow.is_invite_expired(clock.unix_timestamp);

            require!(
                invites_resolved || timeout_passed,
                CompetitionError::InvalidStatus
            );

            // Clear any remaining pending invites (treated as declined due to timeout)
            escrow.pending_invites.clear();
        }
    }

    // For WinnerTakesAll (bet), need at least 2 participants
    // For ThresholdSplit (prize), can be solo (1 participant betting against themselves)
    let min_participants = match escrow.payout_rule {
        PayoutRule::ThresholdSplit { .. } => 1,
        PayoutRule::WinnerTakesAll => 2,
    };

    require!(
        escrow.participants.len() >= min_participants,
        CompetitionError::NoParticipants
    );

    // Activate the competition
    escrow.status = CompetitionStatus::Active;

    emit!(CompetitionStarted {
        escrow: escrow.key(),
        participants: escrow.participants.clone(),
        total_pool: escrow.total_pool,
    });

    Ok(())
}
