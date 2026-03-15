use anchor_lang::prelude::*;
use anchor_lang::system_program;

use crate::error::CompetitionError;
use crate::state::*;

#[derive(Accounts)]
pub struct JoinCompetition<'info> {
    /// The participant accepting the invite
    #[account(mut)]
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

    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<JoinCompetition>) -> Result<()> {
    let escrow = &mut ctx.accounts.escrow;
    let participant = &ctx.accounts.participant;
    let clock = Clock::get()?;

    // Validate status
    require!(
        escrow.status == CompetitionStatus::Pending,
        CompetitionError::InvalidStatus
    );

    // Check invite hasn't expired
    require!(
        !escrow.is_invite_expired(clock.unix_timestamp),
        CompetitionError::InviteExpired
    );

    // Check participant is invited
    require!(
        escrow.is_invited(&participant.key()),
        CompetitionError::NotInvited
    );

    // Check not already joined
    require!(
        !escrow.is_participant(&participant.key()),
        CompetitionError::AlreadyJoined
    );

    // Transfer stake from participant to escrow
    system_program::transfer(
        CpiContext::new(
            ctx.accounts.system_program.to_account_info(),
            system_program::Transfer {
                from: participant.to_account_info(),
                to: escrow.to_account_info(),
            },
        ),
        escrow.stake_per_person,
    )?;

    // Remove from pending, add to participants
    escrow.pending_invites.retain(|p| p != &participant.key());
    escrow.participants.push(participant.key());
    escrow.total_pool = escrow
        .total_pool
        .checked_add(escrow.stake_per_person)
        .ok_or(CompetitionError::ArithmeticOverflow)?;

    emit!(ParticipantJoined {
        escrow: escrow.key(),
        participant: participant.key(),
        total_pool: escrow.total_pool,
        participants_count: escrow.participants.len() as u8,
    });

    Ok(())
}
