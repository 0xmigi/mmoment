use anchor_lang::prelude::*;
use anchor_lang::system_program;

use crate::error::CompetitionError;
use crate::state::*;

#[derive(Accounts)]
#[instruction(args: CreateCompetitionArgs, created_at: i64)]
pub struct CreateCompetition<'info> {
    /// The person initiating/creating the competition
    #[account(mut)]
    pub initiator: Signer<'info>,

    /// The camera authorized to settle this competition (Jetson device key)
    /// CHECK: This is just a pubkey reference, no account data needed
    pub camera: UncheckedAccount<'info>,

    /// The escrow account to create
    #[account(
        init,
        payer = initiator,
        space = CompetitionEscrow::SPACE,
        seeds = [
            b"competition",
            camera.key().as_ref(),
            &created_at.to_le_bytes()
        ],
        bump
    )]
    pub escrow: Account<'info, CompetitionEscrow>,

    pub system_program: Program<'info, System>,
}

pub fn handler(
    ctx: Context<CreateCompetition>,
    args: CreateCompetitionArgs,
    created_at: i64,
) -> Result<()> {
    require!(args.stake_per_person > 0, CompetitionError::InvalidStakeAmount);

    // Allow empty invitees only for ThresholdSplit (prize) mode when initiator participates
    // This enables "bet against yourself" scenarios
    let allows_solo = matches!(args.payout_rule, PayoutRule::ThresholdSplit { .. })
        && args.initiator_participates;

    if !allows_solo {
        require!(!args.invitees.is_empty(), CompetitionError::NoInvitees);
    }

    require!(
        args.invitees.len() <= CompetitionEscrow::MAX_PARTICIPANTS,
        CompetitionError::MaxParticipantsReached
    );

    let escrow = &mut ctx.accounts.escrow;
    let initiator = &ctx.accounts.initiator;

    escrow.initiator = initiator.key();
    escrow.camera = ctx.accounts.camera.key();
    escrow.stake_per_person = args.stake_per_person;
    escrow.payout_rule = args.payout_rule.clone();
    escrow.created_at = created_at;
    escrow.invite_timeout_secs = args.invite_timeout_secs.unwrap_or(60); // Default 60 seconds
    escrow.status = CompetitionStatus::Pending;
    escrow.bump = ctx.bumps.escrow;
    escrow.participants = Vec::new();
    escrow.pending_invites = args.invitees.clone();
    escrow.winners = Vec::new();
    escrow.total_pool = 0;

    // If initiator is participating, they deposit now
    if args.initiator_participates {
        // Transfer stake from initiator to escrow
        system_program::transfer(
            CpiContext::new(
                ctx.accounts.system_program.to_account_info(),
                system_program::Transfer {
                    from: initiator.to_account_info(),
                    to: escrow.to_account_info(),
                },
            ),
            args.stake_per_person,
        )?;

        escrow.participants.push(initiator.key());
        escrow.total_pool = args.stake_per_person;
    }

    emit!(CompetitionCreated {
        escrow: escrow.key(),
        initiator: initiator.key(),
        camera: ctx.accounts.camera.key(),
        stake_per_person: args.stake_per_person,
        invitees: args.invitees,
        payout_rule: args.payout_rule,
        created_at,
    });

    Ok(())
}
