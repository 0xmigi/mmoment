use anchor_lang::prelude::*;

declare_id!("32jXEKF2GDjbezk4x8SkgddeVNMYkFjEh5PiAJijxqLJ");

mod error;
mod instructions;
mod state;

use instructions::*;
use state::*;

#[program]
pub mod competition_escrow {
    use super::*;

    /// Create a new competition with invited participants
    /// Initiator deposits their stake if participating
    ///
    /// # Arguments
    /// * `args` - Competition configuration (invitees, stake, payout rule)
    /// * `created_at` - Unix timestamp (used for PDA derivation)
    pub fn create_competition(
        ctx: Context<CreateCompetition>,
        args: CreateCompetitionArgs,
        created_at: i64,
    ) -> Result<()> {
        instructions::create_competition::handler(ctx, args, created_at)
    }

    /// Accept an invite and deposit stake into escrow
    pub fn join_competition(ctx: Context<JoinCompetition>) -> Result<()> {
        instructions::join_competition::handler(ctx)
    }

    /// Decline an invite (no funds involved)
    pub fn decline_competition(ctx: Context<DeclineCompetition>) -> Result<()> {
        instructions::decline_competition::handler(ctx)
    }

    /// Start the competition after invites are resolved or timeout
    /// Can be called by initiator or camera
    pub fn start_competition(ctx: Context<StartCompetition>) -> Result<()> {
        instructions::start_competition::handler(ctx)
    }

    /// Settle the competition with final results
    /// Only the authorized camera can call this
    /// Distributes funds to winner(s) based on payout rule
    ///
    /// # Arguments
    /// * `results` - Score for each participant (e.g., push-up reps)
    ///
    /// # Remaining Accounts
    /// Pass winner accounts in order for fund distribution
    pub fn settle_competition<'info>(
        ctx: Context<'_, '_, '_, 'info, SettleCompetition<'info>>,
        results: Vec<ParticipantResult>,
    ) -> Result<()> {
        instructions::settle_competition::handler(ctx, results)
    }

    /// Cancel the competition and refund all participants
    /// Only initiator can cancel
    ///
    /// # Arguments
    /// * `reason` - Reason for cancellation
    ///
    /// # Remaining Accounts
    /// Pass participant accounts in order for refunds
    pub fn cancel_competition<'info>(
        ctx: Context<'_, '_, '_, 'info, CancelCompetition<'info>>,
        reason: String,
    ) -> Result<()> {
        instructions::cancel_competition::handler(ctx, reason)
    }
}
