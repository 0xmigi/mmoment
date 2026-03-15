use anchor_lang::prelude::*;

use crate::error::CompetitionError;
use crate::state::*;

/// Settle the competition - only the authorized camera can call this
/// Camera submits results, program determines winners and distributes funds
/// Payer (backend) covers transaction fees, camera authorizes the settlement
#[derive(Accounts)]
pub struct SettleCompetition<'info> {
    /// Fee payer (backend wallet) - pays for transaction fees
    #[account(mut)]
    pub payer: Signer<'info>,

    /// The camera (Jetson device key) authorized to settle - must sign to prove authenticity
    pub camera: Signer<'info>,

    /// The escrow account holding the funds
    #[account(
        mut,
        seeds = [
            b"competition",
            escrow.camera.as_ref(),
            &escrow.created_at.to_le_bytes()
        ],
        bump = escrow.bump,
        constraint = camera.key() == escrow.camera @ CompetitionError::UnauthorizedCamera
    )]
    pub escrow: Account<'info, CompetitionEscrow>,

    /// The camera owner's wallet - receives failed prize funds in ThresholdSplit mode
    /// This is the wallet that owns the camera in camera_network program
    /// CHECK: We trust the camera (Jetson device) to pass the correct owner
    #[account(mut)]
    pub camera_owner: AccountInfo<'info>,

    pub system_program: Program<'info, System>,
    // Note: Winner accounts are passed as remaining_accounts
}

pub fn handler<'info>(
    ctx: Context<'_, '_, '_, 'info, SettleCompetition<'info>>,
    results: Vec<ParticipantResult>,
) -> Result<()> {
    let escrow = &mut ctx.accounts.escrow;

    // Validate status - must be Active
    require!(
        escrow.status == CompetitionStatus::Active,
        CompetitionError::InvalidStatus
    );

    // Validate results cover all participants
    require!(
        results.len() == escrow.participants.len(),
        CompetitionError::IncompleteResults
    );

    // Verify all results are for actual participants
    for result in &results {
        require!(
            escrow.is_participant(&result.participant),
            CompetitionError::ParticipantNotInResults
        );
    }

    // Determine winners based on payout rule
    let winners: Vec<Pubkey> = match &escrow.payout_rule {
        PayoutRule::WinnerTakesAll => {
            // Find highest score
            let max_score = results.iter().map(|r| r.score).max().unwrap_or(0);
            if max_score == 0 {
                // No one scored - could refund or give to initiator
                // For simplicity, we'll treat it as no winners (refund scenario)
                Vec::new()
            } else {
                // Could be ties - all with max score win
                results
                    .iter()
                    .filter(|r| r.score == max_score)
                    .map(|r| r.participant)
                    .collect()
            }
        }
        PayoutRule::ThresholdSplit { min_reps } => {
            // All who hit threshold are winners
            results
                .iter()
                .filter(|r| r.score >= *min_reps)
                .map(|r| r.participant)
                .collect()
        }
    };

    // Get the total pool (what's currently in the escrow account)
    let escrow_lamports = escrow.to_account_info().lamports();
    let rent = Rent::get()?;
    let min_rent = rent.minimum_balance(CompetitionEscrow::SPACE);

    // Available for distribution = total - rent exempt minimum
    let distributable = escrow_lamports
        .checked_sub(min_rent)
        .ok_or(CompetitionError::NoFundsToDistribute)?;

    require!(distributable > 0, CompetitionError::NoFundsToDistribute);

    // Handle payout based on winners and payout rule
    let payout_recipients: Vec<Pubkey>;
    let payout_per_recipient: u64;
    let total_distributed: u64;
    let camera_takes_all: bool;

    if winners.is_empty() {
        // No winners - behavior depends on payout rule
        match &escrow.payout_rule {
            PayoutRule::ThresholdSplit { .. } => {
                // Prize mode: camera owner takes the funds (carnival game model)
                // This incentivizes venues to host challenges
                camera_takes_all = true;
                payout_recipients = vec![ctx.accounts.camera_owner.key()];
                payout_per_recipient = distributable;
                total_distributed = distributable;
            }
            PayoutRule::WinnerTakesAll => {
                // Bet mode with no winners (all scored 0): refund participants
                camera_takes_all = false;
                payout_recipients = escrow.participants.clone();
                payout_per_recipient = distributable
                    .checked_div(payout_recipients.len() as u64)
                    .ok_or(CompetitionError::ArithmeticOverflow)?;
                total_distributed = payout_per_recipient * payout_recipients.len() as u64;
            }
        }
    } else {
        // Pay winners
        camera_takes_all = false;
        payout_recipients = winners.clone();
        payout_per_recipient = distributable
            .checked_div(payout_recipients.len() as u64)
            .ok_or(CompetitionError::ArithmeticOverflow)?;
        total_distributed = payout_per_recipient * payout_recipients.len() as u64;
    }

    // Transfer funds
    if camera_takes_all {
        // Camera owner takes all - transfer to the venue/camera owner
        **escrow.to_account_info().try_borrow_mut_lamports()? -= distributable;
        **ctx.accounts.camera_owner.try_borrow_mut_lamports()? += distributable;
    } else {
        // Transfer to each recipient using remaining_accounts
        // The frontend must pass winner accounts in the same order as payout_recipients
        let remaining_accounts = &ctx.remaining_accounts;
        require!(
            remaining_accounts.len() >= payout_recipients.len(),
            CompetitionError::IncompleteResults
        );

        for (i, recipient) in payout_recipients.iter().enumerate() {
            let recipient_account = &remaining_accounts[i];

            // Verify the account matches the expected recipient
            require!(
                recipient_account.key() == *recipient,
                CompetitionError::ParticipantNotInResults
            );

            // Transfer from escrow PDA to recipient
            **escrow.to_account_info().try_borrow_mut_lamports()? -= payout_per_recipient;
            **recipient_account.try_borrow_mut_lamports()? += payout_per_recipient;
        }
    }

    // Update escrow state
    escrow.status = CompetitionStatus::Settled;
    escrow.winners = winners.clone();
    escrow.total_pool = 0;

    emit!(CompetitionSettled {
        escrow: escrow.key(),
        winners: winners,
        payout_per_winner: payout_per_recipient,
        total_distributed,
    });

    Ok(())
}
