use anchor_lang::prelude::*;

/// Main escrow account for a competition
/// Seeds: ["competition", camera.key(), created_at.to_le_bytes()]
#[account]
pub struct CompetitionEscrow {
    /// Who initiated/created the competition
    pub initiator: Pubkey,
    /// Camera authorized to settle results (Jetson device key)
    pub camera: Pubkey,
    /// Amount each participant must deposit (lamports)
    pub stake_per_person: u64,
    /// Participants who have deposited (includes initiator if they're participating)
    pub participants: Vec<Pubkey>,
    /// Invited participants who haven't accepted yet
    pub pending_invites: Vec<Pubkey>,
    /// Total lamports held in escrow
    pub total_pool: u64,
    /// Current status of the competition
    pub status: CompetitionStatus,
    /// How winnings are distributed
    pub payout_rule: PayoutRule,
    /// Unix timestamp when created
    pub created_at: i64,
    /// Timeout for pending invites (seconds from created_at)
    pub invite_timeout_secs: u32,
    /// Winner(s) after settlement - could be multiple for threshold split
    pub winners: Vec<Pubkey>,
    /// PDA bump
    pub bump: u8,
}

impl CompetitionEscrow {
    /// Calculate space needed for the account
    /// Max 10 participants, max 10 pending invites, max 10 winners
    pub const MAX_PARTICIPANTS: usize = 10;

    pub const SPACE: usize = 8  // discriminator
        + 32  // initiator
        + 32  // camera
        + 8   // stake_per_person
        + 4 + (32 * Self::MAX_PARTICIPANTS)  // participants vec
        + 4 + (32 * Self::MAX_PARTICIPANTS)  // pending_invites vec
        + 8   // total_pool
        + 1   // status (enum)
        + 1 + 4  // payout_rule (enum with optional u32)
        + 8   // created_at
        + 4   // invite_timeout_secs
        + 4 + (32 * Self::MAX_PARTICIPANTS)  // winners vec
        + 1;  // bump

    pub fn is_invite_expired(&self, current_time: i64) -> bool {
        current_time > self.created_at + (self.invite_timeout_secs as i64)
    }

    pub fn is_participant(&self, pubkey: &Pubkey) -> bool {
        self.participants.contains(pubkey)
    }

    pub fn is_invited(&self, pubkey: &Pubkey) -> bool {
        self.pending_invites.contains(pubkey)
    }
}

/// Competition lifecycle status
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum CompetitionStatus {
    /// Waiting for invited participants to accept
    Pending,
    /// All participants joined (or timeout), competition is live
    Active,
    /// Competition ended, funds distributed
    Settled,
    /// Competition cancelled, funds refunded
    Cancelled,
}

impl Default for CompetitionStatus {
    fn default() -> Self {
        CompetitionStatus::Pending
    }
}

/// How competition winnings are distributed
#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum PayoutRule {
    /// Single winner takes entire pool
    WinnerTakesAll,
    /// Split among all who hit the threshold (reps)
    ThresholdSplit { min_reps: u32 },
}

impl Default for PayoutRule {
    fn default() -> Self {
        PayoutRule::WinnerTakesAll
    }
}

/// Result submitted for a participant at settlement
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct ParticipantResult {
    pub participant: Pubkey,
    pub score: u32,  // e.g., push-up reps
}

/// Arguments for creating a competition
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct CreateCompetitionArgs {
    /// Participants to invite (excluding initiator if they're also participating)
    pub invitees: Vec<Pubkey>,
    /// Whether initiator is also a participant
    pub initiator_participates: bool,
    /// Amount each participant stakes
    pub stake_per_person: u64,
    /// Payout rule
    pub payout_rule: PayoutRule,
    /// Timeout for invites in seconds (default: 60)
    pub invite_timeout_secs: Option<u32>,
}

// ==================== Events ====================

#[event]
pub struct CompetitionCreated {
    pub escrow: Pubkey,
    pub initiator: Pubkey,
    pub camera: Pubkey,
    pub stake_per_person: u64,
    pub invitees: Vec<Pubkey>,
    pub payout_rule: PayoutRule,
    pub created_at: i64,
}

#[event]
pub struct ParticipantJoined {
    pub escrow: Pubkey,
    pub participant: Pubkey,
    pub total_pool: u64,
    pub participants_count: u8,
}

#[event]
pub struct CompetitionStarted {
    pub escrow: Pubkey,
    pub participants: Vec<Pubkey>,
    pub total_pool: u64,
}

#[event]
pub struct CompetitionSettled {
    pub escrow: Pubkey,
    pub winners: Vec<Pubkey>,
    pub payout_per_winner: u64,
    pub total_distributed: u64,
}

#[event]
pub struct CompetitionCancelled {
    pub escrow: Pubkey,
    pub reason: String,
    pub refunded_to: Vec<Pubkey>,
}

#[event]
pub struct ParticipantDeclined {
    pub escrow: Pubkey,
    pub participant: Pubkey,
}
