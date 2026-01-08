use anchor_lang::prelude::*;

#[error_code]
pub enum CompetitionError {
    #[msg("Competition is not in the expected status for this operation")]
    InvalidStatus,

    #[msg("Only the authorized camera can settle this competition")]
    UnauthorizedCamera,

    #[msg("Only the initiator can cancel this competition")]
    UnauthorizedInitiator,

    #[msg("Participant is not invited to this competition")]
    NotInvited,

    #[msg("Participant has already joined this competition")]
    AlreadyJoined,

    #[msg("Insufficient funds to join competition")]
    InsufficientFunds,

    #[msg("Maximum number of participants reached")]
    MaxParticipantsReached,

    #[msg("No participants in the competition")]
    NoParticipants,

    #[msg("Results do not include all participants")]
    IncompleteResults,

    #[msg("Invite timeout has expired")]
    InviteExpired,

    #[msg("Cannot cancel - competition is already active or settled")]
    CannotCancel,

    #[msg("No winners determined from results")]
    NoWinners,

    #[msg("Stake amount must be greater than zero")]
    InvalidStakeAmount,

    #[msg("Must invite at least one participant")]
    NoInvitees,

    #[msg("Arithmetic overflow occurred")]
    ArithmeticOverflow,

    #[msg("Competition has no funds to distribute")]
    NoFundsToDistribute,

    #[msg("Participant not found in results")]
    ParticipantNotInResults,
}
