// Auto-generated from target/idl/competition_escrow.json (anchor-lang 0.31.1 format)
export type CompetitionEscrow = {
  "address": "32jXEKF2GDjbezk4x8SkgddeVNMYkFjEh5PiAJijxqLJ",
  "metadata": {
    "name": "competition_escrow",
    "version": "0.1.0",
    "spec": "0.1.0",
    "description": "Solana escrow program for spontaneous IRL competitions with pool/sponsored funding"
  },
  "instructions": [
    {
      "name": "cancel_competition",
      "discriminator": [62, 4, 198, 98, 200, 41, 255, 72],
      "accounts": [
        { "name": "initiator", "writable": true, "signer": true },
        { "name": "escrow", "writable": true },
        { "name": "system_program" }
      ],
      "args": [{ "name": "reason", "type": "string" }]
    },
    {
      "name": "create_competition",
      "discriminator": [110, 212, 234, 212, 118, 128, 158, 244],
      "accounts": [
        { "name": "initiator", "writable": true, "signer": true },
        { "name": "camera" },
        { "name": "escrow", "writable": true },
        { "name": "system_program" }
      ],
      "args": [
        { "name": "args", "type": { "defined": { "name": "CreateCompetitionArgs" } } },
        { "name": "created_at", "type": "i64" }
      ]
    },
    {
      "name": "decline_competition",
      "discriminator": [174, 182, 79, 124, 176, 118, 19, 171],
      "accounts": [
        { "name": "participant", "signer": true },
        { "name": "escrow", "writable": true }
      ],
      "args": []
    },
    {
      "name": "join_competition",
      "discriminator": [9, 202, 251, 16, 34, 54, 85, 243],
      "accounts": [
        { "name": "participant", "writable": true, "signer": true },
        { "name": "escrow", "writable": true },
        { "name": "system_program" }
      ],
      "args": []
    },
    {
      "name": "settle_competition",
      "discriminator": [83, 121, 9, 141, 170, 133, 230, 151],
      "accounts": [
        { "name": "payer", "writable": true, "signer": true },
        { "name": "camera", "signer": true },
        { "name": "escrow", "writable": true },
        { "name": "camera_owner", "writable": true },
        { "name": "system_program" }
      ],
      "args": [
        { "name": "results", "type": { "vec": { "defined": { "name": "ParticipantResult" } } } }
      ]
    },
    {
      "name": "start_competition",
      "discriminator": [213, 149, 189, 30, 12, 88, 146, 54],
      "accounts": [
        { "name": "authority", "signer": true },
        { "name": "escrow", "writable": true }
      ],
      "args": []
    }
  ],
  "accounts": [
    {
      "name": "CompetitionEscrow",
      "discriminator": [160, 59, 187, 35, 242, 15, 134, 246]
    }
  ],
  "errors": [
    { "code": 6000, "name": "InvalidStatus", "msg": "Competition is not in the expected status for this operation" },
    { "code": 6001, "name": "UnauthorizedCamera", "msg": "Only the authorized camera can settle this competition" },
    { "code": 6002, "name": "UnauthorizedInitiator", "msg": "Only the initiator can cancel this competition" },
    { "code": 6003, "name": "NotInvited", "msg": "Participant is not invited to this competition" },
    { "code": 6004, "name": "AlreadyJoined", "msg": "Participant has already joined this competition" },
    { "code": 6005, "name": "InsufficientFunds", "msg": "Insufficient funds to join competition" },
    { "code": 6006, "name": "MaxParticipantsReached", "msg": "Maximum number of participants reached" },
    { "code": 6007, "name": "NoParticipants", "msg": "No participants in the competition" },
    { "code": 6008, "name": "IncompleteResults", "msg": "Results do not include all participants" },
    { "code": 6009, "name": "InviteExpired", "msg": "Invite timeout has expired" },
    { "code": 6010, "name": "CannotCancel", "msg": "Cannot cancel - competition is already active or settled" },
    { "code": 6011, "name": "NoWinners", "msg": "No winners determined from results" },
    { "code": 6012, "name": "InvalidStakeAmount", "msg": "Stake amount must be greater than zero" },
    { "code": 6013, "name": "NoInvitees", "msg": "Must invite at least one participant" },
    { "code": 6014, "name": "ArithmeticOverflow", "msg": "Arithmetic overflow occurred" },
    { "code": 6015, "name": "NoFundsToDistribute", "msg": "Competition has no funds to distribute" },
    { "code": 6016, "name": "ParticipantNotInResults", "msg": "Participant not found in results" }
  ],
  "types": [
    {
      "name": "CompetitionEscrow",
      "type": {
        "kind": "struct",
        "fields": [
          { "name": "initiator", "type": "pubkey" },
          { "name": "camera", "type": "pubkey" },
          { "name": "stake_per_person", "type": "u64" },
          { "name": "participants", "type": { "vec": "pubkey" } },
          { "name": "pending_invites", "type": { "vec": "pubkey" } },
          { "name": "total_pool", "type": "u64" },
          { "name": "status", "type": { "defined": { "name": "CompetitionStatus" } } },
          { "name": "payout_rule", "type": { "defined": { "name": "PayoutRule" } } },
          { "name": "created_at", "type": "i64" },
          { "name": "invite_timeout_secs", "type": "u32" },
          { "name": "winners", "type": { "vec": "pubkey" } },
          { "name": "bump", "type": "u8" }
        ]
      }
    },
    {
      "name": "CompetitionStatus",
      "type": {
        "kind": "enum",
        "variants": [
          { "name": "Pending" },
          { "name": "Active" },
          { "name": "Settled" },
          { "name": "Cancelled" }
        ]
      }
    },
    {
      "name": "CreateCompetitionArgs",
      "type": {
        "kind": "struct",
        "fields": [
          { "name": "invitees", "type": { "vec": "pubkey" } },
          { "name": "initiator_participates", "type": "bool" },
          { "name": "stake_per_person", "type": "u64" },
          { "name": "payout_rule", "type": { "defined": { "name": "PayoutRule" } } },
          { "name": "invite_timeout_secs", "type": { "option": "u32" } }
        ]
      }
    },
    {
      "name": "ParticipantResult",
      "type": {
        "kind": "struct",
        "fields": [
          { "name": "participant", "type": "pubkey" },
          { "name": "score", "type": "u32" }
        ]
      }
    },
    {
      "name": "PayoutRule",
      "type": {
        "kind": "enum",
        "variants": [
          { "name": "WinnerTakesAll" },
          { "name": "ThresholdSplit", "fields": [{ "name": "min_reps", "type": "u32" }] }
        ]
      }
    }
  ]
};

export const IDL: CompetitionEscrow = {
  "address": "32jXEKF2GDjbezk4x8SkgddeVNMYkFjEh5PiAJijxqLJ",
  "metadata": {
    "name": "competition_escrow",
    "version": "0.1.0",
    "spec": "0.1.0",
    "description": "Solana escrow program for spontaneous IRL competitions with pool/sponsored funding"
  },
  "instructions": [
    {
      "name": "cancel_competition",
      "discriminator": [62, 4, 198, 98, 200, 41, 255, 72],
      "accounts": [
        { "name": "initiator", "writable": true, "signer": true },
        { "name": "escrow", "writable": true },
        { "name": "system_program" }
      ],
      "args": [{ "name": "reason", "type": "string" }]
    },
    {
      "name": "create_competition",
      "discriminator": [110, 212, 234, 212, 118, 128, 158, 244],
      "accounts": [
        { "name": "initiator", "writable": true, "signer": true },
        { "name": "camera" },
        { "name": "escrow", "writable": true },
        { "name": "system_program" }
      ],
      "args": [
        { "name": "args", "type": { "defined": { "name": "CreateCompetitionArgs" } } },
        { "name": "created_at", "type": "i64" }
      ]
    },
    {
      "name": "decline_competition",
      "discriminator": [174, 182, 79, 124, 176, 118, 19, 171],
      "accounts": [
        { "name": "participant", "signer": true },
        { "name": "escrow", "writable": true }
      ],
      "args": []
    },
    {
      "name": "join_competition",
      "discriminator": [9, 202, 251, 16, 34, 54, 85, 243],
      "accounts": [
        { "name": "participant", "writable": true, "signer": true },
        { "name": "escrow", "writable": true },
        { "name": "system_program" }
      ],
      "args": []
    },
    {
      "name": "settle_competition",
      "discriminator": [83, 121, 9, 141, 170, 133, 230, 151],
      "accounts": [
        { "name": "payer", "writable": true, "signer": true },
        { "name": "camera", "signer": true },
        { "name": "escrow", "writable": true },
        { "name": "camera_owner", "writable": true },
        { "name": "system_program" }
      ],
      "args": [
        { "name": "results", "type": { "vec": { "defined": { "name": "ParticipantResult" } } } }
      ]
    },
    {
      "name": "start_competition",
      "discriminator": [213, 149, 189, 30, 12, 88, 146, 54],
      "accounts": [
        { "name": "authority", "signer": true },
        { "name": "escrow", "writable": true }
      ],
      "args": []
    }
  ],
  "accounts": [
    {
      "name": "CompetitionEscrow",
      "discriminator": [160, 59, 187, 35, 242, 15, 134, 246]
    }
  ],
  "errors": [
    { "code": 6000, "name": "InvalidStatus", "msg": "Competition is not in the expected status for this operation" },
    { "code": 6001, "name": "UnauthorizedCamera", "msg": "Only the authorized camera can settle this competition" },
    { "code": 6002, "name": "UnauthorizedInitiator", "msg": "Only the initiator can cancel this competition" },
    { "code": 6003, "name": "NotInvited", "msg": "Participant is not invited to this competition" },
    { "code": 6004, "name": "AlreadyJoined", "msg": "Participant has already joined this competition" },
    { "code": 6005, "name": "InsufficientFunds", "msg": "Insufficient funds to join competition" },
    { "code": 6006, "name": "MaxParticipantsReached", "msg": "Maximum number of participants reached" },
    { "code": 6007, "name": "NoParticipants", "msg": "No participants in the competition" },
    { "code": 6008, "name": "IncompleteResults", "msg": "Results do not include all participants" },
    { "code": 6009, "name": "InviteExpired", "msg": "Invite timeout has expired" },
    { "code": 6010, "name": "CannotCancel", "msg": "Cannot cancel - competition is already active or settled" },
    { "code": 6011, "name": "NoWinners", "msg": "No winners determined from results" },
    { "code": 6012, "name": "InvalidStakeAmount", "msg": "Stake amount must be greater than zero" },
    { "code": 6013, "name": "NoInvitees", "msg": "Must invite at least one participant" },
    { "code": 6014, "name": "ArithmeticOverflow", "msg": "Arithmetic overflow occurred" },
    { "code": 6015, "name": "NoFundsToDistribute", "msg": "Competition has no funds to distribute" },
    { "code": 6016, "name": "ParticipantNotInResults", "msg": "Participant not found in results" }
  ],
  "types": [
    {
      "name": "CompetitionEscrow",
      "type": {
        "kind": "struct",
        "fields": [
          { "name": "initiator", "type": "pubkey" },
          { "name": "camera", "type": "pubkey" },
          { "name": "stake_per_person", "type": "u64" },
          { "name": "participants", "type": { "vec": "pubkey" } },
          { "name": "pending_invites", "type": { "vec": "pubkey" } },
          { "name": "total_pool", "type": "u64" },
          { "name": "status", "type": { "defined": { "name": "CompetitionStatus" } } },
          { "name": "payout_rule", "type": { "defined": { "name": "PayoutRule" } } },
          { "name": "created_at", "type": "i64" },
          { "name": "invite_timeout_secs", "type": "u32" },
          { "name": "winners", "type": { "vec": "pubkey" } },
          { "name": "bump", "type": "u8" }
        ]
      }
    },
    {
      "name": "CompetitionStatus",
      "type": {
        "kind": "enum",
        "variants": [
          { "name": "Pending" },
          { "name": "Active" },
          { "name": "Settled" },
          { "name": "Cancelled" }
        ]
      }
    },
    {
      "name": "CreateCompetitionArgs",
      "type": {
        "kind": "struct",
        "fields": [
          { "name": "invitees", "type": { "vec": "pubkey" } },
          { "name": "initiator_participates", "type": "bool" },
          { "name": "stake_per_person", "type": "u64" },
          { "name": "payout_rule", "type": { "defined": { "name": "PayoutRule" } } },
          { "name": "invite_timeout_secs", "type": { "option": "u32" } }
        ]
      }
    },
    {
      "name": "ParticipantResult",
      "type": {
        "kind": "struct",
        "fields": [
          { "name": "participant", "type": "pubkey" },
          { "name": "score", "type": "u32" }
        ]
      }
    },
    {
      "name": "PayoutRule",
      "type": {
        "kind": "enum",
        "variants": [
          { "name": "WinnerTakesAll" },
          { "name": "ThresholdSplit", "fields": [{ "name": "min_reps", "type": "u32" }] }
        ]
      }
    }
  ]
};

export const COMPETITION_ESCROW_IDL = IDL;
