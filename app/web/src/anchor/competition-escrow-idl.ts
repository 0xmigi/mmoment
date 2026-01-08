export type CompetitionEscrow = {
  "version": "0.1.0",
  "name": "competition_escrow",
  "instructions": [
    {
      "name": "cancelCompetition",
      "docs": [
        "Cancel the competition and refund all participants",
        "Only initiator can cancel"
      ],
      "accounts": [
        {
          "name": "initiator",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "reason",
          "type": "string"
        }
      ]
    },
    {
      "name": "createCompetition",
      "docs": [
        "Create a new competition with invited participants",
        "Initiator deposits their stake if participating"
      ],
      "accounts": [
        {
          "name": "initiator",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "camera",
          "isMut": false,
          "isSigner": false
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "args",
          "type": {
            "defined": "CreateCompetitionArgs"
          }
        },
        {
          "name": "createdAt",
          "type": "i64"
        }
      ]
    },
    {
      "name": "declineCompetition",
      "docs": [
        "Decline an invite (no funds involved)"
      ],
      "accounts": [
        {
          "name": "participant",
          "isMut": false,
          "isSigner": true
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        }
      ],
      "args": []
    },
    {
      "name": "joinCompetition",
      "docs": [
        "Accept an invite and deposit stake into escrow"
      ],
      "accounts": [
        {
          "name": "participant",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": []
    },
    {
      "name": "settleCompetition",
      "docs": [
        "Settle the competition with final results",
        "Only the authorized camera can call this"
      ],
      "accounts": [
        {
          "name": "camera",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "cameraOwner",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "results",
          "type": {
            "vec": {
              "defined": "ParticipantResult"
            }
          }
        }
      ]
    },
    {
      "name": "startCompetition",
      "docs": [
        "Start the competition after invites are resolved or timeout",
        "Can be called by initiator or camera"
      ],
      "accounts": [
        {
          "name": "authority",
          "isMut": false,
          "isSigner": true
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        }
      ],
      "args": []
    }
  ],
  "accounts": [
    {
      "name": "competitionEscrow",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "initiator",
            "type": "publicKey"
          },
          {
            "name": "camera",
            "type": "publicKey"
          },
          {
            "name": "stakePerPerson",
            "type": "u64"
          },
          {
            "name": "participants",
            "type": {
              "vec": "publicKey"
            }
          },
          {
            "name": "pendingInvites",
            "type": {
              "vec": "publicKey"
            }
          },
          {
            "name": "totalPool",
            "type": "u64"
          },
          {
            "name": "status",
            "type": {
              "defined": "CompetitionStatus"
            }
          },
          {
            "name": "payoutRule",
            "type": {
              "defined": "PayoutRule"
            }
          },
          {
            "name": "createdAt",
            "type": "i64"
          },
          {
            "name": "inviteTimeoutSecs",
            "type": "u32"
          },
          {
            "name": "winners",
            "type": {
              "vec": "publicKey"
            }
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    }
  ],
  "types": [
    {
      "name": "CreateCompetitionArgs",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "invitees",
            "type": {
              "vec": "publicKey"
            }
          },
          {
            "name": "initiatorParticipates",
            "type": "bool"
          },
          {
            "name": "stakePerPerson",
            "type": "u64"
          },
          {
            "name": "payoutRule",
            "type": {
              "defined": "PayoutRule"
            }
          },
          {
            "name": "inviteTimeoutSecs",
            "type": {
              "option": "u32"
            }
          }
        ]
      }
    },
    {
      "name": "ParticipantResult",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "participant",
            "type": "publicKey"
          },
          {
            "name": "score",
            "type": "u32"
          }
        ]
      }
    },
    {
      "name": "CompetitionStatus",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "Pending"
          },
          {
            "name": "Active"
          },
          {
            "name": "Settled"
          },
          {
            "name": "Cancelled"
          }
        ]
      }
    },
    {
      "name": "PayoutRule",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "WinnerTakesAll"
          },
          {
            "name": "ThresholdSplit",
            "fields": [
              {
                "name": "minReps",
                "type": "u32"
              }
            ]
          }
        ]
      }
    }
  ],
  "errors": [
    {
      "code": 6000,
      "name": "InvalidStatus",
      "msg": "Competition is not in the expected status for this operation"
    },
    {
      "code": 6001,
      "name": "UnauthorizedCamera",
      "msg": "Only the authorized camera can settle this competition"
    },
    {
      "code": 6002,
      "name": "UnauthorizedInitiator",
      "msg": "Only the initiator can cancel this competition"
    },
    {
      "code": 6003,
      "name": "NotInvited",
      "msg": "Participant is not invited to this competition"
    },
    {
      "code": 6004,
      "name": "AlreadyJoined",
      "msg": "Participant has already joined this competition"
    },
    {
      "code": 6005,
      "name": "InsufficientFunds",
      "msg": "Insufficient funds to join competition"
    },
    {
      "code": 6006,
      "name": "MaxParticipantsReached",
      "msg": "Maximum number of participants reached"
    },
    {
      "code": 6007,
      "name": "NoParticipants",
      "msg": "No participants in the competition"
    },
    {
      "code": 6008,
      "name": "IncompleteResults",
      "msg": "Results do not include all participants"
    },
    {
      "code": 6009,
      "name": "InviteExpired",
      "msg": "Invite timeout has expired"
    },
    {
      "code": 6010,
      "name": "CannotCancel",
      "msg": "Cannot cancel - competition is already active or settled"
    },
    {
      "code": 6011,
      "name": "NoWinners",
      "msg": "No winners determined from results"
    },
    {
      "code": 6012,
      "name": "InvalidStakeAmount",
      "msg": "Stake amount must be greater than zero"
    },
    {
      "code": 6013,
      "name": "NoInvitees",
      "msg": "Must invite at least one participant"
    },
    {
      "code": 6014,
      "name": "ArithmeticOverflow",
      "msg": "Arithmetic overflow occurred"
    },
    {
      "code": 6015,
      "name": "NoFundsToDistribute",
      "msg": "Competition has no funds to distribute"
    },
    {
      "code": 6016,
      "name": "ParticipantNotInResults",
      "msg": "Participant not found in results"
    }
  ]
};

export const IDL: CompetitionEscrow = {
  "version": "0.1.0",
  "name": "competition_escrow",
  "instructions": [
    {
      "name": "cancelCompetition",
      "docs": [
        "Cancel the competition and refund all participants",
        "Only initiator can cancel"
      ],
      "accounts": [
        {
          "name": "initiator",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "reason",
          "type": "string"
        }
      ]
    },
    {
      "name": "createCompetition",
      "docs": [
        "Create a new competition with invited participants",
        "Initiator deposits their stake if participating"
      ],
      "accounts": [
        {
          "name": "initiator",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "camera",
          "isMut": false,
          "isSigner": false
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "args",
          "type": {
            "defined": "CreateCompetitionArgs"
          }
        },
        {
          "name": "createdAt",
          "type": "i64"
        }
      ]
    },
    {
      "name": "declineCompetition",
      "docs": [
        "Decline an invite (no funds involved)"
      ],
      "accounts": [
        {
          "name": "participant",
          "isMut": false,
          "isSigner": true
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        }
      ],
      "args": []
    },
    {
      "name": "joinCompetition",
      "docs": [
        "Accept an invite and deposit stake into escrow"
      ],
      "accounts": [
        {
          "name": "participant",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": []
    },
    {
      "name": "settleCompetition",
      "docs": [
        "Settle the competition with final results",
        "Only the authorized camera can call this"
      ],
      "accounts": [
        {
          "name": "camera",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "cameraOwner",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "results",
          "type": {
            "vec": {
              "defined": "ParticipantResult"
            }
          }
        }
      ]
    },
    {
      "name": "startCompetition",
      "docs": [
        "Start the competition after invites are resolved or timeout",
        "Can be called by initiator or camera"
      ],
      "accounts": [
        {
          "name": "authority",
          "isMut": false,
          "isSigner": true
        },
        {
          "name": "escrow",
          "isMut": true,
          "isSigner": false
        }
      ],
      "args": []
    }
  ],
  "accounts": [
    {
      "name": "competitionEscrow",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "initiator",
            "type": "publicKey"
          },
          {
            "name": "camera",
            "type": "publicKey"
          },
          {
            "name": "stakePerPerson",
            "type": "u64"
          },
          {
            "name": "participants",
            "type": {
              "vec": "publicKey"
            }
          },
          {
            "name": "pendingInvites",
            "type": {
              "vec": "publicKey"
            }
          },
          {
            "name": "totalPool",
            "type": "u64"
          },
          {
            "name": "status",
            "type": {
              "defined": "CompetitionStatus"
            }
          },
          {
            "name": "payoutRule",
            "type": {
              "defined": "PayoutRule"
            }
          },
          {
            "name": "createdAt",
            "type": "i64"
          },
          {
            "name": "inviteTimeoutSecs",
            "type": "u32"
          },
          {
            "name": "winners",
            "type": {
              "vec": "publicKey"
            }
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    }
  ],
  "types": [
    {
      "name": "CreateCompetitionArgs",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "invitees",
            "type": {
              "vec": "publicKey"
            }
          },
          {
            "name": "initiatorParticipates",
            "type": "bool"
          },
          {
            "name": "stakePerPerson",
            "type": "u64"
          },
          {
            "name": "payoutRule",
            "type": {
              "defined": "PayoutRule"
            }
          },
          {
            "name": "inviteTimeoutSecs",
            "type": {
              "option": "u32"
            }
          }
        ]
      }
    },
    {
      "name": "ParticipantResult",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "participant",
            "type": "publicKey"
          },
          {
            "name": "score",
            "type": "u32"
          }
        ]
      }
    },
    {
      "name": "CompetitionStatus",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "Pending"
          },
          {
            "name": "Active"
          },
          {
            "name": "Settled"
          },
          {
            "name": "Cancelled"
          }
        ]
      }
    },
    {
      "name": "PayoutRule",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "WinnerTakesAll"
          },
          {
            "name": "ThresholdSplit",
            "fields": [
              {
                "name": "minReps",
                "type": "u32"
              }
            ]
          }
        ]
      }
    }
  ],
  "errors": [
    {
      "code": 6000,
      "name": "InvalidStatus",
      "msg": "Competition is not in the expected status for this operation"
    },
    {
      "code": 6001,
      "name": "UnauthorizedCamera",
      "msg": "Only the authorized camera can settle this competition"
    },
    {
      "code": 6002,
      "name": "UnauthorizedInitiator",
      "msg": "Only the initiator can cancel this competition"
    },
    {
      "code": 6003,
      "name": "NotInvited",
      "msg": "Participant is not invited to this competition"
    },
    {
      "code": 6004,
      "name": "AlreadyJoined",
      "msg": "Participant has already joined this competition"
    },
    {
      "code": 6005,
      "name": "InsufficientFunds",
      "msg": "Insufficient funds to join competition"
    },
    {
      "code": 6006,
      "name": "MaxParticipantsReached",
      "msg": "Maximum number of participants reached"
    },
    {
      "code": 6007,
      "name": "NoParticipants",
      "msg": "No participants in the competition"
    },
    {
      "code": 6008,
      "name": "IncompleteResults",
      "msg": "Results do not include all participants"
    },
    {
      "code": 6009,
      "name": "InviteExpired",
      "msg": "Invite timeout has expired"
    },
    {
      "code": 6010,
      "name": "CannotCancel",
      "msg": "Cannot cancel - competition is already active or settled"
    },
    {
      "code": 6011,
      "name": "NoWinners",
      "msg": "No winners determined from results"
    },
    {
      "code": 6012,
      "name": "InvalidStakeAmount",
      "msg": "Stake amount must be greater than zero"
    },
    {
      "code": 6013,
      "name": "NoInvitees",
      "msg": "Must invite at least one participant"
    },
    {
      "code": 6014,
      "name": "ArithmeticOverflow",
      "msg": "Arithmetic overflow occurred"
    },
    {
      "code": 6015,
      "name": "NoFundsToDistribute",
      "msg": "Competition has no funds to distribute"
    },
    {
      "code": 6016,
      "name": "ParticipantNotInResults",
      "msg": "Participant not found in results"
    }
  ]
};

// Export as COMPETITION_ESCROW_IDL for backwards compatibility
export const COMPETITION_ESCROW_IDL = IDL;
