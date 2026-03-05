import IDL_JSON from "./idl.json";

export const IDL = IDL_JSON;

/**
 * Program IDL in camelCase format in order to be used in JS/TS.
 *
 * Note that this is only a type helper and is not the actual IDL. The original
 * IDL can be found at `target/idl/camera_network.json`.
 */
export type CameraNetwork = {
  "address": "E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL",
  "metadata": {
    "name": "cameraNetwork",
    "version": "0.1.0",
    "spec": "0.1.0",
    "description": "Solana program for camera device network with check-in/check-out functionality"
  },
  "instructions": [
    {
      "name": "createTimelineEntry",
      "docs": [
        "Create a compressed timeline entry for a camera session",
        "Called at checkout — device signs, backend pays gas"
      ],
      "discriminator": [
        243,
        124,
        212,
        170,
        45,
        118,
        145,
        86
      ],
      "accounts": [
        {
          "name": "payer",
          "docs": [
            "Fee payer — backend pays gas"
          ],
          "writable": true,
          "signer": true
        },
        {
          "name": "device",
          "docs": [
            "Device authenticator — must be camera's device key or owner"
          ],
          "signer": true
        },
        {
          "name": "camera",
          "docs": [
            "Camera account — verified against device signer"
          ],
          "writable": true
        }
      ],
      "args": [
        {
          "name": "proof",
          "type": {
            "defined": {
              "name": "validityProof"
            }
          }
        },
        {
          "name": "addressTreeInfo",
          "type": {
            "defined": {
              "name": "packedAddressTreeInfo"
            }
          }
        },
        {
          "name": "outputMerkleTreeIndex",
          "type": "u8"
        },
        {
          "name": "encryptedPayload",
          "type": "bytes"
        },
        {
          "name": "nonce",
          "type": {
            "array": [
              "u8",
              12
            ]
          }
        },
        {
          "name": "accessGrantsBlob",
          "type": "bytes"
        },
        {
          "name": "activityCount",
          "type": "u8"
        }
      ]
    },
    {
      "name": "createUserSessionChain",
      "docs": [
        "Create a user's session chain for storing encrypted access keys",
        "This is the user's \"keychain\" for accessing their session history"
      ],
      "discriminator": [
        81,
        239,
        227,
        141,
        244,
        52,
        171,
        164
      ],
      "accounts": [
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "authority",
          "docs": [
            "The mmoment authority (cron bot) that can also write to this chain"
          ]
        },
        {
          "name": "userSessionChain",
          "writable": true
        },
        {
          "name": "systemProgram"
        }
      ],
      "args": []
    },
    {
      "name": "deleteRecognitionToken",
      "docs": [
        "Delete recognition token and reclaim rent"
      ],
      "discriminator": [
        169,
        195,
        239,
        113,
        254,
        207,
        130,
        37
      ],
      "accounts": [
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "recognitionToken",
          "writable": true
        }
      ],
      "args": []
    },
    {
      "name": "deregisterCamera",
      "docs": [
        "Deregister a camera from the network"
      ],
      "discriminator": [
        191,
        154,
        78,
        182,
        249,
        72,
        207,
        113
      ],
      "accounts": [
        {
          "name": "owner",
          "writable": true,
          "signer": true
        },
        {
          "name": "cameraRegistry",
          "writable": true
        },
        {
          "name": "camera",
          "writable": true
        }
      ],
      "args": []
    },
    {
      "name": "initialize",
      "docs": [
        "Initialize the camera registry (admin only)"
      ],
      "discriminator": [
        175,
        175,
        109,
        31,
        13,
        152,
        155,
        237
      ],
      "accounts": [
        {
          "name": "authority",
          "writable": true,
          "signer": true
        },
        {
          "name": "cameraRegistry",
          "writable": true
        },
        {
          "name": "systemProgram"
        }
      ],
      "args": []
    },
    {
      "name": "registerCamera",
      "docs": [
        "Register a new camera to the network"
      ],
      "discriminator": [
        169,
        161,
        144,
        207,
        102,
        130,
        86,
        62
      ],
      "accounts": [
        {
          "name": "owner",
          "writable": true,
          "signer": true
        },
        {
          "name": "cameraRegistry",
          "writable": true
        },
        {
          "name": "camera",
          "writable": true
        },
        {
          "name": "systemProgram"
        }
      ],
      "args": [
        {
          "name": "args",
          "type": {
            "defined": {
              "name": "registerCameraArgs"
            }
          }
        }
      ]
    },
    {
      "name": "setCameraActive",
      "docs": [
        "Set camera active or inactive status"
      ],
      "discriminator": [
        157,
        63,
        1,
        132,
        85,
        176,
        156,
        74
      ],
      "accounts": [
        {
          "name": "owner",
          "writable": true,
          "signer": true
        },
        {
          "name": "camera",
          "writable": true
        }
      ],
      "args": [
        {
          "name": "isActive",
          "type": "bool"
        }
      ]
    },
    {
      "name": "storeSessionAccessKeys",
      "docs": [
        "Store encrypted session access keys in a user's chain",
        "Can be called by the user OR the mmoment authority (cron bot fallback)"
      ],
      "discriminator": [
        15,
        128,
        124,
        30,
        182,
        222,
        223,
        68
      ],
      "accounts": [
        {
          "name": "signer",
          "docs": [
            "The signer - must be either the user or the authority"
          ],
          "writable": true,
          "signer": true
        },
        {
          "name": "user",
          "docs": [
            "The user whose session chain is being updated"
          ]
        },
        {
          "name": "userSessionChain",
          "writable": true
        },
        {
          "name": "systemProgram"
        }
      ],
      "args": [
        {
          "name": "keys",
          "type": {
            "vec": {
              "defined": {
                "name": "encryptedSessionKey"
              }
            }
          }
        }
      ]
    },
    {
      "name": "updateCamera",
      "docs": [
        "Update camera information"
      ],
      "discriminator": [
        240,
        86,
        145,
        90,
        6,
        185,
        101,
        7
      ],
      "accounts": [
        {
          "name": "owner",
          "writable": true,
          "signer": true
        },
        {
          "name": "camera",
          "writable": true
        }
      ],
      "args": [
        {
          "name": "args",
          "type": {
            "defined": {
              "name": "updateCameraArgs"
            }
          }
        }
      ]
    },
    {
      "name": "upsertRecognitionToken",
      "docs": [
        "Create or regenerate a recognition token (stores encrypted facial embedding)"
      ],
      "discriminator": [
        217,
        76,
        1,
        248,
        153,
        88,
        20,
        188
      ],
      "accounts": [
        {
          "name": "user",
          "writable": true,
          "signer": true
        },
        {
          "name": "recognitionToken",
          "writable": true
        },
        {
          "name": "systemProgram"
        }
      ],
      "args": [
        {
          "name": "encryptedEmbedding",
          "type": "bytes"
        },
        {
          "name": "displayName",
          "type": {
            "option": "string"
          }
        },
        {
          "name": "source",
          "type": "u8"
        }
      ]
    }
  ],
  "accounts": [
    {
      "name": "cameraAccount",
      "discriminator": [
        53,
        116,
        8,
        174,
        44,
        148,
        166,
        34
      ]
    },
    {
      "name": "cameraRegistry",
      "discriminator": [
        242,
        46,
        247,
        175,
        239,
        58,
        125,
        173
      ]
    },
    {
      "name": "recognitionToken",
      "discriminator": [
        185,
        14,
        93,
        76,
        58,
        215,
        68,
        167
      ]
    },
    {
      "name": "userSessionChain",
      "discriminator": [
        94,
        45,
        77,
        120,
        236,
        160,
        216,
        102
      ]
    }
  ],
  "events": [
    {
      "name": "cameraRegistered",
      "discriminator": [
        32,
        240,
        203,
        222,
        43,
        16,
        54,
        81
      ]
    },
    {
      "name": "recognitionTokenCreated",
      "discriminator": [
        24,
        33,
        66,
        21,
        97,
        67,
        214,
        111
      ]
    },
    {
      "name": "recognitionTokenDeleted",
      "discriminator": [
        134,
        32,
        116,
        193,
        89,
        244,
        183,
        153
      ]
    },
    {
      "name": "timelineEntry",
      "discriminator": [
        66,
        96,
        15,
        175,
        192,
        119,
        138,
        5
      ]
    },
    {
      "name": "timelineEntryCreated",
      "discriminator": [
        36,
        131,
        182,
        89,
        193,
        86,
        126,
        30
      ]
    }
  ],
  "errors": [
    {
      "code": 6000,
      "name": "unauthorized",
      "msg": "You are not authorized to perform this action"
    },
    {
      "code": 6001,
      "name": "cameraInactive",
      "msg": "Camera is currently inactive"
    },
    {
      "code": 6002,
      "name": "cameraNotFound",
      "msg": "Camera not found in registry"
    },
    {
      "code": 6003,
      "name": "cameraNameExists",
      "msg": "Camera name already exists"
    },
    {
      "code": 6004,
      "name": "invalidCameraData",
      "msg": "Invalid camera data provided"
    },
    {
      "code": 6005,
      "name": "noActiveSession",
      "msg": "No active session found"
    },
    {
      "code": 6006,
      "name": "sessionExists",
      "msg": "Session already exists"
    },
    {
      "code": 6007,
      "name": "accessDenied",
      "msg": "Access denied to this camera"
    },
    {
      "code": 6008,
      "name": "invalidFaceData",
      "msg": "Face data invalid or not properly formatted"
    },
    {
      "code": 6009,
      "name": "faceDataExists",
      "msg": "Face data already registered for this user"
    },
    {
      "code": 6010,
      "name": "invalidGestureData",
      "msg": "Gesture data invalid or improperly formatted"
    },
    {
      "code": 6011,
      "name": "cameraAlreadyAuthorized",
      "msg": "Camera is already authorized for face recognition"
    },
    {
      "code": 6012,
      "name": "invalidAccessDuration",
      "msg": "Invalid temporary access duration"
    },
    {
      "code": 6013,
      "name": "accessGrantExpired",
      "msg": "Access grant has expired"
    },
    {
      "code": 6014,
      "name": "sessionExpired",
      "msg": "Session has expired"
    },
    {
      "code": 6015,
      "name": "noRecognitionToken",
      "msg": "No recognition token found - please create one first"
    },
    {
      "code": 6016,
      "name": "featureNotAvailable",
      "msg": "Feature not available on this camera"
    },
    {
      "code": 6017,
      "name": "recognitionTokenTooLarge",
      "msg": "Recognition token data too large (max 1024 bytes)"
    }
  ],
  "types": [
    {
      "name": "cameraAccount",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "owner",
            "type": "pubkey"
          },
          {
            "name": "metadata",
            "type": {
              "defined": {
                "name": "cameraMetadata"
              }
            }
          },
          {
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "activityCounter",
            "type": "u64"
          },
          {
            "name": "lastActivityAt",
            "type": "i64"
          },
          {
            "name": "lastActivityType",
            "type": "u8"
          },
          {
            "name": "accessCount",
            "type": "u64"
          },
          {
            "name": "features",
            "type": {
              "defined": {
                "name": "cameraFeatures"
              }
            }
          },
          {
            "name": "bump",
            "type": "u8"
          },
          {
            "name": "devicePubkey",
            "type": {
              "option": "pubkey"
            }
          }
        ]
      }
    },
    {
      "name": "cameraFeatures",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "faceRecognition",
            "type": "bool"
          },
          {
            "name": "gestureControl",
            "type": "bool"
          },
          {
            "name": "videoRecording",
            "type": "bool"
          },
          {
            "name": "liveStreaming",
            "type": "bool"
          },
          {
            "name": "messaging",
            "type": "bool"
          }
        ]
      }
    },
    {
      "name": "cameraMetadata",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "name",
            "type": "string"
          },
          {
            "name": "model",
            "type": "string"
          },
          {
            "name": "location",
            "type": {
              "option": {
                "array": [
                  "i64",
                  2
                ]
              }
            }
          },
          {
            "name": "registrationDate",
            "type": "i64"
          },
          {
            "name": "description",
            "type": "string"
          }
        ]
      }
    },
    {
      "name": "cameraRegistered",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "camera",
            "type": "pubkey"
          },
          {
            "name": "owner",
            "type": "pubkey"
          },
          {
            "name": "name",
            "type": "string"
          },
          {
            "name": "model",
            "type": "string"
          },
          {
            "name": "timestamp",
            "type": "i64"
          }
        ]
      }
    },
    {
      "name": "cameraRegistry",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "cameraCount",
            "type": "u64"
          },
          {
            "name": "feeAccount",
            "type": "pubkey"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "compressedProof",
      "repr": {
        "kind": "c"
      },
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "a",
            "type": {
              "array": [
                "u8",
                32
              ]
            }
          },
          {
            "name": "b",
            "type": {
              "array": [
                "u8",
                64
              ]
            }
          },
          {
            "name": "c",
            "type": {
              "array": [
                "u8",
                32
              ]
            }
          }
        ]
      }
    },
    {
      "name": "encryptedSessionKey",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "keyCiphertext",
            "type": "bytes"
          },
          {
            "name": "nonce",
            "type": {
              "array": [
                "u8",
                12
              ]
            }
          },
          {
            "name": "timestamp",
            "type": "i64"
          }
        ]
      }
    },
    {
      "name": "packedAddressTreeInfo",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "addressMerkleTreePubkeyIndex",
            "type": "u8"
          },
          {
            "name": "addressQueuePubkeyIndex",
            "type": "u8"
          },
          {
            "name": "rootIndex",
            "type": "u16"
          }
        ]
      }
    },
    {
      "name": "recognitionToken",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "pubkey"
          },
          {
            "name": "encryptedEmbedding",
            "type": "bytes"
          },
          {
            "name": "createdAt",
            "type": "i64"
          },
          {
            "name": "version",
            "type": "u8"
          },
          {
            "name": "bump",
            "type": "u8"
          },
          {
            "name": "displayName",
            "type": {
              "option": "string"
            }
          },
          {
            "name": "source",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "recognitionTokenCreated",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "pubkey"
          },
          {
            "name": "token",
            "type": "pubkey"
          },
          {
            "name": "version",
            "type": "u8"
          },
          {
            "name": "source",
            "type": "u8"
          },
          {
            "name": "displayName",
            "type": {
              "option": "string"
            }
          },
          {
            "name": "timestamp",
            "type": "i64"
          }
        ]
      }
    },
    {
      "name": "recognitionTokenDeleted",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "pubkey"
          },
          {
            "name": "token",
            "type": "pubkey"
          },
          {
            "name": "timestamp",
            "type": "i64"
          }
        ]
      }
    },
    {
      "name": "registerCameraArgs",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "name",
            "type": "string"
          },
          {
            "name": "model",
            "type": "string"
          },
          {
            "name": "location",
            "type": {
              "option": {
                "array": [
                  "i64",
                  2
                ]
              }
            }
          },
          {
            "name": "description",
            "type": "string"
          },
          {
            "name": "features",
            "type": {
              "defined": {
                "name": "cameraFeatures"
              }
            }
          },
          {
            "name": "devicePubkey",
            "type": {
              "option": "pubkey"
            }
          }
        ]
      }
    },
    {
      "name": "timelineEntry",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "camera",
            "docs": [
              "Which camera this entry belongs to (for querying all entries by camera)"
            ],
            "type": "pubkey"
          },
          {
            "name": "entryIndex",
            "docs": [
              "Sequential index (camera.activity_counter at time of write)"
            ],
            "type": "u64"
          },
          {
            "name": "timestamp",
            "docs": [
              "When this checkout occurred"
            ],
            "type": "i64"
          },
          {
            "name": "activityCount",
            "docs": [
              "Number of activities in this entry"
            ],
            "type": "u8"
          },
          {
            "name": "encryptedPayload",
            "docs": [
              "All activities from this session, serialized and encrypted together.",
              "Format: AES-256-GCM encrypted JSON array of activities."
            ],
            "type": "bytes"
          },
          {
            "name": "nonce",
            "docs": [
              "AES-GCM nonce for decrypting encrypted_payload"
            ],
            "type": {
              "array": [
                "u8",
                12
              ]
            }
          },
          {
            "name": "accessGrantsBlob",
            "docs": [
              "Flattened access grants blob.",
              "Format: [num_grants (2 bytes BE)] + for each grant: [grant_len (2 bytes BE)] + [grant_bytes]"
            ],
            "type": "bytes"
          }
        ]
      }
    },
    {
      "name": "timelineEntryCreated",
      "docs": [
        "Event emitted when a timeline entry is created (no user info)"
      ],
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "camera",
            "type": "pubkey"
          },
          {
            "name": "entryIndex",
            "type": "u64"
          },
          {
            "name": "activityCount",
            "type": "u8"
          },
          {
            "name": "timestamp",
            "type": "i64"
          }
        ]
      }
    },
    {
      "name": "updateCameraArgs",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "name",
            "type": {
              "option": "string"
            }
          },
          {
            "name": "location",
            "type": {
              "option": {
                "array": [
                  "i64",
                  2
                ]
              }
            }
          },
          {
            "name": "description",
            "type": {
              "option": "string"
            }
          },
          {
            "name": "features",
            "type": {
              "option": {
                "defined": {
                  "name": "cameraFeatures"
                }
              }
            }
          },
          {
            "name": "devicePubkey",
            "type": {
              "option": "pubkey"
            }
          }
        ]
      }
    },
    {
      "name": "userSessionChain",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "pubkey"
          },
          {
            "name": "authority",
            "type": "pubkey"
          },
          {
            "name": "encryptedKeys",
            "type": {
              "vec": {
                "defined": {
                  "name": "encryptedSessionKey"
                }
              }
            }
          },
          {
            "name": "sessionCount",
            "type": "u64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "validityProof",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "option": {
              "defined": {
                "name": "compressedProof"
              }
            }
          }
        ]
      }
    }
  ]
};
