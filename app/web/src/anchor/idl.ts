export type CameraNetwork = {
  "version": "0.1.0",
  "name": "camera_network",
  "instructions": [
    {
      "name": "initialize",
      "docs": [
        "Initialize the camera registry (admin only)"
      ],
      "accounts": [
        {
          "name": "authority",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "cameraRegistry",
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
      "name": "registerCamera",
      "docs": [
        "Register a new camera to the network"
      ],
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "cameraRegistry",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "camera"
              },
              {
                "kind": "arg",
                "type": {
                  "defined": "RegisterCameraArgs"
                },
                "path": "args.name"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "owner"
              }
            ]
          }
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
            "defined": "RegisterCameraArgs"
          }
        }
      ]
    },
    {
      "name": "updateCamera",
      "docs": [
        "Update camera information"
      ],
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "args",
          "type": {
            "defined": "UpdateCameraArgs"
          }
        }
      ]
    },
    {
      "name": "deregisterCamera",
      "docs": [
        "Deregister a camera from the network"
      ],
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "cameraRegistry",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
        }
      ],
      "args": []
    },
    {
      "name": "setCameraActive",
      "docs": [
        "Set camera active or inactive status"
      ],
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
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
      "name": "upsertRecognitionToken",
      "docs": [
        "Create or regenerate a recognition token (stores encrypted facial embedding)"
      ],
      "accounts": [
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "recognitionToken",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "recognition-token"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "user"
              }
            ]
          }
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
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
    },
    {
      "name": "deleteRecognitionToken",
      "docs": [
        "Delete recognition token and reclaim rent"
      ],
      "accounts": [
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "recognitionToken",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "recognition-token"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "user"
              }
            ]
          }
        }
      ],
      "args": []
    },
    {
      "name": "createUserSessionChain",
      "docs": [
        "Create a user's session chain for storing encrypted access keys",
        "This is the user's \"keychain\" for accessing their session history"
      ],
      "accounts": [
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "authority",
          "isMut": false,
          "isSigner": false,
          "docs": [
            "The mmoment authority (cron bot) that can also write to this chain"
          ]
        },
        {
          "name": "userSessionChain",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "user-session-chain"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "user"
              }
            ]
          }
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
      "name": "storeSessionAccessKeys",
      "docs": [
        "Store encrypted session access keys in a user's chain",
        "Can be called by the user OR the mmoment authority (cron bot fallback)"
      ],
      "accounts": [
        {
          "name": "signer",
          "isMut": true,
          "isSigner": true,
          "docs": [
            "The signer - must be either the user or the authority"
          ]
        },
        {
          "name": "user",
          "isMut": false,
          "isSigner": false,
          "docs": [
            "The user whose session chain is being updated"
          ]
        },
        {
          "name": "userSessionChain",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "user-session-chain"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "user"
              }
            ]
          }
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "keys",
          "type": {
            "vec": {
              "defined": "EncryptedSessionKey"
            }
          }
        }
      ]
    },
    {
      "name": "writeToCameraTimeline",
      "docs": [
        "Write encrypted activities to a camera's timeline",
        "Called by Jetson (device key) or camera owner - NO user account involved"
      ],
      "accounts": [
        {
          "name": "signer",
          "isMut": true,
          "isSigner": true,
          "docs": [
            "The signer - must be the camera's device key or owner"
          ]
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "cameraTimeline",
          "isMut": true,
          "isSigner": false,
          "docs": [
            "Camera timeline - created lazily on first write"
          ],
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "camera-timeline"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "account": "CameraAccount",
                "path": "camera"
              }
            ]
          }
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "activities",
          "type": {
            "vec": {
              "defined": "ActivityData"
            }
          }
        }
      ]
    }
  ],
  "accounts": [
    {
      "name": "CameraRegistry",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "authority",
            "type": "publicKey"
          },
          {
            "name": "cameraCount",
            "type": "u64"
          },
          {
            "name": "feeAccount",
            "type": "publicKey"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "CameraAccount",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "owner",
            "type": "publicKey"
          },
          {
            "name": "metadata",
            "type": {
              "defined": "CameraMetadata"
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
              "defined": "CameraFeatures"
            }
          },
          {
            "name": "bump",
            "type": "u8"
          },
          {
            "name": "devicePubkey",
            "type": {
              "option": "publicKey"
            }
          }
        ]
      }
    },
    {
      "name": "RecognitionToken",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "publicKey"
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
      "name": "GestureConfig",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "publicKey"
          },
          {
            "name": "gestureType",
            "type": "u8"
          },
          {
            "name": "dataHash",
            "type": {
              "array": [
                "u8",
                32
              ]
            }
          },
          {
            "name": "createdAt",
            "type": "i64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "CameraMessage",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "publicKey"
          },
          {
            "name": "camera",
            "type": "publicKey"
          },
          {
            "name": "message",
            "type": "string"
          },
          {
            "name": "timestamp",
            "type": "i64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "AccessGrant",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "camera",
            "type": "publicKey"
          },
          {
            "name": "grantor",
            "type": "publicKey"
          },
          {
            "name": "grantee",
            "type": "publicKey"
          },
          {
            "name": "expiresAt",
            "type": "i64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "CameraTimeline",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "camera",
            "type": "publicKey"
          },
          {
            "name": "encryptedActivities",
            "type": {
              "vec": {
                "defined": "EncryptedActivity"
              }
            }
          },
          {
            "name": "activityCount",
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
      "name": "UserSessionChain",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "publicKey"
          },
          {
            "name": "authority",
            "type": "publicKey"
          },
          {
            "name": "encryptedKeys",
            "type": {
              "vec": {
                "defined": "EncryptedSessionKey"
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
    }
  ],
  "types": [
    {
      "name": "CameraMetadata",
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
      "name": "CameraFeatures",
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
      "name": "EncryptedActivity",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "timestamp",
            "type": "i64"
          },
          {
            "name": "activityType",
            "type": "u8"
          },
          {
            "name": "encryptedContent",
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
            "name": "accessGrants",
            "type": {
              "vec": "bytes"
            }
          }
        ]
      }
    },
    {
      "name": "EncryptedSessionKey",
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
      "name": "RegisterCameraArgs",
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
              "defined": "CameraFeatures"
            }
          },
          {
            "name": "devicePubkey",
            "type": {
              "option": "publicKey"
            }
          }
        ]
      }
    },
    {
      "name": "UpdateCameraArgs",
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
                "defined": "CameraFeatures"
              }
            }
          }
        ]
      }
    },
    {
      "name": "ActivityData",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "timestamp",
            "type": "i64"
          },
          {
            "name": "activityType",
            "type": "u8"
          },
          {
            "name": "encryptedContent",
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
            "name": "accessGrants",
            "type": {
              "vec": "bytes"
            }
          }
        ]
      }
    },
    {
      "name": "ActivityType",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "CheckIn"
          },
          {
            "name": "CheckOut"
          },
          {
            "name": "PhotoCapture"
          },
          {
            "name": "VideoRecord"
          },
          {
            "name": "LiveStream"
          },
          {
            "name": "FaceRecognition"
          },
          {
            "name": "CVAppActivity"
          },
          {
            "name": "Other"
          }
        ]
      }
    }
  ],
  "events": [
    {
      "name": "TimelineUpdated",
      "fields": [
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "activitiesAdded",
          "type": "u64",
          "index": false
        },
        {
          "name": "totalActivities",
          "type": "u64",
          "index": false
        },
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        }
      ]
    },
    {
      "name": "CameraRegistered",
      "fields": [
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "owner",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "name",
          "type": "string",
          "index": false
        },
        {
          "name": "model",
          "type": "string",
          "index": false
        },
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        }
      ]
    },
    {
      "name": "RecognitionTokenCreated",
      "fields": [
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "token",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "version",
          "type": "u8",
          "index": false
        },
        {
          "name": "source",
          "type": "u8",
          "index": false
        },
        {
          "name": "displayName",
          "type": {
            "option": "string"
          },
          "index": false
        },
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        }
      ]
    },
    {
      "name": "RecognitionTokenDeleted",
      "fields": [
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "token",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        }
      ]
    }
  ],
  "errors": [
    {
      "code": 6000,
      "name": "Unauthorized",
      "msg": "You are not authorized to perform this action"
    },
    {
      "code": 6001,
      "name": "CameraInactive",
      "msg": "Camera is currently inactive"
    },
    {
      "code": 6002,
      "name": "CameraNotFound",
      "msg": "Camera not found in registry"
    },
    {
      "code": 6003,
      "name": "CameraNameExists",
      "msg": "Camera name already exists"
    },
    {
      "code": 6004,
      "name": "InvalidCameraData",
      "msg": "Invalid camera data provided"
    },
    {
      "code": 6005,
      "name": "NoActiveSession",
      "msg": "No active session found"
    },
    {
      "code": 6006,
      "name": "SessionExists",
      "msg": "Session already exists"
    },
    {
      "code": 6007,
      "name": "AccessDenied",
      "msg": "Access denied to this camera"
    },
    {
      "code": 6008,
      "name": "InvalidFaceData",
      "msg": "Face data invalid or not properly formatted"
    },
    {
      "code": 6009,
      "name": "FaceDataExists",
      "msg": "Face data already registered for this user"
    },
    {
      "code": 6010,
      "name": "InvalidGestureData",
      "msg": "Gesture data invalid or improperly formatted"
    },
    {
      "code": 6011,
      "name": "CameraAlreadyAuthorized",
      "msg": "Camera is already authorized for face recognition"
    },
    {
      "code": 6012,
      "name": "InvalidAccessDuration",
      "msg": "Invalid temporary access duration"
    },
    {
      "code": 6013,
      "name": "AccessGrantExpired",
      "msg": "Access grant has expired"
    },
    {
      "code": 6014,
      "name": "SessionExpired",
      "msg": "Session has expired"
    },
    {
      "code": 6015,
      "name": "NoRecognitionToken",
      "msg": "No recognition token found - please create one first"
    },
    {
      "code": 6016,
      "name": "FeatureNotAvailable",
      "msg": "Feature not available on this camera"
    },
    {
      "code": 6017,
      "name": "RecognitionTokenTooLarge",
      "msg": "Recognition token data too large (max 1024 bytes)"
    }
  ]
};

export const IDL: CameraNetwork = {
  "version": "0.1.0",
  "name": "camera_network",
  "instructions": [
    {
      "name": "initialize",
      "docs": [
        "Initialize the camera registry (admin only)"
      ],
      "accounts": [
        {
          "name": "authority",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "cameraRegistry",
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
      "name": "registerCamera",
      "docs": [
        "Register a new camera to the network"
      ],
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "cameraRegistry",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "camera"
              },
              {
                "kind": "arg",
                "type": {
                  "defined": "RegisterCameraArgs"
                },
                "path": "args.name"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "owner"
              }
            ]
          }
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
            "defined": "RegisterCameraArgs"
          }
        }
      ]
    },
    {
      "name": "updateCamera",
      "docs": [
        "Update camera information"
      ],
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "args",
          "type": {
            "defined": "UpdateCameraArgs"
          }
        }
      ]
    },
    {
      "name": "deregisterCamera",
      "docs": [
        "Deregister a camera from the network"
      ],
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "cameraRegistry",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
        }
      ],
      "args": []
    },
    {
      "name": "setCameraActive",
      "docs": [
        "Set camera active or inactive status"
      ],
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
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
      "name": "upsertRecognitionToken",
      "docs": [
        "Create or regenerate a recognition token (stores encrypted facial embedding)"
      ],
      "accounts": [
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "recognitionToken",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "recognition-token"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "user"
              }
            ]
          }
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
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
    },
    {
      "name": "deleteRecognitionToken",
      "docs": [
        "Delete recognition token and reclaim rent"
      ],
      "accounts": [
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "recognitionToken",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "recognition-token"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "user"
              }
            ]
          }
        }
      ],
      "args": []
    },
    {
      "name": "createUserSessionChain",
      "docs": [
        "Create a user's session chain for storing encrypted access keys",
        "This is the user's \"keychain\" for accessing their session history"
      ],
      "accounts": [
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "authority",
          "isMut": false,
          "isSigner": false,
          "docs": [
            "The mmoment authority (cron bot) that can also write to this chain"
          ]
        },
        {
          "name": "userSessionChain",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "user-session-chain"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "user"
              }
            ]
          }
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
      "name": "storeSessionAccessKeys",
      "docs": [
        "Store encrypted session access keys in a user's chain",
        "Can be called by the user OR the mmoment authority (cron bot fallback)"
      ],
      "accounts": [
        {
          "name": "signer",
          "isMut": true,
          "isSigner": true,
          "docs": [
            "The signer - must be either the user or the authority"
          ]
        },
        {
          "name": "user",
          "isMut": false,
          "isSigner": false,
          "docs": [
            "The user whose session chain is being updated"
          ]
        },
        {
          "name": "userSessionChain",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "user-session-chain"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "user"
              }
            ]
          }
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "keys",
          "type": {
            "vec": {
              "defined": "EncryptedSessionKey"
            }
          }
        }
      ]
    },
    {
      "name": "writeToCameraTimeline",
      "docs": [
        "Write encrypted activities to a camera's timeline",
        "Called by Jetson (device key) or camera owner - NO user account involved"
      ],
      "accounts": [
        {
          "name": "signer",
          "isMut": true,
          "isSigner": true,
          "docs": [
            "The signer - must be the camera's device key or owner"
          ]
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "cameraTimeline",
          "isMut": true,
          "isSigner": false,
          "docs": [
            "Camera timeline - created lazily on first write"
          ],
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "camera-timeline"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "account": "CameraAccount",
                "path": "camera"
              }
            ]
          }
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "activities",
          "type": {
            "vec": {
              "defined": "ActivityData"
            }
          }
        }
      ]
    }
  ],
  "accounts": [
    {
      "name": "CameraRegistry",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "authority",
            "type": "publicKey"
          },
          {
            "name": "cameraCount",
            "type": "u64"
          },
          {
            "name": "feeAccount",
            "type": "publicKey"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "CameraAccount",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "owner",
            "type": "publicKey"
          },
          {
            "name": "metadata",
            "type": {
              "defined": "CameraMetadata"
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
              "defined": "CameraFeatures"
            }
          },
          {
            "name": "bump",
            "type": "u8"
          },
          {
            "name": "devicePubkey",
            "type": {
              "option": "publicKey"
            }
          }
        ]
      }
    },
    {
      "name": "RecognitionToken",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "publicKey"
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
      "name": "GestureConfig",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "publicKey"
          },
          {
            "name": "gestureType",
            "type": "u8"
          },
          {
            "name": "dataHash",
            "type": {
              "array": [
                "u8",
                32
              ]
            }
          },
          {
            "name": "createdAt",
            "type": "i64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "CameraMessage",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "publicKey"
          },
          {
            "name": "camera",
            "type": "publicKey"
          },
          {
            "name": "message",
            "type": "string"
          },
          {
            "name": "timestamp",
            "type": "i64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "AccessGrant",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "camera",
            "type": "publicKey"
          },
          {
            "name": "grantor",
            "type": "publicKey"
          },
          {
            "name": "grantee",
            "type": "publicKey"
          },
          {
            "name": "expiresAt",
            "type": "i64"
          },
          {
            "name": "bump",
            "type": "u8"
          }
        ]
      }
    },
    {
      "name": "CameraTimeline",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "camera",
            "type": "publicKey"
          },
          {
            "name": "encryptedActivities",
            "type": {
              "vec": {
                "defined": "EncryptedActivity"
              }
            }
          },
          {
            "name": "activityCount",
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
      "name": "UserSessionChain",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "user",
            "type": "publicKey"
          },
          {
            "name": "authority",
            "type": "publicKey"
          },
          {
            "name": "encryptedKeys",
            "type": {
              "vec": {
                "defined": "EncryptedSessionKey"
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
    }
  ],
  "types": [
    {
      "name": "CameraMetadata",
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
      "name": "CameraFeatures",
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
      "name": "EncryptedActivity",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "timestamp",
            "type": "i64"
          },
          {
            "name": "activityType",
            "type": "u8"
          },
          {
            "name": "encryptedContent",
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
            "name": "accessGrants",
            "type": {
              "vec": "bytes"
            }
          }
        ]
      }
    },
    {
      "name": "EncryptedSessionKey",
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
      "name": "RegisterCameraArgs",
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
              "defined": "CameraFeatures"
            }
          },
          {
            "name": "devicePubkey",
            "type": {
              "option": "publicKey"
            }
          }
        ]
      }
    },
    {
      "name": "UpdateCameraArgs",
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
                "defined": "CameraFeatures"
              }
            }
          }
        ]
      }
    },
    {
      "name": "ActivityData",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "timestamp",
            "type": "i64"
          },
          {
            "name": "activityType",
            "type": "u8"
          },
          {
            "name": "encryptedContent",
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
            "name": "accessGrants",
            "type": {
              "vec": "bytes"
            }
          }
        ]
      }
    },
    {
      "name": "ActivityType",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "CheckIn"
          },
          {
            "name": "CheckOut"
          },
          {
            "name": "PhotoCapture"
          },
          {
            "name": "VideoRecord"
          },
          {
            "name": "LiveStream"
          },
          {
            "name": "FaceRecognition"
          },
          {
            "name": "CVAppActivity"
          },
          {
            "name": "Other"
          }
        ]
      }
    }
  ],
  "events": [
    {
      "name": "TimelineUpdated",
      "fields": [
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "activitiesAdded",
          "type": "u64",
          "index": false
        },
        {
          "name": "totalActivities",
          "type": "u64",
          "index": false
        },
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        }
      ]
    },
    {
      "name": "CameraRegistered",
      "fields": [
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "owner",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "name",
          "type": "string",
          "index": false
        },
        {
          "name": "model",
          "type": "string",
          "index": false
        },
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        }
      ]
    },
    {
      "name": "RecognitionTokenCreated",
      "fields": [
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "token",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "version",
          "type": "u8",
          "index": false
        },
        {
          "name": "source",
          "type": "u8",
          "index": false
        },
        {
          "name": "displayName",
          "type": {
            "option": "string"
          },
          "index": false
        },
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        }
      ]
    },
    {
      "name": "RecognitionTokenDeleted",
      "fields": [
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "token",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        }
      ]
    }
  ],
  "errors": [
    {
      "code": 6000,
      "name": "Unauthorized",
      "msg": "You are not authorized to perform this action"
    },
    {
      "code": 6001,
      "name": "CameraInactive",
      "msg": "Camera is currently inactive"
    },
    {
      "code": 6002,
      "name": "CameraNotFound",
      "msg": "Camera not found in registry"
    },
    {
      "code": 6003,
      "name": "CameraNameExists",
      "msg": "Camera name already exists"
    },
    {
      "code": 6004,
      "name": "InvalidCameraData",
      "msg": "Invalid camera data provided"
    },
    {
      "code": 6005,
      "name": "NoActiveSession",
      "msg": "No active session found"
    },
    {
      "code": 6006,
      "name": "SessionExists",
      "msg": "Session already exists"
    },
    {
      "code": 6007,
      "name": "AccessDenied",
      "msg": "Access denied to this camera"
    },
    {
      "code": 6008,
      "name": "InvalidFaceData",
      "msg": "Face data invalid or not properly formatted"
    },
    {
      "code": 6009,
      "name": "FaceDataExists",
      "msg": "Face data already registered for this user"
    },
    {
      "code": 6010,
      "name": "InvalidGestureData",
      "msg": "Gesture data invalid or improperly formatted"
    },
    {
      "code": 6011,
      "name": "CameraAlreadyAuthorized",
      "msg": "Camera is already authorized for face recognition"
    },
    {
      "code": 6012,
      "name": "InvalidAccessDuration",
      "msg": "Invalid temporary access duration"
    },
    {
      "code": 6013,
      "name": "AccessGrantExpired",
      "msg": "Access grant has expired"
    },
    {
      "code": 6014,
      "name": "SessionExpired",
      "msg": "Session has expired"
    },
    {
      "code": 6015,
      "name": "NoRecognitionToken",
      "msg": "No recognition token found - please create one first"
    },
    {
      "code": 6016,
      "name": "FeatureNotAvailable",
      "msg": "Feature not available on this camera"
    },
    {
      "code": 6017,
      "name": "RecognitionTokenTooLarge",
      "msg": "Recognition token data too large (max 1024 bytes)"
    }
  ]
};
