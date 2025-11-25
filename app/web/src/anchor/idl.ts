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
      "name": "checkIn",
      "docs": [
        "Check in user to a camera"
      ],
      "accounts": [
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "payer",
          "isMut": true,
          "isSigner": true,
          "docs": [
            "The payer for the session account - usually the user, but can be a sponsor",
            "This allows gas sponsorship where a third party pays for account rent"
          ]
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "recognitionToken",
          "isMut": false,
          "isSigner": false,
          "isOptional": true,
          "docs": [
            "Optional recognition token - required if use_face_recognition is true"
          ]
        },
        {
          "name": "session",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "session"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "user"
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
          "name": "useFaceRecognition",
          "type": "bool"
        }
      ]
    },
    {
      "name": "checkOut",
      "docs": [
        "Check out user from a camera with optional activity bundle"
      ],
      "accounts": [
        {
          "name": "closer",
          "isMut": true,
          "isSigner": true
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
            "Camera timeline - created lazily on first checkout with activities"
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
          "name": "session",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "session"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "account": "UserSession",
                "path": "session.user"
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
          "name": "sessionUser",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "rentDestination",
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
          "name": "activities",
          "type": {
            "vec": {
              "defined": "ActivityData"
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
      "name": "recordActivity",
      "docs": [
        "Record a camera activity (photo, video, stream)"
      ],
      "accounts": [
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "session",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "args",
          "type": {
            "defined": "RecordActivityArgs"
          }
        }
      ]
    }
  ],
  "accounts": [
    {
      "name": "cameraRegistry",
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
      "name": "cameraAccount",
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
      "name": "userSession",
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
            "name": "checkInTime",
            "type": "i64"
          },
          {
            "name": "lastActivity",
            "type": "i64"
          },
          {
            "name": "autoCheckoutAt",
            "type": "i64"
          },
          {
            "name": "enabledFeatures",
            "type": {
              "defined": "SessionFeatures"
            }
          },
          {
            "name": "bump",
            "type": "u8"
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
      "name": "gestureConfig",
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
      "name": "cameraMessage",
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
      "name": "accessGrant",
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
      "name": "cameraTimeline",
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
    }
  ],
  "types": [
    {
      "name": "RecordActivityArgs",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "actionType",
            "type": {
              "defined": "CameraActionType"
            }
          },
          {
            "name": "metadata",
            "type": "string"
          }
        ]
      }
    },
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
      "name": "SessionFeatures",
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
      "name": "ActivityArgs",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "activityType",
            "type": "u8"
          },
          {
            "name": "metadata",
            "type": "string"
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
      "name": "CameraActionType",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "PhotoCapture"
          },
          {
            "name": "VideoRecord"
          },
          {
            "name": "StreamStart"
          },
          {
            "name": "StreamStop"
          },
          {
            "name": "Custom"
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
      "name": "UserCheckedIn",
      "fields": [
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "session",
          "type": "publicKey",
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
      "name": "UserCheckedOut",
      "fields": [
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "session",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "duration",
          "type": "i64",
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
      "name": "ActivityRecorded",
      "fields": [
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "activityType",
          "type": "u8",
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
    },
    {
      "name": "SessionAutoCheckout",
      "fields": [
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "session",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "reason",
          "type": "string",
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
      "name": "checkIn",
      "docs": [
        "Check in user to a camera"
      ],
      "accounts": [
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "payer",
          "isMut": true,
          "isSigner": true,
          "docs": [
            "The payer for the session account - usually the user, but can be a sponsor",
            "This allows gas sponsorship where a third party pays for account rent"
          ]
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "recognitionToken",
          "isMut": false,
          "isSigner": false,
          "isOptional": true,
          "docs": [
            "Optional recognition token - required if use_face_recognition is true"
          ]
        },
        {
          "name": "session",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "session"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "path": "user"
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
          "name": "useFaceRecognition",
          "type": "bool"
        }
      ]
    },
    {
      "name": "checkOut",
      "docs": [
        "Check out user from a camera with optional activity bundle"
      ],
      "accounts": [
        {
          "name": "closer",
          "isMut": true,
          "isSigner": true
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
            "Camera timeline - created lazily on first checkout with activities"
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
          "name": "session",
          "isMut": true,
          "isSigner": false,
          "pda": {
            "seeds": [
              {
                "kind": "const",
                "type": "string",
                "value": "session"
              },
              {
                "kind": "account",
                "type": "publicKey",
                "account": "UserSession",
                "path": "session.user"
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
          "name": "sessionUser",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "rentDestination",
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
          "name": "activities",
          "type": {
            "vec": {
              "defined": "ActivityData"
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
      "name": "recordActivity",
      "docs": [
        "Record a camera activity (photo, video, stream)"
      ],
      "accounts": [
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "camera",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "session",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "args",
          "type": {
            "defined": "RecordActivityArgs"
          }
        }
      ]
    }
  ],
  "accounts": [
    {
      "name": "cameraRegistry",
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
      "name": "cameraAccount",
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
      "name": "userSession",
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
            "name": "checkInTime",
            "type": "i64"
          },
          {
            "name": "lastActivity",
            "type": "i64"
          },
          {
            "name": "autoCheckoutAt",
            "type": "i64"
          },
          {
            "name": "enabledFeatures",
            "type": {
              "defined": "SessionFeatures"
            }
          },
          {
            "name": "bump",
            "type": "u8"
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
      "name": "gestureConfig",
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
      "name": "cameraMessage",
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
      "name": "accessGrant",
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
      "name": "cameraTimeline",
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
    }
  ],
  "types": [
    {
      "name": "RecordActivityArgs",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "actionType",
            "type": {
              "defined": "CameraActionType"
            }
          },
          {
            "name": "metadata",
            "type": "string"
          }
        ]
      }
    },
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
      "name": "SessionFeatures",
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
      "name": "ActivityArgs",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "activityType",
            "type": "u8"
          },
          {
            "name": "metadata",
            "type": "string"
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
      "name": "CameraActionType",
      "type": {
        "kind": "enum",
        "variants": [
          {
            "name": "PhotoCapture"
          },
          {
            "name": "VideoRecord"
          },
          {
            "name": "StreamStart"
          },
          {
            "name": "StreamStop"
          },
          {
            "name": "Custom"
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
      "name": "UserCheckedIn",
      "fields": [
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "session",
          "type": "publicKey",
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
      "name": "UserCheckedOut",
      "fields": [
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "session",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "duration",
          "type": "i64",
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
      "name": "ActivityRecorded",
      "fields": [
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "activityType",
          "type": "u8",
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
    },
    {
      "name": "SessionAutoCheckout",
      "fields": [
        {
          "name": "user",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "session",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "reason",
          "type": "string",
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
