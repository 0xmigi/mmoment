export type MySolanaProject = {
  "version": "0.1.0",
  "name": "my_solana_project",
  "instructions": [
    {
      "name": "initialize",
      "accounts": [
        {
          "name": "authority",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "registry",
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
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "registry",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "camera",
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
            "defined": "RegisterCameraArgs"
          }
        }
      ]
    },
    {
      "name": "updateCamera",
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
      "name": "recordActivity",
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
            "defined": "RecordActivityArgs"
          }
        }
      ]
    },
    {
      "name": "setCameraActive",
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
            "defined": "SetCameraActiveArgs"
          }
        }
      ]
    },
    {
      "name": "deregisterCamera",
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "registry",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "camera",
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
    }
  ],
  "accounts": [
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
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "activityCounter",
            "type": "u64"
          },
          {
            "name": "lastActivityType",
            "type": {
              "option": {
                "defined": "ActivityType"
              }
            }
          },
          {
            "name": "metadata",
            "type": {
              "defined": "CameraMetadata"
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
            "name": "activityType",
            "type": {
              "defined": "ActivityType"
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
            "name": "fee",
            "type": "u64"
          }
        ]
      }
    },
    {
      "name": "SetCameraActiveArgs",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "isActive",
            "type": "bool"
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
            "name": "model",
            "type": {
              "option": "string"
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
            "name": "PhotoCapture"
          },
          {
            "name": "VideoRecord"
          },
          {
            "name": "LiveStream"
          },
          {
            "name": "Custom"
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
            "name": "model",
            "type": "string"
          },
          {
            "name": "registrationDate",
            "type": "i64"
          },
          {
            "name": "lastActivity",
            "type": "i64"
          }
        ]
      }
    }
  ],
  "events": [
    {
      "name": "ActivityRecorded",
      "fields": [
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "name",
          "type": "string",
          "index": false
        },
        {
          "name": "activityNumber",
          "type": "u64",
          "index": false
        },
        {
          "name": "activityType",
          "type": {
            "defined": "ActivityType"
          },
          "index": false
        },
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        },
        {
          "name": "metadata",
          "type": "string",
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
      "name": "InvalidCameraData",
      "msg": "Invalid camera data provided"
    },
    {
      "code": 6003,
      "name": "CameraIdExists",
      "msg": "Camera ID already exists"
    },
    {
      "code": 6004,
      "name": "InsufficientFee",
      "msg": "The fee is insufficient"
    }
  ]
};

export const IDL: MySolanaProject = {
  "version": "0.1.0",
  "name": "my_solana_project",
  "instructions": [
    {
      "name": "initialize",
      "accounts": [
        {
          "name": "authority",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "registry",
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
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "registry",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "camera",
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
            "defined": "RegisterCameraArgs"
          }
        }
      ]
    },
    {
      "name": "updateCamera",
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
      "name": "recordActivity",
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
            "defined": "RecordActivityArgs"
          }
        }
      ]
    },
    {
      "name": "setCameraActive",
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
            "defined": "SetCameraActiveArgs"
          }
        }
      ]
    },
    {
      "name": "deregisterCamera",
      "accounts": [
        {
          "name": "owner",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "registry",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "camera",
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
    }
  ],
  "accounts": [
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
            "name": "isActive",
            "type": "bool"
          },
          {
            "name": "activityCounter",
            "type": "u64"
          },
          {
            "name": "lastActivityType",
            "type": {
              "option": {
                "defined": "ActivityType"
              }
            }
          },
          {
            "name": "metadata",
            "type": {
              "defined": "CameraMetadata"
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
            "name": "activityType",
            "type": {
              "defined": "ActivityType"
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
            "name": "fee",
            "type": "u64"
          }
        ]
      }
    },
    {
      "name": "SetCameraActiveArgs",
      "type": {
        "kind": "struct",
        "fields": [
          {
            "name": "isActive",
            "type": "bool"
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
            "name": "model",
            "type": {
              "option": "string"
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
            "name": "PhotoCapture"
          },
          {
            "name": "VideoRecord"
          },
          {
            "name": "LiveStream"
          },
          {
            "name": "Custom"
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
            "name": "model",
            "type": "string"
          },
          {
            "name": "registrationDate",
            "type": "i64"
          },
          {
            "name": "lastActivity",
            "type": "i64"
          }
        ]
      }
    }
  ],
  "events": [
    {
      "name": "ActivityRecorded",
      "fields": [
        {
          "name": "camera",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "name",
          "type": "string",
          "index": false
        },
        {
          "name": "activityNumber",
          "type": "u64",
          "index": false
        },
        {
          "name": "activityType",
          "type": {
            "defined": "ActivityType"
          },
          "index": false
        },
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        },
        {
          "name": "metadata",
          "type": "string",
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
      "name": "InvalidCameraData",
      "msg": "Invalid camera data provided"
    },
    {
      "code": 6003,
      "name": "CameraIdExists",
      "msg": "Camera ID already exists"
    },
    {
      "code": 6004,
      "name": "InsufficientFee",
      "msg": "The fee is insufficient"
    }
  ]
};
