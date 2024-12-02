export type CameraActivation = {
  "version": "0.1.0",
  "name": "camera_activation",
  "instructions": [
    {
      "name": "initialize",
      "accounts": [
        {
          "name": "cameraAccount",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
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
      "name": "activateCamera",
      "accounts": [
        {
          "name": "cameraAccount",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "fee",
          "type": "u64"
        }
      ]
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
          }
        ]
      }
    }
  ],
  "events": [
    {
      "name": "CameraActivated",
      "fields": [
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        },
        {
          "name": "cameraAccount",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "sessionId",
          "type": "string",
          "index": false
        }
      ]
    }
  ],
  "errors": [
    {
      "code": 6000,
      "name": "InsufficientFee",
      "msg": "The fee is insufficient"
    }
  ]
};

export const IDL: CameraActivation = {
  "version": "0.1.0",
  "name": "camera_activation",
  "instructions": [
    {
      "name": "initialize",
      "accounts": [
        {
          "name": "cameraAccount",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
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
      "name": "activateCamera",
      "accounts": [
        {
          "name": "cameraAccount",
          "isMut": true,
          "isSigner": false
        },
        {
          "name": "user",
          "isMut": true,
          "isSigner": true
        },
        {
          "name": "systemProgram",
          "isMut": false,
          "isSigner": false
        }
      ],
      "args": [
        {
          "name": "fee",
          "type": "u64"
        }
      ]
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
          }
        ]
      }
    }
  ],
  "events": [
    {
      "name": "CameraActivated",
      "fields": [
        {
          "name": "timestamp",
          "type": "i64",
          "index": false
        },
        {
          "name": "cameraAccount",
          "type": "publicKey",
          "index": false
        },
        {
          "name": "sessionId",
          "type": "string",
          "index": false
        }
      ]
    }
  ],
  "errors": [
    {
      "code": 6000,
      "name": "InsufficientFee",
      "msg": "The fee is insufficient"
    }
  ]
};