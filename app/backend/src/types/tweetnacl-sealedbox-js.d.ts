declare module 'tweetnacl-sealedbox-js' {
  interface SealedBox {
    /**
     * Seal a message using a recipient's public key.
     * Creates an ephemeral keypair, performs ECDH, derives key, and encrypts.
     */
    seal(message: Uint8Array, recipientPublicKey: Uint8Array): Uint8Array;

    /**
     * Open a sealed box using the recipient's keypair.
     * Extracts ephemeral public key, performs ECDH, derives key, and decrypts.
     */
    open(
      sealedMessage: Uint8Array,
      recipientPublicKey: Uint8Array,
      recipientSecretKey: Uint8Array
    ): Uint8Array | null;

    /**
     * Overhead added by sealing: ephemeral public key + MAC tag.
     */
    overheadLength: number;
  }

  const sealedBox: SealedBox;
  export = sealedBox;
}
