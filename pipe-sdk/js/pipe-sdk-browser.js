/**
 * Pipe SDK for Browser - Decrypt and display media from MMOMENT cameras
 *
 * This runs in the browser to:
 * 1. Download encrypted media from Pipe
 * 2. Decrypt client-side
 * 3. Display to authorized users
 */

class PipeSDKBrowser {
    constructor(baseUrl = 'https://firestarter.pipenetwork.com') {
        this.baseUrl = baseUrl;
    }

    /**
     * Generate deterministic password from user ID
     * Must match the Rust implementation
     */
    async generateUserPassword(userId) {
        const encoder = new TextEncoder();
        const data = encoder.encode(userId + 'mmoment-pipe-encryption-2024');
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    /**
     * Download a public file (no auth needed)
     */
    async publicDownload(hash) {
        const response = await fetch(`${this.baseUrl}/publicDownload?hash=${hash}`);
        if (!response.ok) {
            throw new Error(`Download failed: ${response.statusText}`);
        }
        return await response.arrayBuffer();
    }

    /**
     * Download a file with authentication
     */
    async download(userId, userAppKey, filename) {
        const params = new URLSearchParams({
            user_id: userId,
            user_app_key: userAppKey,
            file_name: filename
        });

        const response = await fetch(`${this.baseUrl}/download?${params}`);
        if (!response.ok) {
            throw new Error(`Download failed: ${response.statusText}`);
        }
        return await response.arrayBuffer();
    }

    /**
     * Decrypt data using WebCrypto API
     */
    async decrypt(encryptedData, password, metadata) {
        // Decode metadata
        const salt = this.base64ToArrayBuffer(metadata.salt);
        const nonce = this.base64ToArrayBuffer(metadata.nonce);

        // Derive key from password using PBKDF2
        const encoder = new TextEncoder();
        const passwordData = encoder.encode(password);

        const keyMaterial = await crypto.subtle.importKey(
            'raw',
            passwordData,
            'PBKDF2',
            false,
            ['deriveKey']
        );

        const key = await crypto.subtle.deriveKey(
            {
                name: 'PBKDF2',
                salt: salt,
                iterations: metadata.iterations || 100000,
                hash: 'SHA-256'
            },
            keyMaterial,
            { name: 'AES-GCM', length: 256 },
            false,
            ['decrypt']
        );

        // Decrypt using AES-GCM (browser doesn't have ChaCha20)
        // Note: In production, we'd need to handle algorithm differences
        try {
            const decrypted = await crypto.subtle.decrypt(
                {
                    name: 'AES-GCM',
                    iv: nonce,
                    tagLength: 128
                },
                key,
                encryptedData
            );

            return decrypted;
        } catch (e) {
            throw new Error('Decryption failed - wrong password?');
        }
    }

    /**
     * Unpack encrypted file (extract metadata and encrypted data)
     */
    unpackEncryptedFile(packedData) {
        const view = new DataView(packedData);

        // Read metadata length (first 4 bytes, little-endian)
        const metadataLen = view.getUint32(0, true);

        // Extract metadata JSON
        const metadataBytes = new Uint8Array(packedData, 4, metadataLen);
        const metadataJson = new TextDecoder().decode(metadataBytes);
        const metadata = JSON.parse(metadataJson);

        // Extract encrypted data
        const encryptedData = packedData.slice(4 + metadataLen);

        return { encryptedData, metadata };
    }

    /**
     * Download and decrypt a file
     */
    async downloadAndDecrypt(userId, userAppKey, filename, password) {
        // Download the file
        const data = await this.download(userId, userAppKey, filename);

        // Check if encrypted (filename ends with .enc)
        if (filename.endsWith('.enc')) {
            // Unpack the file
            const { encryptedData, metadata } = this.unpackEncryptedFile(data);

            // Use provided password or generate from user ID
            const decryptPassword = password || await this.generateUserPassword(userId);

            // Decrypt
            return await this.decrypt(encryptedData, decryptPassword, metadata);
        }

        return data;
    }

    /**
     * Convert image data to blob URL for display
     */
    createImageUrl(imageData) {
        const blob = new Blob([imageData], { type: 'image/jpeg' });
        return URL.createObjectURL(blob);
    }

    /**
     * Helper: Convert base64 to ArrayBuffer
     */
    base64ToArrayBuffer(base64) {
        const binaryString = atob(base64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }

    /**
     * Helper: Convert ArrayBuffer to base64
     */
    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
}

// ============================================================================
// EXAMPLE USAGE IN MMOMENT WEB APP
// ============================================================================

async function displayUserPhoto() {
    const sdk = new PipeSDKBrowser();

    // Get user credentials (from wallet connection or stored)
    const userId = 'mmoment_wallet_alice_7x8y9z';
    const userAppKey = 'stored_app_key_from_session';
    const filename = 'mmoment_photo_20240101_120000.jpg.enc';

    try {
        // Download and decrypt
        console.log('📥 Downloading encrypted photo...');
        const imageData = await sdk.downloadAndDecrypt(
            userId,
            userAppKey,
            filename,
            null  // Use auto-generated password
        );

        // Create displayable URL
        const imageUrl = sdk.createImageUrl(imageData);

        // Display in img element
        document.getElementById('user-photo').src = imageUrl;
        console.log('✅ Photo decrypted and displayed!');

    } catch (error) {
        console.error('❌ Failed to load photo:', error);
    }
}

// For public shared links
async function displayPublicPhoto(hash, password) {
    const sdk = new PipeSDKBrowser();

    try {
        // Download public file
        const data = await sdk.publicDownload(hash);

        // Unpack and decrypt if needed
        if (password) {
            const { encryptedData, metadata } = sdk.unpackEncryptedFile(data);
            const decrypted = await sdk.decrypt(encryptedData, password, metadata);

            const imageUrl = sdk.createImageUrl(decrypted);
            document.getElementById('shared-photo').src = imageUrl;
        }

    } catch (error) {
        console.error('Failed to load shared photo:', error);
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PipeSDKBrowser;
}
