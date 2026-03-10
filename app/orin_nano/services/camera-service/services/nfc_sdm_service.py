"""
NTAG424 DNA Secure Dynamic Messaging (SUN) verification service.

Validates the cryptographic proof produced by an NTAG424 DNA tag on each tap.
Issues short-lived presence tokens that gate access to the live stream.

Algorithm matches nfc-developer/sdm-backend (github.com/nfc-developer/sdm-backend)
which is the reference implementation for the NFC Developer App by Arx Research.

Crypto: AES-128-CBC (decrypt PICC data) + AES-128-CMAC (verify tap MAC)
Ref: NXP Application Note AN12196
"""

import os
import sqlite3
import secrets
import time
import logging
from typing import Optional, Tuple

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.cmac import CMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

DB_PATH = "/app/identity.db"
PRESENCE_TOKEN_TTL_SECONDS = 8 * 60 * 60  # 8 hours


class NFCSDMService:
    """
    Verifies NTAG424 DNA SUN messages and manages presence tokens.

    Uses a single MASTER_KEY (matching sdm-backend convention).
    With all-zero master key, all derived keys are also all-zero (factory default).
    """

    def __init__(self):
        self._master_key: Optional[bytes] = None
        self._enabled = False
        self._db_path = DB_PATH
        self._load_keys()
        self._init_db()

    def _load_keys(self):
        """Load the master key from env. Matches sdm-backend's MASTER_KEY."""
        master_hex = os.environ.get("NFC_MASTER_KEY", "").strip()

        # Also check the old env var names for backwards compatibility
        if not master_hex:
            enc_hex = os.environ.get("NFC_SDM_ENC_KEY", "").strip()
            mac_hex = os.environ.get("NFC_SDM_MAC_KEY", "").strip()
            if enc_hex:
                master_hex = enc_hex
                logger.info("[NFC] Using NFC_SDM_ENC_KEY as master key (legacy compat)")

        if master_hex:
            try:
                self._master_key = bytes.fromhex(master_hex)
                if len(self._master_key) != 16:
                    raise ValueError("Master key must be 16 bytes (32 hex chars)")
                self._enabled = True
                logger.info("[NFC] SDM master key loaded — NFC verification active")
            except Exception as e:
                logger.error(f"[NFC] Failed to load master key: {e}")
                self._enabled = False
        else:
            logger.info("[NFC] No NFC_MASTER_KEY configured — NFC verification disabled (dev mode)")

    def _derive_key(self, key_no: int, uid: Optional[bytes] = None) -> bytes:
        """
        Derive per-key values from master key.
        Matches sdm-backend: when master key is all zeros, ALL derived keys are all zeros.
        """
        if self._master_key == (b"\x00" * 16):
            return b"\x00" * 16
        # For non-zero master keys, use the master key directly
        # (full key diversification can be added later)
        return self._master_key

    def _init_db(self):
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nfc_tap_counters (
                    tag_uid_hex TEXT PRIMARY KEY,
                    last_counter INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nfc_presence_tokens (
                    token TEXT PRIMARY KEY,
                    camera_pda TEXT NOT NULL,
                    tag_uid_hex TEXT,
                    tap_counter INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                )
            """)
            conn.commit()
            conn.close()
            logger.info("[NFC] Database tables initialized")
        except Exception as e:
            logger.error(f"[NFC] DB init failed: {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------ #
    #  Core SUN Verification (matches sdm-backend algorithm)              #
    # ------------------------------------------------------------------ #

    def _decrypt_picc_data(self, enc_picc_data: bytes):
        """
        Decrypt EncryptedPICCData (16 bytes for AES mode).

        Returns: (picc_data_tag, uid_bytes, ctr_bytes, counter_int)
        """
        sdm_meta_read_key = self._derive_key(1)

        cipher = Cipher(
            algorithms.AES(sdm_meta_read_key),
            modes.CBC(bytes(16)),  # IV = 0x00 * 16
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(enc_picc_data) + decryptor.finalize()

        picc_data_tag = plaintext[0:1]
        uid_mirroring_en = (plaintext[0] & 0x80) == 0x80
        sdm_read_ctr_en = (plaintext[0] & 0x40) == 0x40
        uid_length = plaintext[0] & 0x0F

        if uid_length != 0x07:
            raise ValueError(f"Unsupported UID length: {uid_length}")

        if not uid_mirroring_en:
            raise ValueError("UID mirroring not enabled in PICC data")

        uid_bytes = plaintext[1:8]
        ctr_bytes = plaintext[8:11] if sdm_read_ctr_en else b"\x00\x00\x00"
        counter = int.from_bytes(ctr_bytes, "little")

        # picc_data for MAC calculation = uid + counter bytes (the data portion)
        picc_data = uid_bytes + ctr_bytes if sdm_read_ctr_en else uid_bytes

        return picc_data_tag, uid_bytes, ctr_bytes, counter, picc_data

    def _calculate_sdmmac(self, sdm_file_read_key: bytes, picc_data: bytes,
                          enc_file_data: Optional[bytes] = None) -> bytes:
        """
        Calculate SDMMAC matching sdm-backend's calculate_sdmmac (SEPARATED, AES mode).
        """
        # Build SV2
        sv2 = b"\x3C\xC3\x00\x01\x00\x80" + picc_data
        # Pad to AES block size
        while len(sv2) % 16 != 0:
            sv2 += b"\x00"

        # Session MAC key = CMAC(sdm_file_read_key, SV2)
        c = CMAC(algorithms.AES(sdm_file_read_key), backend=default_backend())
        c.update(sv2)
        session_mac_key = c.finalize()

        # MAC input: for SEPARATED mode with enc_file_data
        mac_input = b""
        if enc_file_data:
            mac_input = enc_file_data.hex().upper().encode("ascii") + b"&cmac="

        # Compute CMAC
        c2 = CMAC(algorithms.AES(session_mac_key), backend=default_backend())
        c2.update(mac_input)
        full_mac = c2.finalize()

        # Truncate: odd-indexed bytes
        return bytes(full_mac[i] for i in range(16) if i % 2 == 1)

    def verify_sun_message(
        self, picc_data_hex: str, cmac_hex: str, enc_file_data_hex: Optional[str] = None
    ) -> Tuple[bool, Optional[str], int, str]:
        """
        Verify an NTAG424 DNA SUN message.

        Args:
            picc_data_hex: Hex string of EncryptedPICCData (32 chars = 16 bytes)
            cmac_hex: Hex string of SDMMAC (16 chars = 8 bytes)
            enc_file_data_hex: Optional hex string of encrypted file data

        Returns:
            (valid, uid_hex, counter, error_message)
        """
        if not self._enabled:
            return False, None, 0, "NFC SDM keys not configured"

        try:
            enc_picc_data = bytes.fromhex(picc_data_hex)
            sdmmac = bytes.fromhex(cmac_hex)
        except ValueError as ex:
            return False, None, 0, f"Invalid hex: {ex}"

        if len(enc_picc_data) != 16:
            return False, None, 0, f"picc_data must be 16 bytes, got {len(enc_picc_data)}"
        if len(sdmmac) != 8:
            return False, None, 0, f"cmac must be 8 bytes, got {len(sdmmac)}"

        enc_file_data = None
        if enc_file_data_hex:
            try:
                enc_file_data = bytes.fromhex(enc_file_data_hex)
            except ValueError:
                pass

        # Step 1: Decrypt PICC data
        try:
            picc_data_tag, uid_bytes, ctr_bytes, counter, picc_data = self._decrypt_picc_data(enc_picc_data)
        except Exception as ex:
            return False, None, 0, f"PICC data decryption failed: {ex}"

        uid_hex = uid_bytes.hex().upper()

        # Step 2: Replay protection
        last_counter = self._get_last_counter(uid_hex)
        if counter <= last_counter:
            return False, uid_hex, counter, (
                f"Stale tap: counter {counter} <= last seen {last_counter}"
            )

        # Step 3: Verify CMAC
        sdm_file_read_key = self._derive_key(2, uid_bytes)
        expected_mac = self._calculate_sdmmac(sdm_file_read_key, picc_data, enc_file_data)

        if expected_mac != sdmmac:
            # Log details for debugging
            logger.warning(f"[NFC] CMAC mismatch: expected={expected_mac.hex()} got={sdmmac.hex()}")
            return False, uid_hex, counter, "SDMMAC verification failed — invalid tap signature"

        # Valid — update counter
        self._update_counter(uid_hex, counter)
        logger.info(f"[NFC] Valid tap: UID={uid_hex} counter={counter}")
        return True, uid_hex, counter, ""

    # ------------------------------------------------------------------ #
    #  Counter Tracking                                                    #
    # ------------------------------------------------------------------ #

    def _get_last_counter(self, uid_hex: str) -> int:
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                "SELECT last_counter FROM nfc_tap_counters WHERE tag_uid_hex = ?",
                (uid_hex,),
            ).fetchone()
            conn.close()
            return row[0] if row else 0
        except Exception:
            return 0

    def _update_counter(self, uid_hex: str, counter: int):
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """
                INSERT INTO nfc_tap_counters (tag_uid_hex, last_counter, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(tag_uid_hex) DO UPDATE SET
                    last_counter = excluded.last_counter,
                    updated_at = excluded.updated_at
                """,
                (uid_hex, counter, int(time.time())),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[NFC] Failed to update counter for {uid_hex}: {e}")

    # ------------------------------------------------------------------ #
    #  Presence Tokens                                                     #
    # ------------------------------------------------------------------ #

    def issue_presence_token(
        self, camera_pda: str, uid_hex: Optional[str], counter: int
    ) -> str:
        token = secrets.token_hex(32)
        now = int(time.time())
        expires_at = now + PRESENCE_TOKEN_TTL_SECONDS

        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """
                INSERT INTO nfc_presence_tokens
                    (token, camera_pda, tag_uid_hex, tap_counter, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (token, camera_pda, uid_hex, counter, now, expires_at),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[NFC] Failed to store presence token: {e}")

        return token

    def validate_presence_token(self, token: str, camera_pda: str) -> bool:
        if not token:
            return False
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                "SELECT expires_at FROM nfc_presence_tokens WHERE token = ? AND camera_pda = ?",
                (token, camera_pda),
            ).fetchone()
            conn.close()
            if not row:
                return False
            return int(time.time()) < row[0]
        except Exception:
            return False

    def cleanup_expired_tokens(self):
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "DELETE FROM nfc_presence_tokens WHERE expires_at < ?",
                (int(time.time()),),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"[NFC] Token cleanup failed: {e}")


# Singleton
_nfc_sdm_service: Optional[NFCSDMService] = None


def get_nfc_sdm_service() -> NFCSDMService:
    global _nfc_sdm_service
    if _nfc_sdm_service is None:
        _nfc_sdm_service = NFCSDMService()
    return _nfc_sdm_service
