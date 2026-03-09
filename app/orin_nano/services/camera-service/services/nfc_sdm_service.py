"""
NTAG424 DNA Secure Dynamic Messaging (SUN) verification service.

Validates the cryptographic proof produced by an NTAG424 DNA tag on each tap.
Issues short-lived presence tokens that gate access to the live stream.

Crypto: AES-128-CBC (decrypt PICC data) + AES-128-CMAC (verify tap MAC)
Ref: NXP Application Note AN12196 — NTAG 424 DNA and NTAG 424 DNA TagTamper
     Section 9: Secure Dynamic Messaging (SDM)
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
# Presence token valid for 8 hours (covers a full day at the event)
PRESENCE_TOKEN_TTL_SECONDS = 8 * 60 * 60


class NFCSDMService:
    """
    Verifies NTAG424 DNA SUN messages and manages presence tokens.

    A presence token is issued after a valid physical tap and lets the holder
    access the live stream without a wallet/account.
    """

    def __init__(self):
        self._enc_key: Optional[bytes] = None
        self._mac_key: Optional[bytes] = None
        self._enabled = False
        self._db_path = DB_PATH
        self._load_keys()
        self._init_db()

    def _load_keys(self):
        enc_hex = os.environ.get("NFC_SDM_ENC_KEY", "").strip()
        mac_hex = os.environ.get("NFC_SDM_MAC_KEY", "").strip()

        if enc_hex and mac_hex:
            try:
                self._enc_key = bytes.fromhex(enc_hex)
                self._mac_key = bytes.fromhex(mac_hex)
                if len(self._enc_key) != 16 or len(self._mac_key) != 16:
                    raise ValueError("Keys must be exactly 16 bytes (32 hex chars)")
                self._enabled = True
                logger.info("[NFC] SDM keys loaded — NFC verification active")
            except Exception as e:
                logger.error(f"[NFC] Failed to load SDM keys: {e}")
                self._enabled = False
        else:
            logger.info("[NFC] No SDM keys configured — NFC verification disabled")

    def _init_db(self):
        """Add NFC tables to the existing identity.db."""
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            # Track last-seen counter per tag UID (replay protection)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nfc_tap_counters (
                    tag_uid_hex TEXT PRIMARY KEY,
                    last_counter INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL DEFAULT 0
                )
            """)

            # Short-lived presence tokens issued after a valid tap
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
    #  Core SUN Verification                                               #
    # ------------------------------------------------------------------ #

    def _decrypt_picc_data(self, e_bytes: bytes) -> Tuple[bytes, bytes, int]:
        """
        Decrypt 16-byte EncryptedPICCData.

        PICC plaintext layout (AN12196 §9.1):
          Byte 0:   0xC7 (PICC data tag, UID + SDMReadCtr present)
          Bytes 1-7:  UID (7 bytes)
          Bytes 8-10: SDMReadCtr (3 bytes, little-endian)
          Bytes 11-15: zero padding

        Returns: (uid_bytes, ctr_bytes, counter_int)
        """
        cipher = Cipher(
            algorithms.AES(self._enc_key),
            modes.CBC(bytes(16)),  # IV = 0x00 * 16
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()
        picc = decryptor.update(e_bytes) + decryptor.finalize()

        if picc[0] != 0xC7:
            raise ValueError(f"Unexpected PICC header byte: 0x{picc[0]:02X} (expected 0xC7)")

        uid_bytes = picc[1:8]       # 7 bytes
        ctr_bytes = picc[8:11]      # 3 bytes
        counter = int.from_bytes(ctr_bytes, "little")
        return uid_bytes, ctr_bytes, counter

    def _derive_session_mac_key(self, uid_bytes: bytes, ctr_bytes: bytes) -> bytes:
        """
        Derive KSesSDMMACKey from SdmMACKey + UID + counter.

        SV2 (Session Vector, AN12196 §6.4.3):
          0x3C || 0xC3 || 0x00 || 0x01 || 0x00 || 0x80  (6 bytes, fixed)
          || UID (7 bytes)
          || SDMReadCtr (3 bytes)
          = 16 bytes total
        """
        sv2 = bytes([0x3C, 0xC3, 0x00, 0x01, 0x00, 0x80]) + uid_bytes + ctr_bytes
        c = CMAC(algorithms.AES(self._mac_key), backend=default_backend())
        c.update(sv2)
        return c.finalize()

    def _compute_sdm_mac(self, k_ses_mac: bytes, mac_input: bytes) -> bytes:
        """
        Compute and truncate the SDMMAC.

        Truncation (AN12196 §9.1): take bytes at odd indices of the full 16-byte CMAC.
        Returns 8 bytes.
        """
        c = CMAC(algorithms.AES(k_ses_mac), backend=default_backend())
        c.update(mac_input)
        full_mac = c.finalize()
        # Take bytes at positions 1, 3, 5, 7, 9, 11, 13, 15
        return bytes(full_mac[i] for i in range(1, 16, 2))

    def verify_sun_message(
        self, e_hex: str, c_hex: str
    ) -> Tuple[bool, Optional[str], int, str]:
        """
        Verify an NTAG424 DNA SUN message.

        Args:
            e_hex: Hex string of EncryptedPICCData (32 chars = 16 bytes)
            c_hex: Hex string of SDMMAC (16 chars = 8 bytes)

        Returns:
            (valid, uid_hex, counter, error_message)
        """
        if not self._enabled:
            return False, None, 0, "NFC SDM keys not configured"

        try:
            e_bytes = bytes.fromhex(e_hex.upper())
            c_bytes = bytes.fromhex(c_hex.upper())
        except ValueError as ex:
            return False, None, 0, f"Invalid hex in e or c param: {ex}"

        if len(e_bytes) != 16:
            return False, None, 0, f"e param must be 16 bytes, got {len(e_bytes)}"
        if len(c_bytes) != 8:
            return False, None, 0, f"c param must be 8 bytes, got {len(c_bytes)}"

        try:
            uid_bytes, ctr_bytes, counter = self._decrypt_picc_data(e_bytes)
        except Exception as ex:
            return False, None, 0, f"PICC data decryption failed: {ex}"

        uid_hex = uid_bytes.hex().upper()

        # Replay protection: counter must be strictly greater than last seen
        last_counter = self._get_last_counter(uid_hex)
        if counter <= last_counter:
            return False, uid_hex, counter, (
                f"Stale tap: counter {counter} <= last seen {last_counter}. "
                "This tap has already been used."
            )

        # Derive session MAC key and verify CMAC
        k_ses_mac = self._derive_session_mac_key(uid_bytes, ctr_bytes)

        # MAC input = ASCII bytes of "e={E_HEX}&c=" in the URL
        # This matches NFC TagWriter's default SDMMACInputOffset placement.
        mac_input = f"e={e_hex.upper()}&c=".encode("ascii")
        expected_mac = self._compute_sdm_mac(k_ses_mac, mac_input)

        if expected_mac != c_bytes:
            return False, uid_hex, counter, "SDMMAC verification failed — invalid tap"

        # Valid — update the counter so this tap can't be replayed
        self._update_counter(uid_hex, counter)
        logger.info(f"[NFC] ✅ Valid tap: UID={uid_hex} counter={counter}")
        return True, uid_hex, counter, ""

    # ------------------------------------------------------------------ #
    #  Counter Tracking (replay protection)                                #
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
        """
        Issue a presence token after a valid tap.
        Token is valid for PRESENCE_TOKEN_TTL_SECONDS (8 hours).
        """
        token = secrets.token_hex(32)  # 64-char hex, 256 bits of randomness
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
        """Return True if the token is valid and not expired for this camera."""
        if not token:
            return False
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                """
                SELECT expires_at FROM nfc_presence_tokens
                WHERE token = ? AND camera_pda = ?
                """,
                (token, camera_pda),
            ).fetchone()
            conn.close()
            if not row:
                return False
            return int(time.time()) < row[0]
        except Exception:
            return False

    def cleanup_expired_tokens(self):
        """Remove expired tokens — call occasionally to keep the DB lean."""
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
