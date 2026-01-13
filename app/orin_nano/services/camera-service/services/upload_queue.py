"""
Upload Queue - Simple background upload tracking.

Tracks files for async upload to Walrus. No file duplication.
SQLite stored in /app/config/ (already mounted).
"""

import os
import sqlite3
import threading
import logging
import time
import base64
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

DB_PATH = "/app/config/upload_queue.db"


class UploadStatus(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class UploadJob:
    id: int
    file_path: str
    file_type: str  # photo or video
    wallet_address: str
    camera_id: str
    device_signature: str
    timestamp: int
    status: str
    blob_id: Optional[str]
    download_url: Optional[str]
    error: Optional[str]
    retry_count: int
    created_at: int


class UploadQueue:
    """Simple upload queue with SQLite tracking."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._init_db()
        self._worker_running = False
        self._worker_thread = None
        self._start_worker()

        logger.info(f"UploadQueue initialized: {DB_PATH}")

    def _init_db(self):
        """Create uploads table if not exists."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    wallet_address TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    device_signature TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    blob_id TEXT,
                    download_url TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at INTEGER NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON uploads(status)")
            conn.commit()

    def add(
        self,
        file_path: str,
        file_type: str,
        wallet_address: str,
        camera_id: str,
        device_signature: str,
        timestamp: int,
    ) -> int:
        """Add file to upload queue. Returns job ID."""
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                """INSERT INTO uploads
                   (file_path, file_type, wallet_address, camera_id, device_signature, timestamp, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (file_path, file_type, wallet_address, camera_id, device_signature, timestamp, int(time.time() * 1000))
            )
            conn.commit()
            job_id = cursor.lastrowid
            logger.info(f"Queued upload #{job_id}: {file_path}")
            return job_id

    def get_status(self, job_id: int) -> Optional[Dict]:
        """Get upload status for a job."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM uploads WHERE id = ?", (job_id,)).fetchone()
            if row:
                return dict(row)
        return None

    def get_pending(self, limit: int = 10) -> List[Dict]:
        """Get pending uploads."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM uploads WHERE status = 'pending' ORDER BY created_at LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_user_uploads(self, wallet_address: str, limit: int = 50) -> List[Dict]:
        """Get uploads for a user."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM uploads WHERE wallet_address = ? ORDER BY created_at DESC LIMIT ?",
                (wallet_address, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def _update_status(self, job_id: int, status: str, blob_id: str = None, download_url: str = None, error: str = None):
        """Update job status."""
        with sqlite3.connect(DB_PATH) as conn:
            if status == UploadStatus.COMPLETED:
                conn.execute(
                    "UPDATE uploads SET status = ?, blob_id = ?, download_url = ? WHERE id = ?",
                    (status, blob_id, download_url, job_id)
                )
            elif status == UploadStatus.FAILED:
                conn.execute(
                    "UPDATE uploads SET status = ?, error = ?, retry_count = retry_count + 1 WHERE id = ?",
                    (status, error, job_id)
                )
            else:
                conn.execute("UPDATE uploads SET status = ? WHERE id = ?", (status, job_id))
            conn.commit()

    def _start_worker(self):
        """Start background upload worker."""
        if self._worker_running:
            return
        self._worker_running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="UploadWorker")
        self._worker_thread.start()
        logger.info("Upload worker started")

    def _worker_loop(self):
        """Process upload queue in background."""
        while self._worker_running:
            try:
                # Get next pending job
                pending = self.get_pending(limit=1)
                if not pending:
                    time.sleep(5)  # Nothing to do, wait
                    continue

                job = pending[0]
                job_id = job["id"]

                logger.info(f"Processing upload #{job_id}: {job['file_path']}")
                self._update_status(job_id, UploadStatus.UPLOADING)

                # Do the upload
                result = self._do_upload(job)

                if result.get("success"):
                    self._update_status(
                        job_id,
                        UploadStatus.COMPLETED,
                        blob_id=result.get("blob_id"),
                        download_url=result.get("download_url")
                    )
                    logger.info(f"Upload #{job_id} completed: {result.get('blob_id')}")
                else:
                    self._update_status(job_id, UploadStatus.FAILED, error=result.get("error"))
                    logger.error(f"Upload #{job_id} failed: {result.get('error')}")

            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(5)

    def _do_upload(self, job: Dict) -> Dict:
        """Actually upload a file to Walrus."""
        try:
            from services.walrus_upload_service import get_walrus_upload_service

            file_path = job["file_path"]
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_path}"}

            # Read and encrypt file
            with open(file_path, 'rb') as f:
                plaintext = f.read()

            content_key = os.urandom(32)
            nonce = os.urandom(12)
            aesgcm = AESGCM(content_key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            encrypted_data = nonce + ciphertext

            # Get checked-in users for access grants
            try:
                from routes import get_checked_in_users
                checked_in_users = get_checked_in_users()
            except:
                checked_in_users = []

            grant_users = list(checked_in_users)
            if job["wallet_address"] not in grant_users:
                grant_users.append(job["wallet_address"])

            # Create access grants
            upload_service = get_walrus_upload_service()
            access_grants = upload_service._create_access_grants(content_key, grant_users)
            nonce_b64 = base64.b64encode(nonce).decode('utf-8')

            # Upload
            result = upload_service.upload_capture(
                wallet_address=job["wallet_address"],
                file_path=file_path,
                camera_id=job["camera_id"],
                device_signature=job["device_signature"],
                checked_in_users=checked_in_users,
                file_type=job["file_type"],
                timestamp=job["timestamp"],
            )

            if result.get("success"):
                return {
                    "success": True,
                    "blob_id": result.get("blob_id"),
                    "download_url": result.get("download_url"),
                }
            else:
                return {"success": False, "error": result.get("error", "Upload failed")}

        except Exception as e:
            logger.error(f"Upload error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_stats(self) -> Dict:
        """Get upload statistics."""
        with sqlite3.connect(DB_PATH) as conn:
            stats = {}
            for status in ["pending", "uploading", "completed", "failed"]:
                count = conn.execute(
                    "SELECT COUNT(*) FROM uploads WHERE status = ?", (status,)
                ).fetchone()[0]
                stats[status] = count
            return stats


# Singleton
_upload_queue: Optional[UploadQueue] = None

def get_upload_queue() -> UploadQueue:
    global _upload_queue
    if _upload_queue is None:
        _upload_queue = UploadQueue()
    return _upload_queue
