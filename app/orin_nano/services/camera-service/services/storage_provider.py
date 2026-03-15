"""
Storage Provider Abstraction

Feature flag to switch between Pipe Network and Walrus storage.
Set STORAGE_PROVIDER=walrus (default) or STORAGE_PROVIDER=pipe.
"""

import os
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.walrus_upload_service import WalrusUploadService
    from services.direct_pipe_upload import DirectPipeUploadService

logger = logging.getLogger(__name__)

# Feature flag: 'walrus' (default) or 'pipe'
STORAGE_PROVIDER = os.environ.get("STORAGE_PROVIDER", "walrus").lower()

logger.info(f"Storage provider configured: {STORAGE_PROVIDER}")


def get_upload_service():
    """
    Get the configured upload service based on STORAGE_PROVIDER env var.

    Returns:
        WalrusUploadService or DirectPipeUploadService singleton
    """
    if STORAGE_PROVIDER == "walrus":
        from services.walrus_upload_service import get_walrus_upload_service
        return get_walrus_upload_service()
    else:
        from services.direct_pipe_upload import get_direct_pipe_upload_service
        return get_direct_pipe_upload_service()


def is_walrus_enabled() -> bool:
    """Check if Walrus storage is enabled."""
    return STORAGE_PROVIDER == "walrus"


def is_pipe_enabled() -> bool:
    """Check if Pipe storage is enabled."""
    return STORAGE_PROVIDER == "pipe"


def get_storage_provider_name() -> str:
    """Get the name of the current storage provider."""
    return STORAGE_PROVIDER
