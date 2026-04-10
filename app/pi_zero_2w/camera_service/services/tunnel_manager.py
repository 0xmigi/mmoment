"""
Tunnel Manager — Pi Zero 2W

Manages the Cloudflare tunnel config that maps {pda}.mmoment.xyz → localhost:5002.
Tunnel credentials are provisioned by the backend during device onboarding —
no Cloudflare API tokens needed on the device.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

CLOUDFLARED_DIR = Path(os.path.expanduser("~/.cloudflared"))
CLOUDFLARED_CONFIG = CLOUDFLARED_DIR / "config.yml"
SERVICE_PORT = 5002


class TunnelManager:
    def __init__(self):
        self.tunnel_id: Optional[str] = os.getenv("CLOUDFLARE_TUNNEL_ID")
        self.credentials_file: Optional[Path] = None
        if self.tunnel_id:
            self.credentials_file = CLOUDFLARED_DIR / f"{self.tunnel_id}.json"

    def set_tunnel_credentials(self, tunnel_id: str, credentials: dict):
        """Write backend-provisioned tunnel credentials to disk."""
        self.tunnel_id = tunnel_id
        CLOUDFLARED_DIR.mkdir(parents=True, exist_ok=True)
        self.credentials_file = CLOUDFLARED_DIR / f"{tunnel_id}.json"
        self.credentials_file.write_text(json.dumps(credentials))
        logger.info(f"Tunnel credentials written to {self.credentials_file}")

    def configure_tunnel(self, full_domain: str) -> bool:
        if not self.tunnel_id or not self.credentials_file:
            logger.error("No tunnel ID or credentials — cannot configure tunnel")
            return False
        try:
            config = {
                "tunnel": self.tunnel_id,
                "credentials-file": str(self.credentials_file),
                "ingress": [
                    {"hostname": full_domain, "service": f"http://localhost:{SERVICE_PORT}"},
                    {"service": "http_status:404"},
                ],
            }
            CLOUDFLARED_DIR.mkdir(parents=True, exist_ok=True)
            with open(CLOUDFLARED_CONFIG, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Tunnel config written for {full_domain}")
            return self._restart()
        except Exception as e:
            logger.error(f"configure_tunnel failed: {e}")
            return False

    def _restart(self) -> bool:
        try:
            result = subprocess.run(
                ["sudo", "systemctl", "restart", "cloudflared"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                logger.info("cloudflared restarted via systemctl")
                return True
            logger.warning(f"systemctl restart failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"systemctl error: {e}")

        # Fallback: kill and relaunch process directly
        try:
            subprocess.run(["sudo", "pkill", "cloudflared"], capture_output=True, timeout=5)
            subprocess.Popen(
                ["cloudflared", "tunnel", "run", self.tunnel_id],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            logger.info("cloudflared process restarted manually")
            return True
        except Exception as e:
            logger.error(f"Manual cloudflared restart failed: {e}")
            return False

    def test_connectivity(self, domain: str) -> bool:
        import requests
        try:
            r = requests.get(f"https://{domain}/api/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "tunnel_id": self.tunnel_id,
            "config_exists": CLOUDFLARED_CONFIG.exists(),
            "running": False,
            "hostname": None,
        }
        if status["config_exists"]:
            try:
                with open(CLOUDFLARED_CONFIG) as f:
                    cfg = yaml.safe_load(f)
                ingress = cfg.get("ingress", [])
                if ingress:
                    status["hostname"] = ingress[0].get("hostname")
            except Exception:
                pass
        try:
            result = subprocess.run(["pgrep", "-f", "cloudflared"], capture_output=True, timeout=5)
            status["running"] = result.returncode == 0
        except Exception:
            pass
        return status


_manager: Optional[TunnelManager] = None


def get_tunnel_manager() -> TunnelManager:
    global _manager
    if _manager is None:
        _manager = TunnelManager()
    return _manager
