"""
Tunnel Manager — Pi Zero 2W

Manages the Cloudflare tunnel config that maps {pda}.mmoment.xyz → localhost:5002.
Same flow as Orin Nano but no Docker — cloudflared runs as a systemd service.
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml

logger = logging.getLogger(__name__)

CLOUDFLARED_CONFIG = Path(os.path.expanduser("~/.cloudflared/config.yml"))
SERVICE_PORT = 5002


class TunnelManager:
    def __init__(self):
        self.tunnel_id = os.getenv("CLOUDFLARE_TUNNEL_ID")
        if not self.tunnel_id:
            logger.warning("CLOUDFLARE_TUNNEL_ID not set — tunnel manager disabled")
        self.credentials_file = (
            Path(os.path.expanduser(f"~/.cloudflared/{self.tunnel_id}.json"))
            if self.tunnel_id else None
        )

    def configure_tunnel(self, full_domain: str) -> bool:
        if not self.tunnel_id:
            logger.error("No CLOUDFLARE_TUNNEL_ID set")
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
            CLOUDFLARED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
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
        try:
            r = requests.get(f"https://{domain}/api/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        status = {
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
