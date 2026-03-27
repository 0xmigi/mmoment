# Cloudflare Tunnel for Camera API

This directory contains scripts and configuration files for setting up a Cloudflare tunnel to expose your Raspberry Pi camera API securely to the internet.

## What is Cloudflare Tunnel?

Cloudflare Tunnel provides a secure way to connect your services to Cloudflare without a publicly routable IP address. It creates encrypted tunnels between your Raspberry Pi and Cloudflare's edge, protecting your device from direct internet access while still making your API accessible through Cloudflare.

## Files in this Directory

- `setup.sh`: Script to automate the Cloudflare tunnel setup
- `cloudflare-tunnel.service`: Systemd service file for running the tunnel
- `config.yml`: Configuration file for the tunnel (created after running setup)
- `*.json`: Tunnel credentials file (created after running setup)

## Setup Instructions

1. Ensure you have a Cloudflare account with a domain added to it
2. Make the setup script executable:
   ```bash
   chmod +x setup.sh
   ```
3. Run the setup script:
   ```bash
   ./setup.sh
   ```
4. Follow the prompts in the script to:
   - Enter your tunnel name
   - Enter your domain name
   - Enter desired subdomain
   - Configure the API port (default: 5000)
   - Authenticate with Cloudflare

## Manual Setup (if script fails)

If you need to set up the tunnel manually:

1. Install Cloudflared:
   ```bash
   wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
   sudo dpkg -i cloudflared-linux-arm64.deb
   ```

2. Login to Cloudflare:
   ```bash
   cloudflared tunnel login
   ```

3. Create a tunnel:
   ```bash
   cloudflared tunnel create <tunnel-name>
   ```

4. Configure DNS:
   ```bash
   cloudflared tunnel route dns <tunnel-name> <subdomain.yourdomain.com>
   ```

5. Create a config file at `~/.cloudflared/config.yml`:
   ```yaml
   tunnel: <tunnel-id>
   credentials-file: /home/azuolas/.cloudflared/<tunnel-id>.json
   ingress:
     - hostname: <subdomain.yourdomain.com>
       service: http://localhost:5000
     - service: http_status:404
   ```

6. Copy the service file:
   ```bash
   sudo cp cloudflare-tunnel.service /etc/systemd/system/
   ```

7. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable cloudflare-tunnel.service
   sudo systemctl start cloudflare-tunnel.service
   ```

## Checking Status

To check the status of your tunnel service:
```bash
sudo systemctl status cloudflare-tunnel.service
```

## Troubleshooting

If the tunnel isn't working:

1. Check the service status:
   ```bash
   sudo systemctl status cloudflare-tunnel.service
   ```

2. View logs:
   ```bash
   sudo journalctl -u cloudflare-tunnel.service -f
   ```

3. Verify your configuration file at `~/.cloudflared/config.yml`

4. Ensure the API service is running and accessible on the configured port

## Backup and Restore

After setup, backup files will be copied to this directory. If you need to restore:

1. Copy the config.yml to ~/.cloudflared/
2. Copy the JSON credentials file to ~/.cloudflared/
3. Install the service file and restart the service 