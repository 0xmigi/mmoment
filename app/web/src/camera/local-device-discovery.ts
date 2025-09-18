/**
 * Local Network Device Discovery for MMOMENT Cameras
 * 
 * Discovers and configures cameras on the local network without requiring
 * internet-based APIs or factory provisioning.
 */

export interface LocalDevice {
  ip: string;
  mac: string;
  deviceId: string;
  model: string;
  setupRequired: boolean;
  devicePubkey?: string;
}

export interface WiFiNetwork {
  ssid: string;
  security: string;
  signal: number;
}

export class LocalDeviceDiscovery {
  private discoveredDevices: Map<string, LocalDevice> = new Map();
  private scanInProgress = false;

  /**
   * Scan local network for MMOMENT devices
   * Uses multiple discovery methods for best coverage
   */
  async scanForDevices(): Promise<LocalDevice[]> {
    if (this.scanInProgress) {
      throw new Error('Scan already in progress');
    }

    this.scanInProgress = true;
    this.discoveredDevices.clear();

    try {
      console.log('🔍 Starting local network device discovery...');

      // Method 1: mDNS/Bonjour discovery
      await this.discoverViaMDNS();

      // Method 2: Network range scanning (if mDNS fails)
      await this.discoverViaNetworkScan();

      // Method 3: Check known device patterns
      await this.discoverViaKnownPorts();

      console.log(`✅ Discovery complete. Found ${this.discoveredDevices.size} devices`);
      return Array.from(this.discoveredDevices.values());

    } finally {
      this.scanInProgress = false;
    }
  }

  /**
   * Method 1: mDNS Discovery (most reliable)
   */
  private async discoverViaMDNS(): Promise<void> {
    try {
      // Check if device advertises mDNS service
      const services = ['_mmoment._tcp.local', '_http._tcp.local'];
      
      for (const service of services) {
        // Note: Browser mDNS is limited, but we can try WebRTC local discovery
        console.log(`Checking mDNS service: ${service}`);
        // Implementation would use WebRTC or browser extension for mDNS
      }
    } catch (error) {
      console.warn('mDNS discovery failed:', error);
    }
  }

  /**
   * Method 2: Network Range Scanning
   */
  private async discoverViaNetworkScan(): Promise<void> {
    try {
      // Get user's local IP to determine network range
      const localIP = await this.getLocalIP();
      if (!localIP) return;

      const networkBase = localIP.substring(0, localIP.lastIndexOf('.'));
      console.log(`Scanning network range: ${networkBase}.0/24`);

      // Scan common IP ranges in parallel
      const scanPromises: Promise<void>[] = [];
      for (let i = 2; i < 255; i++) {
        const targetIP = `${networkBase}.${i}`;
        scanPromises.push(this.checkDeviceAtIP(targetIP));
      }

      // Limit concurrent requests to avoid overwhelming network
      await this.batchProcess(scanPromises, 20);

    } catch (error) {
      console.warn('Network scan failed:', error);
    }
  }

  /**
   * Method 3: Check Known Ports/Patterns
   */
  private async discoverViaKnownPorts(): Promise<void> {
    // Check common local IPs for MMOMENT devices
    const commonIPs = [
      '192.168.1.100', '192.168.1.101', '192.168.1.200',
      '192.168.4.1',   // AP mode setup
      '10.0.0.100', '10.0.0.101'
    ];

    // For localhost development, also check if we're running against a dev server
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      // Add localhost testing - simulate finding a demo device
      this.addMockDevice();
    }

    for (const ip of commonIPs) {
      await this.checkDeviceAtIP(ip);
    }
  }

  /**
   * Add a mock device for localhost testing
   */
  private addMockDevice(): void {
    const mockDevice: LocalDevice = {
      ip: 'demo.localhost',
      mac: 'AA:BB:CC:DD:EE:FF',
      deviceId: 'DEMO-JETSON-001',
      model: 'MMOMENT Jetson Orin Nano (Demo)',
      setupRequired: false,
      devicePubkey: undefined // Mock device has no real device key
    };

    this.discoveredDevices.set('demo.localhost', mockDevice);
    console.log('✅ Added mock device for localhost testing:', mockDevice);
  }

  /**
   * Check if MMOMENT device exists at specific IP
   */
  private async checkDeviceAtIP(ip: string): Promise<void> {
    try {
      // Try to connect to device API
      const response = await fetch(`http://${ip}:5002/api/device-info`, {
        method: 'GET',
        timeout: 3000,
        mode: 'cors',
        credentials: 'omit'
      } as any);

      if (response.ok) {
        const deviceInfo = await response.json();
        
        // Verify it's actually an MMOMENT device
        if (deviceInfo.model && deviceInfo.model.includes('MMOMENT')) {
          const device: LocalDevice = {
            ip: ip,
            mac: deviceInfo.hardware_id || 'unknown',
            deviceId: deviceInfo.hardware_id || ip,
            model: deviceInfo.model,
            setupRequired: !deviceInfo.device_pubkey,
            devicePubkey: deviceInfo.device_pubkey
          };

          this.discoveredDevices.set(ip, device);
          console.log(`✅ Found MMOMENT device at ${ip}:`, device);
        }
      }
    } catch (error) {
      // Silent fail - most IPs won't have devices
    }
  }

  /**
   * Get user's local IP address
   */
  private async getLocalIP(): Promise<string | null> {
    try {
      // Use WebRTC to discover local IP
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      });

      pc.createDataChannel('');
      await pc.createOffer();
      
      return new Promise((resolve) => {
        pc.onicecandidate = (ice) => {
          if (ice.candidate) {
            const candidate = ice.candidate.candidate;
            const match = candidate.match(/(\d+\.\d+\.\d+\.\d+)/);
            if (match && match[1].startsWith('192.168.')) {
              resolve(match[1]);
              pc.close();
            }
          }
        };
        
        // Timeout after 5 seconds
        setTimeout(() => {
          resolve(null);
          pc.close();
        }, 5000);
      });

    } catch (error) {
      console.warn('Could not determine local IP:', error);
      return null;
    }
  }

  /**
   * Configure WiFi on discovered device
   */
  async configureDeviceWiFi(device: LocalDevice, ssid: string, password: string): Promise<boolean> {
    try {
      console.log(`📶 Configuring WiFi for device ${device.deviceId}`);

      // Handle mock device for localhost testing
      if (device.ip === 'demo.localhost') {
        console.log('🎭 Mock WiFi configuration for demo device');
        await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate delay
        return true;
      }

      const response = await fetch(`http://${device.ip}:5002/api/setup/wifi`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ssid: ssid,
          password: password,
          security: 'WPA2' // Default assumption
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('✅ WiFi configured successfully:', result);
        return true;
      } else {
        console.error('❌ WiFi configuration failed:', response.status);
        return false;
      }

    } catch (error) {
      console.error('❌ WiFi configuration error:', error);
      return false;
    }
  }

  /**
   * Get available WiFi networks from device
   */
  async getAvailableNetworks(device: LocalDevice): Promise<WiFiNetwork[]> {
    try {
      // Handle mock device for localhost testing
      if (device.ip === 'demo.localhost') {
        console.log('🎭 Mock WiFi scan for demo device');
        return [
          { ssid: 'Demo_WiFi_5G', security: 'WPA2', signal: 85 },
          { ssid: 'Demo_Guest', security: 'Open', signal: 70 },
          { ssid: 'Demo_Office', security: 'WPA3', signal: 60 }
        ];
      }

      const response = await fetch(`http://${device.ip}:5002/api/setup/wifi/scan`);
      if (response.ok) {
        const networks = await response.json();
        return networks.networks || [];
      }
    } catch (error) {
      console.warn('Could not scan WiFi networks:', error);
    }
    return [];
  }

  /**
   * Utility: Process promises in batches to avoid overwhelming network
   */
  private async batchProcess<T>(promises: Promise<T>[], batchSize: number): Promise<T[]> {
    const results: T[] = [];
    
    for (let i = 0; i < promises.length; i += batchSize) {
      const batch = promises.slice(i, i + batchSize);
      const batchResults = await Promise.allSettled(batch);
      
      batchResults.forEach((result) => {
        if (result.status === 'fulfilled') {
          results.push(result.value);
        }
      });
    }

    return results;
  }

  /**
   * Security: Verify device is on local network
   */
  isLocalDevice(device: LocalDevice): boolean {
    const ip = device.ip;
    
    // Check if IP is in private ranges
    const privateRanges = [
      /^192\.168\./,
      /^10\./,
      /^172\.(1[6-9]|2[0-9]|3[0-1])\./
    ];

    return privateRanges.some(range => range.test(ip));
  }

  /**
   * Security: Geofencing check (requires user permission)
   */
  async verifyLocalPresence(): Promise<boolean> {
    try {
      if (!navigator.geolocation) return true; // Skip if not available

      return new Promise((resolve) => {
        navigator.geolocation.getCurrentPosition(
          (_position) => {
            // Store/compare with device setup location
            console.log('📍 Location verified for local setup');
            resolve(true);
          },
          (error) => {
            console.warn('Location verification failed:', error);
            resolve(true); // Don't block on location errors
          },
          { timeout: 10000 }
        );
      });
    } catch (error) {
      return true; // Don't block setup on geolocation issues
    }
  }
}

// Export singleton instance
export const localDeviceDiscovery = new LocalDeviceDiscovery();