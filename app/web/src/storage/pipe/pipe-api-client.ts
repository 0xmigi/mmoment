/**
 * Pipe Network API Client for MMOMENT Camera System
 * 
 * Core SDK implementation using verified working endpoints
 */

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  expires_at?: Date;
}

export interface PipeUser {
  id: string;
  username: string;
  email?: string;
  app_key: string;
}

export interface UploadOptions {
  encrypted?: boolean;
  password?: string;
  quantum?: boolean;
}

export interface PublicLinkResponse {
  link: string;
  expires_at?: string;
}

export interface TokenUsageResponse {
  total_used: number;
  total_allocated: number;
  percentage_used: number;
}

export interface TierPricingResponse {
  tiers: Array<{
    name: string;
    storage_gb: number;
    price_tokens: number;
    features: string[];
  }>;
}

export class PipeAPIClient {
  private baseUrl = 'https://us-west-00-firestarter.pipenetwork.com';
  private authTokens: AuthTokens | null = null;
  private userId: string | null = null;
  private userAppKey: string | null = null;

  constructor(baseUrl?: string) {
    if (baseUrl) {
      this.baseUrl = baseUrl;
    }
  }

  /**
   * Register a new user account
   */
  async register(
    username: string,
    email: string,
    password: string
  ): Promise<PipeUser> {
    const response = await fetch(`${this.baseUrl}/users`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username,
        email,
        password,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Registration failed: ${error}`);
    }

    const user = await response.json();
    this.userId = user.id;
    this.userAppKey = user.app_key;
    return user;
  }

  /**
   * Login with username/email and password
   */
  async login(usernameOrEmail: string, password: string): Promise<AuthTokens> {
    const response = await fetch(`${this.baseUrl}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: usernameOrEmail,
        password,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Login failed: ${error}`);
    }

    const tokens = await response.json();
    this.authTokens = {
      ...tokens,
      expires_at: new Date(Date.now() + tokens.expires_in * 1000),
    };
    
    // Extract user ID from response if available
    if (tokens.user_id) {
      this.userId = tokens.user_id;
    }
    if (tokens.user_app_key) {
      this.userAppKey = tokens.user_app_key;
    }

    return this.authTokens!;
  }

  /**
   * Login with user_id and user_app_key (legacy auth)
   */
  async loginWithAppKey(userId: string, userAppKey: string): Promise<boolean> {
    this.userId = userId;
    this.userAppKey = userAppKey;
    
    // Test credentials by checking wallet
    try {
      await this.checkWallet();
      return true;
    } catch {
      this.userId = null;
      this.userAppKey = null;
      return false;
    }
  }

  /**
   * Refresh authentication tokens
   */
  async refreshTokens(): Promise<AuthTokens> {
    if (!this.authTokens?.refresh_token) {
      throw new Error('No refresh token available');
    }

    const response = await fetch(`${this.baseUrl}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.authTokens.refresh_token}`,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Token refresh failed: ${error}`);
    }

    const tokens = await response.json();
    this.authTokens = {
      ...tokens,
      expires_at: new Date(Date.now() + tokens.expires_in * 1000),
    };

    return this.authTokens!;
  }

  /**
   * Reset password
   */
  async resetPassword(email: string): Promise<boolean> {
    const response = await fetch(`${this.baseUrl}/auth/reset-password`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email }),
    });

    return response.ok;
  }

  /**
   * Set new password (after reset)
   */
  async setPassword(token: string, newPassword: string): Promise<boolean> {
    const response = await fetch(`${this.baseUrl}/auth/set-password`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        token,
        new_password: newPassword,
      }),
    });

    return response.ok;
  }

  /**
   * Upload a file with priority upload endpoint
   */
  async uploadFile(
    file: File | Blob,
    filename: string,
    options?: UploadOptions
  ): Promise<string> {
    const formData = new FormData();
    formData.append('file', file, filename);

    if (options?.encrypted) {
      formData.append('encrypted', 'true');
      if (options.password) {
        formData.append('password', options.password);
      }
    }

    if (options?.quantum) {
      formData.append('quantum', 'true');
    }

    const headers = this.getAuthHeaders();
    const response = await fetch(`${this.baseUrl}/priorityUpload`, {
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Upload failed: ${error}`);
    }

    const result = await response.text();
    return result; // Returns the file ID/name
  }

  /**
   * Download a file
   */
  async downloadFile(
    fileName: string,
    password?: string
  ): Promise<Blob> {
    const url = new URL(`${this.baseUrl}/download-stream`);
    
    // Add auth params
    if (this.userId && this.userAppKey) {
      url.searchParams.append('user_id', this.userId);
      url.searchParams.append('user_app_key', this.userAppKey);
    }
    
    url.searchParams.append('file_name', fileName);
    
    if (password) {
      url.searchParams.append('password', password);
    }

    const headers = this.getAuthHeaders();
    const response = await fetch(url.toString(), {
      method: 'GET',
      headers,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Download failed: ${error}`);
    }

    return response.blob();
  }

  /**
   * Check wallet balance (SOL)
   */
  async checkWallet(): Promise<{ balance_sol: number; wallet_address: string }> {
    const headers = this.getAuthHeaders();
    const response = await fetch(`${this.baseUrl}/checkWallet`, {
      method: 'POST',
      headers: {
        ...headers,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Wallet check failed: ${error}`);
    }

    return response.json();
  }

  /**
   * Check custom token balance (e.g., PIPE tokens)
   */
  async checkCustomToken(
    tokenMint: string = '35mhJor7qTD212YXdLkB8sRzTbaYRXmTzHTCFSDP5voJ' // PIPE token mint
  ): Promise<{ ui_amount: number; raw_amount: string }> {
    const headers = this.getAuthHeaders();
    const response = await fetch(`${this.baseUrl}/checkCustomToken`, {
      method: 'POST',
      headers: {
        ...headers,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        token_mint: tokenMint,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Token check failed: ${error}`);
    }

    return response.json();
  }

  /**
   * Exchange SOL for PIPE tokens
   */
  async exchangeSolForTokens(
    solAmount: number
  ): Promise<{ tokens_minted: number; transaction_hash: string }> {
    const headers = this.getAuthHeaders();
    const response = await fetch(`${this.baseUrl}/exchangeSolForTokens`, {
      method: 'POST',
      headers: {
        ...headers,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        amount_sol: solAmount,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Token exchange failed: ${error}`);
    }

    return response.json();
  }

  /**
   * Withdraw SOL from account
   */
  async withdrawSol(
    amount: number,
    destinationAddress: string
  ): Promise<{ transaction_hash: string; amount_withdrawn: number }> {
    const headers = this.getAuthHeaders();
    const response = await fetch(`${this.baseUrl}/withdrawSol`, {
      method: 'POST',
      headers: {
        ...headers,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        amount,
        destination_address: destinationAddress,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Withdrawal failed: ${error}`);
    }

    return response.json();
  }

  /**
   * Get token usage statistics
   */
  async getTokenUsage(): Promise<TokenUsageResponse> {
    const headers = this.getAuthHeaders();
    const response = await fetch(`${this.baseUrl}/api/token-usage`, {
      method: 'GET',
      headers,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to get token usage: ${error}`);
    }

    return response.json();
  }

  /**
   * Get tier pricing information
   */
  async getTierPricing(): Promise<TierPricingResponse> {
    const response = await fetch(`${this.baseUrl}/getTierPricing`, {
      method: 'GET',
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to get tier pricing: ${error}`);
    }

    return response.json();
  }

  /**
   * Create a public download link for a file
   */
  async createPublicLink(
    fileName: string,
    expiresIn?: number
  ): Promise<PublicLinkResponse> {
    const headers = this.getAuthHeaders();
    const response = await fetch(`${this.baseUrl}/createPublicLink`, {
      method: 'POST',
      headers: {
        ...headers,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        file_name: fileName,
        expires_in: expiresIn,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to create public link: ${error}`);
    }

    return response.json();
  }

  /**
   * Delete a public link
   */
  async deletePublicLink(linkId: string): Promise<boolean> {
    const headers = this.getAuthHeaders();
    const response = await fetch(`${this.baseUrl}/deletePublicLink`, {
      method: 'POST',
      headers: {
        ...headers,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        link_id: linkId,
      }),
    });

    return response.ok;
  }

  /**
   * Helper to get authentication headers
   */
  private getAuthHeaders(): Record<string, string> {
    const headers: Record<string, string> = {};

    // JWT auth takes precedence
    if (this.authTokens?.access_token) {
      headers['Authorization'] = `Bearer ${this.authTokens.access_token}`;
    } 
    // Fall back to legacy auth
    else if (this.userId && this.userAppKey) {
      headers['X-User-Id'] = this.userId;
      headers['X-User-App-Key'] = this.userAppKey;
    }

    return headers;
  }

  /**
   * Check if tokens need refresh
   */
  shouldRefreshTokens(): boolean {
    if (!this.authTokens?.expires_at) {
      return false;
    }

    // Refresh if less than 5 minutes remaining
    const fiveMinutes = 5 * 60 * 1000;
    return this.authTokens.expires_at.getTime() - Date.now() < fiveMinutes;
  }

  /**
   * Get current authentication status
   */
  isAuthenticated(): boolean {
    return !!(
      (this.authTokens?.access_token && !this.shouldRefreshTokens()) ||
      (this.userId && this.userAppKey)
    );
  }

  /**
   * Clear authentication
   */
  logout(): void {
    this.authTokens = null;
    this.userId = null;
    this.userAppKey = null;
  }
}

