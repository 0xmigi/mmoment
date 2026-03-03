import { Request, Response, NextFunction } from 'express';
import { hashApiKey } from '../keys/api-key-gen';
import { getApiKeyByHash, updateApiKeyLastUsed } from '../../database';

export interface AuthenticatedRequest extends Request {
  apiKey?: {
    id: string;
    walletAddress: string;
    permissions: string[];
    rateLimit: number;
  };
}

export async function apiKeyAuth(
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer mm_sk_')) {
    return res.status(401).json({
      error: 'unauthorized',
      message: 'Missing or invalid API key. Use: Authorization: Bearer mm_sk_...'
    });
  }

  const rawKey = authHeader.slice(7); // strip "Bearer "
  const keyHash = hashApiKey(rawKey);

  try {
    const record = await getApiKeyByHash(keyHash);

    if (!record) {
      return res.status(401).json({
        error: 'unauthorized',
        message: 'Invalid API key'
      });
    }

    if (record.revokedAt) {
      return res.status(401).json({
        error: 'unauthorized',
        message: 'API key has been revoked'
      });
    }

    req.apiKey = {
      id: record.id,
      walletAddress: record.walletAddress,
      permissions: JSON.parse(record.permissions),
      rateLimit: record.rateLimit,
    };

    // Fire-and-forget last_used_at update
    updateApiKeyLastUsed(keyHash).catch(() => {});

    next();
  } catch (err) {
    console.error('[API Auth] Error validating key:', err);
    return res.status(500).json({ error: 'internal', message: 'Auth validation failed' });
  }
}
