import crypto from 'crypto';

export function generateApiKey(): string {
  const random = crypto.randomBytes(16).toString('hex');
  return `mm_sk_${random}`;
}

export function hashApiKey(key: string): string {
  return crypto.createHash('sha256').update(key).digest('hex');
}

export function getKeyPrefix(key: string): string {
  return key.slice(0, 11) + '...';
}
