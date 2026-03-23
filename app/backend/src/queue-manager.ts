// Queue manager — camera-native sign-up/booking system.
// Follows camera-state.ts pattern: module with exported functions, no class.

import { Server } from 'socket.io';
import { randomUUID } from 'crypto';
import {
  getQueueConfig,
  saveQueueConfig,
  getQueueEntries,
  getActiveQueueEntry,
  insertQueueEntry,
  updateQueueEntryStatus,
  reorderQueuePositions,
  getNextQueuePosition,
  getQueueEntryByWallet,
  getUserProfile,
  getUserProfiles,
  getAllActiveQueueEntries,
  QueueConfig,
  QueueEntry,
} from './database';

// Active slot timers: cameraId -> timeout handle
const activeTimers: Map<string, NodeJS.Timeout> = new Map();

let io: Server;

// ── Public API ──────────────────────────────────────────────────────────────

export async function initQueueManager(socketServer: Server): Promise<void> {
  io = socketServer;

  // Recover any active slots that were running when the server restarted
  const activeEntries = await getAllActiveQueueEntries();
  for (const entry of activeEntries) {
    if (!entry.expiresAt) {
      // Malformed — advance past it
      await advanceQueue(entry.cameraId);
      continue;
    }

    const remaining = entry.expiresAt - Date.now();
    if (remaining <= 0) {
      // Expired while server was down
      console.log(`[Queue] Recovering expired slot for camera ${entry.cameraId}`);
      await advanceQueue(entry.cameraId);
    } else {
      // Still valid — set timer for remaining time
      console.log(`[Queue] Restoring timer for camera ${entry.cameraId} (${Math.round(remaining / 1000)}s remaining)`);
      scheduleAdvance(entry.cameraId, remaining);
    }
  }

  console.log(`[Queue] Manager initialized, recovered ${activeEntries.length} active slot(s)`);
}

export async function joinQueue(
  cameraId: string,
  walletAddress: string,
  duration: number,
  displayName?: string,
  title?: string
): Promise<QueueEntry> {
  // Validate against config
  const config = await getQueueConfig(cameraId);
  if (!config.enabled) {
    throw new Error('Queue is not enabled for this camera');
  }
  if (duration < config.minSlotDuration || duration > config.maxSlotDuration) {
    throw new Error(`Duration must be between ${config.minSlotDuration} and ${config.maxSlotDuration} seconds`);
  }

  // Check for existing entry
  const existing = await getQueueEntryByWallet(cameraId, walletAddress);
  if (existing) {
    throw new Error('Already in queue');
  }

  // Resolve display name from user profile if not provided
  let resolvedName = displayName || null;
  if (!resolvedName) {
    try {
      const profile = await getUserProfile(walletAddress);
      resolvedName = profile?.displayName || profile?.username || null;
    } catch {
      // Profile lookup is best-effort
    }
  }

  const position = await getNextQueuePosition(cameraId);
  const activeSlot = await getActiveQueueEntry(cameraId);

  const entry: QueueEntry = {
    id: randomUUID(),
    cameraId,
    walletAddress,
    displayName: resolvedName,
    title: title || null,
    requestedDuration: duration,
    position,
    status: 'waiting',
    joinedAt: Date.now(),
    startedAt: null,
    expiresAt: null,
    completedAt: null,
  };

  await insertQueueEntry(entry);

  // If no one is active, immediately activate this entry
  if (!activeSlot) {
    await activateEntry(entry);
  } else {
    await broadcastQueueUpdate(cameraId);
  }

  // Return the entry with its current state
  return await getQueueEntryByWallet(cameraId, walletAddress) || entry;
}

export async function leaveQueue(cameraId: string, walletAddress: string): Promise<boolean> {
  const entry = await getQueueEntryByWallet(cameraId, walletAddress);
  if (!entry) return false;

  const wasActive = entry.status === 'active';

  await updateQueueEntryStatus(entry.id, 'left', { completedAt: Date.now() });

  if (wasActive) {
    clearTimer(cameraId);
    await advanceQueue(cameraId);
  } else {
    await reorderQueuePositions(cameraId);
    await broadcastQueueUpdate(cameraId);
  }

  return true;
}

export async function skipCurrent(cameraId: string): Promise<boolean> {
  const active = await getActiveQueueEntry(cameraId);
  if (!active) return false;

  clearTimer(cameraId);
  await updateQueueEntryStatus(active.id, 'completed', { completedAt: Date.now() });
  await advanceQueue(cameraId);
  return true;
}

export async function getQueueState(cameraId: string): Promise<{
  config: { enabled: boolean; maxSlotDuration: number; minSlotDuration: number };
  active: (QueueEntry & { remainingSeconds: number }) | null;
  queue: QueueEntry[];
  totalWaiting: number;
}> {
  const config = await getQueueConfig(cameraId);
  const active = await getActiveQueueEntry(cameraId);
  const waiting = await getQueueEntries(cameraId, 'waiting');

  // Batch-resolve display names from user profiles
  const allEntries = [...(active ? [active] : []), ...waiting];
  const wallets = allEntries.map(e => e.walletAddress);
  let profiles = new Map<string, { displayName?: string; username?: string; profileImage?: string }>();
  if (wallets.length > 0) {
    try {
      profiles = await getUserProfiles(wallets);
    } catch {
      // Best-effort
    }
  }

  const enrichEntry = (entry: QueueEntry) => {
    const profile = profiles.get(entry.walletAddress);
    return {
      ...entry,
      displayName: entry.displayName || profile?.displayName || profile?.username || null,
      profileImage: profile?.profileImage || null,
    };
  };

  let activeWithCountdown: (QueueEntry & { remainingSeconds: number; profileImage: string | null }) | null = null;
  if (active && active.expiresAt) {
    activeWithCountdown = {
      ...enrichEntry(active),
      remainingSeconds: Math.max(0, Math.round((active.expiresAt - Date.now()) / 1000)),
    };
  }

  return {
    config: {
      enabled: config.enabled,
      maxSlotDuration: config.maxSlotDuration,
      minSlotDuration: config.minSlotDuration,
      location: config.location,
    },
    active: activeWithCountdown,
    queue: waiting.map(enrichEntry),
    totalWaiting: waiting.length,
  };
}

export async function updateConfig(
  cameraId: string,
  updates: { enabled?: boolean; maxSlotDuration?: number; minSlotDuration?: number; location?: string | null }
): Promise<QueueConfig> {
  const existing = await getQueueConfig(cameraId);
  const config: QueueConfig = {
    cameraId,
    enabled: updates.enabled ?? existing.enabled,
    maxSlotDuration: updates.maxSlotDuration ?? existing.maxSlotDuration,
    minSlotDuration: updates.minSlotDuration ?? existing.minSlotDuration,
    location: updates.location !== undefined ? updates.location : existing.location,
    updatedAt: Date.now(),
  };

  await saveQueueConfig(config);
  await broadcastQueueUpdate(cameraId);
  return config;
}

// ── Internal helpers ────────────────────────────────────────────────────────

async function activateEntry(entry: QueueEntry): Promise<void> {
  const now = Date.now();
  const expiresAt = now + entry.requestedDuration * 1000;

  await updateQueueEntryStatus(entry.id, 'active', { startedAt: now, expiresAt });
  scheduleAdvance(entry.cameraId, entry.requestedDuration * 1000);
  await broadcastQueueUpdate(entry.cameraId);
}

async function advanceQueue(cameraId: string): Promise<void> {
  // Mark any current active as completed (if not already)
  const current = await getActiveQueueEntry(cameraId);
  if (current) {
    await updateQueueEntryStatus(current.id, 'completed', { completedAt: Date.now() });
  }

  // Reorder and find next
  await reorderQueuePositions(cameraId);
  const waiting = await getQueueEntries(cameraId, 'waiting');

  if (waiting.length > 0) {
    await activateEntry(waiting[0]);
  } else {
    // Queue is idle — no timer needed
    await broadcastQueueUpdate(cameraId);
  }
}

function scheduleAdvance(cameraId: string, delayMs: number): void {
  clearTimer(cameraId);
  const timer = setTimeout(async () => {
    activeTimers.delete(cameraId);
    try {
      await advanceQueue(cameraId);
    } catch (err) {
      console.error(`[Queue] Error advancing queue for camera ${cameraId}:`, err);
    }
  }, delayMs);

  activeTimers.set(cameraId, timer);
}

function clearTimer(cameraId: string): void {
  const existing = activeTimers.get(cameraId);
  if (existing) {
    clearTimeout(existing);
    activeTimers.delete(cameraId);
  }
}

async function broadcastQueueUpdate(cameraId: string): Promise<void> {
  if (!io) return;
  try {
    const state = await getQueueState(cameraId);
    io.to(cameraId).emit('queueUpdate', state);
  } catch (err) {
    console.error(`[Queue] Error broadcasting update for ${cameraId}:`, err);
  }
}
