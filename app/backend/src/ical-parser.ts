import ical, { VEvent } from 'node-ical';

export interface ParsedEvent {
  name: string;
  description?: string;
  startTime: number;
  endTime?: number;
  location?: string;
}

/** Extract string value from node-ical ParameterValue (can be string or {val, params}) */
function paramStr(val: unknown): string | undefined {
  if (!val) return undefined;
  if (typeof val === 'string') return val;
  if (typeof val === 'object' && val !== null && 'val' in val) return String((val as any).val);
  return String(val);
}

/**
 * Fetches and parses an iCal/ICS URL, returning the next upcoming or currently-active event.
 * Returns null on any failure — never throws.
 */
export async function fetchAndParseIcal(url: string): Promise<ParsedEvent | null> {
  try {
    const data = await ical.async.fromURL(url);
    const now = Date.now();

    let bestEvent: ParsedEvent | null = null;
    let bestScore = Infinity;

    for (const key of Object.keys(data)) {
      const item = data[key];
      if (!item || item.type !== 'VEVENT') continue;

      const vevent = item as VEvent;
      const start = new Date(vevent.start).getTime();
      const end = vevent.end ? new Date(vevent.end).getTime() : undefined;

      // Skip events that ended more than 24h ago
      if (end && end < now - 86400000) continue;

      // Prefer currently-active events, then nearest upcoming
      let score: number;
      if (end && now >= start && now <= end) {
        score = -1;
      } else if (start >= now) {
        score = start - now;
      } else {
        score = 1e15;
      }

      if (score < bestScore) {
        bestScore = score;
        bestEvent = {
          name: paramStr(vevent.summary) || 'Untitled Event',
          description: paramStr(vevent.description),
          startTime: start,
          endTime: end,
          location: paramStr(vevent.location),
        };
      }
    }

    return bestEvent;
  } catch (err) {
    console.error('[ical-parser] Failed to fetch/parse iCal URL:', url, err);
    return null;
  }
}
