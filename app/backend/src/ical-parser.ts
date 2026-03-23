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
 * Checks if a URL is a Google Calendar share link and scrapes event details from it.
 */
async function parseGoogleShareLink(url: string): Promise<ParsedEvent | null> {
  try {
    // Follow redirects to get the share page
    let shareUrl = url;
    if (url.includes('calendar.app.google')) {
      const redirectRes = await fetch(url, { redirect: 'follow', signal: AbortSignal.timeout(5000) });
      shareUrl = redirectRes.url;
    }

    const res = await fetch(shareUrl, { signal: AbortSignal.timeout(5000) });
    const html = await res.text();

    // Extract event data from the share page HTML
    // Google embeds event details in structured data and meta tags
    const titleMatch = html.match(/<meta\s+property="og:title"\s+content="([^"]+)"/i)
      || html.match(/<title>([^<]+)<\/title>/i);
    const descMatch = html.match(/<meta\s+property="og:description"\s+content="([^"]+)"/i);

    // Try to extract from the page content — Google share pages have event info in text
    const nameFromPage = titleMatch?.[1]?.replace(/ - Google Calendar$/, '').trim();

    if (!nameFromPage) {
      // Try scraping visible text patterns
      const eventNameMatch = html.match(/data-eventchip[^>]*>([^<]+)</i)
        || html.match(/"title":"([^"]+)"/i)
        || html.match(/"summary":"([^"]+)"/i);

      if (!eventNameMatch) return null;
    }

    // Extract date/time — look for structured data
    const startMatch = html.match(/"startDate":"([^"]+)"/i)
      || html.match(/data-start[^"]*"(\d+)"/i);
    const endMatch = html.match(/"endDate":"([^"]+)"/i)
      || html.match(/data-end[^"]*"(\d+)"/i);
    const locationMatch = html.match(/"location":"([^"]+)"/i);

    // Parse dates from various formats Google might use
    let startTime: number | undefined;
    let endTime: number | undefined;

    if (startMatch?.[1]) {
      const parsed = new Date(startMatch[1]).getTime();
      if (!isNaN(parsed)) startTime = parsed;
      // Could be Unix ms
      else if (/^\d+$/.test(startMatch[1])) startTime = parseInt(startMatch[1]);
    }
    if (endMatch?.[1]) {
      const parsed = new Date(endMatch[1]).getTime();
      if (!isNaN(parsed)) endTime = parsed;
      else if (/^\d+$/.test(endMatch[1])) endTime = parseInt(endMatch[1]);
    }

    const name = nameFromPage || 'Untitled Event';
    const description = descMatch?.[1] || undefined;
    const location = locationMatch?.[1] || undefined;

    return {
      name,
      description,
      startTime: startTime || Date.now(),
      endTime,
      location,
    };
  } catch (err) {
    console.error('[ical-parser] Failed to parse Google share link:', url, err);
    return null;
  }
}

/**
 * Checks if URL is a Google Calendar event template link and extracts data from it.
 * Format: https://calendar.google.com/calendar/event?action=TEMPLATE&tmeid=...&tmsrc=...
 */
async function parseGoogleEventLink(url: string): Promise<ParsedEvent | null> {
  // These template links don't have fetchable event data without auth
  // But we can try the share page approach
  return parseGoogleShareLink(url);
}

/**
 * Fetches and parses event data from any supported URL format:
 * - .ics URLs (iCal feeds)
 * - Google Calendar share links (calendar.app.google/...)
 * - Google Calendar event links
 * Returns null on any failure — never throws.
 */
export async function fetchAndParseIcal(url: string): Promise<ParsedEvent | null> {
  // Detect URL type and route to appropriate parser
  if (url.includes('calendar.app.google') || url.includes('/calendar/share')) {
    return parseGoogleShareLink(url);
  }

  if (url.includes('calendar.google.com/calendar/event')) {
    return parseGoogleEventLink(url);
  }

  // Default: treat as ICS feed
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
