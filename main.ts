// photo-geolocator-01 entry for @dirtytesla (make.fun).
// Strategy: ask the allowlisted vision model once per photo (claude-sonnet-5-5
// through the metered gateway, on the entrant's own pushed key). If no key was
// pushed the gateway answers 401 and the call is refused; then, and only then,
// the photo gets the fallback guess below. Every photo always gets an answer,
// so an unanswered photo (20,015 km) never happens. No wall clock, no unseeded
// randomness: every request is a function of the photo bytes and the fixed
// prompt, calls run 4 at a time and answers are posted in photo order, so a
// replay sends the same calls and leaves the same answers.

const MODEL = 'claude-sonnet-5-5';
const MAX_TOKENS = 512;
const CONCURRENCY = 4;
const PROMPT =
  'You are an expert geolocation analyst. Look at this street-level photo and ' +
  'estimate where on Earth it was taken. Use every clue: language and script on ' +
  'signs, road markings and which side of the road traffic drives on, license ' +
  'plates, vegetation and landscape, architecture, sky and light, vehicles. ' +
  'Answer with exactly one line of JSON and nothing else: {"lat": <degrees north>, "lon": <degrees east>}. ' +
  'If you are unsure, still give your single best guess; never refuse and never explain.';
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const anthropicUrl = (process.env.ANTHROPIC_BASE_URL ?? '').replace(/\/+$/, '');
const apiKey = process.env.ANTHROPIC_API_KEY ?? '';
const canCallModel = envUrl !== '' && anthropicUrl !== '' && apiKey !== '';

const sleep = (ms: number): Promise<void> =>
  new Promise((r) => setTimeout(r, ms));

const isLat = (v: unknown): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number =>
  typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

/** Latitude/longitude from a model answer: first JSON object with lat/lon, else first two numbers in range. */
function coordinates(text: string): { lat: number; lon: number } | null {
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = o.lat ?? o.latitude;
      const lon = o.lon ?? o.lng ?? o.longitude;
      if (isLat(lat) && isLon(lon)) return { lat, lon };
    } catch {
      // Not JSON: fall through to the number scan.
    }
  }
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

/** One vision-model call for one photo; null when the gateway refuses it (final: 4xx, incl. 401 with no key pushed). */
async function ask(jpeg: Buffer): Promise<string | null> {
  const body = JSON.stringify({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: { type: 'base64', media_type: 'image/jpeg', data: jpeg.toString('base64') },
          },
          { type: 'text', text: PROMPT },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    let res: Response;
    try {
      res = await fetch(`${anthropicUrl}/v1/messages`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
        },
        body,
      });
    } catch {
      return null;
    }
    if (res.ok) {
      const msg = (await res.json()) as { content?: { type: string; text?: string }[] };
      return (msg.content ?? [])
        .filter((b) => b.type === 'text')
        .map((b) => b.text ?? '')
        .join('\n');
    }
    const text = await res.text();
    if (
      RETRY_STATUSES.includes(res.status) &&
      res.headers.get('x-should-retry') !== 'false' &&
      attempt < 2
    ) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`geolocator: model call refused (HTTP ${res.status}): ${text.slice(0, 200)}\n`);
    return null;
  }
}

/**
 * Model-free fallback, used only when the vision model is unavailable: the
 * single fixed guess that minimises mean error on the public practice pool
 * (75 photos, mean ~6,846 km there, versus ~7,576 km for the pool's plain
 * centroid). Pixel statistics were tried as branches first and measured
 * worse than this fixed guess on the same pool, so they were dropped rather
 * than kept as decoration.
 */
function heuristicGuess(): { lat: number; lon: number } {
  return { lat: 55, lon: 25 };
}

async function main(): Promise<number> {
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  const guesses: (Guess | null)[] = new Array(photos.length).fill(null);
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      const p = photos[i]!;
      const jpeg = Buffer.from(await (await fetch(`${envUrl}${p.url}`)).arrayBuffer());
      if (canCallModel) {
        const text = await ask(jpeg);
        const c = text === null ? null : coordinates(text);
        if (c !== null) {
          guesses[i] = { id: p.id, lat: c.lat, lon: c.lon };
          continue;
        }
        if (text !== null) {
          process.stderr.write(`geolocator: no coordinates in the answer for ${p.id}\n`);
        }
      }
      // Model unavailable, refused, or gave no coordinates: fallback guess.
      const g = heuristicGuess();
      guesses[i] = { id: p.id, lat: g.lat, lon: g.lon };
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
  const answers = guesses.filter((g): g is Guess => g !== null);
  if (answers.length === 0) return 1;
  const res = await fetch(`${envUrl}/answers`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ answers }),
  });
  process.stdout.write(
    `geolocator: ${answers.length} of ${photos.length} photos answered: HTTP ${res.status} ${await res.text()}`,
  );
  return res.ok ? 0 : 1;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`geolocator: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
