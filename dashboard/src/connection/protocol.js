// dashboard/src/connection/protocol.js
/**
 * v1 envelope parsing and v0 fallback detection.
 */

export function isV1Envelope(data) {
  return data && data.v === 1 && typeof data.type === 'string';
}

export function parseEnvelope(raw) {
  try {
    const data = typeof raw === 'string' ? JSON.parse(raw) : raw;
    if (isV1Envelope(data)) {
      return { version: 1, type: data.type, data: data.data, seq: data.seq, ts: data.ts };
    }
    // v0 fallback: legacy single-payload format
    return { version: 0, type: 'legacy', data: data };
  } catch (e) {
    return null;
  }
}
