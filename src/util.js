// util.js — hashing, seeded RNG, varints, stats. Zero deps, deterministic.

import { createHash } from 'node:crypto';

// FNV-1a 32-bit over a JS string's UTF-16 code units (deterministic, fast).
export function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

// mulberry32 — tiny seeded PRNG, returns floats in [0, 1).
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sha256(buf) {
  return createHash('sha256').update(buf).digest('hex');
}

// L2-normalize in place; the zero vector stays zero (documented: it embeds
// nothing, which is honest — its distance math still works).
export function l2normalize(v) {
  let sum = 0;
  for (let i = 0; i < v.length; i += 1) sum += v[i] * v[i];
  if (sum === 0) return v;
  const inv = 1 / Math.sqrt(sum);
  for (let i = 0; i < v.length; i += 1) v[i] *= inv;
  return v;
}

export function isFiniteVector(v, dims) {
  if (!Array.isArray(v) && !ArrayBuffer.isView(v)) return false;
  if (v.length !== dims) return false;
  for (let i = 0; i < v.length; i += 1) {
    if (!Number.isFinite(v[i])) return false;
  }
  return true;
}

// Nearest-rank percentile over a pre-sorted ascending array.
export function percentile(sortedAsc, p) {
  if (sortedAsc.length === 0) return 0;
  const idx = Math.min(
    sortedAsc.length - 1,
    Math.max(0, Math.ceil((p / 100) * sortedAsc.length) - 1),
  );
  return sortedAsc[idx];
}

// LEB128 varints into a plain byte array (build-friendly, no alloc churn).
export function varintEncode(value, out) {
  let v = value;
  if (!Number.isInteger(v) || v < 0) throw new RangeError(`varint: bad value ${value}`);
  while (v >= 0x80) {
    out.push((v & 0x7f) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v);
  return out;
}

// Decode one varint; returns [value, nextPos].
export function varintDecode(buf, pos) {
  let value = 0;
  let shift = 1;
  let p = pos;
  for (;;) {
    if (p >= buf.length) throw new RangeError('varint: truncated');
    const byte = buf[p];
    p += 1;
    value += (byte & 0x7f) * shift;
    if ((byte & 0x80) === 0) break;
    shift *= 128;
    if (shift > 2 ** 53) throw new RangeError('varint: overflow');
  }
  return [value, p];
}

// Stable top-k by (score desc, id asc) — deterministic tie-breaking everywhere.
export function topK(items, k, scoreOf = (x) => x.score, idOf = (x) => x.id) {
  const sorted = items.slice().sort((a, b) => {
    const ds = scoreOf(b) - scoreOf(a);
    if (ds !== 0) return ds;
    return idOf(a) - idOf(b);
  });
  return sorted.slice(0, k);
}

export function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export const DEFAULT_CLOCK = () => Date.now();
export const DEFAULT_HRTIME = () => Number(process.hrtime.bigint()) / 1e6; // ms
export const DEFAULT_MEMORY = () => process.memoryUsage();
