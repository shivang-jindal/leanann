// quantize.js — product quantization: M subspaces, seeded deterministic
// k-means codebooks trained on a reservoir sample of YOUR vectors, Uint8
// codes, and asymmetric distance (ADC) lookup tables per query.
//
// dims must be divisible by M (validated at build, honest error otherwise).

import { mulberry32 } from './util.js';

export function requireDims(dims, m) {
  if (!Number.isInteger(dims) || dims < 2) throw new RangeError(`bad dims ${dims}`);
  if (!Number.isInteger(m) || m < 1) throw new RangeError(`bad --m ${m}`);
  if (dims % m !== 0) {
    throw new RangeError(`--dims ${dims} is not divisible by --m ${m} — pick m that divides dims`);
  }
  return dims / m;
}

// Squared euclidean distance between two equal-length float arrays.
export function sqDist(a, b, from = 0, to = a.length) {
  let sum = 0;
  for (let i = from; i < to; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

// Seeded k-means++ init + Lloyd iterations.
// points: array of Float32Array(sub). Returns Float32Array(k × sub).
export function kmeans(points, k, sub, { seed = 42, iters = 12 } = {}) {
  if (points.length === 0) throw new RangeError('kmeans: no points');
  if (!Number.isInteger(k) || k < 1 || k > 256) throw new RangeError(`k-means k must be 1..256, got ${k}`);
  const rng = mulberry32(seed);
  const n = points.length;
  const centroids = new Array(k);

  // k-means++ seeding (deterministic via seeded rng).
  centroids[0] = Float32Array.from(points[Math.floor(rng() * n)]);
  const d2 = new Float64Array(n).fill(Infinity);
  for (let c = 1; c < k; c += 1) {
    const prev = centroids[c - 1];
    let total = 0;
    for (let i = 0; i < n; i += 1) {
      const dist = sqDist(points[i], prev);
      if (dist < d2[i]) d2[i] = dist;
      total += d2[i];
    }
    if (total === 0) {
      centroids[c] = Float32Array.from(points[Math.floor(rng() * n)]);
      continue;
    }
    let target = rng() * total;
    let pick = n - 1;
    for (let i = 0; i < n; i += 1) {
      target -= d2[i];
      if (target <= 0) { pick = i; break; }
    }
    centroids[c] = Float32Array.from(points[pick]);
  }

  const assign = new Int32Array(n).fill(-1);
  for (let iter = 0; iter < iters; iter += 1) {
    let changed = 0;
    for (let i = 0; i < n; i += 1) {
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c += 1) {
        const d = sqDist(points[i], centroids[c]);
        if (d < bestD) { bestD = d; best = c; }
      }
      if (assign[i] !== best) { assign[i] = best; changed += 1; }
    }
    // recompute means
    const sums = new Float64Array(k * sub);
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i += 1) {
      const c = assign[i];
      counts[c] += 1;
      const base = c * sub;
      for (let s = 0; s < sub; s += 1) sums[base + s] += points[i][s];
    }
    for (let c = 0; c < k; c += 1) {
      if (counts[c] === 0) {
        // empty cluster: reseed to a random point (deterministic via rng order)
        centroids[c] = Float32Array.from(points[Math.floor(rng() * n)]);
        continue;
      }
      const base = c * sub;
      const next = new Float32Array(sub);
      for (let s = 0; s < sub; s += 1) next[s] = sums[base + s] / counts[c];
      centroids[c] = next;
    }
    if (changed === 0) break;
  }

  const flat = new Float32Array(k * sub);
  for (let c = 0; c < k; c += 1) flat.set(centroids[c], c * sub);
  return flat;
}

// Train M codebooks from a sample of full vectors.
// Returns Float32Array(M × k × sub) flattened as [m][c][s].
export function trainCodebooks(vectors, { dims, m, k = 256, seed = 42, iters = 12 } = {}) {
  const sub = requireDims(dims, m);
  if (vectors.length === 0) throw new RangeError('train: no vectors');
  const books = new Float32Array(m * k * sub);
  const bucket = new Array(vectors.length);
  for (let mi = 0; mi < m; mi += 1) {
    for (let i = 0; i < vectors.length; i += 1) {
      const v = vectors[i];
      bucket[i] = v.subarray(mi * sub, (mi + 1) * sub);
    }
    // Distinct but derived seeds keep subspaces independent yet reproducible.
    const flat = kmeans(bucket, k, sub, { seed: (seed + mi * 0x9e3779b1) >>> 0, iters });
    books.set(flat, mi * k * sub);
  }
  return books;
}

// Encode one vector -> Uint8Array(m).
export function encodeVector(v, books, { dims, m, k = 256 } = {}) {
  const sub = requireDims(dims, m);
  const codes = new Uint8Array(m);
  for (let mi = 0; mi < m; mi += 1) {
    const off = mi * sub;
    let best = 0;
    let bestD = Infinity;
    for (let c = 0; c < k; c += 1) {
      const base = mi * k * sub + c * sub;
      let d = 0;
      for (let s = 0; s < sub; s += 1) {
        const diff = v[off + s] - books[base + s];
        d += diff * diff;
      }
      if (d < bestD) { bestD = d; best = c; }
    }
    codes[mi] = best;
  }
  return codes;
}

// ADC tables for a FLOAT query: T[mi * k + c] = dist(querySub_mi, centroid).
export function queryTables(q, books, { dims, m, k = 256 } = {}) {
  const sub = requireDims(dims, m);
  const tables = new Float32Array(m * k);
  for (let mi = 0; mi < m; mi += 1) {
    const off = mi * sub;
    for (let c = 0; c < k; c += 1) {
      const base = mi * k * sub + c * sub;
      let d = 0;
      for (let s = 0; s < sub; s += 1) {
        const diff = q[off + s] - books[base + s];
        d += diff * diff;
      }
      tables[mi * k + c] = d;
    }
  }
  return tables;
}

// Distance from ADC tables to a code.
export function adcDist(tables, codes, k) {
  let sum = 0;
  for (let mi = 0; mi < codes.length; mi += 1) sum += tables[mi * k + codes[mi]];
  return sum;
}

// Node-vs-node distance: tables built from one node's CODES (quantized-quantized).
export function codeTables(codes, books, { dims, m, k = 256 } = {}) {
  const sub = requireDims(dims, m);
  const tables = new Float32Array(m * k);
  for (let mi = 0; mi < m; mi += 1) {
    const src = mi * k * sub + codes[mi] * sub;
    for (let c = 0; c < k; c += 1) {
      const base = mi * k * sub + c * sub;
      let d = 0;
      for (let s = 0; s < sub; s += 1) {
        const diff = books[src + s] - books[base + s];
        d += diff * diff;
      }
      tables[mi * k + c] = d;
    }
  }
  return tables;
}
