// Product quantization tests — seeded determinism, hand-checkable centroids,
// and ADC tables that must agree with direct subspace distance math.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  requireDims, kmeans, trainCodebooks, encodeVector, queryTables, adcDist, codeTables,
} from '../src/quantize.js';
import { mulberry32 } from '../src/util.js';

test('requireDims rejects non-divisible dims with an honest message', () => {
  assert.equal(requireDims(384, 8), 48);
  assert.throws(() => requireDims(10, 3), /not divisible/);
  assert.throws(() => requireDims(10, 3), /--m 3/);
  assert.throws(() => requireDims(1, 1), /bad dims/);
  assert.throws(() => requireDims(8, 0), /bad --m/);
});

test('kmeans is deterministic for a fixed seed', () => {
  const rng = mulberry32(1);
  const points = Array.from({ length: 40 }, () => {
    const v = new Float32Array(2);
    v[0] = rng() * 10;
    v[1] = rng() * 10;
    return v;
  });
  const a = kmeans(points, 4, 2, { seed: 7, iters: 6 });
  const b = kmeans(points, 4, 2, { seed: 7, iters: 6 });
  assert.deepEqual(Array.from(a), Array.from(b));
});

test('kmeans converges to the true centers on a separated mixture', () => {
  const points = [];
  for (let i = 0; i < 20; i += 1) {
    points.push(Float32Array.from([0, 0]));
    points.push(Float32Array.from([10, 10]));
  }
  const c = kmeans(points, 2, 2, { seed: 3, iters: 10 });
  const c0 = [c[0], c[1]];
  const c1 = [c[2], c[3]];
  const near = (a, x, y) => Math.hypot(a[0] - x, a[1] - y) < 1e-3;
  assert.ok((near(c0, 0, 0) && near(c1, 10, 10)) || (near(c0, 10, 10) && near(c1, 0, 0)));
});

test('kmeans validates k range and emptiness', () => {
  const pts = [Float32Array.from([1, 2])];
  assert.throws(() => kmeans([], 2, 2), /no points/);
  assert.throws(() => kmeans(pts, 0, 2), /1\.\.256/);
  assert.throws(() => kmeans(pts, 257, 2), /1\.\.256/);
});

test('trainCodebooks shape: m blocks of k x sub', () => {
  const vectors = [Float32Array.from([1, 2, 3, 4]), Float32Array.from([5, 6, 7, 8])];
  const books = trainCodebooks(vectors, { dims: 4, m: 2, k: 8, seed: 1, iters: 3 });
  assert.equal(books.length, 2 * 8 * 2); // m * k * sub
});

test('encodeVector picks the closest centroid per subspace (hand-checked)', () => {
  // m=1, k=2, sub=2: centroids (0,0) and (10,0)
  const books = Float32Array.from([0, 0, 10, 0]);
  const cfg = { dims: 2, m: 1, k: 2 };
  assert.equal(encodeVector(Float32Array.from([1, 0]), books, cfg)[0], 0);
  assert.equal(encodeVector(Float32Array.from([9, 0]), books, cfg)[0], 1);
  assert.equal(encodeVector(Float32Array.from([4.9, 0]), books, cfg)[0], 0);
  assert.equal(encodeVector(Float32Array.from([5.1, 0]), books, cfg)[0], 1);
});

test('adcDist equals the direct sum of subspace distances', () => {
  const rng = mulberry32(5);
  const dims = 8;
  const m = 2;
  const k = 4;
  const sub = dims / m;
  const books = new Float32Array(m * k * sub);
  for (let i = 0; i < books.length; i += 1) books[i] = Math.round(rng() * 100) / 10;
  const q = new Float32Array(dims);
  for (let i = 0; i < dims; i += 1) q[i] = Math.round(rng() * 100) / 10;
  const codes = Uint8Array.from([1, 3]);

  const tables = queryTables(q, books, { dims, m, k });
  const viaTable = adcDist(tables, codes, k);

  // direct: sum over subspaces of squared distance (q_sub vs centroid_sub)
  let direct = 0;
  for (let mi = 0; mi < m; mi += 1) {
    const qOff = mi * sub;
    const cOff = mi * k * sub + codes[mi] * sub;
    for (let s = 0; s < sub; s += 1) {
      const diff = q[qOff + s] - books[cOff + s];
      direct += diff * diff;
    }
  }
  assert.ok(Math.abs(viaTable - direct) < 1e-4, `${viaTable} vs ${direct}`);
});

test('codeTables are symmetric between nodes', () => {
  const rng = mulberry32(9);
  const dims = 4;
  const m = 2;
  const k = 4;
  const sub = 2;
  const books = new Float32Array(m * k * sub);
  for (let i = 0; i < books.length; i += 1) books[i] = Math.round(rng() * 50) / 10;
  const a = Uint8Array.from([0, 3]);
  const b = Uint8Array.from([2, 1]);
  const cfg = { dims, m, k };
  const d1 = adcDist(codeTables(a, books, cfg), b, k);
  const d2 = adcDist(codeTables(b, books, cfg), a, k);
  assert.ok(Math.abs(d1 - d2) < 1e-4, `${d1} vs ${d2}`);
});

test('encode is deterministic and bounded by k', () => {
  const vectors = Array.from({ length: 16 }, (_, i) => Float32Array.from([i, -i, i * 0.5, 1]));
  const books = trainCodebooks(vectors, { dims: 4, m: 2, k: 4, seed: 11, iters: 5 });
  const cfg = { dims: 4, m: 2, k: 4 };
  const c1 = encodeVector(vectors[3], books, cfg);
  const c2 = encodeVector(vectors[3], books, cfg);
  assert.deepEqual(Array.from(c1), Array.from(c2));
  for (const code of c1) assert.ok(code >= 0 && code < 4);
});
