// Graph tests — HNSW record codec, seeded level assignment, beam-search
// recall against exact ADC ground truth, determinism, and honest behavior on
// disconnected graphs (no fabricated reach).
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {
  Graph, graphStride, parseRecord, serializeRecord, assignLevel, MAX_LEVEL,
} from '../src/graph.js';
import { BufferPool, PagedFile } from '../src/store.js';
import { trainCodebooks, encodeVector, queryTables, adcDist } from '../src/quantize.js';
import { mulberry32 } from '../src/util.js';
import { tmpdir } from './helpers.js';

const DIMS = 8;
const M = 2;
const K = 8;

function clusterVectors(n, seed) {
  const rng = mulberry32(seed);
  const vectors = [];
  const cluster = [];
  for (let i = 0; i < n; i += 1) {
    const c = i % 2; // even -> A (dims 0,1), odd -> B (dims 2,3)
    const v = new Float32Array(DIMS);
    v[c * 2] = 1;
    v[c * 2 + 1] = 1;
    for (let d = 0; d < DIMS; d += 1) v[d] += (rng() - 0.5) * 0.3;
    vectors.push(v);
    cluster.push(c);
  }
  return { vectors, cluster };
}

async function makeFixture({ n = 60, graphM = 8, seed = 42, insert = true } = {}) {
  const dir = tmpdir('graph');
  const codesPath = path.join(dir, 'codes.bin');
  const graphPath = path.join(dir, 'graph.bin');
  fs.writeFileSync(codesPath, Buffer.alloc(0));
  fs.writeFileSync(graphPath, Buffer.alloc(0));
  const pool = new BufferPool({ pageSize: 512, limitBytes: 1 << 20, writable: true });
  const cf = await pool.attach(codesPath, { writable: true });
  const gf = await pool.attach(graphPath, { writable: true });
  const codesFile = new PagedFile({ pool, fid: cf, stride: M, name: 'codes.bin' });
  const graphFile = new PagedFile({ pool, fid: gf, stride: graphStride(graphM), name: 'graph.bin' });

  const { vectors, cluster } = clusterVectors(n, seed + 1);
  const books = trainCodebooks(vectors, { dims: DIMS, m: M, k: K, seed, iters: 6 });
  const cfg = { dims: DIMS, m: M, k: K };
  for (let i = 0; i < n; i += 1) await codesFile.write(i, encodeVector(vectors[i], books, cfg));

  const graph = new Graph({ file: graphFile, codes: codesFile, m: graphM, k: K, seed });
  graph.setBooks(books, cfg);
  if (insert) {
    for (let i = 0; i < n; i += 1) {
      await graph.insert(i, queryTables(vectors[i], books, cfg));
    }
  }
  return { dir, pool, codesFile, graphFile, graph, vectors, cluster, books, cfg, n };
}

test('record codec roundtrips level, degrees, and neighbor ids', () => {
  const m0 = 16;
  const rec = {
    level: 2,
    degrees: Uint8Array.from([3, 1, 0, 0]),
    neighbors: [0, 1, 2, 3].map(() => new Int32Array(m0).fill(-1)),
  };
  rec.neighbors[0].set([4, 7, 9]);
  rec.neighbors[1].set([12]);

  const stride = graphStride(8);
  assert.equal(stride, 1 + 4 * (1 + 2 * m0)); // 133 for m=8
  const buf = serializeRecord(rec, m0);
  assert.equal(buf.length, stride);

  const back = parseRecord(buf, m0);
  assert.equal(back.level, 2);
  assert.deepEqual(Array.from(back.degrees), [3, 1, 0, 0]);
  assert.deepEqual(Array.from(back.neighbors[0].slice(0, 3)), [4, 7, 9]);
  assert.equal(back.neighbors[0][3], -1); // 0xFFFF -> unused
  assert.deepEqual(Array.from(back.neighbors[1].slice(0, 1)), [12]);
  assert.deepEqual(Array.from(back.neighbors[2]), new Array(m0).fill(-1));
});

test('zero-filled record parses as isolated level-0 node', () => {
  const m0 = 16;
  const back = parseRecord(new Uint8Array(graphStride(8)), m0);
  assert.equal(back.level, 0);
  assert.deepEqual(Array.from(back.degrees), [0, 0, 0, 0]);
  // neighbor slots are only consulted up to their degree; with degree 0 the
  // node is isolated regardless of slot contents (which is what keeps a
  // zero-filled record safe — beam never reads the slots)
  const serialized = parseRecord(serializeRecord({
    level: 0,
    degrees: Uint8Array.from([0, 0, 0, 0]),
    neighbors: [0, 1, 2, 3].map(() => new Int32Array(m0).fill(-1)),
  }, m0), m0);
  assert.ok(Array.from(serialized.neighbors[0]).every((x) => x === -1),
    'properly serialized unused slots are -1');
});

test('graphStride matches the SPEC formula for common m', () => {
  assert.equal(graphStride(8), 133);
  assert.equal(graphStride(16), 261);
});

test('assignLevel is seeded-deterministic, geometric, and clamped', () => {
  const a = mulberry32(42);
  const b = mulberry32(42);
  for (let i = 0; i < 50; i += 1) assert.equal(assignLevel(a, 16), assignLevel(b, 16));

  const rng = mulberry32(1);
  let upper = 0;
  for (let i = 0; i < 1000; i += 1) {
    const l = assignLevel(rng, 16);
    assert.ok(l >= 0 && l <= MAX_LEVEL, `level ${l} in range`);
    if (l >= 1) upper += 1;
  }
  assert.ok(upper > 0, 'geometric distribution must produce some level >= 1');
});

test('insert then search: ANN finds the right cluster vs exact ADC', async () => {
  const fx = await makeFixture({ n: 60, graphM: 8, seed: 42 });
  assert.ok(fx.graph.entry, 'entry assigned');
  assert.equal(fx.graphFile.count, 60);

  // query = a cluster-A vector (node 0)
  const q = fx.vectors[0];
  const tables = queryTables(q, fx.books, fx.cfg);
  const beam = await fx.graph.search(tables, 16);
  assert.ok(beam.length >= 5, `beam returned ${beam.length}`);

  // exact top-5 by brute-force ADC over all codes
  const all = [];
  for (let id = 0; id < fx.n; id += 1) {
    const codes = await fx.codesFile.read(id);
    all.push({ id, dist: adcDist(tables, codes, K) });
  }
  all.sort((x, y) => (x.dist !== y.dist ? x.dist - y.dist : x.id - y.id));
  const exact = all.slice(0, 5).map((x) => x.id);
  const got = new Set(beam.slice(0, 5).map((r) => r.id));
  const hits = exact.filter((id) => got.has(id)).length;
  assert.ok(hits >= 3, `recall@5 ${hits}/5 (beam: ${[...got].join(',')}, exact: ${exact.join(',')})`);

  // semantic sanity: top-5 should be overwhelmingly cluster A
  const topCluster = beam.slice(0, 5).map((r) => fx.cluster[r.id]);
  assert.ok(topCluster.filter((c) => c === 0).length >= 4, `top5 clusters: ${topCluster.join(',')}`);

  await fx.pool.close();
});

test('search is deterministic (identical sequence on repeat)', async () => {
  const fx = await makeFixture({ n: 40, seed: 7 });
  const tables = queryTables(fx.vectors[3], fx.books, fx.cfg);
  const first = (await fx.graph.search(tables, 12)).map((r) => r.id);
  const second = (await fx.graph.search(tables, 12)).map((r) => r.id);
  assert.deepEqual(first, second);
  await fx.pool.close();
});

test('full build is byte-identical across runs (determinism gate)', async () => {
  const a = await makeFixture({ n: 30, seed: 42 });
  const b = await makeFixture({ n: 30, seed: 42 });
  await a.pool.flush();
  await b.pool.flush();
  const bytesA = fs.readFileSync(path.join(a.dir, 'graph.bin'));
  const bytesB = fs.readFileSync(path.join(b.dir, 'graph.bin'));
  assert.ok(bytesA.length > 0);
  assert.equal(Buffer.compare(bytesA, bytesB), 0, 'graph.bin must be byte-identical');
  const codesA = fs.readFileSync(path.join(a.dir, 'codes.bin'));
  const codesB = fs.readFileSync(path.join(b.dir, 'codes.bin'));
  assert.equal(Buffer.compare(codesA, codesB), 0, 'codes.bin must be byte-identical');
  await a.pool.close();
  await b.pool.close();
});

test('single-node graph returns exactly that node', async () => {
  const fx = await makeFixture({ n: 1, seed: 5 });
  const tables = queryTables(fx.vectors[0], fx.books, fx.cfg);
  const res = await fx.graph.search(tables, 8);
  assert.deepEqual(res.map((r) => r.id), [0]);
  await fx.pool.close();
});

test('disconnected nodes are never fabricated into results', async () => {
  // Hand-craft a graph where node 1 exists but nothing points to it.
  const fx = await makeFixture({ n: 2, seed: 5, insert: false });
  const rec = { level: 0, degrees: Uint8Array.from([0, 0, 0, 0]), neighbors: [0, 1, 2, 3].map(() => new Int32Array(16).fill(-1)) };
  await fx.graphFile.write(0, serializeRecord(rec, 16));
  await fx.graphFile.write(1, serializeRecord(rec, 16));
  fx.graph.entry = { id: 0, level: 0 };

  const tables = queryTables(fx.vectors[1], fx.books, fx.cfg); // closest to unreachable node 1
  const res = await fx.graph.search(tables, 8);
  assert.deepEqual(res.map((r) => r.id), [0], 'only the reachable entry is returned');
  await fx.pool.close();
});
