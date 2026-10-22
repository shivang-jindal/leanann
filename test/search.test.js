// Query-path tests — RRF math by hand, fusion behavior on a real index,
// honesty rails (embed-none, OOV, budget, corruption), and stats shape.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { rrfFuse, search, searchOpened, openIndex, stats } from '../src/search.js';
import { buildIndex } from '../src/indexer.js';
import { BudgetExceeded, BudgetExceeded as BudgetErr } from '../src/budget.js';
import { StoreError } from '../src/store.js';
import { FAST, clusteredCorpus, corpusDir, tmpdir } from './helpers.js';

const RRF_FIXTURE = [
  { name: 'bm25', items: [{ id: 'A' }, { id: 'B' }, { id: 'C' }] },
  { name: 'vector', items: [{ id: 'B' }, { id: 'C' }, { id: 'D' }] },
];

test('rrfFuse: hand-computed scores with rrfK=1', () => {
  const { fused, detail } = rrfFuse(RRF_FIXTURE, 1);
  const byId = Object.fromEntries(fused.map((f) => [f.id, f.score]));
  // A: only bm25 #1  => 1/2
  // B: bm25 #2 + vec #1 => 1/3 + 1/2
  // C: bm25 #3 + vec #2 => 1/4 + 1/3
  // D: only vec #3  => 1/4
  assert.ok(Math.abs(byId.A - 0.5) < 1e-12);
  assert.ok(Math.abs(byId.B - (1 / 3 + 1 / 2)) < 1e-12);
  assert.ok(Math.abs(byId.C - (1 / 4 + 1 / 3)) < 1e-12);
  assert.ok(Math.abs(byId.D - 0.25) < 1e-12);
  // order: B > C > A > D
  assert.deepEqual(fused.map((f) => f.id), ['B', 'C', 'A', 'D']);
  // detail carries per-source rank + contribution
  assert.deepEqual(detail.get('B').bm25, { rank: 2, contrib: 1 / 3 });
  assert.deepEqual(detail.get('B').vector, { rank: 1, contrib: 0.5 });
  assert.deepEqual(detail.get('A').vector, undefined);
});

test('rrfFuse: default rrfK=60 term', () => {
  const { fused } = rrfFuse([{ name: 'x', items: [{ id: 'only' }] }], 60);
  assert.ok(Math.abs(fused[0].score - 1 / 61) < 1e-12);
});

test('rrfFuse: ties break deterministically by doc id', () => {
  const { fused } = rrfFuse([
    { name: 'a', items: [{ id: 'x' }, { id: 'y' }] },
    { name: 'b', items: [{ id: 'y' }, { id: 'x' }] },
  ], 60);
  // x and y get identical scores -> stable topK order must be reproducible
  const again = rrfFuse([
    { name: 'a', items: [{ id: 'x' }, { id: 'y' }] },
    { name: 'b', items: [{ id: 'y' }, { id: 'x' }] },
  ], 60);
  assert.deepEqual(fused.map((f) => f.id), again.fused.map((f) => f.id));
  assert.ok(Math.abs(fused[0].score - fused[1].score) < 1e-12);
});

async function buildQueryFixture(prefix) {
  const docs = clusteredCorpus({
    perTopic: 10,
    topics: [
      ['database', 'pool', 'connection'],
      ['memory', 'heap', 'pressure'],
      ['render', 'layout', 'modal'],
      ['deploy', 'rollback', 'artifact'],
    ],
    seed: 7,
  });
  const { dir, file } = corpusDir(prefix, docs);
  const out = path.join(dir, 'idx');
  await buildIndex({ input: file, out, ...FAST });
  return { dir, docs, out };
}

test('self-query: the queried document ranks at the top in hybrid', async () => {
  const { out, docs } = await buildQueryFixture('sr-self');
  for (const i of [0, 5, 17, 33]) {
    const res = await search({
      dir: out, query: docs[i].text, k: 3, budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
    });
    const ids = res.results.map((r) => r.id);
    assert.ok(
      ids.slice(0, 3).includes(docs[i].id),
      `query ${i} should surface ${docs[i].id} in top3, got ${ids.join(',')}`,
    );
  }
});

test('hybrid >= bm25 recall on the fixture (fusion never hurts here)', async () => {
  const { out, docs } = await buildQueryFixture('sr-recall');
  let hybridHits = 0;
  let bm25Hits = 0;
  for (const doc of docs) {
    const h = await search({
      dir: out, query: doc.text, k: 3, mode: 'hybrid', budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
    });
    const b = await search({
      dir: out, query: doc.text, k: 3, mode: 'bm25', budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
    });
    if (h.results.some((r) => r.id === doc.id)) hybridHits += 1;
    if (b.results.some((r) => r.id === doc.id)) bm25Hits += 1;
  }
  assert.ok(hybridHits >= bm25Hits, `hybrid ${hybridHits} >= bm25 ${bm25Hits} self-hits`);
  assert.ok(hybridHits >= docs.length - 2, `hybrid self-hit rate high: ${hybridHits}/${docs.length}`);
});

test('explain: both sources carry rank + contribution', async () => {
  const { out, docs } = await buildQueryFixture('sr-explain');
  const res = await search({
    dir: out, query: docs[0].text, k: 3, mode: 'hybrid', explain: true,
    budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
  });
  const top = res.results[0];
  assert.ok(top.sources, 'explain adds sources');
  assert.ok(top.sources.bm25, 'bm25 source present');
  assert.equal(top.sources.bm25.rank, 1);
  assert.ok(top.sources.bm25.contrib > 0);
  assert.ok(top.sources.vector, 'vector source present');
  assert.equal(top.sources.vector.rank, 1);

  // without explain there are no sources
  const plain = await search({
    dir: out, query: docs[0].text, k: 3, budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
  });
  assert.equal(plain.results[0].sources, undefined);
});

test('modes: vector-only and bm25-only both answer', async () => {
  const { out, docs } = await buildQueryFixture('sr-modes');
  const v = await search({
    dir: out, query: 'completely unseen tokens zzzqqq', k: 3, mode: 'vector',
    budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
  });
  assert.equal(v.mode, 'vector');
  assert.ok(v.results.length > 0, 'ANN always returns nearest docs');

  const b = await search({
    dir: out, query: 'completely unseen tokens zzzqqq', k: 3, mode: 'bm25',
    budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
  });
  assert.equal(b.results.length, 0, 'bm25 with no matching term honestly returns nothing');
  assert.equal(b.degraded, false);
});

test('k is honored', async () => {
  const { out } = await buildQueryFixture('sr-k');
  for (const k of [1, 2, 5]) {
    const res = await search({
      dir: out, query: 'database pool connection', k, budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
    });
    assert.ok(res.results.length <= k, `k=${k} gave ${res.results.length}`);
  }
});

test('--embed none: vector mode errors, hybrid degrades with a reason', async () => {
  const dims = FAST.dims;
  const docs = clusteredCorpus({ perTopic: 4, topics: [['alpha', 'beta'], ['gamma', 'delta']], seed: 3 })
    .map((d, i) => ({
      ...d,
      vector: Array.from({ length: dims }, (_, j) => ((i * 5 + j) % 9) - 4),
    }));
  const { dir, file } = corpusDir('sr-none', docs);
  const out = path.join(dir, 'idx');
  await buildIndex({ input: file, out, ...FAST, embed: 'none' });

  await assert.rejects(
    () => search({ dir: out, query: 'alpha', mode: 'vector', budgetMb: 8, pageKb: 4 }),
    (err) => err instanceof StoreError && /built with --embed none/.test(err.message),
  );

  const hybrid = await search({
    dir: out, query: 'alpha beta gamma', mode: 'hybrid', budgetMb: 8, pageKb: 4,
  });
  assert.equal(hybrid.degraded, true, 'hybrid without query embedder must flag degraded');
  assert.match(hybrid.degradedReason, /BM25-only/);
  assert.ok(hybrid.results.length > 0, 'still answers via BM25');

  const bm25 = await search({ dir: out, query: 'alpha beta', mode: 'bm25', budgetMb: 8, pageKb: 4 });
  assert.equal(bm25.degraded, false, 'pure bm25 mode is not degraded — it is the mode');
});

test('openIndex: impossible budget fails with a compute-your-minimum message', async () => {
  const { out } = await buildQueryFixture('sr-budget');
  await assert.rejects(
    () => openIndex({ dir: out, budgetMb: 1, pageKb: 4096, ef: 64 }),
    (err) => err instanceof BudgetErr && /raise --budget-mb to at least/.test(err.message),
  );
});

test('openIndex: truncated data file is caught before queries run', async () => {
  const { out } = await buildQueryFixture('sr-trunc');
  const codes = path.join(out, 'codes.bin');
  const full = fs.readFileSync(codes);
  fs.writeFileSync(codes, full.subarray(0, Math.floor(full.length / 2)));
  await assert.rejects(
    () => openIndex({ dir: out, budgetMb: FAST.budgetMb, pageKb: FAST.pageKb }),
    (err) => err instanceof StoreError && /manifest says/.test(err.message),
  );
});

test('openIndex: missing index directory names the fix', async () => {
  const missing = path.join(tmpdir('sr-missing'), 'nope');
  await assert.rejects(
    () => openIndex({ dir: missing }),
    (err) => err instanceof StoreError && /run `leanann index` first/.test(err.message),
  );
});

test('stats: shape covers docs, byte breakdown, embedder, corpus hash', async () => {
  const { out } = await buildQueryFixture('sr-stats');
  const s = await stats({ dir: out });
  assert.equal(s.docs, 40);
  assert.ok(s.terms > 0);
  assert.equal(s.dims, FAST.dims);
  assert.equal(s.m, FAST.m);
  assert.equal(s.k, FAST.k);
  assert.equal(s.graphM, FAST.graphM);
  assert.equal(s.embed.kind, 'hash');
  assert.ok(s.entry && s.entry.id >= 0);
  assert.ok(s.createdAt);
  assert.ok(s.corpus.hash);
  assert.ok(s.diskBytes > 0);
  const sum = Object.values(s.files).reduce((a, f) => a + f.bytes, 0);
  assert.equal(s.diskBytes, sum);
  for (const [name, f] of Object.entries(s.files)) {
    assert.ok(f.bytes > 0, `${name} has bytes`);
    assert.match(f.sha256, /^[0-9a-f]{64}$/, `${name} has a real sha256`);
  }
});

test('searchOpened: pool stays within budget after a query (accounting holds)', async () => {
  const { out } = await buildQueryFixture('sr-pool');
  const idx = await openIndex({ dir: out, budgetMb: FAST.budgetMb, pageKb: FAST.pageKb });
  try {
    const res = await searchOpened(idx, { query: 'database pool connection render modal', k: 5 });
    assert.equal(res.cache.evictions, 0, '8 MB is plenty for a 40-doc index');
    const s = idx.pool.stats();
    assert.ok(s.used <= s.limit, `pool used ${s.used} <= limit ${s.limit}`);
    assert.equal(res.docs, 40);
  } finally {
    await idx.pool.close();
  }
});
