// Benchmark honesty tests — recall math, report shape, injected clocks so the
// numbers are deterministic, and scratch-index cleanup.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { bench, recallAtK } from '../src/bench.js';
import { buildIndex } from '../src/indexer.js';
import { FAST, clusteredCorpus, corpusDir } from './helpers.js';

const FIX_NOW = () => Date.parse('2026-01-02T03:04:05.000Z');

function fixtureDocs(perTopic = 10) {
  return clusteredCorpus({
    perTopic,
    topics: [
      ['database', 'pool', 'connection'],
      ['memory', 'heap', 'pressure'],
      ['render', 'layout', 'modal'],
      ['deploy', 'rollback', 'artifact'],
    ],
    seed: 7,
  });
}

async function benchFixture(prefix, { perTopic = 10, n = null } = {}) {
  const docs = fixtureDocs(perTopic);
  const { dir, file } = corpusDir(prefix, n ? docs.slice(0, n) : docs);
  const out = path.join(dir, 'idx');
  await buildIndex({ input: file, out, ...FAST, now: FIX_NOW });
  return { dir, file, out, docs };
}

test('recallAtK: the number the benchmark hangs on', () => {
  const expected = [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }];
  assert.equal(recallAtK([{ id: 'a' }, { id: 'b' }, { id: 'z' }, { id: 'y' }], expected), 0.5);
  assert.equal(recallAtK(expected, expected), 1);
  assert.equal(recallAtK([{ id: 'nope' }], expected), 0);
  assert.equal(recallAtK([], []), 0, 'no expected -> 0, never NaN');
  assert.equal(recallAtK([], expected), 0);
});

test('bench on an existing index: honest recall table + enforced budget', async () => {
  const { file, out } = await benchFixture('bench-existing');
  const msgs = [];
  const report = await bench({
    input: file,
    index: out,
    queries: 8,
    k: 3,
    budgetMb: FAST.budgetMb,
    pageKb: FAST.pageKb,
    ef: 32,
    now: FIX_NOW,
    progress: (m) => msgs.push(m),
  });

  assert.equal(report.docs, 40);
  assert.equal(report.k, 3);
  assert.equal(report.queries, 8);
  assert.equal(report.embed, 'hash');
  assert.equal(report.budgetMb, FAST.budgetMb);

  // recall numbers are real and sane
  for (const [name, v] of Object.entries(report.recall)) {
    assert.ok(v !== null && v >= 0 && v <= 1, `${name} recall in [0,1], got ${v}`);
  }
  assert.ok(report.recall.vector > 0.7, `vector recall ${report.recall.vector}`);
  assert.ok(
    report.recall.hybrid >= report.recall.bm25,
    `fusion must not lose to keywords alone: hybrid=${report.recall.hybrid} bm25=${report.recall.bm25}`,
  );

  // ground truth is exact over all docs here
  assert.equal(report.groundTruth.full, true);
  assert.equal(report.groundTruth.docs, 40);
  assert.match(report.groundTruth.method, /exact-ADC/);

  // latency + memory, with RSS kept SEPARATE from the logical budget
  assert.ok(report.latencyMs.p50 >= 0);
  assert.ok(report.latencyMs.p95 >= report.latencyMs.p50);
  assert.equal(report.memory.budgetEnforced, true);
  assert.ok(report.memory.poolUsedBytes <= report.memory.poolLimitBytes);
  assert.ok(report.memory.rssBytes > 0);
  assert.match(report.memory.rssHuman, /MB|KB|B/);

  assert.ok(report.cache.hitRate >= 0 && report.cache.hitRate <= 1);
  assert.equal(report.degradedQueries, 0, '8 MB over 40 docs must not thrash');
  assert.ok(report.diskBytes > 0);
  assert.equal(report.elapsedMs, 0, 'fixed clock -> zero elapsed');

  // progress reported at least the header + summary
  assert.ok(msgs.some((m) => m.startsWith('bench:')), `header in ${JSON.stringify(msgs)}`);
  assert.ok(msgs.some((m) => m.startsWith('recall@3')), 'summary in progress');
});

test('bench: ground-truth cap marks the table as sampled/conservative', async () => {
  const { file, out } = await benchFixture('bench-cap', { perTopic: 5 }); // 20 docs
  const report = await bench({
    input: file, index: out, queries: 4, k: 3,
    exactCap: 5, budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
  });
  assert.equal(report.groundTruth.full, false);
  assert.equal(report.groundTruth.docs, 5);
  assert.match(report.groundTruth.note, /sampled/);
  assert.match(report.groundTruth.note, /conservative/);
});

test('bench: injected hrtime/memory drive the reported numbers exactly', async () => {
  const { file, out } = await benchFixture('bench-inject', { perTopic: 3, n: 12 });
  let t = 0;
  const report = await bench({
    input: file, index: out, queries: 4, k: 3,
    budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
    hrtime: () => { t += 10; return t; },
    memory: () => ({ rss: 12345678 }),
    now: () => 5000,
  });
  assert.equal(report.latencyMs.p50, 10, 'two hrtime reads per query -> 10ms each');
  assert.equal(report.latencyMs.p95, 10);
  assert.equal(report.memory.rssBytes, 12345678);
  assert.equal(report.memory.rssHuman, '11.8 MB');
  assert.equal(report.elapsedMs, 0, 'constant clock');
});

test('bench: scratch index is built and removed (no temp litter)', async () => {
  const { file } = await benchFixture('bench-scratch', { perTopic: 3, n: 12 });
  const listScratch = () => fs.readdirSync(os.tmpdir()).filter((f) => f.startsWith('leanann-bench-'));
  const before = listScratch();
  const report = await bench({
    input: file, index: null, queries: 3, k: 3,
    dims: FAST.dims, m: FAST.m, codebookK: FAST.k, iters: FAST.iters, graphM: FAST.graphM,
    budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
  });
  assert.ok(report.docs > 0);
  assert.equal(report.recall.vector !== null, true, 'scratch build still measures recall');
  const added = listScratch().filter((f) => !before.includes(f));
  assert.deepEqual(added, [], 'scratch dirs must be cleaned up');
});

test('bench: recallAtK id-space rule (hydrated docId vs string id)', async () => {
  // regression guard for the bug where hybrid recall compared "db-1" against 4
  const { file, out } = await benchFixture('bench-idspace', { perTopic: 4, n: 16 });
  const report = await bench({
    input: file, index: out, queries: 4, k: 3,
    budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
  });
  // both sides now compare numeric docIds: bm25 alone can't be all-zero when
  // queries are corpus docs (they always match their own terms)
  assert.ok(report.recall.bm25 > 0, `bm25 recall ${report.recall.bm25} must be measured in docId space`);
});
