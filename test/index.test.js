// End-to-end build tests — two-pass indexer output, reopen, byte-identical
// determinism, staleness checking, and honest failure modes.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { buildIndex, checkIndex, BuildError } from '../src/indexer.js';
import { readManifest } from '../src/store.js';
import { search } from '../src/search.js';
import { BudgetExceeded } from '../src/budget.js';
import { CorpusError } from '../src/corpus.js';
import { FAST, corpusDir, tmpdir, writeJsonl } from './helpers.js';

const FIX_NOW = () => Date.parse('2026-01-02T03:04:05.000Z');

function plainDocs(n = 20) {
  const topics = [
    ['database', 'connection', 'pool'],
    ['memory', 'heap', 'pressure'],
    ['render', 'button', 'modal'],
    ['deploy', 'rollback', 'cache'],
  ];
  const docs = [];
  for (let i = 0; i < n; i += 1) {
    const pool = topics[i % topics.length];
    docs.push({
      id: `d-${i}`,
      text: `${pool[i % 3]} ${pool[(i + 1) % 3]} issue number ${i} with token t${i}`,
    });
  }
  return docs;
}

async function buildFixture(prefix, { n = 20, now = FIX_NOW, ...opts } = {}) {
  const { dir, file, docs } = corpusDir(prefix, plainDocs(n));
  const out = path.join(dir, 'idx');
  const result = await buildIndex({ input: file, out, ...FAST, now, ...opts });
  return { dir, file, docs, out, result };
}

test('build: produces every manifest file with exact sizes and sane fields', async () => {
  const { out, result } = await buildFixture('idx-build');
  assert.equal(result.docs, 20);
  assert.ok(result.terms > 0);
  assert.ok(result.elapsedMs >= 0);

  const m = await readManifest(out);
  assert.equal(m.version, 1);
  assert.equal(m.docs, 20);
  assert.equal(m.dims, FAST.dims);
  assert.equal(m.m, FAST.m);
  assert.equal(m.k, FAST.k);
  assert.equal(m.createdAt, '2026-01-02T03:04:05.000Z'); // injected clock
  assert.equal(m.embed.kind, 'hash');
  assert.ok(m.entry && m.entry.id >= 0);
  assert.ok(Array.isArray(m.corpus.hash) || typeof m.corpus.hash === 'string');

  const expectedFiles = [
    'codes.bin', 'graph.bin', 'postings.bin', 'terms.bin', 'termoff.bin',
    'postoff.bin', 'doclens.bin', 'meta.bin', 'metaoff.bin', 'codebooks.bin',
  ];
  for (const name of expectedFiles) {
    assert.ok(m.files[name], `${name} listed in manifest`);
    assert.equal(
      fs.statSync(path.join(out, name)).size,
      m.files[name].bytes,
      `${name} on-disk size matches manifest`,
    );
  }
  assert.equal(m.files['codebooks.bin'].bytes, FAST.dims * FAST.k * 4);
});

test('build: staging and run files are cleaned up', async () => {
  const { out } = await buildFixture('idx-clean');
  const entries = fs.readdirSync(out);
  assert.ok(!entries.includes('vectors.f32'), 'vectors.f32 staging removed');
  assert.ok(!entries.some((f) => f.startsWith('.run-')), 'run files removed');
  assert.ok(entries.includes('manifest.json'));
});

test('reopen: built index answers queries immediately', async () => {
  const { file, out, docs } = await buildFixture('idx-reopen');
  void file;
  const res = await search({
    dir: out, query: docs[3].text, k: 3, budgetMb: FAST.budgetMb, pageKb: FAST.pageKb,
  });
  assert.ok(res.results.length > 0);
  assert.equal(res.degraded, false);
  const ids = res.results.map((r) => r.id);
  assert.ok(ids.includes(docs[3].id), `self-query should find ${docs[3].id}, got ${ids.join(',')}`);
});

test('determinism: same input + config + clock => byte-identical index', async () => {
  const { dir, file } = corpusDir('idx-det', plainDocs(20));
  const outA = path.join(dir, 'a');
  const outB = path.join(dir, 'b');
  await buildIndex({ input: file, out: outA, ...FAST, now: FIX_NOW });
  await buildIndex({ input: file, out: outB, ...FAST, now: FIX_NOW });

  const m = JSON.parse(fs.readFileSync(path.join(outA, 'manifest.json'), 'utf8'));
  assert.equal(
    fs.readFileSync(path.join(outA, 'manifest.json'), 'utf8'),
    fs.readFileSync(path.join(outB, 'manifest.json'), 'utf8'),
    'manifest byte-identical',
  );
  for (const name of Object.keys(m.files)) {
    const a = fs.readFileSync(path.join(outA, name));
    const b = fs.readFileSync(path.join(outB, name));
    assert.equal(Buffer.compare(a, b), 0, `${name} byte-identical`);
  }
});

test('determinism: different seed produces a different graph (seed matters)', async () => {
  const { dir, file } = corpusDir('idx-seed', plainDocs(20));
  const outA = path.join(dir, 'a');
  const outB = path.join(dir, 'b');
  await buildIndex({ input: file, out: outA, ...FAST, seed: 1, now: FIX_NOW });
  await buildIndex({ input: file, out: outB, ...FAST, seed: 2, now: FIX_NOW });
  const ga = fs.readFileSync(path.join(outA, 'graph.bin'));
  const gb = fs.readFileSync(path.join(outB, 'graph.bin'));
  assert.notEqual(Buffer.compare(ga, gb), 0, 'graph must depend on seed');
});

test('checkIndex: fresh => stale false; corpus change => stale true', async () => {
  const { dir, file, out } = await buildFixture('idx-check');
  const fresh = await checkIndex({ input: file, out });
  assert.equal(fresh.stale, false);
  assert.equal(fresh.docs, 20);
  assert.equal(fresh.expected, fresh.actual);

  fs.appendFileSync(file, JSON.stringify({ id: 'd-99', text: 'changed corpus' }) + '\n');
  const stale = await checkIndex({ input: file, out });
  assert.equal(stale.stale, true);
  assert.notEqual(stale.expected, stale.actual);
  void dir;
});

test('build: empty corpus fails honestly (no documents)', async () => {
  const dir = tmpdir('idx-empty');
  const file = path.join(dir, 'empty.jsonl');
  fs.writeFileSync(file, '\n');
  await assert.rejects(
    () => buildIndex({ input: file, out: path.join(dir, 'idx'), ...FAST }),
    (err) => err instanceof CorpusError && /no documents/.test(err.message),
  );
});

test('build: --embed none without vectors fails naming the flag', async () => {
  const { file, dir } = corpusDir('idx-none', plainDocs(3));
  await assert.rejects(
    () => buildIndex({ input: file, out: path.join(dir, 'idx'), ...FAST, embed: 'none' }),
    (err) => err instanceof BuildError && /--embed none: a document has no "vector" field/.test(err.message),
  );
});

test('build: --embed none with supplied vectors succeeds and records kind', async () => {
  const dims = FAST.dims;
  const docs = plainDocs(6).map((d, i) => ({
    ...d,
    vector: Array.from({ length: dims }, (_, j) => ((i * 7 + j * 3) % 11) - 5),
  }));
  const { dir, file } = corpusDir('idx-vecs', docs);
  const out = path.join(dir, 'idx');
  const res = await buildIndex({ input: file, out, ...FAST, embed: 'none' });
  assert.equal(res.docs, 6);
  const m = await readManifest(out);
  assert.equal(m.embed.kind, 'none');
});

test('build: vector dims mismatch is a line-honest corpus error', async () => {
  const docs = [{ id: 'x', text: 'has vector', vector: [1, 2, 3] }];
  const { dir, file } = corpusDir('idx-dim', docs);
  await assert.rejects(
    () => buildIndex({ input: file, out: path.join(dir, 'idx'), ...FAST }), // FAST.dims = 16
    (err) => err instanceof CorpusError && /"vector" must be 16 finite numbers \(got 3\)/.test(err.message),
  );
});

test('build: page size exceeding the budget fails before any work', async () => {
  const { file, dir } = corpusDir('idx-budget', plainDocs(3));
  await assert.rejects(
    () => buildIndex({ input: file, out: path.join(dir, 'idx'), ...FAST, budgetMb: 1, pageKb: 2048 }),
    (err) => err instanceof BudgetExceeded && /raise --budget-mb/.test(err.message),
  );
  assert.ok(!fs.existsSync(path.join(dir, 'idx', 'manifest.json')), 'no index committed on failure');
});

test('build: exclude patterns drop matching files from a directory corpus', async () => {
  const dir = tmpdir('idx-excl');
  fs.writeFileSync(path.join(dir, 'keep.md'), 'keep this document about pools');
  fs.writeFileSync(path.join(dir, 'skip.txt'), 'skip this one about pools');
  const out = path.join(dir, 'idx');
  const res = await buildIndex({ input: dir, out, ...FAST, exclude: ['skip.txt'] });
  assert.equal(res.docs, 1);
  const m = await readManifest(out);
  assert.equal(m.docs, 1);
});

test('build: id defaulting and meta roundtrip survive reopen', async () => {
  const dir = tmpdir('idx-meta');
  const file = path.join(dir, 'c.jsonl');
  // no explicit ids -> doc-0..N; meta must still hydrate them at query time
  writeJsonl(file, [
    { text: 'first doc about turtles' },
    { text: 'second doc about turtles' },
  ]);
  const out = path.join(dir, 'idx');
  await buildIndex({ input: file, out, ...FAST });
  const res = await search({ dir: out, query: 'turtles', mode: 'bm25', k: 2, budgetMb: 8, pageKb: 4 });
  assert.deepEqual(res.results.map((r) => r.id), ['doc-0', 'doc-1']);
  assert.ok(res.results.every((r) => r.preview.includes('turtles')));
});
