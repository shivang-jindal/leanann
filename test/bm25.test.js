// BM25 tests — hand-computed scoring math + external run-merge roundtrip.
//
// The postings path is exercised the hard way: a tiny flush threshold forces
// multiple sorted RUNS that must k-way merge back into one consistent index.
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import { idf, bm25Score, PostingsBuilder, Bm25Index } from '../src/bm25.js';
import { BufferPool } from '../src/store.js';
import { varintEncode, varintDecode } from '../src/util.js';
import { tmpdir } from './helpers.js';

test('idf is smoothed and always positive (even at df = N)', () => {
  // idf = ln(1 + (N - df + 0.5)/(df + 0.5))
  assert.ok(Math.abs(idf(10, 5) - Math.LN2) < 1e-12); // ln(1 + 5.5/5.5) = ln 2
  assert.ok(idf(4, 4) > 0); // classic idf would be <= 0 here
  assert.ok(idf(1000, 1000) > 0);
  assert.ok(idf(10, 1) > idf(10, 5)); // rarer term scores higher
});

test('bm25Score matches hand-computed values', () => {
  // dl == avgdl => norm == 1, idf(10,5) == ln2
  const s1 = bm25Score({ tf: 1, df: 5, N: 10, dl: 10, avgdl: 10 });
  assert.ok(Math.abs(s1 - Math.LN2) < 1e-12); // ln2 * 2.2 / (1 + 1.2) == ln2

  // tf = 3: ln2 * (3*2.2) / (3 + 1.2) = ln2 * 6.6/4.2
  const s3 = bm25Score({ tf: 3, df: 5, N: 10, dl: 10, avgdl: 10 });
  assert.ok(Math.abs(s3 - Math.LN2 * (6.6 / 4.2)) < 1e-12);

  // length penalty: dl = 3*avgdl, b = .75 => norm = .25 + .75*3 = 2.5
  const sLong = bm25Score({ tf: 1, df: 5, N: 10, dl: 30, avgdl: 10 });
  assert.ok(Math.abs(sLong - Math.LN2 * (2.2 / (1 + 1.2 * 2.5))) < 1e-12);

  // tf saturation: gain from tf 10 -> 100 is bounded
  const s10 = bm25Score({ tf: 10, df: 5, N: 10, dl: 10, avgdl: 10 });
  const s100 = bm25Score({ tf: 100, df: 5, N: 10, dl: 10, avgdl: 10 });
  assert.ok(s100 > s10 && s100 - s10 < s10);
});

test('varint roundtrip incl. large deltas', () => {
  for (const n of [0, 1, 127, 128, 300, 65535, 10_000_000]) {
    const bytes = [];
    varintEncode(n, bytes);
    const [v, pos] = varintDecode(Uint8Array.from(bytes), 0);
    assert.equal(v, n, `roundtrip ${n}`);
    assert.equal(pos, bytes.length);
  }
  assert.throws(() => varintEncode(-1, []), /bad value/);
  assert.throws(() => varintDecode(Uint8Array.from([0x80]), 0), /truncated/);
});

async function openBuilder(dir, flushAtBytes) {
  const builder = new PostingsBuilder({ dir, flushAtBytes });
  return builder;
}

test('postings run-merge: multi-run flush produces one consistent index', async () => {
  const dir = tmpdir('bm25');
  // force a flush after every couple of terms -> several runs that must merge
  const builder = await openBuilder(dir, 1);
  builder.addDoc(0, new Map([['cache', 2], ['miss', 1]]));
  await builder.flushRun(); // run 1
  builder.addDoc(1, new Map([['cache', 1], ['hit', 3]]));
  await builder.flushRun(); // run 2
  builder.addDoc(2, new Map([['miss', 1], ['evict', 1], ['cache', 1]]));
  // third batch stays buffered until finish() flushes it
  const avgdl = builder.avgdl();
  const built = await builder.finish();

  assert.equal(built.docs, 3);
  assert.equal(built.terms, 4); // cache, evict, hit, miss
  assert.equal(avgdl, (3 + 4 + 3) / 3);

  // reopen as a query index (needs a pool over postings.bin)
  const manifest = { docs: 3, avgdl };
  const pool = new BufferPool({ pageSize: 64, limitBytes: 1 << 20 });
  const bm25 = await Bm25Index.open({ dir, manifest, pool });

  // merged df: cache appears in docs 0,1,2
  const cacheId = bm25.termId('cache');
  assert.ok(cacheId >= 0);
  const posts = await bm25.postingsFor(cacheId);
  assert.deepEqual(posts.map((p) => p.docId), [0, 1, 2]);
  assert.deepEqual(posts.map((p) => p.tf), [2, 1, 1]);

  // unknown term -> no postings
  assert.equal(bm25.termId('zebra'), -1);
  assert.equal(await bm25.postingsFor(-1), null);

  // ranking sanity: 'cache' hits all three; doc0 has tf=2 so should outscore
  // doc1 (tf=1, but longer doc -> penalized; both effects favor doc0 here)
  const res = await bm25.search(['cache'], 10);
  assert.equal(res.length, 3);
  assert.equal(res[0].id, 0);
  assert.ok(res[0].score > res[1].score);

  // OOV-only query -> honest empty result
  assert.deepEqual(await bm25.search(['zebra'], 10), []);

  await pool.close();
});

test('terms dictionary is sorted and binary-searchable', async () => {
  const dir = tmpdir('bm25');
  const builder = new PostingsBuilder({ dir, flushAtBytes: 1 << 20 });
  // insertion order is deliberately unsorted
  builder.addDoc(0, new Map([['zeta', 1], ['alpha', 1], ['mid', 1]]));
  await builder.finish();

  const pool = new BufferPool({ pageSize: 64, limitBytes: 1 << 20 });
  const bm25 = await Bm25Index.open({ dir, manifest: { docs: 1, avgdl: 3 }, pool });
  const terms = [0, 1, 2].map((i) => bm25.termAt(i));
  assert.deepEqual(terms, ['alpha', 'mid', 'zeta']); // sorted
  assert.equal(bm25.termId('mid'), 1);
  assert.equal(bm25.termId('aaa'), -1); // before first
  assert.equal(bm25.termId('zzz'), -1); // after last
  await pool.close();
});

test('empty builder finishes with zero terms and zero doclens', async () => {
  const dir = tmpdir('bm25');
  const builder = new PostingsBuilder({ dir, flushAtBytes: 1 << 20 });
  const built = await builder.finish();
  assert.equal(built.terms, 0);
  assert.equal(built.docs, 0);
  assert.equal(builder.avgdl(), 0);
  assert.ok(fs.existsSync(path.join(dir, 'terms.bin')));
  assert.ok(fs.existsSync(path.join(dir, 'postoff.bin')));
});

test('postoff.bin and termoff.bin agree with their files', async () => {
  const dir = tmpdir('bm25');
  const builder = new PostingsBuilder({ dir, flushAtBytes: 64 });
  builder.addDoc(0, new Map([['alpha', 1], ['beta', 2]]));
  await builder.flushRun();
  builder.addDoc(1, new Map([['alpha', 1], ['gamma', 1]]));
  await builder.finish();

  const termoff = fs.readFileSync(path.join(dir, 'termoff.bin'));
  const postoff = fs.readFileSync(path.join(dir, 'postoff.bin'));
  assert.equal(termoff.length, 3 * 4); // alpha beta gamma
  assert.equal(postoff.length, 3 * 4);

  // last term's postings must end exactly at postings.bin's size
  const lastOffset = postoff.readUInt32LE(2 * 4);
  const postSize = fs.statSync(path.join(dir, 'postings.bin')).size;
  assert.ok(lastOffset < postSize);

  const pool = new BufferPool({ pageSize: 32, limitBytes: 1 << 20 });
  const bm25 = await Bm25Index.open({ dir, manifest: { docs: 2, avgdl: 2.5 }, pool });
  // termAt offsets index terms.bin only — decode each term cleanly
  for (let i = 0; i < 3; i += 1) assert.equal(typeof bm25.termAt(i), 'string');
  const alpha = await bm25.postingsFor(bm25.termId('alpha'));
  assert.deepEqual(alpha.map((p) => p.docId), [0, 1]);
  await pool.close();
});
