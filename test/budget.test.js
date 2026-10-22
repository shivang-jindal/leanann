// Budget engine tests — the memory contract: hard caps, honest LRU, dirty
// writeback on eviction, and impossible configs that must FAIL loudly.
import test from 'node:test';
import assert from 'node:assert/strict';
import { Budget, BudgetExceeded, PageCache, validateQueryBudget } from '../src/budget.js';

function makeCache({ pageSize = 10, limit = 30, writes = null } = {}) {
  const reads = [];
  const cache = new PageCache({
    pageSize,
    limitBytes: limit,
    readPage: async (key) => {
      reads.push(key);
      return new Uint8Array(pageSize);
    },
    writePage: writes
      ? async (key, bytes) => writes.push({ key, len: bytes.length })
      : async (key) => { throw new Error(`unexpected write for ${key}`); },
  });
  return { cache, reads, writes };
}

test('Budget charges, refunds, tracks peak, and refuses overflow', () => {
  const b = new Budget(100);
  b.charge(40);
  assert.equal(b.used, 40);
  b.charge(40);
  b.charge(20);
  assert.equal(b.used, 100);
  assert.equal(b.peak, 100);
  b.refund(50);
  assert.equal(b.used, 50);
  assert.equal(b.peak, 100);
  assert.throws(() => b.charge(101), BudgetExceeded);
  assert.throws(() => new Budget(0), RangeError);
  assert.throws(() => new Budget(NaN), RangeError);
});

test('page larger than the budget fails at construction with a fix hint', () => {
  assert.throws(
    () => new PageCache({ pageSize: 4096, limitBytes: 1024, readPage: async () => new Uint8Array(0) }),
    (err) => err instanceof BudgetExceeded && /raise --budget-mb/.test(err.message),
  );
});

test('LRU: oldest page is evicted, a hit refreshes recency', async () => {
  const { cache, reads } = makeCache({ pageSize: 10, limit: 30 });
  await cache.get('0:0');
  await cache.get('0:1');
  await cache.get('0:2');
  assert.equal(cache.map.size, 3);
  assert.equal(cache.used, 30);

  await cache.get('0:0'); // touch 0 -> now most recent; 1 is oldest
  await cache.get('0:3'); // 4th distinct page exceeds limit -> evict 1
  assert.equal(cache.map.size, 3);
  assert.equal(cache.used, 30);
  assert.ok(cache.evictions >= 1);
  assert.ok(!cache.map.has('0:1'), 'page 0:1 should be the LRU victim');
  assert.ok(cache.map.has('0:0'), 'recently used page must survive');
  // readPage called exactly once per distinct page (0:0 read once, re-get was a hit)
  assert.equal(reads.filter((k) => k === '0:0').length, 1);
  assert.equal(reads.filter((k) => k === '0:1').length, 1);
});

test('hit/miss accounting and hitRate', async () => {
  const { cache, reads } = makeCache();
  await cache.get('1:0');
  await cache.get('1:0'); // hit
  await cache.get('1:1'); // miss
  const s = cache.stats();
  assert.equal(s.hits, 1);
  assert.equal(s.misses, 2);
  assert.equal(s.pages, 2);
  assert.ok(Math.abs(s.hitRate - 1 / 3) < 1e-12);
  assert.equal(reads.length, 2); // only misses reach readPage
});

test('composite keys are independent across files', async () => {
  const { cache } = makeCache({ pageSize: 8, limit: 16 });
  const a = await cache.get('1:0');
  const b = await cache.get('2:0');
  assert.notEqual(a, b);
  assert.equal(cache.map.size, 2);
  assert.ok(cache.map.has('1:0') && cache.map.has('2:0'));
});

test('dirty pages are flushed on eviction, not dropped', async () => {
  const { cache, writes } = makeCache({ pageSize: 10, limit: 30, writes: [] });
  await cache.put('0:0', new Uint8Array(10)); // dirty
  await cache.put('0:1', new Uint8Array(10));
  await cache.put('0:2', new Uint8Array(10));
  await cache.put('0:3', new Uint8Array(10)); // forces eviction of 0:0
  assert.ok(writes.some((w) => w.key === '0:0'), 'evicted dirty page must be written back');
  assert.ok(cache.dirtyFlushes >= 1);
});

test('flush() writes every dirty page', async () => {
  const { cache, writes } = makeCache({ writes: [] });
  await cache.put('0:0', new Uint8Array(10));
  await cache.put('0:1', new Uint8Array(10));
  await cache.flush();
  assert.equal(writes.length, 2);
  await cache.flush(); // idempotent: nothing dirty anymore
  assert.equal(writes.length, 2);
});

test('pages may change length (tail pages grow), used stays accurate', async () => {
  const { cache } = makeCache({ pageSize: 16, limit: 64 });
  await cache.put('0:0', new Uint8Array(4));
  assert.equal(cache.used, 4);
  await cache.put('0:0', new Uint8Array(16)); // grown
  assert.equal(cache.used, 16);
  const back = await cache.get('0:0');
  assert.equal(back.length, 16);
});

test('a payload that cannot fit even after evicting everything fails loudly', async () => {
  const { cache } = makeCache({ pageSize: 10, limit: 15 });
  await assert.rejects(
    () => cache.put('0:0', new Uint8Array(20)),
    (err) => err instanceof BudgetExceeded && /raise --budget-mb/.test(err.message),
  );
});

test('validateQueryBudget: page vs budget, then codebooks + beam vs budget', () => {
  const tooSmallPage = validateQueryBudget({
    budgetBytes: 1024, pageBytes: 2048, ef: 64, dims: 384, k: 256,
  });
  assert.match(tooSmallPage, /< page size/);
  assert.match(tooSmallPage, /raise --budget-mb/);

  const tooSmallBeam = validateQueryBudget({
    budgetBytes: 100_000, pageBytes: 4096, ef: 64, dims: 384, k: 256,
  });
  assert.match(tooSmallBeam, /codebooks/);
  assert.match(tooSmallBeam, /at least \d+ MB/);

  const ok = validateQueryBudget({
    budgetBytes: 128 * 1024 * 1024, pageBytes: 65536, ef: 64, dims: 384, k: 256,
  });
  assert.equal(ok, null);
});
