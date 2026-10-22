// Store tests — buffer pool, paged records, byte-exact file growth,
// truncation detection, manifest validation, and write confinement.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {
  BufferPool, PagedFile, StoreError, readManifest, writeManifest, verifySizes,
  hashTree, confinedPath, ensureOutDir, removeDir, FORMAT_VERSION,
} from '../src/store.js';
import { tmpdir } from './helpers.js';

function scratchFile(name, bytes) {
  const dir = tmpdir('store');
  const file = path.join(dir, name);
  fs.writeFileSync(file, bytes);
  return { dir, file };
}

test('BufferPool readRange: exact, page-boundary, and cross-page reads', async () => {
  const payload = Uint8Array.from({ length: 40 }, (_, i) => i); // 0..39
  const { file } = scratchFile('data.bin', Buffer.from(payload));
  const pool = new BufferPool({ pageSize: 16, limitBytes: 1 << 16 });
  const fid = await pool.attach(file);

  assert.equal(pool.fileSize(fid), 40);
  const whole = await pool.readRange(fid, 0, 40);
  assert.deepEqual(Array.from(whole), Array.from(payload));

  const spanning = await pool.readRange(fid, 10, 20); // crosses pages 0|1
  assert.deepEqual(Array.from(spanning), Array.from(payload.slice(10, 30)));

  const exactTail = await pool.readRange(fid, 32, 8); // exactly the last page
  assert.deepEqual(Array.from(exactTail), Array.from(payload.slice(32)));

  await pool.close();
});

test('readRange beyond EOF is a truncation error with sizes in the message', async () => {
  const { file } = scratchFile('short.bin', Buffer.alloc(10));
  const pool = new BufferPool({ pageSize: 8, limitBytes: 1 << 16 });
  const fid = await pool.attach(file);
  await assert.rejects(
    () => pool.readRange(fid, 4, 12),
    (err) => err instanceof StoreError && /outside the 10-B file/.test(err.message) && /truncated/.test(err.message),
  );
  await pool.close();
});

test('writeRange extends files to EXACT size (no page padding)', async () => {
  const { file } = scratchFile('grow.bin', Buffer.alloc(0));
  const pool = new BufferPool({ pageSize: 4096, limitBytes: 1 << 16, writable: true });
  const fid = await pool.attach(file, { writable: true });

  const row = Uint8Array.from([9, 8, 7]);
  await pool.writeRange(fid, 0, row);
  assert.equal(pool.fileSize(fid), 3);

  await pool.writeRange(fid, 3, Uint8Array.from([6, 5])); // append
  assert.equal(pool.fileSize(fid), 5);
  await pool.flush();
  assert.equal(fs.statSync(file).size, 5, 'file must be exactly 5 bytes');
  assert.deepEqual(Array.from(fs.readFileSync(file)), [9, 8, 7, 6, 5]);
  await pool.close();
});

test('dirty pages written on flush; data survives a reopen', async () => {
  const { file } = scratchFile('persist.bin', Buffer.alloc(0));
  const pool = new BufferPool({ pageSize: 8, limitBytes: 1 << 16, writable: true });
  const fid = await pool.attach(file, { writable: true });
  await pool.writeRange(fid, 0, Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
  await pool.flush();
  await pool.close();

  const pool2 = new BufferPool({ pageSize: 8, limitBytes: 1 << 16 });
  const fid2 = await pool2.attach(file);
  const back = await pool2.readRange(fid2, 0, 10);
  assert.deepEqual(Array.from(back), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  await pool2.close();
});

test('eviction during build still lands every byte on disk (small pool)', async () => {
  // 200 records of stride 32 with a pool that holds ~8 pages forces churn.
  const { file } = scratchFile('churn.bin', Buffer.alloc(0));
  const pool = new BufferPool({ pageSize: 32, limitBytes: 32 * 4, writable: true });
  const fid = await pool.attach(file, { writable: true });
  const stride = 32;
  for (let i = 0; i < 200; i += 1) {
    const rec = new Uint8Array(stride);
    rec[0] = i & 0xff;
    rec[1] = (i >> 8) & 0xff;
    await pool.writeRange(fid, i * stride, rec);
  }
  await pool.flush();
  assert.equal(fs.statSync(file).size, 200 * stride);
  await pool.close();

  const buf = fs.readFileSync(file);
  for (const i of [0, 1, 127, 199]) {
    assert.equal(buf[i * stride], i & 0xff, `record ${i} low byte`);
    assert.equal(buf[i * stride + 1], (i >> 8) & 0xff, `record ${i} high byte`);
  }
  assert.ok(pool.stats().evictions > 0, 'pool must actually have evicted');
});

test('PagedFile: roundtrip, count refresh, stride validation, range errors', async () => {
  const { file } = scratchFile('recs.bin', Buffer.alloc(0));
  const pool = new BufferPool({ pageSize: 64, limitBytes: 1 << 16, writable: true });
  const fid = await pool.attach(file, { writable: true });
  const pf = new PagedFile({ pool, fid, stride: 8, name: 'recs.bin' });
  assert.equal(pf.count, 0);

  await pf.write(0, Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]));
  await pf.write(1, Uint8Array.from([9, 9, 9, 9, 9, 9, 9, 9]));
  assert.equal(pf.count, 2);
  assert.deepEqual(Array.from(await pf.read(1)), [9, 9, 9, 9, 9, 9, 9, 9]);
  assert.deepEqual(Array.from(await pf.read(0)), [1, 2, 3, 4, 5, 6, 7, 8]);
  await assert.rejects(() => pf.read(2), /out of range/);
  await assert.rejects(() => pf.write(0, Uint8Array.from([1])), /!= stride/);
  await pool.close();

  // a file whose size is not a multiple of stride must be rejected
  const { file: bad } = scratchFile('bad.bin', Buffer.alloc(5));
  const pool2 = new BufferPool({ pageSize: 16, limitBytes: 1 << 16 });
  const fid2 = await pool2.attach(bad);
  assert.throws(
    () => new PagedFile({ pool: pool2, fid: fid2, stride: 3, name: 'bad.bin' }),
    (err) => err instanceof StoreError && /not a multiple of stride/.test(err.message),
  );
  await pool2.close();
});

test('read-only pool refuses writes', async () => {
  const { file } = scratchFile('ro.bin', Buffer.alloc(4));
  const pool = new BufferPool({ pageSize: 8, limitBytes: 1 << 16 });
  const fid = await pool.attach(file);
  await assert.rejects(() => pool.writeRange(fid, 0, Uint8Array.from([1, 2])), /read-only/);
  await pool.close();
});

test('manifest: roundtrip, missing, invalid JSON, version mismatch', async () => {
  const dir = tmpdir('manifest');
  const manifest = {
    version: FORMAT_VERSION, docs: 3, dims: 8, files: { 'a.bin': { bytes: 5, sha256: 'ab' } },
  };
  await writeManifest(dir, manifest);
  const back = await readManifest(dir);
  assert.deepEqual(back, manifest);

  const empty = tmpdir('manifest');
  await assert.rejects(() => readManifest(empty), /no index at .* run `leanann index` first/);

  fs.writeFileSync(path.join(dir, 'manifest.json'), '{not json');
  await assert.rejects(() => readManifest(dir), /not valid JSON/);

  await writeManifest(dir, { ...manifest, version: 999 });
  await assert.rejects(() => readManifest(dir), /v999 is not this build's v1/);
});

test('verifySizes catches missing and truncated index files', async () => {
  const dir = tmpdir('sizes');
  await writeManifest(dir, {
    version: FORMAT_VERSION, files: { 'x.bin': { bytes: 10, sha256: 'aa' } },
  });
  fs.writeFileSync(path.join(dir, 'x.bin'), Buffer.alloc(10));
  await verifySizes(dir, { files: { 'x.bin': { bytes: 10 } } }); // ok

  fs.writeFileSync(path.join(dir, 'x.bin'), Buffer.alloc(7));
  await assert.rejects(
    () => verifySizes(dir, { files: { 'x.bin': { bytes: 10 } } }),
    /is 7 B, manifest says 10 B/,
  );
  fs.rmSync(path.join(dir, 'x.bin'));
  await assert.rejects(() => verifySizes(dir, { files: { 'x.bin': { bytes: 10 } } }), /missing x.bin/);
});

test('hashTree: stable, path-order independent, content sensitive', async () => {
  const a = tmpdir('hash');
  fs.writeFileSync(path.join(a, 'one.txt'), 'hello');
  fs.mkdirSync(path.join(a, 'sub'));
  fs.writeFileSync(path.join(a, 'sub', 'two.txt'), 'world');

  const h1 = await hashTree(a);
  const h2 = await hashTree(a);
  assert.equal(h1, h2);

  fs.writeFileSync(path.join(a, 'one.txt'), 'HELLO');
  const h3 = await hashTree(a);
  assert.notEqual(h1, h3);

  // single file hashes as content directly
  const f = path.join(a, 'one.txt');
  const { createHash } = await import('node:crypto');
  assert.equal(await hashTree(f), createHash('sha256').update('HELLO').digest('hex'));
});

test('confinedPath refuses traversal, allows nested paths', () => {
  const base = path.resolve(tmpdir('confined'), 'out');
  const ok = confinedPath(base, path.join('sub', 'file.bin'));
  assert.ok(ok.startsWith(base));
  assert.throws(() => confinedPath(base, '../evil.bin'), /outside the index dir/);
  assert.throws(() => confinedPath(base, path.join('..', '..', 'evil.bin')), /outside the index dir/);
});

test('ensureOutDir creates and removeDir cleans up', async () => {
  const parent = tmpdir('mkdir');
  const out = path.join(parent, 'nested', 'idx');
  const abs = await ensureOutDir(out);
  assert.ok(fs.existsSync(abs));
  await removeDir(path.join(parent, 'nested'));
  assert.ok(!fs.existsSync(path.join(parent, 'nested')));
});
