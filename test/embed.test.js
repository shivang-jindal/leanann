// Embedder tests — deterministic hash embedding, validation rails, and the
// user-trusted --embed-cmd path driven by an injected fake exec.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { hashEmbed, validateEmbedding, embedViaCommand, makeEmbedder } from '../src/embed.js';
import { fakeEmbedExec, tmpdir } from './helpers.js';

test('hashEmbed is deterministic per text', () => {
  const a = hashEmbed('database connection pool', 32);
  const b = hashEmbed('database connection pool', 32);
  assert.deepEqual(Array.from(a), Array.from(b));
  assert.equal(a.length, 32);
});

test('hashEmbed L2-normalizes (unit length for non-empty text)', () => {
  const v = hashEmbed('page cache thrashing under load', 64);
  let sum = 0;
  for (const x of v) sum += x * x;
  assert.ok(Math.abs(Math.sqrt(sum) - 1) < 1e-6, `norm ${Math.sqrt(sum)}`);
});

test('hashEmbed: empty text is the honest zero vector', () => {
  const v = hashEmbed('', 16);
  assert.ok(Array.from(v).every((x) => x === 0));
});

test('hashEmbed: different texts produce different vectors', () => {
  const a = hashEmbed('alpha beta', 32);
  const b = hashEmbed('gamma delta', 32);
  assert.notDeepEqual(Array.from(a), Array.from(b));
});

test('hashEmbed validates dims', () => {
  assert.throws(() => hashEmbed('x', 1), /bad dims/);
  assert.throws(() => hashEmbed('x', 2.5), /bad dims/);
});

test('validateEmbedding: dims, finiteness, type', () => {
  const ok = validateEmbedding([1, 2, 3], 3, 'where');
  assert.ok(ok instanceof Float32Array);
  assert.throws(() => validateEmbedding([1, 2], 3, 'where'), /where: expected a finite vector of 3 dims, got 2/);
  assert.throws(() => validateEmbedding([1, NaN, 3], 3, 'where'), /finite vector/);
  assert.throws(() => validateEmbedding('nope', 3, 'where'), /got string/);
});

test('embedViaCommand: happy path returns vectors in input order', async () => {
  const dir = tmpdir('embed');
  const texts = ['first document', 'second document', 'third'];
  const vecs = await embedViaCommand('fake-embed {input} {output}', texts, {
    dims: 16,
    exec: fakeEmbedExec({ dims: 16 }),
    makeTmpDir: () => dir,
  });
  assert.equal(vecs.length, 3);
  for (const v of vecs) assert.equal(v.length, 16);
  // different texts -> different vectors
  assert.notDeepEqual(Array.from(vecs[0]), Array.from(vecs[1]));
  // both temp exchange files are removed
  const leftovers = fs.readdirSync(dir).filter((f) => f.includes('leanann-embed-'));
  assert.deepEqual(leftovers, [], 'temp exchange files must be cleaned up');
});

test('embedViaCommand: requires {input} and {output} placeholders', async () => {
  await assert.rejects(
    () => embedViaCommand('embed everything', ['a'], { dims: 4, exec: fakeEmbedExec() }),
    /must contain \{input\} and \{output\}/,
  );
  await assert.rejects(
    () => embedViaCommand('embed {input} only', ['a'], { dims: 4, exec: fakeEmbedExec() }),
    /must contain \{input\} and \{output\}/,
  );
  await assert.rejects(
    () => embedViaCommand('', ['a'], { dims: 4, exec: fakeEmbedExec() }),
    /empty command/,
  );
});

test('embedViaCommand: exec failure propagates with cause', async () => {
  await assert.rejects(
    () => embedViaCommand('bad {input} {output}', ['a'], { dims: 4, exec: fakeEmbedExec({ fail: true }) }),
    /embedder exploded/,
  );
});

test('embedViaCommand: wrong dims from the embedder is caught', async () => {
  await assert.rejects(
    () => embedViaCommand('e {input} {output}', ['a', 'b'], {
      dims: 16, exec: fakeEmbedExec({ dims: 16, wrongDims: true }),
    }),
    /expected a finite vector of 16 dims, got 18/,
  );
});

test('embedViaCommand: missing output rows are counted honestly', async () => {
  const dir = tmpdir('embed');
  const shortExec = async (cmd) => {
    const parts = cmd.trim().split(/\s+/);
    fs.writeFileSync(parts[parts.length - 1], JSON.stringify({ i: 0, vector: [1, 2] }) + '\n');
  };
  await assert.rejects(
    () => embedViaCommand('s {input} {output}', ['a', 'b'], { dims: 2, exec: shortExec, makeTmpDir: () => dir }),
    /expected 2 vectors, got 1/,
  );
});

test('embedViaCommand: output size cap is enforced', async () => {
  const dir = tmpdir('embed');
  const bigExec = async (cmd) => {
    const parts = cmd.trim().split(/\s+/);
    const row = { i: 0, vector: Array(16).fill(1) };
    fs.writeFileSync(parts[parts.length - 1], (JSON.stringify(row) + '\n').repeat(50));
  };
  await assert.rejects(
    () => embedViaCommand('b {input} {output}', ['a'], {
      dims: 16, exec: bigExec, makeTmpDir: () => dir, maxOutputBytes: 64,
    }),
    /exceeds 64 bytes/,
  );
});

test('makeEmbedder: hash and embed-cmd wiring', async () => {
  const hashFn = await makeEmbedder({ embed: 'hash', dims: 8 });
  const vecs = await hashFn(['hello world']);
  assert.equal(vecs.length, 1);
  assert.equal(vecs[0].length, 8);

  const cmdFn = await makeEmbedder({ embed: 'vector', embedCmd: 'e {input} {output}', dims: 16, exec: fakeEmbedExec({ dims: 16 }) });
  const viaCmd = await cmdFn(['text']);
  assert.equal(viaCmd[0].length, 16);

  await assert.rejects(() => makeEmbedder({ embed: 'bogus', dims: 8 }), /unknown embedder/);
  await assert.rejects(() => makeEmbedder({ embed: 'none', dims: 8 }), /--embed none/);
});
