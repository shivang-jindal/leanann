// Safety tests — the write-confinement and read-only guarantees, no-network
// static check, and temp-file hygiene on FAILURE paths too.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { buildIndex, checkIndex } from '../src/indexer.js';
import { search, stats } from '../src/search.js';
import { embedViaCommand } from '../src/embed.js';
import { confinedPath } from '../src/store.js';
import { main } from '../src/cli.js';
import { TEST_DIR, capture, tmpdir, writeJsonl, fingerprintTree } from './helpers.js';

const FAST = {
  dims: 16, m: 4, k: 16, iters: 4, graphM: 8, budgetMb: 8, pageKb: 4, seed: 42,
};

function docs(n = 12) {
  return Array.from({ length: n }, (_, i) => ({
    id: `s-${i}`,
    text: `safety document number ${i} about pools and caches`,
  }));
}

async function builtIndex(prefix) {
  const dir = tmpdir(prefix);
  const file = path.join(dir, 'corpus.jsonl');
  writeJsonl(file, docs());
  const out = path.join(dir, 'idx');
  await buildIndex({ input: file, out, ...FAST });
  return { dir, file, out };
}

test('search, stats, and --check never write to the index (fingerprints equal)', async () => {
  const { dir, file, out } = await builtIndex('safe-ro');
  const before = fingerprintTree(out);

  const s = await search({ dir: out, query: 'pools and caches', k: 5, budgetMb: 8, pageKb: 4 });
  assert.ok(s.results.length > 0);
  await search({ dir: out, query: 'number 3', k: 3, mode: 'bm25', budgetMb: 8, pageKb: 4 });
  const st = await stats({ dir: out });
  assert.equal(st.docs, 12);
  const ck = await checkIndex({ input: file, out });
  assert.equal(ck.stale, false);

  const after = fingerprintTree(out);
  assert.equal(after, before, 'read commands must leave every byte untouched');
  void dir;
});

test('build writes ONLY into --out (parent listing gains exactly one entry)', async () => {
  const parent = tmpdir('safe-parent');
  const file = path.join(parent, 'corpus.jsonl');
  writeJsonl(file, docs(6));
  const parentBefore = fs.readdirSync(parent).sort();

  const out = path.join(parent, 'nested-out');
  await buildIndex({ input: file, out, ...FAST });

  const parentAfter = fs.readdirSync(parent).sort();
  assert.deepEqual(parentAfter, [...parentBefore, 'nested-out'].sort(), 'only the out dir appears');
});

test('confinedPath: traversal attempts are refused with a StoreError-shaped error', () => {
  const base = path.join(tmpdir('safe-confined'), 'out');
  assert.ok(confinedPath(base, 'codes.bin').startsWith(base));
  assert.throws(
    () => confinedPath(base, path.join('..', 'escaped.bin')),
    (err) => /refusing to write outside the index dir/.test(err.message),
  );
  assert.throws(() => confinedPath(base, path.join('..', '..', 'x')), /outside the index dir/);
});

test('no network surfaces in src (zero-dependency promise kept)', () => {
  const srcDir = path.join(TEST_DIR, '..', 'src');
  const banned = [
    /node:net\b/, /node:http\b/, /node:https\b/, /node:dgram\b/, /node:tls\b/,
    /\bfetch\s*\(/, /require\(['"]undici/, /XMLHttpRequest/,
  ];
  for (const name of fs.readdirSync(srcDir)) {
    if (!name.endsWith('.js')) continue;
    const text = fs.readFileSync(path.join(srcDir, name), 'utf8');
    for (const re of banned) {
      assert.ok(!re.test(text), `${name} must not match ${re}`);
    }
  }
});

test('helpers.tmpdir lands under the system temp dir (tests write nowhere else)', () => {
  const d = tmpdir('safe-tmp');
  assert.ok(d.startsWith(os.tmpdir()), `${d} under ${os.tmpdir()}`);
  fs.rmSync(d, { recursive: true, force: true });
});

test('embed-cmd temp exchange files are removed even when exec FAILS', async () => {
  const dir = tmpdir('safe-embed');
  const failingExec = async () => { throw new Error('boom'); };
  await assert.rejects(
    () => embedViaCommand('bad {input} {output}', ['a', 'b'], {
      dims: 4, exec: failingExec, makeTmpDir: () => dir,
    }),
    /boom/,
  );
  const leftovers = fs.readdirSync(dir).filter((f) => f.includes('leanann-embed-'));
  assert.deepEqual(leftovers, [], 'failed embed must still clean up');
});

test('CLI pipeline in tmp leaves the tool source tree untouched', async () => {
  const srcBefore = fingerprintTree(path.join(TEST_DIR, '..', 'src'));

  const dir = tmpdir('safe-cli');
  writeJsonl(path.join(dir, 'corpus.jsonl'), docs(6));
  const cap = capture();
  const flags = ['--dims', '16', '--m', '4', '--k', '16', '--iters', '4', '--graph-m', '8', '--budget-mb', '8', '--page-kb', '4'];
  const build = await main(['index', 'corpus.jsonl', '--out', 'idx', ...flags], {
    stdout: cap.stdout, stderr: cap.stderr, cwd: () => dir,
  });
  assert.equal(build, 0, cap.errors());
  const query = await main(['search', 'pools and caches', '--out', 'idx'], {
    stdout: cap.stdout, stderr: cap.stderr, cwd: () => dir,
  });
  assert.equal(query, 0, cap.errors());

  const srcAfter = fingerprintTree(path.join(TEST_DIR, '..', 'src'));
  assert.equal(srcAfter, srcBefore, 'the tool never modifies its own source');
});
