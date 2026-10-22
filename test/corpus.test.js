// Corpus ingestion tests — JSONL parsing rules, directory walking, exclude
// globs, and honest line-numbered errors.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { readCorpus, countCorpus, globToRegExp, makeExcluder, CorpusError } from '../src/corpus.js';
import { tmpdir, writeJsonl } from './helpers.js';

async function collect(iter) {
  const out = [];
  for await (const item of iter) out.push(item);
  return out;
}

test('JSONL: id defaults to doc-N, custom ids preserved, text/content/body accepted', async () => {
  const dir = tmpdir('corpus');
  const file = path.join(dir, 'c.jsonl');
  fs.writeFileSync(file, [
    JSON.stringify({ text: 'first' }),
    JSON.stringify({ id: 'custom', content: 'second' }),
    JSON.stringify({ id: 'b', body: 'third' }),
  ].join('\n') + '\n');

  const docs = await collect(readCorpus({ input: file }));
  assert.deepEqual(docs.map((d) => d.id), ['doc-0', 'custom', 'b']);
  assert.deepEqual(docs.map((d) => d.text), ['first', 'second', 'third']);
  assert.equal(docs[0].vector, null);
});

test('JSONL: vector passthrough validated against dims', async () => {
  const dir = tmpdir('corpus');
  const good = path.join(dir, 'good.jsonl');
  writeJsonl(good, [{ id: 'v', text: 'has vector', vector: [1, 0, -1] }]);
  const docs = await collect(readCorpus({ input: good, dims: 3 }));
  assert.deepEqual(docs[0].vector, [1, 0, -1]);

  const bad = path.join(dir, 'bad.jsonl');
  writeJsonl(bad, [{ id: 'v', text: 'wrong dims', vector: [1, 0] }]);
  await assert.rejects(
    () => collect(readCorpus({ input: bad, dims: 3 })),
    (err) => err instanceof CorpusError && /"vector" must be 3 finite numbers \(got 2\)/.test(err.message),
  );

  const nan = path.join(dir, 'nan.jsonl');
  writeJsonl(nan, [{ id: 'v', text: 'nan', vector: [1, null, 3] }]);
  await assert.rejects(() => collect(readCorpus({ input: nan, dims: 3 })), /finite numbers/);
});

test('JSONL: invalid JSON and missing text report the line number', async () => {
  const dir = tmpdir('corpus');
  const broken = path.join(dir, 'broken.jsonl');
  fs.writeFileSync(broken, '{"text":"ok"}\n{oops\n');
  await assert.rejects(
    () => collect(readCorpus({ input: broken })),
    (err) => err instanceof CorpusError && /broken\.jsonl:2: invalid JSON/.test(err.message),
  );

  const noText = path.join(dir, 'notext.jsonl');
  fs.writeFileSync(noText, '{"id":"x"}\n');
  await assert.rejects(() => collect(readCorpus({ input: noText })), /needs a "text" field/);
});

test('JSONL: empty corpus and missing input are honest errors', async () => {
  const dir = tmpdir('corpus');
  const empty = path.join(dir, 'empty.jsonl');
  fs.writeFileSync(empty, '\n\n');
  await assert.rejects(() => collect(readCorpus({ input: empty })), /no documents/);
  await assert.rejects(() => collect(readCorpus({ input: path.join(dir, 'nope.jsonl') })), /input not found/);
});

test('directory input: only txt/md, deterministic order, skips excludes', async () => {
  const dir = tmpdir('corpusdir');
  fs.writeFileSync(path.join(dir, 'b.md'), '# b doc');
  fs.writeFileSync(path.join(dir, 'a.txt'), 'a doc');
  fs.mkdirSync(path.join(dir, 'nested'));
  fs.writeFileSync(path.join(dir, 'nested', 'c.txt'), 'c doc');
  fs.writeFileSync(path.join(dir, 'ignored.png'), 'not text');
  fs.writeFileSync(path.join(dir, 'skip.txt'), 'excluded doc');

  const docs = await collect(readCorpus({ input: dir, exclude: ['skip.txt', 'nested/**'] }));
  // readdir is sorted and dirs are stacked, so traversal order is deterministic
  assert.deepEqual(docs.map((d) => d.id), ['a.txt', 'b.md']);
  const all = await collect(readCorpus({ input: dir, exclude: ['**/node_modules/**'] }));
  assert.deepEqual(all.map((d) => d.id), ['a.txt', 'b.md', 'skip.txt', 'nested/c.txt']);
  assert.ok(!all.some((d) => d.id.endsWith('.png')));
  const texts = Object.fromEntries(all.map((d) => [d.id, d.text]));
  assert.equal(texts['nested/c.txt'], 'c doc');
});

test('directory input: no matching files after excludes -> error', async () => {
  const dir = tmpdir('corpusdir');
  fs.writeFileSync(path.join(dir, 'only.bin'), 'binary-ish');
  await assert.rejects(
    () => collect(readCorpus({ input: dir })),
    /no \.txt\/\.md files under/,
  );
});

test('globToRegExp: ** crosses separators, * does not, ? one char', () => {
  assert.ok(globToRegExp('**/*.md').test('a/b/c.md'));
  assert.ok(globToRegExp('**/*.md').test('c.md'));
  assert.ok(!globToRegExp('*.md').test('a/c.md'));
  assert.ok(globToRegExp('*.md').test('c.md'));
  assert.ok(globToRegExp('doc?.txt').test('doc1.txt'));
  assert.ok(!globToRegExp('doc?.txt').test('doc12.txt'));
  assert.ok(globToRegExp('a.b.txt').test('a.b.txt'));
  assert.ok(!globToRegExp('axb.txt').test('a.b.txt'), 'dot is literal, not any-char');
});

test('makeExcluder: any pattern matches, empty pattern list excludes nothing', () => {
  const ex = makeExcluder(['build/**', '*.tmp']);
  assert.ok(ex('build/out/x.js'));
  assert.ok(ex('thing.tmp'));
  assert.ok(!ex('src/thing.js'));
  const none = makeExcluder([]);
  assert.ok(!none('anything.txt'));
});

test('countCorpus counts both formats', async () => {
  const dir = tmpdir('corpus');
  const file = path.join(dir, 'c.jsonl');
  writeJsonl(file, [{ text: 'a' }, { text: 'b' }, { text: 'c' }]);
  assert.equal(await countCorpus(file), 3);
  const dir2 = tmpdir('corpusdir');
  fs.writeFileSync(path.join(dir2, 'x.txt'), 'x');
  assert.equal(await countCorpus(dir2), 1);
});
