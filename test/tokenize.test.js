// tokenizer tests — determinism is the contract (BM25 + hash embedder share it).
import test from 'node:test';
import assert from 'node:assert/strict';
import { tokenize, termFreq, STOPWORDS } from '../src/tokenize.js';

test('tokenize lowercases and splits on non-alphanumeric', () => {
  assert.deepEqual(tokenize('Hello, World!'), ['hello', 'world']);
});

test('tokenize drops short non-numeric tokens', () => {
  assert.deepEqual(tokenize('a bb c ddd'), ['bb', 'ddd']);
});

test('tokenize keeps short numerics', () => {
  assert.deepEqual(tokenize('port 80 data'), ['port', '80', 'data']);
  assert.deepEqual(tokenize('x 7'), ['7']); // 1-char non-numeric dropped, numeric kept
});

test('tokenize keeps underscores (identifiers matter for code corpora)', () => {
  assert.deepEqual(tokenize('connect pool_size now'), ['connect', 'pool_size', 'now']);
});

test('tokenize drops stopwords by default, keeps them with stop:false', () => {
  assert.deepEqual(tokenize('the quick fox'), ['quick', 'fox']);
  assert.deepEqual(tokenize('the quick fox', { stop: false }), ['the', 'quick', 'fox']);
});

test('stopword list contains the usual suspects', () => {
  for (const w of ['the', 'and', 'with', 'this']) assert.ok(STOPWORDS.has(w), w);
});

test('tokenize emits CJK codepoints as unigrams', () => {
  const toks = tokenize('数据库连接');
  assert.deepEqual(toks, ['数', '据', '库', '连', '接']);
});

test('tokenize handles empty and nullish input', () => {
  assert.deepEqual(tokenize(''), []);
  assert.deepEqual(tokenize(null), []);
  assert.deepEqual(tokenize(undefined), []);
});

test('tokenize is deterministic', () => {
  const a = tokenize('Reconnect the connection pool after a storm');
  const b = tokenize('Reconnect the connection pool after a storm');
  assert.deepEqual(a, b);
});

test('termFreq counts repeats', () => {
  const tf = termFreq(tokenize('cache cache miss'));
  assert.equal(tf.get('cache'), 2);
  assert.equal(tf.get('miss'), 1);
  assert.ok(!tf.has('the'));
});
