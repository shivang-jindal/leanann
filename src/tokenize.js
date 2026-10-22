// tokenize.js — deterministic tokenizer shared by BM25 and the hash embedder.
//
// v0 scope (documented in SPEC §4.1): case folding, [a-z0-9_]+ runs, CJK
// unigrams, drop short tokens (unless numeric), English stopword list.
// No stemming — deferred to v1, and honestly labeled.

export const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', 'did',
  'do', 'does', 'for', 'from', 'had', 'has', 'have', 'how', 'i', 'if', 'in',
  'into', 'is', 'it', 'its', 'me', 'my', 'no', 'not', 'of', 'on', 'or', 'our',
  'out', 'she', 'so', 'that', 'the', 'their', 'them', 'then', 'there', 'these',
  'they', 'this', 'to', 'too', 'up', 'us', 'was', 'we', 'were', 'what', 'when',
  'which', 'who', 'why', 'will', 'with', 'you', 'your',
]);

const CJK = /[㐀-䶿一-鿿豈-﫿぀-ヿ]/;
const RUN = /[a-z0-9_]+/g;

// Returns an array of tokens. Deterministic for a given input.
export function tokenize(text, { stop = true } = {}) {
  const src = String(text ?? '').toLowerCase();
  const out = [];
  let m;
  RUN.lastIndex = 0;
  while ((m = RUN.exec(src)) !== null) {
    const tok = m[0];
    const numeric = /^\d+$/.test(tok);
    if (tok.length < 2 && !numeric) continue;
    if (stop && STOPWORDS.has(tok)) continue;
    out.push(tok);
  }
  // CJK: one token per character (unigrams). Honest v0 limitation — no
  // segmentation, no n-grams — but deterministic and better than dropping.
  for (const ch of src) {
    if (CJK.test(ch)) out.push(ch);
  }
  return out;
}

export function termFreq(tokens) {
  const tf = new Map();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  return tf;
}
