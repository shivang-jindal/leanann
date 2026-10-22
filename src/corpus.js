// corpus.js — input ingestion: JSONL ({"id","text","vector"?}) or a directory
// of .txt/.md files. Deterministic ordering, exclude globs, honest errors.

import { readFile, readdir, stat } from 'node:fs/promises';
import { join, relative, resolve } from 'node:path';
import { isFiniteVector } from './util.js';

export class CorpusError extends Error {
  constructor(message) {
    super(message);
    this.name = 'CorpusError';
    this.code = 'CORPUS_ERROR';
  }
}

// Minimal glob: ** (cross-sep), * (within-sep), ? (one char). Anchored.
export function globToRegExp(glob) {
  let re = '';
  let i = 0;
  const g = glob.replaceAll('\\', '/');
  while (i < g.length) {
    const c = g[i];
    if (c === '*') {
      if (g[i + 1] === '*') {
        // '**/' or trailing '**'
        if (g[i + 2] === '/') { re += '(?:.*/)?'; i += 3; continue; }
        re += '.*'; i += 2; continue;
      }
      re += '[^/]*'; i += 1; continue;
    }
    if (c === '?') { re += '[^/]'; i += 1; continue; }
    re += c.replace(/[.+^${}()|[\]\\]/g, '\\$&');
    i += 1;
  }
  return new RegExp(`^${re}$`);
}

export function makeExcluder(patterns) {
  const regs = patterns.filter(Boolean).map(globToRegExp);
  return (relPath) => {
    const p = relPath.replaceAll('\\', '/');
    return regs.some((r) => r.test(p));
  };
}

const TEXT_EXT = new Set(['.txt', '.md', '.markdown']);

async function* walkDir(root, ex) {
  const stack = [root];
  while (stack.length > 0) {
    const dir = stack.pop();
    const names = (await readdir(dir)).sort();
    for (const name of names) {
      const abs = join(dir, name);
      const st = await stat(abs);
      if (st.isDirectory()) { stack.push(abs); continue; }
      const ext = name.slice(name.lastIndexOf('.')).toLowerCase();
      if (!TEXT_EXT.has(ext)) continue;
      const rel = relative(root, abs).replaceAll('\\', '/');
      if (ex(rel)) continue;
      yield { id: rel, path: abs };
    }
  }
}

// Async generator of { id, text, vector } in deterministic order.
// Called twice by the indexer (pass 1 encode/meta, pass 2 codes+graph).
export async function* readCorpus({ input, exclude = [], dims = null }) {
  const ex = makeExcluder(exclude);
  const abs = resolve(input);
  const st = await stat(abs).catch(() => null);
  if (!st) throw new CorpusError(`input not found: ${input}`);

  if (st.isDirectory()) {
    let n = 0;
    for await (const file of walkDir(abs, ex)) {
      const text = await readFile(file.path, 'utf8');
      const doc = { id: file.id, text, vector: null };
      yield doc;
      n += 1;
    }
    if (n === 0) throw new CorpusError(`no .txt/.md files under ${input} (after excludes)`);
    return;
  }

  // JSONL file
  const raw = await readFile(abs, 'utf8');
  let n = 0;
  let lineNo = 0;
  for (const line of raw.split('\n')) {
    lineNo += 1;
    if (line.trim() === '') continue;
    let row;
    try {
      row = JSON.parse(line);
    } catch {
      throw new CorpusError(`${input}:${lineNo}: invalid JSON line`);
    }
    const text = typeof row.text === 'string' ? row.text
      : typeof row.content === 'string' ? row.content
        : typeof row.body === 'string' ? row.body : null;
    if (text === null && !row.vector) {
      throw new CorpusError(`${input}:${lineNo}: line needs a "text" field (or a "vector")`);
    }
    let vector = null;
    if (row.vector != null) {
      if (dims != null && !isFiniteVector(row.vector, dims)) {
        throw new CorpusError(
          `${input}:${lineNo}: "vector" must be ${dims} finite numbers (got ` +
          `${Array.isArray(row.vector) ? row.vector.length : typeof row.vector})`,
        );
      }
      vector = row.vector;
    }
    const id = typeof row.id === 'string' && row.id !== '' ? row.id : `doc-${n}`;
    yield { id, text: text ?? '', vector };
    n += 1;
  }
  if (n === 0) throw new CorpusError(`${input}: no documents`);
}

export async function countCorpus(input, exclude = []) {
  let n = 0;
  for await (const doc of readCorpus({ input, exclude })) {
    void doc;
    n += 1;
  }
  return n;
}
