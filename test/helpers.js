// test/helpers.js — shared utilities for the leanann test suite.
//
// Everything is test-only: throwaway dirs under os.tmpdir() (the only place
// tests ever write), deterministic corpus generators, output capture for the
// injectable CLI streams, and a fake `exec` that impersonates an external
// embedding command — no network, no monkey-patching, no fixture mutation.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { mulberry32 } from '../src/util.js';

export const TEST_DIR = path.dirname(fileURLToPath(import.meta.url));
export const FIXTURES_DIR = path.join(TEST_DIR, 'fixtures');
export const fixturePath = (name) => path.join(FIXTURES_DIR, name);

/** Fresh throwaway directory under the system temp dir. */
export function tmpdir(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), `leanann-${prefix}-`));
}

/** Write [{id, text, vector?}, ...] as JSONL and return the file path. */
export function writeJsonl(file, docs) {
  const body = docs.map((d) => JSON.stringify(d)).join('\n') + '\n';
  fs.writeFileSync(file, body, 'utf8');
  return file;
}

/** Create a JSONL corpus in a fresh tmpdir; returns { dir, file, docs }. */
export function corpusDir(prefix, docs) {
  const dir = tmpdir(prefix);
  const file = path.join(dir, 'corpus.jsonl');
  writeJsonl(file, docs);
  return { dir, file, docs };
}

// Fast build defaults for tests: tiny dims/codebooks, small budget so the
// page cache actually does its job. Every test still exercises the same code.
export const FAST = {
  dims: 16,
  m: 4,
  k: 16,
  iters: 4,
  graphM: 8,
  budgetMb: 8,
  pageKb: 4,
  seed: 42,
};

/**
 * Topic-clustered corpus generator (deterministic for a given seed).
 * `topics` is an array of token pools; docs mix one pool's tokens with noise.
 * Used for recall/hybrid tests where lexical structure matters.
 */
const NOISE = ['alpha', 'beta', 'gamma', 'delta', 'zeta', 'sigma', 'kappa', 'omega'];

export function clusteredCorpus({ perTopic = 10, topics, seed = 7 } = {}) {
  const rng = mulberry32(seed);
  const docs = [];
  topics.forEach((pool, ti) => {
    for (let i = 0; i < perTopic; i += 1) {
      const words = [];
      // two tokens from the topic pool (shared within topic)
      words.push(pool[Math.floor(rng() * pool.length)]);
      words.push(pool[Math.floor(rng() * pool.length)]);
      // one noise token (shared across corpus)
      words.push(NOISE[Math.floor(rng() * NOISE.length)]);
      // one instance-specific token (keeps docs distinguishable)
      words.push(`t${ti}x${i}`);
      docs.push({ id: `topic${ti}-${i}`, text: words.join(' ') });
    }
  });
  return docs;
}

/** Stable content fingerprint of a directory: sorted relpaths + sha256. */
export function fingerprintTree(root) {
  const entries = [];
  const walk = (dir, rel) => {
    let names;
    try {
      names = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    names.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    for (const e of names) {
      const relPath = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) walk(path.join(dir, e.name), relPath);
      else {
        const sha = createHash('sha256')
          .update(fs.readFileSync(path.join(dir, e.name)))
          .digest('hex');
        entries.push(`${relPath} ${sha}`);
      }
    }
  };
  walk(root, '');
  return entries.join('\n');
}

/** Capture injectable stdout/stderr for CLI tests. */
export function capture() {
  const out = [];
  const err = [];
  return {
    stdout: (s) => out.push(s),
    stderr: (s) => err.push(s),
    text: () => out.join(''),
    errors: () => err.join(''),
    json: () => JSON.parse(out.join('')),
    out,
    err,
  };
}

/**
 * Fake `exec` impersonating an external embedder command.
 * embedViaCommand substitutes {input}/{output} BEFORE calling exec, so the
 * command arrives as `<something> <inPath> <outPath>`; the fake reads the
 * input JSONL and writes deterministic hash vectors of `dims` dimensions.
 */
export function fakeEmbedExec({ dims = 16, fail = false, wrongDims = false } = {}) {
  return async (cmd) => {
    if (fail) throw new Error('embedder exploded');
    const parts = cmd.trim().split(/\s+/);
    const inPath = parts[parts.length - 2];
    const outPath = parts[parts.length - 1];
    const lines = fs.readFileSync(inPath, 'utf8').split('\n').filter((l) => l.trim());
    const rows = lines.map((line) => {
      const { i, text } = JSON.parse(line);
      const d = wrongDims ? dims + 2 : dims;
      const v = new Array(d).fill(0);
      for (const tok of String(text).toLowerCase().split(/\W+/).filter(Boolean)) {
        let h = 0x811c9dc5;
        h = Math.imul(h ^ tok.charCodeAt(0), 0x01000193) >>> 0;
        v[h % d] += 1;
      }
      return { i, vector: v };
    });
    fs.writeFileSync(outPath, rows.map((r) => JSON.stringify(r)).join('\n') + '\n', 'utf8');
    return undefined;
  };
}

/** Read a file as text or null when missing. */
export function readMaybe(file) {
  try {
    return fs.readFileSync(file, 'utf8');
  } catch {
    return null;
  }
}
