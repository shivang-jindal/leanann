// embed.js — three vector sources: hash (built-in), embed-cmd (user-trusted
// external model), passthrough (vector field in the input).

import { spawn } from 'node:child_process';
import { readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fnv1a, l2normalize, isFiniteVector } from './util.js';
import { tokenize } from './tokenize.js';

// Deterministic feature-hash embedding: each token votes ±1 into one bucket,
// then L2-normalize. Lexical, instant, zero-dep. Honest about what it is:
// good for dedup/lexical similarity; use --embed-cmd for semantics.
export function hashEmbed(text, dims = 384) {
  if (!Number.isInteger(dims) || dims < 2) throw new RangeError(`bad dims ${dims}`);
  const v = new Float32Array(dims);
  for (const tok of tokenize(text)) {
    const h = fnv1a(tok);
    v[h % dims] += h & 0x80000000 ? -1 : 1;
  }
  return l2normalize(v);
}

export function validateEmbedding(v, dims, where) {
  if (!isFiniteVector(v, dims)) {
    const arrayish = Array.isArray(v) || ArrayBuffer.isView(v);
    const got = arrayish ? `${v.length}` : (v === null ? 'null' : typeof v);
    throw new Error(`${where}: expected a finite vector of ${dims} dims, got ${got}`);
  }
  return Float32Array.from(v);
}

// Run the user's embedder over texts.
//   {input}  -> JSONL  [{"i":0,"text":"..."}, ...]
//   {output} -> JSONL  [{"i":0,"vector":[...]}, ...]
// Command is user-trusted (their own), shell-executed, with timeout + caps.
export async function embedViaCommand(cmd, texts, {
  dims,
  exec = null,
  timeoutMs = 120_000,
  maxOutputBytes = 64 * 1024 * 1024,
  makeTmpDir = () => tmpdir(),
} = {}) {
  if (!cmd || typeof cmd !== 'string') throw new Error('embed-cmd: empty command');
  if (!cmd.includes('{input}') || !cmd.includes('{output}')) {
    throw new Error('embed-cmd: command must contain {input} and {output} placeholders');
  }
  const dir = makeTmpDir();
  const inputPath = join(dir, `leanann-embed-${process.pid}-${Date.now()}.in.jsonl`);
  const outputPath = join(dir, `leanann-embed-${process.pid}-${Date.now()}.out.jsonl`);
  const finalCmd = cmd.replace('{input}', inputPath).replace('{output}', outputPath);

  await writeFile(inputPath, texts.map((t, i) => JSON.stringify({ i, text: t })).join('\n') + '\n', 'utf8');
  try {
    if (exec) {
      await exec(finalCmd, { timeout: timeoutMs });
    } else {
      await runShell(finalCmd, timeoutMs);
    }
    const raw = await readFile(outputPath, 'utf8');
    if (Buffer.byteLength(raw, 'utf8') > maxOutputBytes) {
      throw new Error(`embed-cmd: output exceeds ${maxOutputBytes} bytes cap`);
    }
    const vectors = new Array(texts.length).fill(null);
    let seen = 0;
    for (const line of raw.split('\n')) {
      if (!line.trim()) continue;
      const row = JSON.parse(line);
      if (!Number.isInteger(row.i) || row.i < 0 || row.i >= texts.length) {
        throw new Error(`embed-cmd: bad index ${row.i} in output`);
      }
      vectors[row.i] = validateEmbedding(row.vector, dims, `embed-cmd row ${row.i}`);
      seen += 1;
    }
    if (seen !== texts.length) {
      throw new Error(`embed-cmd: expected ${texts.length} vectors, got ${seen}`);
    }
    return vectors;
  } finally {
    await rm(inputPath, { force: true }).catch(() => {});
    await rm(outputPath, { force: true }).catch(() => {});
  }
}

function runShell(cmd, timeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, { shell: true, stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    let done = false;
    const finish = (fn, arg) => { if (!done) { done = true; fn(arg); } };
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(reject, new Error(`embed-cmd: timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    child.stderr.on('data', (d) => { if (stderr.length < 8192) stderr += d; });
    child.on('error', (err) => { clearTimeout(timer); finish(reject, err); });
    child.on('close', (code) => {
      clearTimeout(timer);
      if (code === 0) finish(resolve, undefined);
      else finish(reject, new Error(`embed-cmd: exit ${code}${stderr ? `: ${stderr.trim()}` : ''}`));
    });
  });
}

// Resolve which embedder a build uses. Returns a function (texts) => vectors.
export async function makeEmbedder({ embed = 'hash', embedCmd = null, dims = 384, exec = null }) {
  if (embedCmd) {
    return (texts) => embedViaCommand(embedCmd, texts, { dims, exec });
  }
  if (embed === 'hash') {
    return async (texts) => texts.map((t) => hashEmbed(t, dims));
  }
  if (embed === 'none') {
    throw new Error('--embed none: input must carry a "vector" field per document');
  }
  throw new Error(`unknown embedder: ${embed}`);
}
