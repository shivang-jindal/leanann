// indexer.js — two-pass build that never exceeds the RAM budget.
//
//   pass 1: tokenize -> BM25 sorted RUNS + meta sidecar + staged vectors.f32
//           (dims × 4 × N bytes staged on DISK, never in RAM)
//   train:  codebooks from a budget-capped reservoir sample (resident, then
//           freed before the pool is constructed)
//   pass 2: encode -> codes.bin + HNSW inserts, every byte through the
//           shared BufferPool (one budget, one LRU)
//   finish: manifest with per-file content hashes; staging file removed.
//
// Same input + config => byte-identical output files (determinism gate).

import { open, writeFile, readFile, rm, readdir } from 'node:fs/promises';
import { join, resolve, sep, relative, isAbsolute } from 'node:path';
import { readCorpus } from './corpus.js';
import { tokenize, termFreq } from './tokenize.js';
import { makeEmbedder } from './embed.js';
import { trainCodebooks, encodeVector, queryTables, requireDims } from './quantize.js';
import { Graph, graphStride } from './graph.js';
import { PostingsBuilder } from './bm25.js';
import {
  BufferPool, PagedFile, AppendFile, ensureOutDir, writeManifest, readManifest, hashTree,
  StoreError, FORMAT_VERSION,
} from './store.js';
import { BudgetExceeded } from './budget.js';
import { sha256, mulberry32, l2normalize, varintEncode } from './util.js';

const VECTORS_STAGING = 'vectors.f32';
const EMBED_BATCH = 256;

export class BuildError extends Error {
  constructor(message) {
    super(message);
    this.name = 'BuildError';
    this.code = 'BUILD_ERROR';
  }
}

function float32LE(v) {
  const buf = Buffer.alloc(v.length * 4);
  for (let i = 0; i < v.length; i += 1) buf.writeFloatLE(v[i], i * 4);
  return buf;
}

function readFloat32LE(buf, offset, dims) {
  const v = new Float32Array(dims);
  for (let i = 0; i < dims; i += 1) v[i] = buf.readFloatLE(offset + i * 4);
  return v;
}

// When --out lives INSIDE a directory input, exclude it from the corpus
// fingerprint: hashing the index's own output would make --check stale forever.
function outExclude(input, outAbs) {
  // path.relative, not startsWith: on win32 resolve() may produce different
  // drive letter cases (c:\ vs C:\), and startsWith also treats C:\srcbar as
  // inside C:\src — either one would corrupt the --check fingerprint.
  const inputAbs = resolve(input);
  const relPath = relative(inputAbs, resolve(outAbs));
  if (relPath === '' || relPath.startsWith('..') || isAbsolute(relPath)) return null;
  const rel = relPath.split(/[\\/]/)[0];
  if (!rel) return null;
  return (p) => p === rel || p.startsWith(`${rel}/`);
}

export async function buildIndex({
  input,
  out = '.leanann',
  embed = 'hash',
  embedCmd = null,
  dims = 384,
  m = 8,
  k = 256,
  sample = 20000,
  iters = 12,
  graphM = 16,
  budgetMb = 128,
  pageKb = 64,
  seed = 42,
  exclude = [],
  exec = null,
  now = () => Date.now(),
  progress = null,
} = {}) {
  const started = now();
  requireDims(dims, m);
  if (!Number.isInteger(k) || k < 2 || k > 256) throw new BuildError(`--k must be 2..256, got ${k}`);
  // graph records store per-level degree as u8 and neighbor ids as u16, so
  // m0 = 2*graphM must fit in 255 — a larger --graph-m would silently wrap
  // the degree byte and corrupt every long adjacency list.
  if (!Number.isInteger(graphM) || graphM < 2 || graphM > 127) {
    throw new BuildError(`--graph-m must be 2..127 (u8 degree limit; default 16), got ${graphM}`);
  }

  const budgetBytes = Math.floor(budgetMb * 1024 * 1024);
  const pageSize = pageKb * 1024;
  if (pageSize > budgetBytes) {
    throw new BudgetExceeded(
      `page size ${pageSize} B does not fit in budget ${budgetBytes} B — raise --budget-mb to at least ` +
      `${Math.ceil(pageSize / (1024 * 1024))} MB`,
    );
  }

  const outDir = await ensureOutDir(out);
  const say = (msg) => { if (progress) progress(msg); };
  const corpusHash = await hashTree(input, outExclude(input, outDir));

  // Resident bound for the sample: budget/4 (the rest of pass-1 is the
  // postings buffer at budget/4 plus a 256-doc embed batch).
  const maxSampleRows = Math.max(1, Math.floor(budgetBytes / 4 / (dims * 4)));
  const sampleTarget = Math.min(sample, maxSampleRows);

  const builder = new PostingsBuilder({ dir: outDir, flushAtBytes: Math.max(4096, Math.floor(budgetBytes / 4)) });
  const embedder = embed === 'none'
    ? async () => { throw new BuildError('--embed none: a document has no "vector" field (embedding disabled)'); }
    : await makeEmbedder({ embed, embedCmd, dims, exec });

  const vectorsFd = await open(join(outDir, VECTORS_STAGING), 'w');
  const metaFd = await open(join(outDir, 'meta.bin'), 'w');
  const offFd = await open(join(outDir, 'metaoff.bin'), 'w');
  const metaAppend = new AppendFile(metaFd);
  const offAppend = new AppendFile(offFd);
  // Declared here (not inside the try) because train/pass-2 read them after
  // pass 1 completes.
  let sampleVectors = [];
  let docCount = 0;
  let avgdl = 0;
  let built = null;

  try {
    const rng = mulberry32((seed ^ 0x5eed) >>> 0);
    let sampleSeen = 0;
    docCount = 0;
    sampleVectors = [];

    const stageVector = (vec) => {
      sampleSeen += 1;
      if (sampleVectors.length < sampleTarget) sampleVectors.push(vec);
      else {
        const j = Math.floor(rng() * sampleSeen);
        if (j < sampleVectors.length) sampleVectors[j] = vec;
      }
    };

    const flushBatch = async (batch) => {
      if (batch.length === 0) return;
      const needEmbed = batch.filter((b) => b.vector === null);
      const embedded = needEmbed.length > 0 ? await embedder(needEmbed.map((b) => b.text)) : [];
      let ei = 0;
      const rows = [];
      for (const b of batch) {
        let vec;
        // ei advances ONLY for docs that actually consumed an embedding:
        // `embedded` is the filtered subsequence of vector-less docs, so
        // incrementing for every batch element would hand doc k+1's embedding
        // to doc k (or undefined → crash) in mixed vector/text batches.
        if (b.vector !== null) vec = Float32Array.from(b.vector);
        else { vec = embedded[ei]; ei += 1; }
        if (!(vec instanceof Float32Array)) vec = Float32Array.from(vec);
        if (vec.length !== dims) throw new BuildError(`embedder returned ${vec.length} dims, expected ${dims}`);
        l2normalize(vec);
        rows.push(float32LE(vec));
        stageVector(vec);
      }
      for (const row of rows) await vectorsFd.write(row, 0, row.length, null);
    };

    let batch = [];
    say(`pass 1: indexing ${input} ...`);
    for await (const doc of readCorpus({ input, exclude, dims })) {
      // graph neighbor ids are u16 with 0xFFFF as the "unused" sentinel
      // (graph.js:8), so 65535 docs (ids 0..65534) is the hard v0 ceiling —
      // fail loudly instead of wrapping ids into a corrupted graph.
      if (docCount >= 65535) {
        throw new BuildError('corpus exceeds 65535 docs — v0 graph node ids are u16; split the corpus (multi-segment is v1)');
      }
      // meta sidecar (offset written before the record so it stays u32-indexable)
      const off = metaAppend.offset;
      if (off > 0xffffff00) {
        throw new BuildError('meta.bin exceeds the 4 GB v0 offset limit — split the corpus (multi-segment is v1)');
      }
      const offBuf = Buffer.alloc(4);
      offBuf.writeUInt32LE(off);
      offAppend.write(offBuf);
      const idBytes = Buffer.from(doc.id, 'utf8');
      const pvBytes = Buffer.from(String(doc.text ?? '').slice(0, 120), 'utf8');
      const parts = [];
      const tmp = [];
      varintEncode(idBytes.length, tmp);
      parts.push(Buffer.from(tmp), idBytes);
      const tmp2 = [];
      varintEncode(pvBytes.length, tmp2);
      parts.push(Buffer.from(tmp2), pvBytes);
      metaAppend.write(Buffer.concat(parts));

      builder.addDoc(docCount, termFreq(tokenize(doc.text)));
      if (builder.needsFlush()) await builder.flushRun();

      batch.push({ text: doc.text, vector: doc.vector });
      if (batch.length >= EMBED_BATCH) { await flushBatch(batch); batch = []; }

      docCount += 1;
      if (docCount % 5000 === 0) say(`  ${docCount} docs ...`);
    }
    if (docCount === 0) throw new BuildError('empty corpus — nothing to index');
    await flushBatch(batch);
    batch = [];

    built = await builder.finish();
    avgdl = builder.avgdl();
    await metaAppend.flush();
    await offAppend.flush();
    await vectorsFd.close();
    await metaFd.close();
    await offFd.close();
    say(`pass 1: ${docCount} docs, ${built.terms} terms`);
  } catch (err) {
    // Abort path: close staging handles and drop partial runs — an aborted
    // build must not leak open fds or leave half-written shards behind.
    await vectorsFd.close().catch(() => {});
    await metaFd.close().catch(() => {});
    await offFd.close().catch(() => {});
    await builder.abort().catch(() => {});
    throw err;
  }

  // --- train ---------------------------------------------------------------
  say(`training ${k}-means (k=${k}, m=${m}, sample=${sampleVectors.length}) ...`);
  let codebooks = null;
  try {
    codebooks = trainCodebooks(sampleVectors, { dims, m, k, seed, iters });
  } catch (err) {
    throw new BuildError(`codebook training failed: ${err.message}`);
  }
  const sampleUsed = sampleVectors.length;
  sampleVectors.length = 0; // free before the pool takes its budget

  await writeFile(join(outDir, 'codebooks.bin'), Buffer.from(codebooks.buffer, codebooks.byteOffset, codebooks.byteLength));
  await writeFile(join(outDir, 'codes.bin'), Buffer.alloc(0));
  await writeFile(join(outDir, 'graph.bin'), Buffer.alloc(0));

  // --- pass 2 ---------------------------------------------------------------
  say('pass 2: encoding + graph insert ...');
  const pool = new BufferPool({ pageSize, limitBytes: budgetBytes, writable: true });
  const codesFid = await pool.attach(join(outDir, 'codes.bin'), { writable: true });
  const graphFid = await pool.attach(join(outDir, 'graph.bin'), { writable: true });
  const codesFile = new PagedFile({ pool, fid: codesFid, stride: m, name: 'codes.bin' });
  const graphFile = new PagedFile({ pool, fid: graphFid, stride: graphStride(graphM), name: 'graph.bin' });
  const graph = new Graph({ file: graphFile, codes: codesFile, m: graphM, k, seed });
  graph.setBooks(codebooks, { dims, m, k });

  const vecFd = await open(join(outDir, VECTORS_STAGING), 'r');
  const rowBytes = dims * 4;
  const rowBuf = Buffer.alloc(rowBytes);
  for (let id = 0; id < docCount; id += 1) {
    await vecFd.read(rowBuf, 0, rowBytes, id * rowBytes);
    const vec = readFloat32LE(rowBuf, 0, dims);
    const codes = encodeVector(vec, codebooks, { dims, m, k });
    await codesFile.write(id, codes);
    const tables = queryTables(vec, codebooks, { dims, m, k });
    await graph.insert(id, tables);
    if (id > 0 && id % 5000 === 0) say(`  ${id}/${docCount} inserted ...`);
  }
  await vecFd.close();
  await pool.flush();
  await pool.close();
  if (!graph.entry) throw new BuildError('graph entry missing after build (internal error)');
  say(`pass 2: ${docCount} docs, entry=${graph.entry.id}@L${graph.entry.level}`);

  // --- manifest -------------------------------------------------------------
  const fileNames = [
    'codes.bin', 'graph.bin', 'postings.bin', 'terms.bin', 'termoff.bin',
    'postoff.bin', 'doclens.bin', 'meta.bin', 'metaoff.bin', 'codebooks.bin',
  ];
  const files = {};
  for (const name of fileNames) {
    const buf = await readFile(join(outDir, name));
    files[name] = { bytes: buf.length, sha256: sha256(buf) };
  }
  const manifest = {
    version: FORMAT_VERSION,
    createdAt: new Date(now()).toISOString(),
    tool: 'leanann@0.1.0',
    docs: docCount,
    terms: built.terms,
    dims, m, k, graphM, avgdl,
    entry: graph.entry,
    seed,
    embed: { kind: embedCmd ? 'cmd' : embed, dims, cmd: embedCmd ?? null },
    train: { sampleUsed, iters },
    corpus: { path: String(input), hash: corpusHash },
    files,
  };
  await writeManifest(outDir, manifest);
  await rm(join(outDir, VECTORS_STAGING), { force: true });
  for (const f of await readdir(outDir)) {
    if (f.startsWith('.run-')) await rm(join(outDir, f), { force: true }).catch(() => {});
  }

  const totalBytes = Object.values(files).reduce((a, f) => a + f.bytes, 0);
  say(`index built: ${docCount} docs, ${built.terms} terms, ${(totalBytes / (1024 * 1024)).toFixed(1)} MB on disk`);
  return {
    out: resolve(outDir),
    docs: docCount,
    terms: built.terms,
    entry: graph.entry,
    sampleUsed,
    totalBytes,
    files,
    elapsedMs: now() - started,
  };
}

// --check: is the committed index stale vs the corpus? (ctxc-style)
export async function checkIndex({ input, out = '.leanann', exclude = [] }) {
  const manifest = await readManifest(out);
  const current = await hashTree(input, outExclude(input, resolve(out)));
  void exclude;
  return {
    stale: manifest.corpus?.hash !== current,
    expected: manifest.corpus?.hash ?? null,
    actual: current,
    docs: manifest.docs,
  };
}
