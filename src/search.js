// search.js — the query path: BM25, ANN beam, RRF fusion, --explain.
//
// Honesty rails:
//   * index with `--embed none` + vector mode  -> explicit error (no query embedder)
//   * budget thrashing during a query          -> degraded: true + minimum budget
//   * no results                               -> "no results", never filler

import { open, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { readManifest, verifySizes, BufferPool, PagedFile, StoreError } from './store.js';
import { Budget, BudgetExceeded } from './budget.js';
import { Graph, graphStride } from './graph.js';
import { Bm25Index } from './bm25.js';
import { queryTables } from './quantize.js';
import { tokenize } from './tokenize.js';
import { hashEmbed, embedViaCommand } from './embed.js';
import { l2normalize, topK, varintDecode } from './util.js';

const DEGRADE_EVICTION_FACTOR = 2; // evictions > ef × factor => degraded

export async function openIndex({ dir, budgetMb = null, pageKb = null, k = 10, ef = 64 }) {
  const manifest = await readManifest(dir);
  await verifySizes(dir, manifest);
  manifest.dir = dir;

  const budgetBytes = Math.floor((budgetMb ?? 128) * 1024 * 1024);
  const pageSize = (pageKb ?? 64) * 1024;

  // Precise accounting BEFORE any allocation: resident = codebooks + doclens,
  // then the pool must hold at least one page plus the beam's working set.
  const codebooksBuf = await readFile(join(dir, 'codebooks.bin'));
  const expected = manifest.dims * manifest.k * 4;
  if (codebooksBuf.length !== expected) {
    throw new StoreError(
      `codebooks.bin is ${codebooksBuf.length} B, expected ${expected} B for dims=${manifest.dims} k=${manifest.k} (rebuild)`,
    );
  }
  const doclensBytes = manifest.docs * 4;
  const fm = manifest.files ?? {};
  const resident = codebooksBuf.length
    + (fm['doclens.bin']?.bytes ?? doclensBytes)
    + (fm['terms.bin']?.bytes ?? 0)
    + (fm['termoff.bin']?.bytes ?? 0)
    + (fm['postoff.bin']?.bytes ?? 0);
  const beamWorking = Math.max(ef, k) * 64;
  const needed = resident + pageSize + beamWorking;
  if (budgetBytes < needed) {
    throw new BudgetExceeded(
      `budget ${budgetBytes} B is too small: resident structures ${resident} B (codebooks + doclens + terms) + ` +
      `page ${pageSize} B + beam (ef=${ef}) ${beamWorking} B need ${needed} B — ` +
      `raise --budget-mb to at least ${Math.max(1, Math.ceil(needed / (1024 * 1024)))} MB`,
    );
  }

  const books = new Float32Array(
    codebooksBuf.buffer.slice(codebooksBuf.byteOffset, codebooksBuf.byteOffset + codebooksBuf.length),
  );

  const budget = new Budget(budgetBytes);
  budget.charge(resident);

  const pool = new BufferPool({ pageSize, limitBytes: budgetBytes - resident });
  const codesFid = await pool.attach(join(dir, 'codes.bin'));
  const graphFid = await pool.attach(join(dir, 'graph.bin'));
  const codesFile = new PagedFile({ pool, fid: codesFid, stride: manifest.m, name: 'codes.bin' });
  const graphFile = new PagedFile({
    pool, fid: graphFid, stride: graphStride(manifest.graphM), name: 'graph.bin',
  });
  if (graphFile.count !== manifest.docs || codesFile.count !== manifest.docs) {
    throw new StoreError(
      `graph has ${graphFile.count} nodes, codes has ${codesFile.count}, manifest says ${manifest.docs} — corrupt (rebuild)`,
    );
  }
  const graph = new Graph({
    file: graphFile, codes: codesFile, m: manifest.graphM, k: manifest.k, entry: manifest.entry,
  });
  graph.setBooks(books, { dims: manifest.dims, m: manifest.m, k: manifest.k });

  // Bm25Index loads doclens/terms/offsets (already covered by `resident` above)
  // and attaches postings.bin to the pool.
  const bm25 = await Bm25Index.open({ dir, manifest, pool });

  return { manifest, pool, graph, bm25, books, budget, codesFile };
}

// Read one doc's meta {id, preview} via the metaoff u32 sidecar.
async function readMeta(dir, docId) {
  const offFd = await open(join(dir, 'metaoff.bin'), 'r');
  try {
    const buf = Buffer.alloc(4);
    const { bytesRead } = await offFd.read(buf, 0, 4, docId * 4);
    if (bytesRead !== 4) throw new StoreError(`metaoff: doc ${docId} out of range`);
    const off = buf.readUInt32LE(0);
    const mf = await open(join(dir, 'meta.bin'), 'r');
    try {
      const st = await mf.stat();
      const head = Buffer.alloc(Math.min(8, st.size - off));
      await mf.read(head, 0, head.length, off);
      let [idLen, p] = varintDecode(head, 0);
      const need = p + idLen;
      const full = Buffer.alloc(need + 4);
      await mf.read(full, 0, Math.min(full.length, st.size - off), off);
      const id = full.subarray(p, p + idLen).toString('utf8');
      p += idLen;
      const [pvLen, p2] = varintDecode(full, p);
      const pvStart = off + p2;
      const pvBuf = Buffer.alloc(pvLen);
      await mf.read(pvBuf, 0, pvLen, pvStart);
      return { id, preview: pvBuf.toString('utf8') };
    } finally {
      await mf.close();
    }
  } finally {
    await offFd.close();
  }
}

// Reciprocal Rank Fusion: score = sum over sources of 1/(rrfK + rank), rank 1-based.
export function rrfFuse(lists, rrfK = 60) {
  const acc = new Map();
  const detail = new Map();
  for (const { name, items } of lists) {
    items.forEach((item, idx) => {
      const rank = idx + 1;
      const contrib = 1 / (rrfK + rank);
      acc.set(item.id, (acc.get(item.id) ?? 0) + contrib);
      if (!detail.has(item.id)) detail.set(item.id, {});
      detail.get(item.id)[name] = { rank, contrib };
    });
  }
  const fused = [...acc.entries()].map(([id, score]) => ({ id, score }));
  return { fused: topK(fused, fused.length), detail };
}

export async function queryEmbedding({ manifest, query, exec = null, dims }) {
  const kind = manifest.embed?.kind ?? 'hash';
  if (kind === 'none') return null; // caller handles honestly per mode
  if (kind === 'cmd') {
    const vecs = await embedViaCommand(manifest.embed.cmd, [query], { dims, exec });
    return vecs[0];
  }
  return hashEmbed(query, dims);
}

// Run one query against an ALREADY-OPEN index (bench opens once, not per query).
export async function searchOpened(idx, {
  query,
  k = 10,
  mode = 'hybrid',
  ef = 64,
  rrfK = 60,
  explain = false,
  exec = null,
} = {}) {
  if (mode !== 'hybrid' && mode !== 'vector' && mode !== 'bm25') {
    throw new StoreError(`unknown --mode ${mode} (hybrid|vector|bm25)`);
  }
  const { manifest, pool, graph, bm25, books } = idx;
  const dir = manifest.dir;
  const dims = manifest.dims;
  const searchK = Math.max(k * 3, 30); // candidate lists before fusion
  const tokens = tokenize(query);
  const before = pool.stats();

  // --- BM25 side ---
  let bm25Items = [];
  if (mode === 'hybrid' || mode === 'bm25') {
    bm25Items = tokens.length > 0 ? await bm25.search(tokens, searchK) : [];
  }

  // --- ANN side ---
  let annItems = [];
  let queryVec = null;
  if (mode === 'hybrid' || mode === 'vector') {
    queryVec = await queryEmbedding({ manifest, query, exec, dims });
    if (queryVec === null) {
      if (mode === 'vector') {
        throw new StoreError(
          'index was built with --embed none: there is no query embedder — use --mode bm25 or rebuild with an embedder',
        );
      }
      // hybrid degrades to bm25-only, flagged below
    } else {
      l2normalize(queryVec);
      if (queryVec.length !== dims) {
        throw new StoreError(`query embedder returned ${queryVec.length} dims, index is ${dims}`);
      }
      const tables = queryTables(queryVec, books, { dims, m: manifest.m, k: manifest.k });
      const beam = await graph.search(tables, Math.max(ef, k));
      annItems = beam.slice(0, searchK).map((r) => ({ id: r.id, score: 1 / (1 + r.dist) }));
    }
  }

  // --- fusion ---
  let results = [];
  const detail = new Map();
  const listNames = [];
  if (mode === 'bm25' || mode === 'hybrid') listNames.push({ name: 'bm25', items: bm25Items });
  if ((mode === 'vector' || mode === 'hybrid') && queryVec !== null) listNames.push({ name: 'vector', items: annItems });

  if (listNames.length === 1 && listNames[0].name === 'bm25') {
    results = topK(bm25Items.map((r) => ({ id: r.id, score: r.score })), k);
    bm25Items.forEach((r, i) => detail.set(r.id, { bm25: { rank: i + 1, contrib: r.score } }));
  } else if (listNames.length === 1 && listNames[0].name === 'vector') {
    results = topK(annItems.map((r) => ({ id: r.id, score: r.score })), k);
    annItems.forEach((r, i) => detail.set(r.id, { vector: { rank: i + 1, contrib: r.score } }));
  } else if (listNames.length === 2) {
    const fused = rrfFuse(listNames, rrfK);
    results = fused.fused.slice(0, k);
    for (const [id, d] of fused.detail) detail.set(id, d);
  }

  // --- degrade detection: thrashing page cache during this query ---
  const after = pool.stats();
  const evictions = after.evictions - before.evictions;
  const degraded = evictions > Math.max(ef, 1) * DEGRADE_EVICTION_FACTOR;
  let minBudgetMb = null;
  if (degraded) {
    // minimum budget that would hold the touched pages: rough but honest
    const touchedBytes = (after.misses - before.misses) * pool.pageSize;
    minBudgetMb = Math.max(1, Math.ceil(touchedBytes / (1024 * 1024)));
  }

  // --- hydrate results ---
  const out = [];
  for (const r of results) {
    const meta = await readMeta(dir, r.id);
    out.push({
      docId: r.id,
      id: meta.id,
      score: r.score,
      preview: meta.preview,
      ...(explain ? { sources: detail.get(r.id) ?? {} } : {}),
    });
  }

  const hybridNoVector = mode === 'hybrid' && queryVec === null;
  return {
    results: out,
    mode,
    tokens: tokens.length,
    degraded: degraded || hybridNoVector,
    degradedReason: degraded
      ? `page cache thrashed (${evictions} evictions for ef=${ef}); raise --budget-mb to at least ${minBudgetMb}`
      : hybridNoVector
        ? 'index has no query embedder; result is BM25-only'
        : null,
    minBudgetMb,
    cache: {
      evictions,
      pages: after.pages,
      hitRate: pool.stats().hitRate,
    },
    docs: manifest.docs,
  };
}

// Convenience: open + one query + (caller may close). Used by the CLI.
export async function search({
  dir,
  budgetMb = null,
  pageKb = null,
  ...queryOpts
} = {}) {
  const idx = await openIndex({
    dir, budgetMb, pageKb, k: queryOpts.k ?? 10, ef: queryOpts.ef ?? 64,
  });
  try {
    return await searchOpened(idx, queryOpts);
  } finally {
    await idx.pool.close();
  }
}

export async function stats({ dir }) {
  const manifest = await readManifest(dir);
  const sizeSum = Object.values(manifest.files ?? {}).reduce((a, f) => a + (f.bytes ?? 0), 0);
  return {
    dir,
    docs: manifest.docs,
    terms: manifest.terms,
    dims: manifest.dims,
    m: manifest.m,
    k: manifest.k,
    graphM: manifest.graphM,
    entry: manifest.entry,
    embed: manifest.embed,
    train: manifest.train,
    createdAt: manifest.createdAt,
    diskBytes: sizeSum,
    files: manifest.files,
    corpus: manifest.corpus,
  };
}
