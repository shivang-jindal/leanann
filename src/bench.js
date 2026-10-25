// bench.js — the honest benchmark: recall@k measured against an EXACT
// brute-force ground truth on YOUR corpus, plus latency percentiles, peak
// RSS (reported separately from the logical budget — never conflated), disk
// size, cache accounting, and degraded-query counts.
//
// Same shape as AirLLM's "X GB model on Y GB RAM at Z tok/s" table, for an
// index: recall / p95 / RSS / on-disk / budget / evictions.

import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { buildIndex } from './indexer.js';
import { openIndex, searchOpened } from './search.js';
import { readCorpus } from './corpus.js';
import { queryTables, adcDist } from './quantize.js';
import { l2normalize, mulberry32, percentile, topK, formatBytes } from './util.js';

// Ground truth over stored PQ codes with the FLOAT query (ADC, exact w.r.t.
// the index's own code space — which is the only space any method can rank).
async function exactTopK(idx, tables, k, cap) {
  const { manifest, codesFile } = idx;
  const n = manifest.docs;
  const scan = Math.min(n, cap);
  const out = new Array(scan);
  for (let id = 0; id < scan; id += 1) {
    const codes = await codesFile.read(id);
    out[id] = { id, score: -adcDist(tables, codes, manifest.k) };
  }
  return topK(out, k);
}

// Fraction of expected (ground-truth) ids present in retrieved. Exported for
// direct unit tests — this is the number the whole benchmark hangs on.
export function recallAtK(retrieved, expected) {
  const got = new Set(retrieved.map((r) => r.id));
  let hits = 0;
  for (const e of expected) if (got.has(e.id)) hits += 1;
  return expected.length === 0 ? 0 : hits / expected.length;
}

export async function bench({
  input,
  index = null,
  // Parent dir for a scratch (index: null) build. Injectable so tests can
  // watch a PRIVATE directory — the shared os.tmpdir() races when another
  // process legitimately mkdtemps its own leanann-bench-* scratch at the
  // same moment (cli.test.js spawns the real CLI bench concurrently).
  tmpRoot = tmpdir(),
  queries = 32,
  k = 10,
  budgetMb = 128,
  pageKb = 64,
  exactCap = 50000,
  seed = 42,
  ef = 64,
  embed = 'hash',
  embedCmd = null,
  dims = 384,
  m = 8,
  codebookK = 256,
  graphM = 16,
  exclude = [],
  exec = null,
  now = () => Date.now(),
  hrtime = () => Number(process.hrtime.bigint()) / 1e6,
  memory = () => process.memoryUsage(),
  progress = null,
} = {}) {
  const started = now();
  const say = (msg) => { if (progress) progress(msg); };
  let scratch = null;
  let idxDir = index;

  if (!idxDir) {
    scratch = await mkdtemp(join(tmpRoot, 'leanann-bench-'));
    idxDir = scratch;
    say(`building scratch index under ${scratch} ...`);
    await buildIndex({
      input, out: idxDir, embed, embedCmd, dims, m, k: codebookK,
      budgetMb, pageKb, seed, exclude, exec, now, progress,
    });
  }

  // try/finally: every exit path (embedder failure, query crash, budget
  // violation) must close the pool and remove a scratch index — leaked
  // handles + a full built index in os.tmpdir() is not acceptable.
  let idx = null;
  try {
    idx = await openIndex({ dir: idxDir, budgetMb, pageKb, k, ef });
    const { manifest } = idx;
    say(`bench: ${manifest.docs} docs, k=${k}, queries=${queries}, budget=${budgetMb} MB`);

  // --- sample query docs (seeded reservoir over the corpus texts) ------------
  const rng = mulberry32(seed >>> 0);
  const reservoir = [];
  let seen = 0;
  for await (const doc of readCorpus({ input, exclude, dims: manifest.dims })) {
    seen += 1;
    if (reservoir.length < queries) reservoir.push({ text: doc.text, hasVector: doc.vector != null, vector: doc.vector });
    else {
      const j = Math.floor(rng() * seen);
      if (j < reservoir.length) reservoir[j] = { text: doc.text, hasVector: doc.vector != null, vector: doc.vector };
    }
  }
  const queryDocs = reservoir.slice(0, queries);

  // --- query embeddings (manifest's embedder, so vectors match the index) ---
  const kind = manifest.embed?.kind ?? 'hash';
  let queryVecs = null;
  if (kind === 'none') {
    const withVec = queryDocs.filter((d) => d.hasVector);
    queryVecs = withVec.length === queryDocs.length && queryDocs.length > 0
      ? withVec.map((d) => { const v = Float32Array.from(d.vector); l2normalize(v); return v; })
      : null;
  } else {
    const vecs = await queryEmbeddingMany(manifest, queryDocs.map((d) => d.text), exec, manifest.dims);
    queryVecs = vecs;
  }

  const groundTruthFull = manifest.docs <= exactCap;
  const groundTruthDocs = groundTruthFull ? manifest.docs : exactCap;
  if (!groundTruthFull) {
    say(`note: ${manifest.docs} docs > --exact-cap ${exactCap}; ground truth sampled to first ${exactCap} docs`);
  }

  // --- run --------------------------------------------------------------------
  const latencies = [];
  let vectorRecallSum = 0;
  let hybridRecallSum = 0;
  let bm25RecallSum = 0;
  let degradedQueries = 0;
  let measured = 0;

  for (let qi = 0; qi < queryDocs.length; qi += 1) {
    const qDoc = queryDocs[qi];
    let exact = [];
    let qVec = null;
    if (queryVecs) {
      qVec = queryVecs[qi];
      const tables = queryTables(qVec, idx.books, { dims: manifest.dims, m: manifest.m, k: manifest.k });
      exact = await exactTopK(idx, tables, k, groundTruthDocs);
    }
    const hasTokens = qDoc.text && qDoc.text.trim().length > 0;

    // ANN (timed)
    let ann = [];
    if (qVec) {
      const tables = queryTables(qVec, idx.books, { dims: manifest.dims, m: manifest.m, k: manifest.k });
      const t0 = hrtime();
      const beam = await idx.graph.search(tables, Math.max(ef, k));
      latencies.push(hrtime() - t0);
      ann = beam.slice(0, k).map((r) => ({ id: r.id, score: -r.dist }));
    }

    if (exact.length > 0) {
      vectorRecallSum += recallAtK(ann, exact);
      measured += 1;
    }

    // hybrid + bm25 against the same ground truth (only meaningful when the
    // corpus texts carry the query terms). Hydrated results carry the string
    // `id`; the ground truth is numeric docIds — compare on docId.
    if (exact.length > 0 && hasTokens) {
      const h = await searchOpened(idx, { query: qDoc.text, k, mode: 'hybrid', ef, exec });
      const b = await searchOpened(idx, { query: qDoc.text, k, mode: 'bm25', ef, exec });
      if (h.degraded) degradedQueries += 1;
      hybridRecallSum += recallAtK(h.results.map((r) => ({ id: r.docId })), exact);
      bm25RecallSum += recallAtK(b.results.map((r) => ({ id: r.docId })), exact);
    }
  }

  const after = memory();
  const cache = idx.pool.stats();
  const disk = Object.values(manifest.files ?? {}).reduce((a, f) => a + (f.bytes ?? 0), 0);

  const latSorted = latencies.slice().sort((a, b) => a - b);
  const report = {
    docs: manifest.docs,
    k,
    queries: queryDocs.length,
    budgetMb,
    budgetBytes: Math.floor(budgetMb * 1024 * 1024),
    embed: kind,
    recall: {
      vector: measured > 0 ? round(vectorRecallSum / measured) : null,
      hybrid: measured > 0 ? round(hybridRecallSum / measured) : null,
      bm25: measured > 0 ? round(bm25RecallSum / measured) : null,
    },
    groundTruth: {
      method: 'exact-ADC brute force over stored codes',
      docs: groundTruthDocs,
      full: groundTruthFull,
      ...(groundTruthFull ? {} : { note: `sampled: corpus > --exact-cap ${exactCap}; recall is conservative` }),
    },
    latencyMs: {
      p50: round(percentile(latSorted, 50)),
      p95: round(percentile(latSorted, 95)),
      max: round(latSorted.length ? latSorted[latSorted.length - 1] : 0),
    },
    memory: {
      // RSS includes the Node runtime — reported SEPARATELY from the
      // enforced logical budget, never as "the budget".
      rssBytes: after.rss,
      rssHuman: formatBytes(after.rss),
      poolUsedBytes: cache.used,
      poolLimitBytes: cache.limit,
      poolPages: cache.pages,
      budgetEnforced: cache.used <= cache.limit,
    },
    cache: {
      hits: cache.hits,
      misses: cache.misses,
      evictions: cache.evictions,
      hitRate: round(cache.hitRate),
    },
    diskBytes: disk,
    diskHuman: formatBytes(disk),
    degradedQueries,
    elapsedMs: now() - started,
  };

    say(`recall@${k}: vector=${report.recall.vector} hybrid=${report.recall.hybrid} bm25=${report.recall.bm25}`);
    say(`p95=${report.latencyMs.p95}ms rss=${report.memory.rssHuman} disk=${report.diskHuman} degraded=${degradedQueries}`);
    return report;
  } finally {
    if (idx) await idx.pool.close().catch(() => {});
    if (scratch) await rm(scratch, { recursive: true, force: true }).catch(() => {});
  }
}

async function queryEmbeddingMany(manifest, texts, exec, dims) {
  if (manifest.embed?.kind === 'cmd') {
    const { embedViaCommand } = await import('./embed.js');
    return embedViaCommand(manifest.embed.cmd, texts, { dims, exec });
  }
  const { hashEmbed } = await import('./embed.js');
  return texts.map((t) => hashEmbed(t, dims));
}

function round(x) {
  return Number.isFinite(x) ? Math.round(x * 1e4) / 1e4 : x;
}
