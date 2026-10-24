#!/usr/bin/env node
// cli.js — leanann | index, search, bench, stats.
//
// Exit codes: 0 ok · 1 completed-with-warnings (degraded, stale --check)
//             2 usage · 3 operational (missing/corrupt index, impossible
//             budget, bad corpus, embed failure)
//
// All side effects are injectable (stdout/stderr/exec/cwd/now/hrtime/memory)
// so tests drive the real code paths without monkey-patching globals.

import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { buildIndex, checkIndex } from './indexer.js';
import { search, stats } from './search.js';
import { bench } from './bench.js';
import { BudgetExceeded } from './budget.js';
import { StoreError } from './store.js';
import { CorpusError } from './corpus.js';
import { BuildError } from './indexer.js';
import { formatBytes } from './util.js';

const HELP = `leanann — vector search that fits your RAM budget

Hybrid BM25 + product-quantized ANN over a disk-paged HNSW graph.
Zero dependencies: Node >= 20, no npm install, no server.

Usage:
  leanann index <input> [options]      build/refresh an index
  leanann search <query...> [options]  query an index
  leanann bench <input> [options]      recall vs exact, latency, RSS, disk
  leanann stats [options]              manifest, counts, byte breakdown
  leanann --help | --version

Input:  a .jsonl file ({"id","text","vector"?} per line) or a directory
        of .txt/.md files.

index options:
  --out <dir>          output dir (default .leanann; only place written)
  --embed hash|none    built-in feature-hash embedder (default: hash)
  --embed-cmd <cmd>    external embedder; {input}/{output} JSONL placeholders
  --dims <n>           vector dimensions (default 384)
  --m <n>              product-quantization subspaces (default 8)
  --k <n>              k-means centroids, 2..256 (default 256)
  --sample <n>         codebook training sample (default 20000)
  --iters <n>          k-means iterations (default 12)
  --graph-m <n>        HNSW max edges (default 16; level 0 uses 2x)
  --budget-mb <n>      hard RAM budget in MB (default 128)
  --page-kb <n>        page-cache page size (default 64)
  --exclude <glob>     repeatable; merged with .leanannignore
  --seed <n>           RNG seed (default 42)
  --check              verify the index against the corpus; no writes
  --json, --quiet      machine summary / silence

search options:
  --out <dir>          index dir (default .leanann)
  --k <n>              results (default 10)
  --mode hybrid|vector|bm25   (default hybrid)
  --ef <n>             beam width (default 64)
  --rrf-k <n>          RRF constant (default 60)
  --budget-mb <n>      page-cache budget for this query
  --explain            per-source ranks + fusion contribution
  --json               JSON output

bench options:
  --index <dir>        bench an existing index (else builds a scratch one)
  --queries <n>        query count (default 32)
  --exact-cap <n>      ground-truth cap (default 50000)
  --k-means-k <n>      codebook centroids when building (default 256)
  (plus index flags: --budget-mb --page-kb --ef --embed --dims --m ...)

Exit codes: 0 ok · 1 warnings (degraded / stale) · 2 usage · 3 operational
`;

const VERSION = '0.1.0';

function parseArgs(argv) {
  const positional = [];
  const flags = new Map();
  const repeatable = new Set(['exclude']);
  const booleans = new Set([
    'json', 'quiet', 'explain', 'check', 'help', 'version',
  ]);
  let i = 0;
  while (i < argv.length) {
    const a = argv[i];
    if (!a.startsWith('--')) {
      positional.push(a);
      i += 1;
      continue;
    }
    const eq = a.indexOf('=');
    const name = eq === -1 ? a.slice(2) : a.slice(2, eq);
    if (booleans.has(name)) {
      if (eq !== -1) throw new UsageError(`--${name} takes no value`);
      flags.set(name, true);
      i += 1;
      continue;
    }
    let value = null;
    if (eq !== -1) value = a.slice(eq + 1);
    else {
      const next = argv[i + 1];
      if (next === undefined || (next.startsWith('--') && !/^--\d/.test(next))) {
        throw new UsageError(`--${name} requires a value`);
      }
      value = next;
      i += 1;
    }
    if (repeatable.has(name)) {
      const list = flags.get(name) ?? [];
      list.push(value);
      flags.set(name, list);
    } else {
      flags.set(name, value);
    }
    i += 1;
  }
  return { positional, flags };
}

class UsageError extends Error {
  constructor(message) {
    super(message);
    this.name = 'UsageError';
    this.code = 'USAGE';
  }
}

function intFlag(flags, name, def, { min = null, max = null } = {}) {
  if (!flags.has(name)) return def;
  const raw = flags.get(name);
  const v = Number(raw);
  if (!Number.isInteger(v)) throw new UsageError(`--${name} must be an integer, got "${raw}"`);
  if (min !== null && v < min) throw new UsageError(`--${name} must be >= ${min}, got ${v}`);
  if (max !== null && v > max) throw new UsageError(`--${name} must be <= ${max}, got ${v}`);
  return v;
}

function strFlag(flags, name, def) {
  return flags.has(name) ? String(flags.get(name)) : def;
}

async function loadConfig(cwd) {
  try {
    const text = await readFile(join(cwd, 'leanann.config.json'), 'utf8');
    const cfg = JSON.parse(text);
    if (typeof cfg !== 'object' || cfg === null) throw new Error('not an object');
    return cfg;
  } catch (err) {
    if (err && err.code === 'ENOENT') return {};
    throw new UsageError(`leanann.config.json: ${err.message}`);
  }
}

async function ignorePatterns(cwd, flags, config = {}) {
  const fromFlags = flags.get('exclude') ?? [];
  let file = [];
  try {
    file = (await readFile(join(cwd, '.leanannignore'), 'utf8'))
      .split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('#'));
  } catch { /* no ignore file — fine */ }
  const fromConfig = Array.isArray(config.exclude) ? config.exclude : [];
  return [...file, ...fromConfig, ...fromFlags];
}

export async function main(argv, deps = {}) {
  const out = deps.stdout ?? ((s) => process.stdout.write(s));
  const err = deps.stderr ?? ((s) => process.stderr.write(s));
  const cwd = () => (deps.cwd ? deps.cwd() : process.cwd());

  let parsed;
  try {
    parsed = parseArgs(argv);
  } catch (e) {
    err(`leanann: ${e.message}\n`);
    return 2;
  }
  const { positional, flags } = parsed;

  if (flags.get('version')) { out(`leanann ${VERSION}\n`); return 0; }
  if (flags.get('help') || positional.length === 0) { out(HELP); return flags.get('help') ? 0 : 2; }

  const command = positional[0];
  const json = Boolean(flags.get('json'));
  const quiet = Boolean(flags.get('quiet'));
  const say = (msg) => { if (!quiet && !json) out(`${msg}\n`); };

  try {
    const config = await loadConfig(cwd());
    const merged = (name, def) => (flags.has(name) ? flags.get(name) : (config[name] ?? def));

    if (command === 'index') {
      if (positional.length < 2) throw new UsageError('index requires <input> (jsonl file or directory)');
      const input = resolve(cwd(), positional[1]);
      const check = Boolean(flags.get('check'));
      const exclude = await ignorePatterns(cwd(), flags, config);
      // resolve --out against the injected cwd (process.cwd() when not injected)
      const outDir = resolve(cwd(), String(merged('out', '.leanann')));

      if (check) {
        const res = await checkIndex({ input, out: outDir, exclude });
        if (json) out(`${JSON.stringify(res, null, 2)}\n`);
        else if (res.stale) {
          // expected is null when the manifest is missing/unreadable — a
          // truncated or hand-deleted index still reports stale, and slicing
          // null would crash the reporting path itself.
          const expected = res.expected ? res.expected.slice(0, 12) : '<no manifest>';
          say(`stale: corpus changed since the index was built (expected ${expected}, got ${res.actual.slice(0, 12)})`);
          say('re-run: leanann index <input> --out ...');
        } else say(`ok: index matches the corpus (${res.docs} docs)`);
        return res.stale ? 1 : 0;
      }

      const result = await buildIndex({
        input,
        out: outDir,
        embed: String(merged('embed', 'hash')),
        embedCmd: merged('embedCmd', merged('embed-cmd', null)) ?? null,
        dims: intFlag(flags, 'dims', Number(merged('dims', 384)), { min: 2 }),
        m: intFlag(flags, 'm', Number(merged('m', 8)), { min: 1 }),
        k: intFlag(flags, 'k', Number(merged('k', 256)), { min: 2, max: 256 }),
        sample: intFlag(flags, 'sample', Number(merged('sample', 20000)), { min: 1 }),
        iters: intFlag(flags, 'iters', Number(merged('iters', 12)), { min: 1 }),
        graphM: intFlag(flags, 'graph-m', Number(merged('graphM', merged('graph-m', 16))), { min: 2 }),
        budgetMb: intFlag(flags, 'budget-mb', Number(merged('budgetMb', merged('budget-mb', 128))), { min: 1 }),
        pageKb: intFlag(flags, 'page-kb', Number(merged('pageKb', merged('page-kb', 64))), { min: 1 }),
        seed: intFlag(flags, 'seed', Number(merged('seed', 42))),
        exclude,
        exec: deps.exec ?? null,
        now: deps.now ?? undefined,
        progress: quiet || json ? null : (m) => say(m),
      });
      if (json) out(`${JSON.stringify(result, null, 2)}\n`);
      else say(`indexed ${result.docs} docs -> ${result.out} (${formatBytes(result.totalBytes)}) in ${result.elapsedMs}ms`);
      return 0;
    }

    if (command === 'search') {
      if (positional.length < 2) throw new UsageError('search requires <query...>');
      const query = positional.slice(1).join(' ');
      const mode = String(merged('mode', 'hybrid'));
      if (!['hybrid', 'vector', 'bm25'].includes(mode)) {
        throw new UsageError(`--mode must be hybrid|vector|bm25, got "${mode}"`);
      }
      const res = await search({
        dir: resolve(cwd(), String(merged('out', '.leanann'))),
        query,
        k: intFlag(flags, 'k', Number(merged('k', 10)), { min: 1 }),
        mode,
        ef: intFlag(flags, 'ef', Number(merged('ef', 64)), { min: 1 }),
        rrfK: intFlag(flags, 'rrf-k', Number(merged('rrfK', merged('rrf-k', 60))), { min: 1 }),
        budgetMb: intFlag(flags, 'budget-mb',
          Number(merged('budgetMb', merged('budget-mb', 128))), { min: 1 }),
        explain: Boolean(flags.get('explain')),
        exec: deps.exec ?? null,
      });
      if (json) out(`${JSON.stringify(res, null, 2)}\n`);
      else {
        if (res.results.length === 0) say('no results');
        for (const r of res.results) {
          out(`${r.score.toFixed(4)}  ${r.id}  ${r.preview}\n`);
          if (flags.get('explain') && r.sources) {
            const parts = Object.entries(r.sources)
              .map(([name, d]) => `${name} #${d.rank} (+${d.contrib.toFixed(4)})`);
            if (parts.length) out(`            via ${parts.join(', ')}\n`);
          }
        }
        if (res.degraded) {
          err(`leanann: degraded — ${res.degradedReason}\n`);
        }
      }
      return res.degraded ? 1 : 0;
    }

    if (command === 'bench') {
      if (positional.length < 2) throw new UsageError('bench requires <input> (jsonl file or directory)');
      const input = resolve(cwd(), positional[1]);
      const exclude = await ignorePatterns(cwd(), flags, config);
      const report = await bench({
        input,
        index: flags.has('index') ? resolve(cwd(), String(flags.get('index'))) : null,
        queries: intFlag(flags, 'queries', Number(merged('queries', 32)), { min: 1 }),
        k: intFlag(flags, 'k', Number(merged('k', 10)), { min: 1 }),
        budgetMb: intFlag(flags, 'budget-mb', Number(merged('budgetMb', merged('budget-mb', 128))), { min: 1 }),
        pageKb: intFlag(flags, 'page-kb', Number(merged('pageKb', merged('page-kb', 64))), { min: 1 }),
        exactCap: intFlag(flags, 'exact-cap', Number(merged('exactCap', merged('exact-cap', 50000))), { min: 1 }),
        ef: intFlag(flags, 'ef', Number(merged('ef', 64)), { min: 1 }),
        seed: intFlag(flags, 'seed', Number(merged('seed', 42))),
        embed: String(merged('embed', 'hash')),
        embedCmd: merged('embedCmd', merged('embed-cmd', null)) ?? null,
        dims: intFlag(flags, 'dims', Number(merged('dims', 384)), { min: 2 }),
        m: intFlag(flags, 'm', Number(merged('m', 8)), { min: 1 }),
        // k-means k, NOT `--k`: bench's --k is the top-k for the report
        // (line 284), so defaulting this from merged('k') made `--k 50` also
        // silently rebuild the codebooks with k=50.
        codebookK: intFlag(flags, 'k-means-k', Number(merged('k-means-k', 256)), { min: 2, max: 256 }),
        graphM: intFlag(flags, 'graph-m', Number(merged('graphM', merged('graph-m', 16))), { min: 2 }),
        exclude,
        exec: deps.exec ?? null,
        now: deps.now ?? undefined,
        hrtime: deps.hrtime ?? undefined,
        memory: deps.memory ?? undefined,
        progress: quiet || json ? null : (m) => say(m),
      });
      if (json) out(`${JSON.stringify(report, null, 2)}\n`);
      else {
        out('\n  measure            value\n');
        out('  ---------------------------\n');
        out(`  docs               ${report.docs}\n`);
        out(`  recall@${report.k} vector      ${report.recall.vector ?? 'n/a'}\n`);
        out(`  recall@${report.k} hybrid      ${report.recall.hybrid ?? 'n/a'}\n`);
        out(`  recall@${report.k} bm25        ${report.recall.bm25 ?? 'n/a'}\n`);
        out(`  ground truth        ${report.groundTruth.docs} docs (${report.groundTruth.full ? 'full' : 'sampled'})\n`);
        out(`  latency p50/p95     ${report.latencyMs.p50} / ${report.latencyMs.p95} ms\n`);
        out(`  peak RSS            ${report.memory.rssHuman} (runtime + index; NOT the budget)\n`);
        out(`  budget              ${formatBytes(report.memory.poolUsedBytes)} used / ${formatBytes(report.memory.poolLimitBytes)} capped: ${report.memory.budgetEnforced}\n`);
        out(`  cache               hit ${Math.round(report.cache.hitRate * 100)}%, ${report.cache.evictions} evictions\n`);
        out(`  on disk             ${report.diskHuman}\n`);
        out(`  degraded queries    ${report.degradedQueries}\n\n`);
      }
      return report.degradedQueries > 0 ? 1 : 0;
    }

    if (command === 'stats') {
      const s = await stats({ dir: resolve(cwd(), String(merged('out', '.leanann'))) });
      if (json) out(`${JSON.stringify(s, null, 2)}\n`);
      else {
        out(`index          ${s.dir}\n`);
        out(`docs / terms   ${s.docs} / ${s.terms}\n`);
        out(`dims/m/k/graph ${s.dims}/${s.m}/${s.k}/${s.graphM}\n`);
        out(`embedder       ${s.embed?.kind ?? 'hash'}\n`);
        out(`entry          node ${s.entry?.id} @ level ${s.entry?.level}\n`);
        out(`trained on     ${s.train?.sampleUsed} vectors, ${s.train?.iters} iters\n`);
        out(`created        ${s.createdAt}\n`);
        out(`corpus hash    ${(s.corpus?.hash ?? '').slice(0, 16)}\n`);
        out(`on disk        ${formatBytes(s.diskBytes)}\n`);
        for (const [name, f] of Object.entries(s.files ?? {})) {
          out(`  ${name.padEnd(16)} ${formatBytes(f.bytes)}\n`);
        }
      }
      return 0;
    }

    err(`leanann: unknown command "${command}" (index|search|bench|stats)\n`);
    return 2;
  } catch (e) {
    if (e instanceof UsageError) {
      err(`leanann: ${e.message}\n`);
      return 2;
    }
    if (e instanceof BudgetExceeded || e instanceof StoreError
      || e instanceof CorpusError || e instanceof BuildError) {
      err(`leanann: ${e.message}\n`);
      return 3;
    }
    if (e && e.code === 'BUDGET_EXCEEDED') {
      err(`leanann: ${e.message}\n`);
      return 3;
    }
    err(`leanann: ${e?.stack ?? e}\n`);
    return 3;
  }
}

// Run only when invoked as `node src/cli.js ...` (or via the bin link), not on import.
const invokedDirectly = process.argv[1]
  && import.meta.url === pathToFileURL(resolve(process.argv[1])).href;

if (invokedDirectly) {
  process.exitCode = await main(process.argv.slice(2));
}

export const _internal = { parseArgs, UsageError, intFlag, strFlag, HELP, VERSION };
