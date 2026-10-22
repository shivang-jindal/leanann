// CLI tests — exit-code contract (0/1/2/3), JSON shapes, config merge, quiet,
// and one subprocess run proving the bin wiring itself.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { main } from '../src/cli.js';
import { TEST_DIR, capture, tmpdir, writeJsonl } from './helpers.js';

const FAST_FLAGS = [
  '--dims', '16', '--m', '4', '--k', '16', '--iters', '4',
  '--graph-m', '8', '--budget-mb', '8', '--page-kb', '4',
];

async function run(argv, opts = {}) {
  const cap = capture();
  const code = await main(argv, {
    stdout: cap.stdout,
    stderr: cap.stderr,
    cwd: opts.cwd ?? (() => process.cwd()),
    ...(opts.exec !== undefined ? { exec: opts.exec } : {}),
    ...(opts.now !== undefined ? { now: opts.now } : {}),
  });
  return { code, cap };
}

function plainDocs(n = 20) {
  const topics = [
    ['database', 'connection', 'pool'],
    ['memory', 'heap', 'pressure'],
    ['render', 'button', 'modal'],
    ['deploy', 'rollback', 'cache'],
  ];
  return Array.from({ length: n }, (_, i) => {
    const pool = topics[i % topics.length];
    return { id: `d-${i}`, text: `${pool[i % 3]} ${pool[(i + 1) % 3]} issue ${i} token t${i}` };
  });
}

async function pipelineFixture(prefix, n = 20) {
  const dir = tmpdir(prefix);
  const file = path.join(dir, 'corpus.jsonl');
  writeJsonl(file, plainDocs(n));
  return { dir, file };
}

test('help exits 0 and prints usage', async () => {
  const { code, cap } = await run(['--help']);
  assert.equal(code, 0);
  assert.match(cap.text(), /Usage:/);
  assert.match(cap.text(), /Exit codes:/);
  assert.equal(cap.errors(), '');
});

test('version exits 0', async () => {
  const { code, cap } = await run(['--version']);
  assert.equal(code, 0);
  assert.match(cap.text(), /^leanann \d+\.\d+\.\d+/);
});

test('no arguments: exit 2 with help on stdout', async () => {
  const { code, cap } = await run([]);
  assert.equal(code, 2);
  assert.match(cap.text(), /Usage:/);
});

test('unknown command: exit 2, names the valid ones', async () => {
  const { code, cap } = await run(['frobnicate']);
  assert.equal(code, 2);
  assert.match(cap.errors(), /unknown command "frobnicate" \(index\|search\|bench\|stats\)/);
});

test('usage errors: missing operands and bad flag values -> 2', async () => {
  const cases = [
    { argv: ['index'], msg: /requires <input>/ },
    { argv: ['search'], msg: /requires <query/ },
    { argv: ['bench'], msg: /requires <input>/ },
    { argv: ['search', 'q', '--mode', 'sideways'], msg: /--mode must be hybrid\|vector\|bm25/ },
    { argv: ['search', 'q', '--k', 'abc'], msg: /--k must be an integer/ },
    { argv: ['search', 'q', '--k'], msg: /--k requires a value/ },
    { argv: ['search', 'q', '--json=1'], msg: /--json takes no value/ },
    { argv: ['index', 'x.jsonl', '--dims', 'nope'], msg: /--dims must be an integer/ },
  ];
  for (const { argv, msg } of cases) {
    const { code, cap } = await run(argv);
    assert.equal(code, 2, `${argv.join(' ')} -> exit 2 (got ${code})`);
    assert.match(cap.errors(), msg, `${argv.join(' ')}`);
  }
});

test('search on a missing index: exit 3 with the fix', async () => {
  const dir = tmpdir('cli-missing');
  const { code, cap } = await run(['search', 'anything', '--out', path.join(dir, 'nope')]);
  assert.equal(code, 3);
  assert.match(cap.errors(), /run `leanann index` first/);
});

test('full pipeline through main(): index, search, stats, --check', async () => {
  const { dir, file } = await pipelineFixture('cli-pipe');
  const cwd = () => dir;

  // index (relative --out resolved against injected cwd)
  const idx = await run(['index', 'corpus.jsonl', '--out', 'idx', ...FAST_FLAGS], { cwd });
  assert.equal(idx.code, 0, idx.cap.errors());
  assert.ok(fs.existsSync(path.join(dir, 'idx', 'manifest.json')));

  // search --json
  const s = await run(['search', 'database connection pool', '--out', 'idx', '--k', '3', '--json'], { cwd });
  assert.equal(s.code, 0, s.cap.errors());
  const payload = s.cap.json();
  assert.ok(Array.isArray(payload.results));
  assert.ok(payload.results.length > 0 && payload.results.length <= 3);
  assert.equal(payload.mode, 'hybrid');
  assert.equal(payload.degraded, false);
  for (const r of payload.results) {
    assert.equal(typeof r.id, 'string');
    assert.equal(typeof r.docId, 'number');
    assert.equal(typeof r.score, 'number');
    assert.equal(typeof r.preview, 'string');
  }

  // stats --json
  const st = await run(['stats', '--out', 'idx', '--json'], { cwd });
  assert.equal(st.code, 0);
  const sm = st.cap.json();
  assert.equal(sm.docs, 20);
  assert.ok(sm.files['manifest.json'] === undefined);
  assert.ok(sm.files['graph.bin'].bytes > 0);

  // --check fresh -> 0
  const ok = await run(['index', 'corpus.jsonl', '--out', 'idx', '--check'], { cwd });
  assert.equal(ok.code, 0, ok.cap.errors());
  assert.match(ok.cap.text(), /ok: index matches the corpus/);

  // mutate the corpus -> stale, exit 1
  fs.appendFileSync(file, JSON.stringify({ id: 'd-99', text: 'brand new line' }) + '\n');
  const stale = await run(['index', 'corpus.jsonl', '--out', 'idx', '--check'], { cwd });
  assert.equal(stale.code, 1);
  assert.match(stale.cap.text(), /stale: corpus changed/);

  // stale --json shape
  const staleJson = await run(['index', 'corpus.jsonl', '--out', 'idx', '--check', '--json'], { cwd });
  assert.equal(staleJson.code, 1);
  const sj = staleJson.cap.json();
  assert.equal(sj.stale, true);
  assert.equal(typeof sj.expected, 'string');
  assert.equal(typeof sj.actual, 'string');
});

test('search human output prints score + id and --explain adds sources', async () => {
  const { dir, file } = await pipelineFixture('cli-human');
  const cwd = () => dir;
  await run(['index', 'corpus.jsonl', '--out', 'idx', ...FAST_FLAGS, '--quiet'], { cwd });
  const s = await run(['search', 'database connection pool', '--out', 'idx', '--k', '2', '--explain'], { cwd });
  assert.equal(s.code, 0, s.cap.errors());
  const lines = s.cap.text().trim().split('\n');
  assert.match(lines[0], /^\d\.\d{4}\s+d-\d+\s+/);
  assert.ok(lines.some((l) => l.includes('via bm25 #')), `explain line in ${JSON.stringify(lines)}`);
});

test('quiet: successful index prints nothing', async () => {
  const { dir } = await pipelineFixture('cli-quiet', 6);
  const cwd = () => dir;
  const { code, cap } = await run(
    ['index', 'corpus.jsonl', '--out', 'idx', ...FAST_FLAGS, '--quiet'],
    { cwd },
  );
  assert.equal(code, 0, cap.errors());
  assert.equal(cap.text(), '', 'quiet must be silent on success');
});

test('config file: out/mode defaults merge; flags override config', async () => {
  const { dir } = await pipelineFixture('cli-config', 12);
  const cwd = () => dir;
  fs.writeFileSync(
    path.join(dir, 'leanann.config.json'),
    JSON.stringify({ out: 'cfg-out', mode: 'bm25', k: 3 }),
  );

  // no --out flag: config drives output dir
  const idx = await run(['index', 'corpus.jsonl', ...FAST_FLAGS], { cwd });
  assert.equal(idx.code, 0, idx.cap.errors());
  assert.ok(fs.existsSync(path.join(dir, 'cfg-out', 'manifest.json')), 'config out honored');

  // no --mode flag: config drives mode
  const bm = await run(['search', 'database connection', '--json'], { cwd });
  assert.equal(bm.code, 0, bm.cap.errors());
  assert.equal(bm.cap.json().mode, 'bm25');

  // flag beats config
  const vec = await run(['search', 'database connection', '--mode', 'vector', '--json'], { cwd });
  assert.equal(vec.code, 0, vec.cap.errors());
  assert.equal(vec.cap.json().mode, 'vector');

  // config k honored when flag absent (3 results), overridden by flag (1)
  assert.ok(bm.cap.json().results.length <= 3);
  const one = await run(['search', 'database connection', '--k', '1', '--json'], { cwd });
  assert.equal(one.code, 0);
  assert.ok(one.cap.json().results.length <= 1);
});

test('broken config file: exit 2 with the parse error', async () => {
  const dir = tmpdir('cli-badconfig');
  fs.writeFileSync(path.join(dir, 'leanann.config.json'), '{oops');
  const { code, cap } = await run(['stats'], { cwd: () => dir });
  assert.equal(code, 2);
  assert.match(cap.errors(), /leanann\.config\.json/);
});

test('embed-none corpus: hybrid search degrades -> exit 1 + stderr note', async () => {
  const dir = tmpdir('cli-degraded');
  const dims = 16;
  const docs = plainDocs(8).map((d, i) => ({
    ...d,
    vector: Array.from({ length: dims }, (_, j) => ((i * 5 + j) % 9) - 4),
  }));
  writeJsonl(path.join(dir, 'corpus.jsonl'), docs);
  const cwd = () => dir;

  const idx = await run(['index', 'corpus.jsonl', '--out', 'idx', '--embed', 'none', ...FAST_FLAGS], { cwd });
  assert.equal(idx.code, 0, idx.cap.errors());

  const s = await run(['search', 'database connection', '--out', 'idx'], { cwd });
  assert.equal(s.code, 1, 'degraded must exit 1');
  assert.match(s.cap.errors(), /degraded/);
  assert.match(s.cap.errors(), /BM25-only/);
  assert.ok(s.cap.text().trim().length > 0, 'still prints results');

  // vector mode on the same index is an operational error, not a warning
  const v = await run(['search', 'database', '--out', 'idx', '--mode', 'vector'], { cwd });
  assert.equal(v.code, 3);
  assert.match(v.cap.errors(), /--embed none/);
});

test('bench --json through main(): valid report, exit 0', async () => {
  const { dir } = await pipelineFixture('cli-bench', 12);
  const cwd = () => dir;
  const { code, cap } = await run(
    ['bench', 'corpus.jsonl', '--json', '--queries', '3', '--k', '3', ...FAST_FLAGS],
    { cwd },
  );
  assert.equal(code, 0, cap.errors());
  const report = cap.json();
  assert.equal(report.docs, 12);
  assert.equal(report.queries, 3);
  assert.ok(report.recall.vector !== null && report.recall.vector >= 0);
  assert.equal(report.memory.budgetEnforced, true);
  assert.equal(report.groundTruth.full, true);
});

test('bench human table renders and mentions RSS separately from budget', async () => {
  const { dir } = await pipelineFixture('cli-benchhuman', 10);
  const cwd = () => dir;
  const { code, cap } = await run(
    ['bench', 'corpus.jsonl', '--queries', '3', '--k', '3', ...FAST_FLAGS],
    { cwd },
  );
  assert.equal(code, 0, cap.errors());
  const text = cap.text();
  assert.match(text, /measure\s+value/);
  assert.match(text, /peak RSS\s+\S+/);
  assert.match(text, /NOT the budget/);
  assert.ok(!text.includes('undefined'), 'no undefined anywhere in the table');
});

test('subprocess: node src/cli.js --version runs as a real process', async () => {
  const cliPath = path.join(TEST_DIR, '..', 'src', 'cli.js');
  const out = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [cliPath, '--version'], { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d; });
    child.stderr.on('data', (d) => { stderr += d; });
    child.on('error', reject);
    child.on('close', (exit) => resolve({ exit, stdout, stderr }));
  });
  assert.equal(out.exit, 0, out.stderr);
  assert.match(out.stdout, /^leanann 0\.1\.0/);
  assert.equal(out.stderr, '');
});
