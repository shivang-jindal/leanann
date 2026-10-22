// Test bootstrap.
//
// Node >= 21 treats positional `--test` arguments as GLOB PATTERNS, so
// `node --test test/` matches the DIRECTORY `test` itself and executes it as a
// single entry (`node <dir>`, resolved here as index.js). This file therefore
// loads every test module explicitly so the mandated `npm test` command stays
// green on Node 24.
//
// In all other contexts (bare `node --test`, explicit file globs such as
// `node --test test/*.test.js`, Node 20 directory search) each *.test.js runs
// on its own and this file is either not executed or intentionally inert — no
// test ever runs twice.
//
// House style: same guard as oss-ideas/minrepro/test/index.js.

import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const invokedDirectly = Boolean(
  process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href,
);

if (!invokedDirectly) {
  await import('./tokenize.test.js');
  await import('./bm25.test.js');
  await import('./quantize.test.js');
  await import('./budget.test.js');
  await import('./store.test.js');
  await import('./graph.test.js');
  await import('./embed.test.js');
  await import('./corpus.test.js');
  await import('./index.test.js');
  await import('./search.test.js');
  await import('./bench.test.js');
  await import('./cli.test.js');
  await import('./safety.test.js');
}
