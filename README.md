# leanann — vector search that fits your RAM budget

> **The AirLLM trick, for indices.** AirLLM runs a 70B model on a 4GB laptop by
> keeping only the *active layer* in RAM and paging the rest between disk and
> CPU. leanann does the same thing to a search index: a 10-million-vector index
> that never loads more than your `--budget-mb`, because the codes live on
> disk, the graph is paged, and only the search beam stays hot.
>
> Hybrid **BM25 + product-quantized ANN** over a **disk-paged HNSW graph**.
> Zero dependencies. No server, no native build, no `npm install`.

**Archetype:** CLI + library + GitHub Action · **Buzz:** 🔥🔥🔥🔥🔥 · **Moat:** budget-enforced correctness

---

## The problem

Every local RAG setup, semantic-dedup script, and "find similar issues" tool
hits the same wall the moment the corpus gets real:

| Tool | What it assumes |
|------|-----------------|
| FAISS / hnswlib | You have the RAM (or a GPU) to hold the whole index |
| Qdrant / Weaviate / Milvus | You want to run a *server*, forever, in memory |
| Chroma / LanceDB | A few hundred thousand rows and several GB of headroom |
| pgvector | A Postgres instance with room to spare |

So the honest workflow becomes: rent a bigger box, or shard the corpus by
hand, or silently cap it at "the part that fits." On a 512MB CI runner, a
free-tier container, a laptop with 20 browser tabs, or a shared build machine,
none of those are answers.

The default assumption across the field is that **RAM is the index**. That
assumption is now the bottleneck — not the vectors.

## The idea

Stop treating RAM as the index. Treat it as a *cache* for the index, the same
way CPU caches treat RAM. Three ideas do all the work:

1. **Quantize before you store.** Product quantization (PQ) squeezes each
   vector into `M` bytes (default: 8 bytes for a 384-dim vector — a 48×
   reduction) by learning `M` small codebooks from *your* data. The index
   stops being `N × dims × 4` bytes and becomes `N × M` bytes.

2. **Page the graph, don't load it.** The HNSW graph — the part that makes ANN
   fast and the part that eats RAM — lives in a fixed-stride record file read
   through a **hard-budgeted LRU page cache**. During a beam search only the
   nodes actually visited touch RAM. The "active layer" is the beam, exactly
   like AirLLM's active transformer layer.

3. **Enforce the budget, don't hope for it.** `--budget-mb 128` is not a hint.
   A `Budget` + `PageCache` accounting layer *evicts to stay under it* and
   **hard-errors** if a single page cannot fit. When a budget is so small the
   beam thrashes, leanann says `degraded: true` and tells you the minimum
   budget — it never pretends a slow, evicting search was a clean one.

Then, because a semantic index alone is a lie for most real corpora, leanann
also builds a **BM25 inverted index** from the same pass and fuses the two
with **Reciprocal Rank Fusion**. Short queries, rare tokens, identifiers,
typos — the keyword side catches what the quantized vector side blurs, and
`--mode` lets you pick either.

## Why it wins

- **It runs where the others won't.** Zero dependencies, no native build, no
  server process. `node src/cli.js index corpus.jsonl` on a stock Node ≥ 20.
- **The budget is a contract.** Enforced at the allocator, measured by
  `bench`, reported per query. Other engines *aim* for a memory target;
  leanann *cannot exceed* one.
- **Honesty is the feature.** `bench` measures real recall@k against an exact
  brute-force ground truth on your corpus — not a benchmark cherry-picked on
  SIFT1M. `degraded`, stale-index, and thrashing states are surfaced, never
  hidden.
- **Hybrid is the default, not an add-on.** One pass builds both indexes; one
  query fuses both. No second engine, no glue code.
- **It is a primitive, not a platform.** A CLI with `--json` on everything and
  deterministic, byte-identical output — designed to be *embedded* in other
  tools (this vault's own `ctxc ask` and `issue-radar` are candidates).

## Who wants it

- **Local RAG builders** who want a 1M-chunk index on a laptop without a
  vector DB running as a daemon.
- **Maintainers of large monorepos** doing semantic dedup of issues, PRs, and
  comments (see `issue-radar`, `reviewlab`).
- **CI pipelines** that need similarity search inside a small memory envelope
  — dedup test failures, cluster logs, find near-duplicate reports.
- **Anyone embedding search in a CLI or Action** who cannot ship a server
  dependency or a native module.
- **The long tail of "I just want similar-text search"** — researchers,
  journalists, hobbyists — for whom an install that *just runs* beats a
  config file with a memory section.

## Features (v0, all shipped)

- **`leanann index <input>`** — build from JSONL (`{"id","text","vector"?}`)
  or a directory of `.txt`/`.md` files. Streaming, budget-aware build.
- **Three vector sources:**
  - `--embed hash` (default) — deterministic feature-hash embedding. Instant,
    zero-dep, honest: good for lexical similarity and dedup.
  - `--embed-cmd "<cmd>"` — shell out to *any* embedding model
    (`ollama`, `text-embeddings-inference`, a Python script…). leanann ships
    no model and makes no model claims; you bring embeddings, it runs them
    cheap. `{input}`/`{output}` JSONL placeholders, timeout + output caps.
  - pass-through — a `vector` field in your JSONL is used directly.
- **`leanann search <query>`** — `--mode hybrid|vector|bm25`, `--k`, `--ef`,
  `--explain` (per-source ranks + fusion), `--json`.
- **`leanann bench`** — recall@k vs exact brute-force ground truth, p50/p95
  latency, peak RSS, on-disk size, cache hit rate. The AirLLM benchmark
  table, for your corpus.
- **`leanann stats`** — index manifest, counts, byte breakdown, cache
  accounting.
- **Hard memory budget** — `--budget-mb` (default 128) enforced by a
  `Budget`/`PageCache` accounting layer with LRU eviction and honest
  `degraded` reporting.
- **Product quantization** — seeded, deterministic k-means codebooks trained
  on a reservoir sample of *your* vectors; asymmetric distance (ADC) lookup
  tables per query.
- **HNSW graph, paged** — layered small-world graph with geometric level
  assignment and the HNSW neighbor-selection heuristic; adjacency stored in a
  fixed-stride record file behind the same page cache.
- **BM25 inverted index** — k1/b saturation, smoothed idf, delta+varint
  postings built via external run-merge so *build* respects the budget too.
- **Reciprocal Rank Fusion** — parameter-free hybrid fusion (`--rrf-k`,
  default 60).
- **`--embed-cmd` + `--exclude`** — glob excludes merged with `.leanannignore`.
- **Determinism gate** — same input + config ⇒ byte-identical index files;
  asserted in the suite.
- **Staleness check** — `leanann index --check` (content-hash manifest, the
  `ctxc` pattern) exits 1 when the corpus moved.
- **Safety** — writes confined to the out dir, temp exchanges in `os.tmpdir()`,
  read commands (`search`/`stats`) leave the target byte-identical (fingerprint
  proofs in tests).

## Architecture

```
                       ┌─────────────────────────────────────────┐
   corpus (JSONL/dir)  │                leanann index            │
        │              └─────────────────────────────────────────┘
        ▼                        │
  ┌───────────┐   tokenize   ┌───▼──────────────┐   reservoir   ┌──────────────┐
  │  ingest   ├─────────────►│  BM25 postings   │   sample      │ k-means      │
  │ (streaming)│              │  (run-merge)     ├──────────────►│ codebooks    │
  └─────┬─────┘              └──────────────────┘               └──────┬───────┘
        │ embed (hash | --embed-cmd | passthrough)                     │
        ▼                                                              ▼
  ┌───────────┐      encode (M bytes/doc)                      ┌──────────────┐
  │  vectors  ├────────────────────────────────────────────────►│ codes.bin    │
  └───────────┘                                                  │ (fixed stride)│
                                                                 └──────┬───────┘
   .leanann/  (everything below respects --budget-mb)                    │ insert
  ┌───────────────┬───────────────┬───────────────┬──────────────┐  ┌────▼──────┐
  │ manifest.json │ terms/postings│ codes.bin     │ graph.bin    │  │ HNSW      │
  │ (hashes,meta) │ (varint)      │ (PQ codes)    │ (adjacency)  │  │ (paged)   │
  └───────────────┴───────┬───────┴───────────────┴──────┬───────┘  └───────────┘
                          │                    ▲         │
                          │   ┌────────────────┴─────────┴──────────┐
                          │   │   Budget + PageCache (hard cap,     │
                          │   │   LRU eviction, hit/miss/eviction   │
                          │   │   accounting)  ◄── the AirLLM layer │
                          │   └────────────────┬───────────────────┘
                          ▼                    ▼
   leanann search  ┌──────────┐   ┌──────────┐   ┌────────────┐
   <query>  ──────►│  BM25    │   │ ANN beam │   │ RRF fusion │──► results
                   │ top-N    │   │ (ef)     │   │ + --explain│
                   └──────────┘   └──────────┘   └────────────┘
```

The memory story is the whole product: only `codes.bin` stride reads, graph
node records, and postings ranges pass through the **Budget/PageCache**, which
enforces `--budget-mb` by LRU eviction and hard-fails on impossible configs
(page larger than budget, beam larger than budget). `bench` proves it by
measuring peak RSS *and* logical cache bytes separately — never conflating
"under budget" with "low RSS."

## Growth loop

1. **`bench` output is the meme.** A table with *your* corpus: recall@k, p95
   latency, peak RSS, index size. Post the before/after of `--budget-mb 16`
   vs `512`. That table is the launch image.
2. **`--json` everywhere** means it drops into scripts; every user's first
   week is a tweet about the 6-line wrapper they wrote.
3. **The Action** (`leanann` on a schedule) turns a repo's issues/comments
   into a dedup report — visible, recurring, shareable.
4. **Composition.** The other tools in this vault are the first users:
   `ctxc ask` over a real corpus, `issue-radar` similarity, `reviewlab`
   "this PR resembles these merged ones."

## Competitors & wedge

- **FAISS / hnswlib / DiskANN (C++).** Faster, more algorithms, and they need
  a build toolchain, a native module, or a GPU. leanann's wedge is *"it runs
  anywhere Node runs, and it cannot exceed your budget."*
- **Qdrant / Weaviate / Milvus.** Real servers with real features. Wedge: no
  daemon, no config file, no ops. It's a CLI and a library.
- **sqlite-vec / LanceDB / Chroma.** The honest closest neighbors — embedded
  and embeddable. Wedge: a *hard, reported* memory budget, hybrid BM25 as a
  first-class citizen, and a `bench` that measures recall against exact
  ground truth instead of asserting speed on a leaderboard.
- **pgvector.** If you already run Postgres, it's great. leanann is for
  everyone who doesn't want to.

The wedge is not "faster ANN." It is **search that is honest about, and
capped by, its own memory** — a category the incumbents treat as an
implementation detail.

## 48-hour MVP (already built)

v0 is complete and green. The 48-hour loop now is *launch*: pick a headline
corpus (a public GitHub issue dump, the docs of a popular package), run
`bench`, publish the table and the JSONL, ship the Action.

## Monetization (open core)

- **Free forever:** the CLI, the library, the budget engine, hybrid search.
- **Lean:** hosted corpora — index a public dataset, get an API + badge.
- **Lean Pro:** scheduled re-index Actions, org-wide dedup, recall SLOs.
- **Support/SSO:** for teams running it in CI at scale.
- Never: a license that makes the core non-free for individuals.

## Risks

- **"Just use FAISS" from power users.** Correct — for them. The wedge is the
  envelope (zero-dep, capped RAM, no server), not raw QPS.
- **PQ recall is not exact recall.** Mitigate: `bench` *measures* recall@k on
  your data and `--explain` shows when the vector side contributed nothing;
  BM25 is always there as the honest fallback.
- **A Node ANN library can't match C++ throughput.** True, and the README says
  so. The pitch is the budget and the install story, not a FLOPs chart.
- **Embedding model churn.** `--embed-cmd` makes leanann model-agnostic by
  design; we ship no model and claim no quality.

## Build & test

Zero runtime dependencies — Node ≥ 20 only. No `npm install`, no lockfile.

```bash
# run from the leanann directory
node src/cli.js --help
node src/cli.js index corpus.jsonl --out .leanann --embed hash
node src/cli.js search "connection reset" --out .leanann --k 5 --explain
node src/cli.js bench corpus.jsonl --budget-mb 64 --k 10   # recall vs exact, RSS, disk
node src/cli.js stats --out .leanann --json
node --test test/                                          # 134 tests
```

(`test/index.js` is the Node ≥ 21 bootstrap: `node --test test/` matches the
directory as a glob, so the shim imports every `*.test.js`. Inert elsewhere.)

## v0 status

**Shipped, covered by 134 passing tests:**

- **Budget engine** — `Budget` + `PageCache` with LRU eviction, hard caps,
  hit/miss/eviction/bytes accounting, and config-time validation (page > budget
  or beam > budget ⇒ honest error with the computed minimum).
- **Store** — fixed-stride record files (`codes.bin`, `graph.bin`) + offset
  sidecar read via page-aligned `readRange`; append-only writer; truncation and
  stride-mismatch detected with expected/actual sizes.
- **Product quantization** — seeded deterministic k-means (k up to 256),
  codebooks trained on a reservoir sample, `Uint8` codes, ADC query tables,
  subspace-distance math verified against direct computation.
- **HNSW graph** — geometric level assignment, beam search, HNSW neighbor
  heuristic, paged adjacency, disconnected/degraded detection.
- **BM25** — smoothed idf, k1/b saturation, delta+varint postings, external
  run-merge so build stays under budget.
- **Hybrid** — RRF fusion with `--explain`; `--mode hybrid|vector|bm25`.
- **Vectors** — hash embedder, `--embed-cmd` (JSONL placeholders, timeout,
  caps), direct pass-through; dimension validation.
- **CLI** — `index|search|bench|stats`, `--json`, `--budget-mb`, `--check`,
  exit codes 0/1/2/3, injectable stdout/stderr/exec/cwd/clock.
- **Honesty rails** — determinism (byte-identical rebuilds), staleness
  (`--check`, honest even when `--out` lives *inside* the input dir),
  `degraded` reporting, RSS-vs-logical-budget separation, read commands
  fingerprinted read-only, writes confined to the out dir (path-traversal
  refused, separator-safe on Windows), no network anywhere in `src/`.

**Deferred (v1+, documented, not silently dropped):** a scalar-int8 fast path,
disk-backed *build* for corpora whose postings exceed RAM without run-merge,
MIPS/MAX inner-product mode, `leanann serve` (a thin HTTP façade), a GitHub
Action wrapper, and per-language stemmers.
