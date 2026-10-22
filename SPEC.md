# leanann — SPEC (v0.1.0)

The build document. `README.md` is the pitch; this file is the contract the
code and tests hold themselves to.

## 0. Non-negotiables

1. **Zero runtime dependencies.** Node builtins only (`node:fs`, `node:crypto`,
   `node:child_process`, `node:os`, `node:path`). No `npm install`, no lockfile.
2. **ESM, Node ≥ 20.** `"type": "module"`, no transpile step.
3. **The budget is enforced, not approximated.** Every byte of index data that
   enters RAM during build and query passes through `Budget`/`PageCache`.
4. **Determinism.** Same corpus + config ⇒ byte-identical index files and
   byte-identical reports (timing fields excluded and excluded-by-design).
5. **Injectable side effects.** exec/clock/fs-out/stdout/stderr are parameters
   with real defaults; tests inject fakes and never patch globals.
6. **Honesty rails.** Degraded searches, stale indexes, impossible budgets and
   missing models are surfaced with distinct exit codes and messages. No
   silent truncation, no fabricated results.

## 1. CLI surface

```
leanann index <input> [options]        build/refresh an index
leanann search <query...> [options]    query an index
leanann bench <input> [options]        measure recall/latency/RSS vs exact
leanann stats [options]                manifest, counts, byte breakdown
leanann --help | --version
```

### 1.1 `index`

| Flag | Default | Meaning |
|------|---------|---------|
| `--out <dir>` | `.leanann` | output dir (only place ever written) |
| `--embed hash\|none` | `hash` | built-in embedder; `none` = require vectors |
| `--embed-cmd <cmd>` | — | external embedder; `{input}`/`{output}` JSONL placeholders |
| `--dims <n>` | `384` | vector dimensions (hash embedder + validation) |
| `--m <n>` | `8` | PQ subspaces (code bytes per vector) |
| `--k <n>` | `256` | k-means centroids per subspace (≤ 256; codes are uint8) |
| `--sample <n>` | `20000` | reservoir sample size for codebook training |
| `--iters <n>` | `12` | k-means Lloyd iterations |
| `--graph-m <n>` | `16` | HNSW max edges per node (upper layers); level 0 uses `2m` |
| `--budget-mb <n>` | `128` | hard RAM budget for index data |
| `--page-kb <n>` | `64` | page-cache page size |
| `--exclude <glob>` | — | repeatable; merged with `.leanannignore` |
| `--check` | — | verify manifest hashes against the corpus; no writes |
| `--seed <n>` | `42` | RNG seed (k-means init, level assignment, sampling) |
| `--json` | — | machine-readable progress/summary |
| `--quiet` | — | no stdout on success |

### 1.2 `search`

| Flag | Default | Meaning |
|------|---------|---------|
| `--out <dir>` | `.leanann` | index dir |
| `--k <n>` | `10` | results |
| `--mode hybrid\|vector\|bm25` | `hybrid` | retrieval source(s) |
| `--ef <n>` | `64` | beam width (must be ≥ k) |
| `--rrf-k <n>` | `60` | RRF constant |
| `--budget-mb <n>` | manifest | page-cache budget (may *raise*, not lower below minimum) |
| `--explain` | — | per-source ranks + fusion contribution per result |
| `--json` | — | JSON results |

### 1.3 `bench`

| Flag | Default | Meaning |
|------|---------|---------|
| `<input>` | (positional) | corpus used for queries (sampled) |
| `--index <dir>` | — | bench against an existing index (else builds in temp) |
| `--queries <n>` | `32` | query count (seeded sample of corpus docs) |
| `--k <n>` | `10` | recall@k cutoff |
| `--budget-mb <n>` | `128` | budget under test |
| `--exact-cap <n>` | `50000` | brute-force ground-truth cap (honest note if exceeded) |
| `--json` | — | JSON report |

### 1.4 Exit codes (all commands)

| Code | Meaning |
|------|---------|
| `0` | success |
| `1` | completed with warnings (search degraded, index stale in `--check`) |
| `2` | usage error (unknown flag, missing arg, bad value) |
| `3` | operational failure (missing/corrupt index, impossible budget, embed failure, I/O) |

## 2. Config file

`leanann.config.json` in cwd (optional). Flags override file; file overrides
defaults.

```json
{
  "out": ".leanann",
  "embed": "hash",
  "embedCmd": null,
  "dims": 384, "m": 8, "k": 256, "sample": 20000, "iters": 12, "graphM": 16,
  "budgetMb": 128, "pageKb": 64, "rrfK": 60, "seed": 42,
  "exclude": ["**/node_modules/**"]
}
```

`.leanannignore`: newline globs, merged with `--exclude`.

## 3. Data model & on-disk formats (`.leanann/`)

All binary formats little-endian, fixed stride where noted. `manifest.json`
carries format versions; a mismatch is exit 3 with the expected version.

### 3.1 `manifest.json`

```json
{
  "version": 1, "createdAt": "...", "tool": "leanann@0.1.0",
  "docs": 12345, "dims": 384, "m": 8, "k": 256, "graphM": 16,
  "entryId": 4211, "entryLevel": 3, "terms": 42110, "avgdl": 18.4,
  "embed": {"kind": "hash", "dims": 384},
  "seed": 42,
  "files": {"codes.bin": {"sha256": "...", "bytes": 98760},
            "graph.bin": {"sha256": "...", "bytes": 3221184},
            "postings.bin": {"sha256": "..."}, "terms.bin": {"sha256": "..."},
            "doclens.bin": {"sha256": "..."}, "meta.bin": {"sha256": "..."},
            "metaoff.bin": {"sha256": "..."}, "codebooks.bin": {"sha256": "..."}},
  "corpus": {"kind": "jsonl", "hash": "sha256-of-content", "path": "corpus.jsonl"}
}
```

`corpus.hash` powers `--check` (the `ctxc` staleness pattern).

### 3.2 `codebooks.bin`

`M` blocks of `k × sub × 4` float32, `sub = dims / M`. `dims % M === 0` is
validated at build (honest error otherwise).

### 3.3 `codes.bin`

Fixed stride `M` bytes per doc, docId = line number. Random access
`offset = docId × M`.

### 3.4 `graph.bin`

Per node (fixed stride), `MAXL = 4`:

```
u8  level            (clamped 0..3)
per level l in 0..3:
  u8  degree
  u16 neighbors[M0]  (M0 = 2 × graphM, unused slots = 0xFFFF)
```

stride = `1 + 4 × (1 + 2 × M0)`. Nodes above level 3 are clamped (documented;
probability of level ≥ 4 with `mL = 1/ln(M)` is negligible and clamping is
reported in `stats`).

### 3.5 Postings

- `terms.bin`: sorted terms, `varint(byteLen) + utf8` concatenated;
  `termoff.bin`: u32 offsets per term.
- `postings.bin`: per term — `varint(df)` then `df` entries of
  `varint(ΔdocId) + varint(tf)` (Δ ≥ 1 for non-first entries).
- `doclens.bin`: u32 per doc. Loaded under budget accounting
  (`4 × docs ≤ budget/4`, else exit 3 with the computed minimum).

### 3.6 Metadata

- `meta.bin`: per doc, `varint(idLen)+id + varint(previewLen)+preview(utf8)`.
- `metaoff.bin`: u32 per doc → byte offset in `meta.bin`. Only result docs are
  fetched (paged `readRange`).

## 4. Algorithms

### 4.1 Tokenizer

`[a-z0-9_]+` runs after case folding; CJK codepoints → per-char unigrams;
tokens `< 2` chars dropped unless numeric; English stopword list dropped
(`stop=true`). **No stemming in v0** (documented deferral).

### 4.2 Embedding

- **hash** (default): for each token, `h = fnv1a32(token)`; `v[h % dims] +=
  (h & 0x80000000) ? -1 : +1`; L2-normalize. Deterministic, lexical.
- **embed-cmd**: write `{input}` JSONL (`{"i":n,"text":...}`), run the command
  (shell, user-trusted, documented), read `{output}` JSONL
  (`{"i":n,"vector":[...]}`), enforce dims and count, cap output, timeout ⇒
  exit 3.
- **passthrough**: `vector` field in input JSONL used directly (validated).

All vectors L2-normalized before PQ so ADC distances are comparable.

### 4.3 Product quantization + ADC

`sub = dims / M`. k-means per subspace over the reservoir sample: k-means++
init (seeded mulberry32), Lloyd iterations, empty-cluster reseed. Query: build
`T[m][256]` subspace-distance tables (cost `256 × dims`), distance =
`Σ_m T[m][code[m]]`. Node-vs-node (heuristic): table from one node's code.

### 4.4 HNSW (paged)

- `mL = 1 / ln(graphM)`; level = `min(3, floor(-ln(u) × mL))`, `u` from the
  seeded RNG.
- Insert: greedy descent from entry to level `l+1`, then `ef = 2m` beam at
  level `l`; neighbors via the **HNSW heuristic** (diversity: accept `c` only
  if `dist(c, e) < dist(c, selected)` for the current selected set); bidirectional
  edges with over-cap pruning on the neighbor side.
- Search: greedy descent above level 0, `ef`-beam on level 0.
- Determinism: stable sorts with docId tie-break; seeded RNG; visited sets are
  insertion-ordered.
- **Degraded detection**: if page evictions during one query exceed
  `ef × 2`, the result is flagged `degraded` with the minimum budget that
  would avoid it (measured from graph.bin size + ef working set).

### 4.5 BM25

`idf = ln(1 + (N − df + 0.5)/(df + 0.5))` (always > 0);
`score += idf × tf(k1+1) / (tf + k1(1 − b + b·dl/avgdl))`, k1 = 1.2, b = 0.75.

Build: postings accumulate per batch; when estimated bytes exceed
`budget/4`, sort and flush a **run**; runs are k-way merged into
`postings.bin` (external-memory pattern ⇒ build respects the budget).

### 4.6 Fusion

`score(d) = Σ_{list ∈ sources} 1 / (rrfK + rank_list(d))`. Ties break by
docId. `--explain` emits per-source rank and contribution.

## 5. Security posture

- **Untrusted input**: corpus lines and query text are data, never code.
  Embeddings validated (finite numbers, exact dims) or rejected.
- **`--embed-cmd` is explicitly user-trusted** (their own command; shell).
  Documented; timeout + output caps; no env leakage beyond a clean allowlist.
- **Writes confined** to `--out` (resolved `startsWith` guard, no traversal);
  temp exchanges in `os.tmpdir()` and removed.
- **No network** in core, ever. `bench`/`search`/`index` never open sockets.
- **Read commands are read-only** — asserted by tree fingerprints.

## 6. Performance budgets

| Metric | Budget (v0) |
|--------|-------------|
| Query RAM | ≤ `--budget-mb`, enforced |
| Build RAM | ≤ `--budget-mb` + one batch buffer (bounded by `budget/4`) |
| Page size | 64 KB default; ≤ budget validated |
| `bench` sample | exact ground truth capped at `--exact-cap` (honest note if the corpus is larger) |

## 7. Failure modes → behavior

| Failure | Behavior |
|---------|----------|
| dims not divisible by M | exit 3, message with the constraint |
| budget < page size | exit 3, message with computed minimum |
| budget < beam working set | exit 3, message with computed minimum |
| missing `.leanann/` | exit 3, `run leanann index first` |
| manifest version mismatch | exit 3, expected vs found |
| truncated/short file vs manifest | exit 3, expected vs actual bytes |
| stale corpus (`--check`) | exit 1 + diff summary |
| empty corpus | exit 3, honest message |
| no results | exit 0, `no results` (never fabricated) |
| thrashing search | exit 1 + `degraded: true` + minimum budget |
| embed-cmd fail/timeout/wrong dims | exit 3, cause + counts |

## 8. Test strategy

`node:test` + `node:assert/strict`, zero deps, no network, tmpdir-only writes,
injected exec/clock for externalities. Per-file coverage: tokenizer; BM25 math
vs hand-computed values + postings roundtrip; k-means determinism + ADC vs
direct math; budget/LRU accounting + impossible configs; store roundtrip +
truncation detection + cache hit/miss; graph recall on synthetic clusters vs
exact ground truth + determinism; hash/cmd embedders (fake exec); end-to-end
build + reopen + search; hybrid ≥ either mode on a constructed fixture; bench
recall math; CLI exit codes; safety (confined writes, read-only proofs, the
byte-identical rebuild determinism gate).

## 9. v1 → v2 cut line

**v1:** scalar-int8 fast path, disk-run build without the in-RAM batch buffer,
MIPS/inner-product search, `serve` (HTTP façade over `search --json`), the
GitHub Action, per-language stemmers, a `reindex --partial` for appends.

**v2:** sharded multi-segment indexes (append without rebuild), learned
(anisotropic) quantization, a WASM SIMD distance kernel, hybrid reranking with
cross-encoders behind `--embed-cmd`, and a hosted corpus tier.
