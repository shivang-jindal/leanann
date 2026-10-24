# Contributing to leanann

Zero dependencies, by design. This project has no runtime dependencies and no build step: clone it, run `node --test test/` (or `node src/cli.js --help`), and you are productive. Contributing never requires `npm install` -- PRs must not add runtime dependencies, lockfiles, or network calls in tests.

Before opening a PR, make sure `node --test test/` is green on your machine (Node 20+). Keep the invariants the codebase already guarantees: injectable exec/clock/fetch in everything that touches the outside world, deterministic output (byte-order sorts, never locale-dependent comparison), and honest exit codes. If you grow a suite, update the test-count claims in the README so they match reality.

The `## v0 status` section of the README is part of the spec, not a to-do list: Deferred items are deliberately deferred. Read it before proposing features, and do not implement a deferred v1 item to satisfy a request -- link to that section instead. Small, well-tested PRs that fix bugs or improve docs are the fastest to merge.
