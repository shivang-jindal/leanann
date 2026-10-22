// graph.js — HNSW-style layered small-world graph over a FIXED-STRIDE paged
// file. The adjacency — the RAM-hungry part of any ANN index — never loads
// wholesale: only the nodes the beam actually visits touch the page cache,
// which is the AirLLM trick (keep the active set hot, page the rest).
//
// Layout per node (graphStride(m)), little-endian:
//   u8  level (clamped 0..MAX_LEVEL)
//   per level l in 0..MAX_LEVEL: u8 degree; u16 m0 neighbors (0xFFFF = unused)
// MAX_LEVEL = 4 recorded levels; geometric assignment clamps above 3 (SPEC 3.4).

import { mulberry32 } from './util.js';
import { codeTables, queryTables, adcDist } from './quantize.js';

export const MAX_LEVEL = 3;

export function graphStride(m) {
  const m0 = 2 * m;
  return 1 + 4 * (1 + 2 * m0);
}

export function assignLevel(rng, m) {
  const mL = 1 / Math.log(m);
  const u = Math.max(rng(), 1e-9);
  return Math.min(MAX_LEVEL, Math.floor(-Math.log(u) * mL));
}

function cmpDistAsc(a, b) {
  if (a.dist !== b.dist) return a.dist - b.dist;
  return a.id - b.id;
}

function cmpDistDesc(a, b) {
  if (b.dist !== a.dist) return b.dist - a.dist;
  return a.id - b.id;
}

export function parseRecord(buf, m0) {
  let p = 0;
  const level = buf[p];
  p += 1;
  const degrees = new Uint8Array(4);
  const neighbors = [];
  for (let l = 0; l < 4; l += 1) {
    degrees[l] = buf[p];
    p += 1;
    const list = new Int32Array(m0).fill(-1);
    for (let i = 0; i < m0; i += 1) {
      const v = buf[p] | (buf[p + 1] << 8);
      p += 2;
      if (v !== 0xffff) list[i] = v;
    }
    neighbors.push(list);
  }
  return { level, degrees, neighbors };
}

export function serializeRecord(rec, m0) {
  const stride = graphStride(m0 / 2);
  const out = new Uint8Array(stride);
  let p = 0;
  out[p] = rec.level;
  p += 1;
  for (let l = 0; l < 4; l += 1) {
    const deg = Math.min(rec.degrees[l], m0);
    out[p] = deg;
    p += 1;
    const list = rec.neighbors[l];
    for (let i = 0; i < m0; i += 1) {
      const v = i < deg && list[i] >= 0 ? list[i] : 0xffff;
      out[p] = v & 0xff;
      out[p + 1] = (v >> 8) & 0xff;
      p += 2;
    }
  }
  if (p !== stride) throw new Error(`graph record serialize: ${p} != ${stride}`);
  return out;
}

export class Graph {
  constructor({ file, codes, m, k, seed = 42, entry = null }) {
    this.file = file;   // PagedFile over graph.bin (writable during build)
    this.codesFile = codes; // PagedFile over codes.bin
    this.m = m;
    this.m0 = 2 * m;
    this.k = k;
    this.rng = mulberry32(seed);
    this.entry = entry; // {id, level} | null
    this.stride = graphStride(m);
  }

  get count() {
    return this.file.count;
  }

  async nodeCodes(id) {
    return this.codesFile.read(id);
  }

  // float-query distance to a node (ADC).
  async distTo(tables, id) {
    return adcDist(tables, await this.nodeCodes(id), this.k);
  }

  // node-vs-node distance in code space (symmetric).
  async distNodes(a, b) {
    if (a === b) return 0;
    const ca = await this.nodeCodes(a);
    const cb = await this.nodeCodes(b);
    return adcDist(codeTables(ca, this.books, this.cfg), cb, this.k);
  }

  setBooks(books, cfg) {
    this.books = books;
    this.cfg = cfg;
  }

  async record(id) {
    const buf = await this.file.read(id);
    return parseRecord(buf, this.m0);
  }

  async neighborsAt(id, level) {
    const rec = await this.record(id);
    const out = [];
    if (level > rec.level) return out;
    const deg = Math.min(rec.degrees[level], this.m0);
    for (let i = 0; i < deg; i += 1) {
      const n = rec.neighbors[level][i];
      if (n >= 0) out.push(n);
    }
    return out;
  }

  async writeRecord(id, rec) {
    await this.file.write(id, serializeRecord(rec, this.m0));
  }

  // --- beam search -----------------------------------------------------------
  // candidates/result sorted (dist asc, id asc); visited is insertion-ordered.
  async beam(tables, entryId, entryDist, ef, level) {
    const visited = new Set([entryId]);
    const W = [{ id: entryId, dist: entryDist }];
    const C = [{ id: entryId, dist: entryDist }];
    while (C.length > 0) {
      C.sort(cmpDistAsc);
      const c = C.shift();
      if (W.length >= ef && c.dist > W[W.length - 1].dist) break;
      const nbrs = await this.neighborsAt(c.id, level);
      nbrs.sort((x, y) => x - y); // deterministic iteration
      for (const n of nbrs) {
        if (visited.has(n)) continue;
        visited.add(n);
        const d = await this.distTo(tables, n);
        if (W.length < ef || d < W[W.length - 1].dist) {
          W.push({ id: n, dist: d });
          C.push({ id: n, dist: d });
          W.sort(cmpDistAsc);
          if (W.length > ef) W.pop();
        }
      }
    }
    W.sort(cmpDistAsc);
    return W;
  }

  // HNSW neighbor-selection heuristic (diversity): keep candidate c only when
  // it is closer to the anchor (cand.dist) than to every already-selected
  // neighbor. Ties break deterministically by (dist, id).
  async selectHeuristic(candidates, cap) {
    const sorted = candidates.slice().sort(cmpDistAsc);
    const result = [];
    for (const cand of sorted) {
      if (result.length >= cap) break;
      let keep = true;
      for (const sel of result) {
        const d = await this.distNodes(cand.id, sel.id);
        if (d < cand.dist) { keep = false; break; }
      }
      if (keep) result.push(cand);
    }
    // If the heuristic filtered too aggressively, backfill with the closest
    // remaining candidates (never leave a node isolated when options exist).
    if (result.length < Math.min(cap, sorted.length)) {
      const chosen = new Set(result.map((r) => r.id));
      for (const cand of sorted) {
        if (result.length >= cap) break;
        if (!chosen.has(cand.id)) { result.push(cand); chosen.add(cand.id); }
      }
    }
    return result;
  }

  // --- insert ----------------------------------------------------------------
  // tables: ADC tables built from the NEW node's float vector (anchor).
  async insert(id, tables) {
    const level = assignLevel(this.rng, this.m);
    const rec = { level, degrees: new Uint8Array(4), neighbors: [0, 1, 2, 3].map(() => new Int32Array(this.m0).fill(-1)) };

    if (!this.entry) {
      await this.writeRecord(id, rec);
      this.entry = { id, level };
      return { level, edges: 0 };
    }

    let enterId = this.entry.id;
    let enterDist = await this.distTo(tables, enterId);
    // greedy descent above the node's own level
    for (let l = this.entry.level; l > level; l -= 1) {
      const W = await this.beam(tables, enterId, enterDist, 1, l);
      if (W.length === 0) break;
      enterId = W[0].id;
      enterDist = W[0].dist;
    }

    let edges = 0;
    const top = Math.min(level, this.entry.level);
    for (let l = top; l >= 0; l -= 1) {
      const ef = l === 0 ? this.m0 : this.m;
      const W = await this.beam(tables, enterId, enterDist, ef, l);
      const cap = l === 0 ? this.m0 : this.m;
      const withEnter = W.slice();
      if (!withEnter.some((x) => x.id === enterId)) {
        withEnter.push({ id: enterId, dist: enterDist });
      }
      const selected = await this.selectHeuristic(withEnter, cap);
      const list = selected.filter((s) => s.id !== id);
      rec.degrees[l] = list.length;
      for (let i = 0; i < list.length; i += 1) rec.neighbors[l][i] = list[i].id;
      edges += list.length;

      // backlink: add id to each selected neighbor's level-l list, prune if over cap
      for (const s of list) {
        const nrec = await this.record(s.id);
        if (l > nrec.level) continue;
        const deg = Math.min(nrec.degrees[l], this.m0);
        const listN = [];
        for (let i = 0; i < deg; i += 1) if (nrec.neighbors[l][i] >= 0) listN.push(nrec.neighbors[l][i]);
        if (listN.includes(id)) continue;
        listN.push(id);
        if (listN.length > this.m0) {
          // prune: heuristic with s as anchor (dist(s, x) via s's code tables)
          const cand = [];
          const sTables = codeTables(await this.nodeCodes(s.id), this.books, this.cfg);
          for (const nid of listN) {
            const d = adcDist(sTables, await this.nodeCodes(nid), this.k);
            cand.push({ id: nid, dist: d });
          }
          const kept = await this.selectHeuristic(cand, this.m0);
          nrec.degrees[l] = kept.length;
          for (let i = 0; i < this.m0; i += 1) nrec.neighbors[l][i] = -1;
          for (let i = 0; i < kept.length; i += 1) nrec.neighbors[l][i] = kept[i].id;
        } else {
          nrec.degrees[l] = listN.length;
          nrec.neighbors[l][listN.length - 1] = id;
        }
        await this.writeRecord(s.id, nrec);
      }

      if (W.length > 0) {
        enterId = W[0].id;
        enterDist = W[0].dist;
      }
    }

    await this.writeRecord(id, rec);
    if (level > this.entry.level) this.entry = { id, level };
    return { level, edges };
  }

  // --- search ----------------------------------------------------------------
  async search(tables, ef) {
    if (!this.entry) return [];
    let cur = this.entry.id;
    let curDist = await this.distTo(tables, cur);
    const rec = await this.record(cur);
    for (let l = rec.level; l > 0; l -= 1) {
      const W = await this.beam(tables, cur, curDist, 1, l);
      if (W.length === 0) break;
      cur = W[0].id;
      curDist = W[0].dist;
    }
    return this.beam(tables, cur, curDist, ef, 0);
  }

  static buildTablesFor(v, cfg) {
    return queryTables(v, cfg.books, cfg);
  }
}
