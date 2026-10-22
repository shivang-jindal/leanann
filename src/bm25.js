// bm25.js — inverted index: k1/b saturation, smoothed idf (always > 0),
// delta+varint postings. BUILD stays under budget by flushing sorted RUNS and
// k-way merging them (external-memory pattern) instead of holding every
// posting list in RAM.
//
// Files: terms.bin (varint len + utf8, sorted), termoff.bin (u32 per term),
//        postings.bin (per term: varint df, then df × [varint ΔdocId, varint tf]),
//        doclens.bin (u32 per doc).

import { open, writeFile, readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { varintEncode, varintDecode, topK } from './util.js';
import { StoreError } from './store.js';

export const K1 = 1.2;
export const B = 0.75;

export function idf(N, df) {
  return Math.log(1 + (N - df + 0.5) / (df + 0.5));
}

export function bm25Score({ tf, df, N, dl, avgdl, k1 = K1, b = B }) {
  const norm = 1 - b + b * (avgdl === 0 ? 0 : dl / avgdl);
  return idf(N, df) * ((tf * (k1 + 1)) / (tf + k1 * norm));
}

// --- build side -------------------------------------------------------------

export class PostingsBuilder {
  // Writes ONLY inside `dir` (caller guarantees confinement). Runs flush when
  // estimated bytes reach flushAtBytes (the caller passes budget/4).
  constructor({ dir, flushAtBytes = 8 * 1024 * 1024 } = {}) {
    this.dir = dir;
    this.flushAtBytes = flushAtBytes;
    this.map = new Map(); // term -> Map(docId -> tf)
    this.docLens = [];
    this.estimatedBytes = 0;
    this.runFiles = [];
    this.runSeq = 0;
    this.closed = false;
  }

  addDoc(docId, termFreqs) {
    let len = 0;
    for (const [term, tf] of termFreqs) {
      len += tf;
      let docs = this.map.get(term);
      if (!docs) {
        docs = new Map();
        this.map.set(term, docs);
        this.estimatedBytes += term.length + 8;
      } else {
        this.estimatedBytes += 8;
      }
      docs.set(docId, tf);
    }
    this.docLens[docId] = len;
    return len;
  }

  needsFlush() {
    return this.estimatedBytes >= this.flushAtBytes;
  }

  // Serialize one term block: varint len + utf8 term, varint df, entries.
  static #termBlock(term, merged) {
    const parts = [];
    const tb = Buffer.from(term, 'utf8');
    const lenBuf = [];
    varintEncode(tb.length, lenBuf);
    parts.push(Buffer.from(lenBuf), tb);
    const pbuf = [];
    varintEncode(merged.size, pbuf);
    const ids = [...merged.keys()].sort((a, b) => a - b);
    let prev = 0;
    for (const id of ids) {
      varintEncode(id - prev, pbuf); // Δ >= 1 for non-first
      varintEncode(merged.get(id), pbuf);
      prev = id;
    }
    parts.push(Buffer.from(pbuf));
    return Buffer.concat(parts);
  }

  // Sort the current buffer by term, append a run file.
  async flushRun() {
    if (this.map.size === 0) return;
    const terms = [...this.map.keys()].sort();
    const chunks = [];
    for (const term of terms) {
      chunks.push(PostingsBuilder.#termBlock(term, this.map.get(term)));
    }
    const path = join(this.dir, `.run-${this.runSeq}.bin`);
    this.runSeq += 1;
    const fd = await open(path, 'w');
    const buf = Buffer.concat(chunks);
    await fd.write(buf, 0, buf.length, 0);
    await fd.close();
    this.runFiles.push(path);
    this.map = new Map();
    this.estimatedBytes = 0;
  }

  // K-way merge runs -> terms.bin / termoff.bin / postings.bin / postoff.bin
  // / doclens.bin. termoff[i] indexes terms.bin; postoff[i] indexes
  // postings.bin (two independent offset spaces, same term order).
  async finish() {
    if (this.closed) throw new StoreError('builder already finished');
    await this.flushRun();

    const runs = [];
    for (const path of this.runFiles) {
      const fd = await open(path, 'r');
      const st = await fd.stat();
      const buf = Buffer.alloc(st.size);
      await fd.read(buf, 0, st.size, 0);
      await fd.close();
      runs.push({ buf, pos: 0, head: null });
    }

    const readEntry = (run) => {
      if (run.pos >= run.buf.length) { run.head = null; return null; }
      let [len, p] = varintDecode(run.buf, run.pos);
      const term = run.buf.subarray(p, p + len).toString('utf8');
      p += len;
      const [df, p2] = varintDecode(run.buf, p);
      const postings = [];
      let q = p2;
      let prev = 0;
      for (let i = 0; i < df; i += 1) {
        let d; let tf;
        [d, q] = varintDecode(run.buf, q);
        [tf, q] = varintDecode(run.buf, q);
        prev += d;
        postings.push({ docId: prev, tf });
      }
      run.pos = q;
      run.head = { term, postings };
      return run.head;
    };
    for (const run of runs) readEntry(run);

    const termsChunks = [];
    const termoff = [];
    const postChunks = [];
    const postoff = [];
    let termsCount = 0;
    let termBytes = 0;
    let postBytes = 0;

    for (;;) {
      // smallest term across run heads (ties: all runs merge, deterministic)
      let smallest = null;
      for (const run of runs) {
        if (!run.head) continue;
        if (smallest === null || run.head.term < smallest) smallest = run.head.term;
      }
      if (smallest === null) break;

      const merged = new Map();
      for (const run of runs) {
        if (run.head && run.head.term === smallest) {
          for (const { docId, tf } of run.head.postings) {
            merged.set(docId, (merged.get(docId) ?? 0) + tf);
          }
          readEntry(run);
        }
      }

      termoff[termsCount] = termBytes;
      termsCount += 1;
      // terms.bin holds ONLY term bytes; postings.bin holds ONLY entries
      const tb = Buffer.from(smallest, 'utf8');
      const lenBuf = [];
      varintEncode(tb.length, lenBuf);
      termsChunks.push(Buffer.from(lenBuf), tb);
      termBytes += lenBuf.length + tb.length;

      const pbuf = [];
      varintEncode(merged.size, pbuf);
      const ids = [...merged.keys()].sort((a, b) => a - b);
      let prev = 0;
      for (const id of ids) {
        varintEncode(id - prev, pbuf);
        varintEncode(merged.get(id), pbuf);
        prev = id;
      }
      postoff[termsCount - 1] = postBytes;
      const postBlock = Buffer.from(pbuf);
      postChunks.push(postBlock);
      postBytes += postBlock.length;
    }

    await writeFile(join(this.dir, 'terms.bin'), Buffer.concat(termsChunks));
    await writeFile(join(this.dir, 'postings.bin'), Buffer.concat(postChunks));
    const offBuf = Buffer.alloc(termoff.length * 4);
    for (let i = 0; i < termoff.length; i += 1) offBuf.writeUInt32LE(termoff[i] ?? 0, i * 4);
    await writeFile(join(this.dir, 'termoff.bin'), offBuf);
    const postOffBuf = Buffer.alloc(postoff.length * 4);
    for (let i = 0; i < postoff.length; i += 1) postOffBuf.writeUInt32LE(postoff[i] ?? 0, i * 4);
    await writeFile(join(this.dir, 'postoff.bin'), postOffBuf);

    const lens = Buffer.alloc(this.docLens.length * 4);
    for (let i = 0; i < this.docLens.length; i += 1) lens.writeUInt32LE(this.docLens[i] ?? 0, i * 4);
    await writeFile(join(this.dir, 'doclens.bin'), lens);

    for (const path of this.runFiles) await rm(path, { force: true }).catch(() => {});
    this.runFiles = [];
    this.closed = true;
    return { terms: termsCount, docs: this.docLens.length };
  }

  async abort() {
    for (const path of this.runFiles) await rm(path, { force: true }).catch(() => {});
    this.runFiles = [];
  }

  avgdl() {
    if (this.docLens.length === 0) return 0;
    let sum = 0;
    for (const l of this.docLens) sum += l ?? 0;
    return sum / this.docLens.length;
  }
}

// --- query side ---------------------------------------------------------------

export class Bm25Index {
  constructor({ manifest, pool, fid, doclens, termOffsets, postOffsets, termsBuf, postingsSize }) {
    this.manifest = manifest;
    this.pool = pool;
    this.fid = fid;
    this.doclens = doclens;
    this.termOffsets = termOffsets; // into terms.bin
    this.postOffsets = postOffsets; // into postings.bin
    this.termsBuf = termsBuf;
    this.postingsSize = postingsSize;
    this.N = manifest.docs;
    this.avgdl = manifest.avgdl;
    this.termCount = termOffsets.length;
    if (postOffsets.length !== termOffsets.length) {
      throw new StoreError('termoff.bin and postoff.bin disagree on term count (corrupt)');
    }
  }

  // Resident here (doclens/terms/offsets); the CALLER (openIndex) charges the
  // budget so that codebooks + these + the pool all fit one hard limit.
  // Postings themselves are paged through the shared pool.
  static async open({ dir, manifest, pool }) {
    const doclensBuf = await readFile(join(dir, 'doclens.bin'));
    if (doclensBuf.length % 4 !== 0) throw new StoreError('doclens.bin: not a multiple of 4 (corrupt)');
    const docs = doclensBuf.length / 4;
    if (docs !== manifest.docs) {
      throw new StoreError(`doclens has ${docs} entries, manifest says ${manifest.docs} — corrupt (rebuild)`);
    }
    const doclens = new Uint32Array(
      doclensBuf.buffer.slice(doclensBuf.byteOffset, doclensBuf.byteOffset + docs * 4),
    );
    const readU32 = async (name) => {
      let buf;
      try {
        buf = await readFile(join(dir, name));
      } catch {
        throw new StoreError(`leanann: ${name} missing from ${dir} — index is incomplete (rebuild with \`leanann index\`)`);
      }
      if (buf.length % 4 !== 0) throw new StoreError(`${name}: corrupt`);
      return new Uint32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length));
    };
    const termOffsets = await readU32('termoff.bin');
    const postOffsets = await readU32('postoff.bin');
    const termsBuf = await readFile(join(dir, 'terms.bin'));
    const fid = await pool.attach(join(dir, 'postings.bin'));
    const postingsSize = pool.fileSize(fid);
    return new Bm25Index({ manifest, pool, fid, doclens, termOffsets, postOffsets, termsBuf, postingsSize });
  }

  termAt(i) {
    if (i < 0 || i >= this.termCount) return null;
    const off = this.termOffsets[i];
    const [len, p] = varintDecode(this.termsBuf, off);
    return this.termsBuf.subarray(p, p + len).toString('utf8');
  }

  // Binary search the sorted term dictionary. Deterministic.
  termId(term) {
    let lo = 0;
    let hi = this.termCount - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const t = this.termAt(mid);
      if (t === term) return mid;
      if (t < term) lo = mid + 1;
      else hi = mid - 1;
    }
    return -1;
  }

  async postingsFor(termId) {
    if (termId < 0 || termId >= this.termCount) return null;
    const off = this.postOffsets[termId];
    const next = termId + 1 < this.termCount ? this.postOffsets[termId + 1] : this.postingsSize;
    const span = Math.max(1, next - off);
    const block = await this.pool.readRange(this.fid, off, span);
    const [df, p] = varintDecode(block, 0);
    const out = new Array(df);
    let q = p;
    let prev = 0;
    for (let i = 0; i < df; i += 1) {
      let d; let tf;
      [d, q] = varintDecode(block, q);
      [tf, q] = varintDecode(block, q);
      prev += d;
      out[i] = { docId: prev, tf };
    }
    return out;
  }

  async search(tokens, k) {
    const scores = new Map();
    for (const tok of new Set(tokens)) {
      const tid = this.termId(tok);
      if (tid < 0) continue;
      const posts = await this.postingsFor(tid);
      if (!posts || posts.length === 0) continue;
      for (const { docId, tf } of posts) {
        const s = bm25Score({
          tf, df: posts.length, N: this.N,
          dl: this.doclens[docId] ?? 0, avgdl: this.avgdl,
        });
        scores.set(docId, (scores.get(docId) ?? 0) + s);
      }
    }
    const items = [...scores.entries()].map(([id, score]) => ({ id, score }));
    return topK(items, k);
  }
}
