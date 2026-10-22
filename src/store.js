// store.js — on-disk formats behind ONE shared buffer pool: manifest,
// fixed-stride record files, byte ranges, hashing.
//
// BufferPool is the memory story's single chokepoint: every byte of index
// data that enters RAM during build or query is a page here, under one hard
// budget, one LRU, one set of accounting counters. Files are attached by id;
// pages are keyed `${fid}:${pageNo}`.

import { open, readFile, writeFile, stat, readdir, mkdir, rm } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { sha256 } from './util.js';
import { PageCache } from './budget.js';

export const FORMAT_VERSION = 1;
export const MANIFEST_NAME = 'manifest.json';

export class StoreError extends Error {
  constructor(message) {
    super(message);
    this.name = 'StoreError';
    this.code = 'STORE_ERROR';
  }
}

// --- buffer pool ----------------------------------------------------------------

export class BufferPool {
  constructor({ pageSize, limitBytes, writable = false }) {
    this.files = new Map(); // fid -> { fd, path, size, writable }
    this.nextFid = 1;
    this.cache = new PageCache({
      pageSize,
      limitBytes,
      readPage: (key) => this.#readPage(key),
      writePage: (key, bytes) => this.#writePage(key, bytes),
    });
    this.pageSize = pageSize;
    this.limit = limitBytes;
    this.writable = writable;
  }

  async attach(path, { writable = false } = {}) {
    // 'r+' (NOT 'a+' — append mode ignores explicit write positions and would
    // corrupt paged writes). File must pre-exist for read-write; build creates
    // it first. Writes past EOF extend the file exactly (see writeRange).
    const mode = writable ? 'r+' : 'r';
    const fd = await open(path, mode);
    const st = await fd.stat();
    const fid = this.nextFid;
    this.nextFid += 1;
    this.files.set(fid, { fd, path, size: st.size, writable: writable && this.writable });
    return fid;
  }

  fileSize(fid) {
    const f = this.files.get(fid);
    if (!f) throw new StoreError(`buffer-pool: unknown file ${fid}`);
    return f.size;
  }

  #parse(key) {
    const i = key.lastIndexOf(':');
    return [Number(key.slice(0, i)), Number(key.slice(i + 1))];
  }

  async #readPage(key) {
    const [fid, pageNo] = this.#parse(key);
    const f = this.files.get(fid);
    if (!f) throw new StoreError(`buffer-pool: unknown file for key ${key}`);
    const position = pageNo * this.pageSize;
    if (position >= f.size) return new Uint8Array(0);
    const buf = Buffer.alloc(this.pageSize);
    const { bytesRead } = await f.fd.read(buf, 0, this.pageSize, position);
    return new Uint8Array(buf.subarray(0, bytesRead));
  }

  async #writePage(key, bytes) {
    const [fid, pageNo] = this.#parse(key);
    const f = this.files.get(fid);
    if (!f) throw new StoreError(`buffer-pool: unknown file for key ${key}`);
    if (!f.writable) throw new StoreError(`buffer-pool: ${f.path} is read-only`);
    const position = pageNo * this.pageSize;
    await f.fd.write(Buffer.from(bytes), 0, bytes.length, position);
    if (position + bytes.length > f.size) f.size = position + bytes.length;
  }

  // Assemble `len` bytes at `start` (may span page boundaries).
  async readRange(fid, start, len) {
    const size = this.fileSize(fid);
    if (len === 0) return new Uint8Array(0);
    if (start < 0 || start + len > size) {
      throw new StoreError(
        `leanann: read range [${start}, ${start + len}) is outside the ${size}-B file — ` +
        'index is truncated (rebuild)',
      );
    }
    const out = new Uint8Array(len);
    let filled = 0;
    while (filled < len) {
      const abs = start + filled;
      const pageNo = Math.floor(abs / this.pageSize);
      const pageStart = pageNo * this.pageSize;
      const page = await this.cache.get(`${fid}:${pageNo}`);
      const offsetInPage = abs - pageStart;
      const take = Math.min(page.length - offsetInPage, len - filled);
      if (take <= 0) {
        throw new StoreError(
          `leanann: page ${pageNo} ended before byte ${abs} — short file (rebuild)`,
        );
      }
      out.set(page.subarray(offsetInPage, offsetInPage + take), filled);
      filled += take;
    }
    return out;
  }

  // Update a byte range in place through the pool (dirty, evicted-flushed).
  // Past-EOF writes extend the file to EXACTLY the needed length — no
  // page-boundary padding, because record counts derive from file size.
  async writeRange(fid, start, bytes) {
    const f = this.files.get(fid);
    if (!f || !f.writable) throw new StoreError('buffer-pool: file is read-only');
    let filled = 0;
    while (filled < bytes.length) {
      const abs = start + filled;
      const pageNo = Math.floor(abs / this.pageSize);
      const pageStart = pageNo * this.pageSize;
      let page;
      try {
        page = (await this.cache.get(`${fid}:${pageNo}`)).slice();
      } catch (err) {
        if (err && err.name === 'StoreError') page = new Uint8Array(0); // beyond EOF
        else throw err;
      }
      const offsetInPage = abs - pageStart;
      const needed = offsetInPage + Math.min(this.pageSize - offsetInPage, bytes.length - filled);
      if (page.length < needed) {
        const grown = new Uint8Array(needed);
        grown.set(page);
        page = grown;
      }
      const take = Math.min(page.length - offsetInPage, bytes.length - filled);
      page.set(bytes.subarray(filled, filled + take), offsetInPage);
      await this.cache.put(`${fid}:${pageNo}`, page);
      filled += take;
    }
    if (start + bytes.length > f.size) f.size = start + bytes.length;
  }

  stats() {
    return this.cache.stats();
  }

  async flush() {
    await this.cache.flush();
  }

  async detach(fid) {
    const f = this.files.get(fid);
    if (!f) return;
    await this.cache.flush(); // flush all dirty pages before closing
    await f.fd.close();
    this.files.delete(fid);
  }

  async close() {
    await this.cache.flush();
    for (const [, f] of this.files) await f.fd.close();
    this.files.clear();
  }
}

// --- fixed-stride record file over the pool --------------------------------------

export class PagedFile {
  constructor({ pool, fid, stride, name = '' }) {
    this.pool = pool;
    this.fid = fid;
    this.stride = stride;
    this.name = name;
    if (stride <= 0) throw new StoreError(`bad stride ${stride}`);
    const size = pool.fileSize(fid);
    if (size % stride !== 0) {
      throw new StoreError(
        `leanann: ${name} is ${size} B which is not a multiple of stride ${stride} — corrupt (rebuild)`,
      );
    }
    this.size = size;
    this.count = Math.floor(size / stride);
  }

  refresh() {
    this.size = this.pool.fileSize(this.fid);
    this.count = Math.floor(this.size / this.stride);
  }

  async read(i) {
    if (i < 0 || i >= this.count) {
      throw new StoreError(`${this.name}: record ${i} out of range 0..${this.count - 1}`);
    }
    return this.pool.readRange(this.fid, i * this.stride, this.stride);
  }

  async write(i, bytes) {
    if (bytes.length !== this.stride) throw new StoreError('record length != stride');
    await this.pool.writeRange(this.fid, i * this.stride, bytes);
    this.refresh();
  }
}

// --- sequential append (build-time postings/meta writes) ---------------------------

export class AppendFile {
  constructor(fd) {
    this.fd = fd;
    this.offset = 0;
    this.chunks = [];
    this.buffered = 0;
  }

  write(bytes) {
    this.chunks.push(Buffer.from(bytes));
    this.buffered += bytes.length;
    this.offset += bytes.length;
  }

  async flush() {
    if (this.chunks.length === 0) return;
    const buf = Buffer.concat(this.chunks, this.buffered);
    await this.fd.write(buf, 0, buf.length, this.offset - buf.length);
    this.chunks = [];
    this.buffered = 0;
  }
}

// --- manifest ------------------------------------------------------------------

export async function readManifest(dir) {
  let text;
  try {
    text = await readFile(join(dir, MANIFEST_NAME), 'utf8');
  } catch {
    throw new StoreError(`leanann: no index at ${dir} — run \`leanann index\` first`);
  }
  let manifest;
  try {
    manifest = JSON.parse(text);
  } catch {
    throw new StoreError(`leanann: ${MANIFEST_NAME} in ${dir} is not valid JSON`);
  }
  if (manifest.version !== FORMAT_VERSION) {
    throw new StoreError(
      `leanann: index format v${manifest.version} is not this build's v${FORMAT_VERSION} — rebuild with \`leanann index\``,
    );
  }
  return manifest;
}

export async function writeManifest(dir, manifest) {
  await writeFile(join(dir, MANIFEST_NAME), JSON.stringify(manifest, null, 2) + '\n', 'utf8');
}

// Byte-size verification against the manifest (fast, no hashing).
export async function verifySizes(dir, manifest) {
  for (const [name, meta] of Object.entries(manifest.files ?? {})) {
    const p = join(dir, name);
    let st;
    try {
      st = await stat(p);
    } catch {
      throw new StoreError(`leanann: missing ${name} in ${dir} — index is incomplete`);
    }
    if (st.size !== meta.bytes) {
      throw new StoreError(
        `leanann: ${name} is ${st.size} B, manifest says ${meta.bytes} B — index is corrupt (rebuild)`,
      );
    }
  }
}

// Deep content-hash verification (--check staleness path).
export async function verifyHashes(dir, manifest) {
  const problems = [];
  for (const [name, meta] of Object.entries(manifest.files ?? {})) {
    if (!meta.sha256) continue;
    const buf = await readFile(join(dir, name));
    const got = sha256(buf);
    if (got !== meta.sha256) {
      problems.push(`${name}: content hash ${got.slice(0, 12)} != ${meta.sha256.slice(0, 12)}`);
    }
  }
  return problems;
}

// `ignore(relPath)` may veto entries (used to skip the index's own output when
// --out lives inside a directory input, which would otherwise make --check
// permanently stale).
export async function hashTree(path, ignore = null) {
  const st = await stat(path);
  if (st.isFile()) return sha256(await readFile(path));
  const entries = [];
  await walk(path, path, entries, ignore);
  entries.sort((a, b) => (a.rel < b.rel ? -1 : a.rel > b.rel ? 1 : 0));
  const h = [];
  for (const e of entries) {
    h.push(e.rel);
    h.push(sha256(await readFile(e.abs)));
  }
  return sha256(h.join('\n'));
}

async function walk(root, dir, out, ignore) {
  for (const name of (await readdir(dir)).sort()) {
    const abs = join(dir, name);
    const rel = abs.slice(root.length + 1).replaceAll('\\', '/');
    if (ignore && ignore(rel)) continue;
    const st = await stat(abs);
    if (st.isDirectory()) await walk(root, abs, out, ignore);
    else out.push({ rel, abs });
  }
}

// --- safety ----------------------------------------------------------------------

// Confined writer: everything under `out`, path-traversal-proof.
export async function ensureOutDir(out) {
  const abs = resolve(out);
  await mkdir(abs, { recursive: true });
  return abs;
}

export function confinedPath(outDir, name) {
  const root = resolve(outDir);
  const target = resolve(outDir, name);
  // compare with normalized separators (Windows resolve() returns backslashes)
  const norm = (p) => p.replaceAll('\\', '/');
  const r = norm(root);
  const t = norm(target);
  const prefix = r.endsWith('/') ? r : `${r}/`;
  if (t !== r && !t.startsWith(prefix)) {
    throw new StoreError(`refusing to write outside the index dir: ${name}`);
  }
  return target;
}

export async function removeDir(dir) {
  await rm(dir, { recursive: true, force: true });
}

export function fileEntry(buf, bytes) {
  return { bytes, sha256: sha256(buf) };
}
