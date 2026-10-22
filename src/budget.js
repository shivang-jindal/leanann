// budget.js — the AirLLM layer: a hard memory budget enforced by accounting,
// LRU page caching, and honest failure when the budget is impossible.
//
// Every byte of index data that enters RAM during build or query goes through
// Budget/PageCache. This is a contract, not a target.

export class BudgetExceeded extends Error {
  constructor(message) {
    super(message);
    this.name = 'BudgetExceeded';
    this.code = 'BUDGET_EXCEEDED';
  }
}

export class Budget {
  constructor(limitBytes) {
    if (!Number.isFinite(limitBytes) || limitBytes <= 0) {
      throw new RangeError(`budget: limit must be > 0, got ${limitBytes}`);
    }
    this.limit = Math.floor(limitBytes);
    this.used = 0;
    this.peak = 0;
  }

  charge(bytes) {
    if (bytes > this.limit) {
      throw new BudgetExceeded(
        `budget: ${bytes} bytes exceeds the whole ${this.limit}-byte budget`,
      );
    }
    this.used += bytes;
    if (this.used > this.peak) this.peak = this.used;
  }

  refund(bytes) {
    this.used = Math.max(0, this.used - bytes);
  }
}

// Hard-capped LRU page cache.
//   key: opaque composite (BufferPool uses `${fid}:${pageNo}`) — one shared
//        LRU + one shared limit across every file of the index.
//   readPage(key) -> Promise<Uint8Array>       (short final page allowed)
//   writePage(key, bytes)?  (optional: makes the cache read-write; dirty pages
//                            are flushed on eviction and on flush())
// LRU order rides on Map insertion order: a hit re-inserts the key.
export class PageCache {
  constructor({ pageSize, limitBytes, readPage, writePage = null, onEvict = null }) {
    if (!Number.isInteger(pageSize) || pageSize <= 0) {
      throw new RangeError(`page-cache: bad pageSize ${pageSize}`);
    }
    if (pageSize > limitBytes) {
      throw new BudgetExceeded(
        `page-cache: page size ${pageSize} B does not fit in budget ${limitBytes} B — ` +
        `raise --budget-mb to at least ${Math.ceil(pageSize / (1024 * 1024))} MB`,
      );
    }
    this.pageSize = pageSize;
    this.limit = limitBytes;
    this.readPage = readPage;
    this.writePage = writePage;
    this.onEvict = onEvict;
    this.map = new Map(); // key -> { data: Uint8Array, dirty: boolean }
    this.used = 0;
    this.hits = 0;
    this.misses = 0;
    this.evictions = 0;
    this.bytesRead = 0;
    this.bytesWritten = 0;
    this.dirtyFlushes = 0;
  }

  get pageCount() {
    return this.map.size;
  }

  stats() {
    return {
      pageSize: this.pageSize,
      limit: this.limit,
      used: this.used,
      pages: this.map.size,
      hits: this.hits,
      misses: this.misses,
      evictions: this.evictions,
      bytesRead: this.bytesRead,
      bytesWritten: this.bytesWritten,
      dirtyFlushes: this.dirtyFlushes,
      hitRate: this.hits + this.misses === 0
        ? 0
        : this.hits / (this.hits + this.misses),
    };
  }

  async get(key) {
    const entry = this.map.get(key);
    if (entry) {
      this.hits += 1;
      // re-insert = most-recently used
      this.map.delete(key);
      this.map.set(key, entry);
      return entry.data;
    }
    this.misses += 1;
    const data = await this.readPage(key);
    this.bytesRead += data.length;
    this.map.set(key, { data, dirty: false });
    this.used += data.length;
    await this.#evictIfNeeded(data.length, key);
    return data;
  }

  // Update a page in place (marks dirty). Page may CHANGE length — tail pages
  // start short and grow as records are appended past EOF.
  async put(key, data) {
    const existing = this.map.get(key);
    if (existing) {
      this.used -= existing.data.length;
      this.map.delete(key);
    }
    this.map.set(key, { data, dirty: true });
    this.used += data.length;
    await this.#evictIfNeeded(data.length, key);
  }

  // Evict least-recently-used pages until under the limit. The page being
  // inserted right now (`protectKey`) is NEVER evicted — silently dropping it
  // would lose data; if it alone cannot fit, fail loudly instead.
  async #evictIfNeeded(incoming, protectKey) {
    if (this.used <= this.limit) return;
    for (const [key, entry] of this.map) {
      if (this.used <= this.limit) break;
      if (key === protectKey) continue;
      if (entry.dirty) await this.#flush(key, entry);
      this.map.delete(key);
      this.used -= entry.data.length;
      this.evictions += 1;
      if (this.onEvict) this.onEvict(key, entry);
    }
    if (this.used > this.limit) {
      throw new BudgetExceeded(
        `page-cache: ${incoming}-byte page cannot fit the ${this.limit}-byte budget even after evicting ` +
        'everything else — raise --budget-mb',
      );
    }
  }

  async #flush(key, entry) {
    if (!this.writePage) {
      throw new BudgetExceeded('page-cache: dirty page in a read-only cache');
    }
    await this.writePage(key, entry.data);
    entry.dirty = false;
    this.bytesWritten += entry.data.length;
    this.dirtyFlushes += 1;
  }

  async flush() {
    for (const [key, entry] of this.map) {
      if (entry.dirty) await this.#flush(key, entry);
    }
  }

  clear() {
    this.map.clear();
    this.used = 0;
  }
}

// Config-time validation: can this budget serve queries at all?
// Returns null when valid, or a human message with the computed minimum.
//   codebooks = M blocks × k centroids × sub × 4 B == dims × k × 4 (resident)
//   beamWorking ≈ ef live candidates × record slack
export function validateQueryBudget({ budgetBytes, pageBytes, ef, dims, k }) {
  const codebooks = dims * k * 4;
  const beamWorking = ef * 64;
  if (budgetBytes < pageBytes) {
    return `budget ${budgetBytes} B < page size ${pageBytes} B — raise --budget-mb to at least ${formatMin(pageBytes)}`;
  }
  const needed = Math.ceil(codebooks / 2) + beamWorking;
  if (budgetBytes < needed) {
    return `budget ${budgetBytes} B is too small for codebooks (${codebooks} B) + beam (ef=${ef}) — ` +
      `raise --budget-mb to at least ${formatMin(needed)}`;
  }
  return null;
}

function formatMin(bytes) {
  const mb = bytes / (1024 * 1024);
  return `${Math.max(1, Math.ceil(mb))} MB`;
}
