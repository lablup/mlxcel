// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CACHE_LIMIT, cached, formatBytes } from './format';

afterEach(() => { vi.restoreAllMocks(); });

describe('shared number formatting', () => {
  it('reuses one formatter per locale and precision', () => {
    const constructed = vi.spyOn(Intl, 'NumberFormat');
    expect(formatBytes(1, 'en')).toBe('1 B');
    expect(formatBytes(1, 'en')).toBe('1 B');
    expect(formatBytes(2048, 'en')).toBe('2 KiB');
    expect(formatBytes(2048, 'en')).toBe('2 KiB');
    // Bytes use 0 fraction digits, KiB and up 1: two formatters, never one per call.
    expect(constructed).toHaveBeenCalledTimes(2);
  });
});

describe('cached', () => {
  it('holds at most CACHE_LIMIT keys, evicting the oldest first', () => {
    const cache = new Map<string, number>();
    for (let index = 0; index <= CACHE_LIMIT; index += 1) cached(cache, `key-${index}`, () => index);
    expect(cache.size).toBe(CACHE_LIMIT);
    expect(cache.has('key-0')).toBe(false);
    expect(cache.has('key-1')).toBe(true);
    const create = vi.fn(() => -1);
    expect(cached(cache, `key-${CACHE_LIMIT}`, create)).toBe(CACHE_LIMIT);
    expect(create).not.toHaveBeenCalled();
    expect(cache.size).toBe(CACHE_LIMIT);
  });
});
