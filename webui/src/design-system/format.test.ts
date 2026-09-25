// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { formatBytes } from './format';

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
