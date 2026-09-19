// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { isApplePlatform, isPrimaryModifier } from './keyboard';

function stubPlatform(platform: string): void {
  vi.spyOn(window.navigator, 'platform', 'get').mockReturnValue(platform);
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('isApplePlatform', () => {
  it('is false in this jsdom test environment, which reports no Apple platform hint', () => {
    expect(isApplePlatform()).toBe(false);
  });

  it('reads navigator.platform, case-insensitively, for macOS and iOS device strings', () => {
    stubPlatform('MacIntel');
    expect(isApplePlatform()).toBe(true);
    stubPlatform('iPhone');
    expect(isApplePlatform()).toBe(true);
    stubPlatform('iPad');
    expect(isApplePlatform()).toBe(true);
    stubPlatform('iPod touch');
    expect(isApplePlatform()).toBe(true);
    stubPlatform('Win32');
    expect(isApplePlatform()).toBe(false);
    stubPlatform('Linux x86_64');
    expect(isApplePlatform()).toBe(false);
  });

  it('prefers userAgentData.platform over the legacy properties when present', () => {
    const withUaData = window.navigator as Navigator & { userAgentData?: { platform: string } };
    const original = withUaData.userAgentData;
    stubPlatform('Win32');
    Object.defineProperty(withUaData, 'userAgentData', { value: { platform: 'macOS' }, configurable: true });
    expect(isApplePlatform()).toBe(true);
    if (original === undefined) {
      Object.defineProperty(withUaData, 'userAgentData', { value: undefined, configurable: true });
    } else {
      Object.defineProperty(withUaData, 'userAgentData', { value: original, configurable: true });
    }
  });

  it('returns false when navigator is unavailable', () => {
    const original = globalThis.navigator;
    // @ts-expect-error deleting a required global to exercise the SSR / missing-navigator guard
    delete globalThis.navigator;
    try {
      expect(isApplePlatform()).toBe(false);
    } finally {
      globalThis.navigator = original;
    }
  });
});

describe('isPrimaryModifier', () => {
  it('is the meta key on Apple platforms', () => {
    stubPlatform('MacIntel');
    expect(isPrimaryModifier({ metaKey: true, ctrlKey: false })).toBe(true);
    expect(isPrimaryModifier({ metaKey: false, ctrlKey: true })).toBe(false);
  });

  it('is the ctrl key on non-Apple platforms', () => {
    stubPlatform('Win32');
    expect(isPrimaryModifier({ metaKey: true, ctrlKey: false })).toBe(false);
    expect(isPrimaryModifier({ metaKey: false, ctrlKey: true })).toBe(true);
  });
});
