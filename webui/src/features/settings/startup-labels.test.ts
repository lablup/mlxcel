// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Every read-only value the server sends on Settings > Server needs a catalog label in both
// locales; a raw field name is not user copy (#2334). The fixture is the server's own field list:
// src/server/runtime_settings_values_tests.rs asserts it equals the /settings schema, so a new
// server field fails there until it is listed, and fails here until it has a label.
import { describe, expect, it } from 'vitest';
import readOnlyFields from '../../../../tests/fixtures/settings/read-only-fields.json';
import { startupLabel } from './setting-control';
import { startupSettingStrings } from './startup-strings';

const expectedKeys = readOnlyFields.fields.flatMap(({ name, sub_fields }) => [`settings.startup.${name}`, ...sub_fields.map((sub) => `settings.startup.${name}.${sub}`)]);
/** A snake_case identifier such as `tp_size`: what a raw field name looks like. */
const RAW_NAME = /[a-z0-9]_[a-z0-9]/i;

describe('Settings > Server startup labels', () => {
  it('has a catalog key for every read-only field and sub-field the server sends', () => {
    const keys = new Set<string>(startupSettingStrings.map((entry) => entry.key));
    expect(readOnlyFields.fields.length).toBeGreaterThan(50);
    expect(expectedKeys.filter((key) => !keys.has(key))).toEqual([]);
  });

  it('labels each of them in English and Korean, never with the raw name', () => {
    const byKey = new Map<string, { en: string; ko: string }>(startupSettingStrings.map((entry) => [entry.key, entry]));
    const missing = expectedKeys.flatMap((key) => (['en', 'ko'] as const).filter((locale) => (byKey.get(key)?.[locale] ?? '').trim() === '').map((locale) => `${key} (${locale})`));
    expect(missing).toEqual([]);
    const raw = startupSettingStrings.flatMap((entry) => [entry.en, entry.ko]).filter((text) => RAW_NAME.test(text));
    expect(raw).toEqual([]);
  });

  it('keeps no key for a field the server does not send', () => {
    const expected = new Set(expectedKeys);
    expect(startupSettingStrings.map((entry) => entry.key).filter((key) => !expected.has(key))).toEqual([]);
  });

  it('resolves labels by path and returns null for an unknown name', () => {
    expect(startupLabel('en', ['tensor_parallel'])).toBe('Tensor parallelism');
    expect(startupLabel('ko', ['tensor_parallel', 'tp_size'])).toBe('텐서 병렬 크기');
    expect(startupLabel('en', ['some_future_field'])).toBeNull();
    expect(startupLabel('en', ['tensor_parallel', 'some_future_sub_field'])).toBeNull();
  });
});
