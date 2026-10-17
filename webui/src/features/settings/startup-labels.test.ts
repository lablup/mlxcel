// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Every read-only value the server sends on Settings > Server needs a catalog label in both
// locales; a raw field name is not user copy (#2334). The fixture is the server's own field list:
// src/server/runtime_settings_values_tests.rs asserts it equals the /settings schema, so a new
// server field fails there until it is listed, and fails here until it has a label. The same holds
// for each enum spelling a read-only value can take (#2419): the fixture's `enum_values` lists them
// by path, and each needs a `settings.startup.<path>.values.<spelling>` label.
import { describe, expect, it } from 'vitest';
import readOnlyFields from '../../../../tests/fixtures/settings/read-only-fields.json';
import { startupLabel } from './setting-control';
import { startupSettingStrings, startupValueStrings } from './startup-strings';
import { hasUnknownSpelling, startupStringText, valueKeySegment } from './value-text';

const expectedKeys = readOnlyFields.fields.flatMap(({ name, sub_fields }) => [`settings.startup.${name}`, ...sub_fields.map((sub) => `settings.startup.${name}.${sub}`)]);
const enumValues = Object.entries(readOnlyFields.enum_values);
const expectedValueKeys = enumValues.flatMap(([path, values]) => values.map((value) => `settings.startup.${path}.values.${valueKeySegment(value)}`));
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

  it('keeps no label key for a field the server does not send', () => {
    const expected = new Set(expectedKeys);
    expect(startupSettingStrings.map((entry) => entry.key).filter((key) => !expected.has(key))).toEqual([]);
  });

  it('has an English and Korean label for every enum spelling the server sends, and none for a spelling it does not', () => {
    expect(enumValues.length).toBeGreaterThan(10);
    const byKey = new Map<string, { en: string; ko: string }>(startupValueStrings.map((entry) => [entry.key, entry]));
    const missing = expectedValueKeys.flatMap((key) => (['en', 'ko'] as const).filter((locale) => (byKey.get(key)?.[locale] ?? '').trim() === '').map((locale) => `${key} (${locale})`));
    expect(missing).toEqual([]);
    const expected = new Set(expectedValueKeys);
    expect(startupValueStrings.map((entry) => entry.key).filter((key) => !expected.has(key))).toEqual([]);
    expect(startupValueStrings.flatMap((entry) => [entry.en, entry.ko]).filter((text) => RAW_NAME.test(text))).toEqual([]);
  });

  it('gives each spelling of a path its own key, and no field a sub-field named like the spelling table', () => {
    // The key segment folds characters outside the key alphabet into `_`, so two spellings of one path must not fold together.
    expect(enumValues.filter(([, values]) => new Set(values.map(valueKeySegment)).size !== values.length).map(([path]) => path)).toEqual([]);
    expect(readOnlyFields.fields.filter(({ sub_fields }) => sub_fields.some((sub) => sub.split('.').includes('values'))).map(({ name }) => name)).toEqual([]);
    // Every enum path is a field or sub-field the server sends.
    const paths = new Set(readOnlyFields.fields.flatMap(({ name, sub_fields }) => [name, ...sub_fields.map((sub) => `${name}.${sub}`)]));
    expect(enumValues.map(([path]) => path).filter((path) => !paths.has(path))).toEqual([]);
  });

  it('resolves a value by path: a label for a spelling, data as sent, null for an unknown spelling', () => {
    expect(startupStringText('en', ['tensor_parallel', 'moe_mode'], 'expert_parallel')).toBe('Expert parallel (whole experts per rank)');
    expect(startupStringText('ko', ['cors_policy', 'origin_policy'], 'allow_list')).toBe('허용 목록');
    expect(startupStringText('en', ['kv_cache_mode'], 'fp16+turbo4')).toBe('FP16 keys, Turbo4 values');
    expect(startupStringText('en', ['tensor_parallel', 'moe_mode'], 'future_mode')).toBeNull();
    expect(hasUnknownSpelling('future_mode', ['tensor_parallel', 'moe_mode'])).toBe(true);
    expect(startupStringText('en', ['kv_cache_budget'], '4294967296')).toBe('4294967296');
    expect(startupStringText('en', ['model_alias'], 'expert_parallel')).toBe('expert_parallel');
    expect(startupStringText('en', null, 'expert_parallel')).toBe('expert_parallel');
    expect(startupStringText('en', ['tensor_parallel.moe_mode'], 'expert_parallel')).toBe('expert_parallel');
    expect(hasUnknownSpelling('future_mode', null)).toBe(false);
  });

  it('resolves labels by path and returns null for an unknown name', () => {
    expect(startupLabel('en', ['tensor_parallel'])).toBe('Tensor parallelism');
    expect(startupLabel('ko', ['tensor_parallel', 'tp_size'])).toBe('텐서 병렬 크기');
    expect(startupLabel('en', ['some_future_field'])).toBeNull();
    expect(startupLabel('en', ['tensor_parallel', 'some_future_sub_field'])).toBeNull();
    // A name holding the separator must not borrow another field's label.
    expect(startupLabel('en', ['tensor_parallel.tp_size'])).toBeNull();
  });
});
