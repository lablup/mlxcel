// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Drift gate: user-facing copy resolves through t(locale, key), and confirmations
// use the design-system dialogs. Patterns stay RegExp objects (and this file lives
// in the skipped i18n directory) so the gate never matches itself.
import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';
import stringsFixture from '../../../tests/fixtures/webui/strings.json';
import { activityStrings } from '../features/activity/strings';
import { chatStrings } from '../features/chat/strings';
import { settingsStrings } from '../features/settings/strings';
import { entries } from './catalog';

const SRC = join(import.meta.dirname, '..');
const BYPASSES: readonly { pattern: RegExp; fix: string }[] = [
  { pattern: /window\.(confirm|alert|prompt)\b/, fix: 'native dialog; use ConfirmDialog or Dialog from design-system/primitives' },
  { pattern: /locale === 'ko'/, fix: 'inline locale branch; add a catalog key and call t(locale, key)' },
  // \s keeps the literal phrase out of this file, so the issue's plain grep stays empty.
  { pattern: /const\swords = /, fix: 'inline en/ko helper; add catalog keys and call t(locale, key)' },
];
const INLINE_KEYS: readonly string[] = ['models.next_profile.body', 'models.next_profile.edit', 'format.unknown', 'format.not_measured'];

function sourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return relative(SRC, path) === 'i18n' ? [] : sourceFiles(path);
    return /\.m?[jt]sx?$/.test(entry.name) && !/\.test\.[jt]sx?$/.test(entry.name) ? [path] : [];
  });
}

describe('i18n drift gate', () => {
  it('rejects native dialogs and inline locale branches in webui/src', () => {
    const files = sourceFiles(SRC);
    expect(files.length).toBeGreaterThan(50);
    const violations = files.flatMap((file) => readFileSync(file, 'utf8').split('\n').flatMap((line, index) => BYPASSES
      .filter(({ pattern }) => pattern.test(line))
      .map(({ fix }) => `webui/src/${relative(SRC, file)}:${index + 1}: ${fix}`)));
    expect(violations).toEqual([]);
  });

  it('keeps the feature string modules and inline keys identical to the checked fixture', () => {
    const fixture = new Map(stringsFixture.strings.map((entry) => [entry.key, entry]));
    const inline = entries.filter((entry) => INLINE_KEYS.includes(entry.key));
    expect(inline).toHaveLength(INLINE_KEYS.length);
    for (const { key, en, ko, test_id } of [...chatStrings, ...settingsStrings, ...activityStrings, ...inline]) {
      expect(fixture.get(key)).toEqual({ key, en, ko, test_id });
    }
  });
});
