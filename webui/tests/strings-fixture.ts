import { readFileSync } from 'node:fs';

export type FixtureLocale = 'en' | 'ko';
type StringEntry = { key: string; en: string; ko: string; test_id: string };

const catalog = JSON.parse(readFileSync(new URL('../../tests/fixtures/webui/strings.json', import.meta.url), 'utf8')) as { strings: StringEntry[] };
const table = new Map(catalog.strings.map((entry) => [entry.key, entry]));

/** Resolve a key from the checked string fixture the way t(locale, key, values) does in the app. */
export function fixtureString(locale: FixtureLocale, key: string, values: Record<string, string> = {}): string {
  const entry = table.get(key);
  if (!entry) throw new Error(`Unknown string fixture key: ${key}`);
  return entry[locale].replace(/\{(\w+)\}/g, (_, name: string) => values[name] ?? `{${name}}`);
}

/** Assert-friendly pair: the copy for the active locale and the copy that must be absent. */
export function localizedPair(locale: FixtureLocale, key: string): { shown: string; hidden: string } {
  return { shown: fixtureString(locale, key), hidden: fixtureString(locale === 'ko' ? 'en' : 'ko', key) };
}
