// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Display text for setting values on Settings (#2419). Read-only startup rows and the "Server
// value" hint show a value as user copy: a localized yes or no, the catalog label of an enum
// spelling, a plain list as one comma-separated line. Editable inputs and their edited check never
// use this: they keep the wire spelling from formatSettingValue, which parseSettingInput reads back.
import { formatSettingValue, type SettingSpec } from '../../api/settings';
import { t, type Locale, type StringKey } from '../../i18n/catalog';
import { STARTUP_VALUE_KEYS } from './strings';

const PREFIX = 'settings.startup.';
const TABLE = '.values.';

/** Where a value sits, for its catalog text; `null` inside the raw-name disclosure, where names and spellings stay raw. */
export type ValuePath = readonly string[] | null;

/** The key segment of an enum spelling: the catalog key alphabet is `[a-z0-9_.-]`, so `.` and any character outside it read as `_`. */
export const valueKeySegment = (spelling: string): string => spelling.replace(/[^a-z0-9_-]/g, '_');

/** Dotted paths (`tensor_parallel.moe_mode`) whose strings are enum spellings with catalog labels. */
const ENUM_PATHS: ReadonlySet<string> = new Set([...STARTUP_VALUE_KEYS].map((key) => key.slice(PREFIX.length, key.indexOf(TABLE))));

const isRecord = (value: unknown): boolean => typeof value === 'object' && value !== null && !Array.isArray(value);
/** A record, or a list holding one at any depth: a value shown as sub-rows rather than one line. */
export const containsRecord = (value: unknown): boolean => isRecord(value) || Array.isArray(value) && value.some(containsRecord);

/** What a string at `path` is: an enum spelling with a label, data shown as sent, or a spelling this UI has no label for. */
function spelling(path: ValuePath, text: string): StringKey | 'data' | 'unknown' {
  // A name holding the separator would address another path's spellings.
  if (path === null || path.some((segment) => segment.includes('.'))) return 'data';
  const table = path.join('.');
  if (!ENUM_PATHS.has(table)) return 'data';
  const key = `${PREFIX}${table}${TABLE}${valueKeySegment(text)}`;
  if (STARTUP_VALUE_KEYS.has(key)) return key as StringKey;
  // A decimal count (the KV cache budget in bytes) is data, not a spelling.
  return /^\d+$/.test(text) ? 'data' : 'unknown';
}

/** The display text of a string at `path`: its catalog label for an enum spelling, otherwise the string itself; `null` for an unlabelled spelling. */
export function startupStringText(locale: Locale, path: ValuePath, text: string): string | null {
  const kind = spelling(path, text);
  return kind === 'unknown' ? null : kind === 'data' ? text : t(locale, kind);
}

/** True when `value` holds an enum spelling this UI has no label for; its row then moves into the raw-name disclosure. */
export function hasUnknownSpelling(value: unknown, path: ValuePath): boolean {
  if (typeof value === 'string') return spelling(path, value) === 'unknown';
  return Array.isArray(value) && value.some((item) => hasUnknownSpelling(item, path));
}

const yesNo = (locale: Locale, value: boolean): string => t(locale, value ? 'settings.value.yes' : 'settings.value.no');

/** One list item. Data a comma-separated line would hide (empty, a comma, a control character, edge whitespace) stays quoted. */
function itemText(locale: Locale, item: unknown, path: ValuePath): string {
  if (typeof item === 'string' && spelling(path, item) === 'data') return item === '' || /[,\p{Cc}]/u.test(item) || item.trim() !== item ? JSON.stringify(item) : item;
  if (Array.isArray(item)) return `(${listText(locale, item, path)})`;
  return plainValueText(locale, item, path);
}

/** A list of plain values as one comma-separated line; an empty list is unset. */
export function listText(locale: Locale, items: readonly unknown[], path: ValuePath): string {
  return items.length === 0 ? t(locale, 'settings.value.unset') : items.map((item) => itemText(locale, item, path)).join(', ');
}

/** Display text for a value that is not shown as sub-rows. An unlabelled spelling shows as sent; callers move its row into the disclosure first. */
export function plainValueText(locale: Locale, value: unknown, path: ValuePath): string {
  if (value === null || value === undefined) return t(locale, 'settings.value.unset');
  if (typeof value === 'boolean') return yesNo(locale, value);
  if (typeof value === 'string') return startupStringText(locale, path, value) ?? value;
  if (Array.isArray(value)) return listText(locale, value, path);
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}

/** The "Server value" hint of an editable setting. A structured value stays JSON, as its JSON field shows it. */
export function settingValueText(locale: Locale, spec: SettingSpec, value: unknown): string {
  if (typeof value === 'number' || containsRecord(value)) return formatSettingValue(spec, value);
  return plainValueText(locale, value, null);
}
