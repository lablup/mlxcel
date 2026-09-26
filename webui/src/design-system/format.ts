import { t, type Locale } from '../i18n/catalog';

// BCP-47 tags for Intl formatting; these are identifiers, not user-facing copy.
const LOCALE_TAGS: Record<Locale, string> = { en: 'en-US', ko: 'ko-KR' };
export const localeTag = (locale: Locale): string => LOCALE_TAGS[locale];

/** The most keys one `cached` map holds; inserting past it evicts the oldest key first. */
export const CACHE_LIMIT = 32;

/**
 * The cached value for `key`, created on first use. Intl formatters are costly to build and pages
 * format on every poll. Each map is bounded by CACHE_LIMIT, so a caller keyed by server data cannot
 * grow it without end.
 */
export function cached<K, T>(cache: Map<K, T>, key: K, create: () => T): T {
  let value = cache.get(key);
  if (value === undefined) {
    value = create();
    if (cache.size >= CACHE_LIMIT) {
      // A Map iterates in insertion order, so the first key is the oldest.
      const oldest = cache.keys().next();
      if (!oldest.done) cache.delete(oldest.value);
    }
    cache.set(key, value);
  }
  return value;
}

// Keyed by locale and precision: formatBytes uses 0 digits for bytes and 1 above.
const numberFormats = new Map<string, Intl.NumberFormat>();
function numberFormat(locale: Locale, maximumFractionDigits: number): Intl.NumberFormat {
  return cached(numberFormats, `${locale}:${maximumFractionDigits}`, () => new Intl.NumberFormat(localeTag(locale), { maximumFractionDigits }));
}

export function formatBytes(bytes: number, locale: Locale): string {
  if (!Number.isFinite(bytes) || bytes < 0) return t(locale, 'format.unknown');
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = unitIndex === 0 ? 0 : 1;
  return `${numberFormat(locale, digits).format(value)} ${units[unitIndex]}`;
}

export function formatTokensPerSecond(value: number | null, locale: Locale): string {
  if (value === null) return t(locale, 'format.not_measured');
  return `${numberFormat(locale, 1).format(value)} tokens/s`;
}
