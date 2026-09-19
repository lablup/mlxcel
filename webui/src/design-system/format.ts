import { t, type Locale } from '../i18n/catalog';

// BCP-47 tags for Intl formatting; these are identifiers, not user-facing copy.
const LOCALE_TAGS: Record<Locale, string> = { en: 'en-US', ko: 'ko-KR' };
const localeTag = (locale: Locale): string => LOCALE_TAGS[locale];

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
  return `${new Intl.NumberFormat(localeTag(locale), { maximumFractionDigits: digits }).format(value)} ${units[unitIndex]}`;
}

export function formatTokensPerSecond(value: number | null, locale: Locale): string {
  if (value === null) return t(locale, 'format.not_measured');
  return `${new Intl.NumberFormat(localeTag(locale), { maximumFractionDigits: 1 }).format(value)} tokens/s`;
}
