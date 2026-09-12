import type { Locale } from '../i18n/catalog';

const localeTag = (locale: Locale): string => (locale === 'ko' ? 'ko-KR' : 'en-US');

export function formatBytes(bytes: number, locale: Locale): string {
  if (!Number.isFinite(bytes) || bytes < 0) return locale === 'ko' ? '알 수 없음' : 'unknown';
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
  if (value === null) return locale === 'ko' ? '아직 측정되지 않음' : 'not yet measured';
  return `${new Intl.NumberFormat(localeTag(locale), { maximumFractionDigits: 1 }).format(value)} tokens/s`;
}
