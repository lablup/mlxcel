import type { Locale } from '../i18n/catalog';
import {
  DEFAULT_COLOR_SCHEME,
  DEFAULT_THEME_FAMILY,
  getSystemColorScheme,
  isColorSchemePreference,
  isThemeFamily,
  resolveThemeSelection,
  type ColorScheme,
  type ColorSchemePreference,
  type ThemeFamily,
} from './theme';

export type { ColorSchemePreference, ThemeFamily } from './theme';
export type MaterialPreference = 'glass' | 'tinted' | 'opaque';
export type ContrastPreference = 'system' | 'on' | 'off';

export type AppearancePreferences = {
  themeFamily: ThemeFamily;
  colorScheme: ColorSchemePreference;
  material: MaterialPreference;
  glassIntensity: number;
  reduceMotion: boolean;
  reduceTransparency: boolean;
  highContrast: ContrastPreference;
  locale: Locale;
};

export const DEFAULT_APPEARANCE: AppearancePreferences = {
  themeFamily: DEFAULT_THEME_FAMILY,
  colorScheme: DEFAULT_COLOR_SCHEME,
  material: 'glass',
  glassIntensity: 35,
  reduceMotion: false,
  reduceTransparency: false,
  highContrast: 'system',
  locale: 'en',
};

// Also read by public/theme-bootstrap.js before the app mounts; keep the key and
// the family/scheme fields in step with it (theme-bootstrap.test.ts checks).
export const APPEARANCE_STORAGE_KEY = 'mlxcel.webui.appearance';

function clampIntensity(value: unknown): number {
  const numeric = typeof value === 'number' && Number.isFinite(value) ? value : DEFAULT_APPEARANCE.glassIntensity;
  return Math.max(0, Math.min(100, Math.round(numeric)));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isMaterial(value: unknown): value is MaterialPreference {
  return value === 'glass' || value === 'tinted' || value === 'opaque';
}

function isContrast(value: unknown): value is ContrastPreference {
  return value === 'system' || value === 'on' || value === 'off';
}

function normalizeContrast(value: unknown): ContrastPreference {
  if (isContrast(value)) return value;
  if (value === true) return 'on';
  if (value === false) return 'off';
  return DEFAULT_APPEARANCE.highContrast;
}

// Stored appearance written before #1903 carried the color scheme in a flat
// `theme` field ('system' | 'light' | 'dark') and had no family. It is still
// honored when `colorScheme` is absent, so an upgrade keeps the user's choice.
function normalizeColorScheme(parsed: Record<string, unknown>): ColorSchemePreference {
  if (isColorSchemePreference(parsed.colorScheme)) return parsed.colorScheme;
  if (parsed.colorScheme === undefined && isColorSchemePreference(parsed.theme)) return parsed.theme;
  return DEFAULT_APPEARANCE.colorScheme;
}

function isLocale(value: unknown): value is Locale {
  return value === 'en' || value === 'ko';
}

export function loadAppearance(): AppearancePreferences {
  if (typeof window === 'undefined') return DEFAULT_APPEARANCE;
  let raw: string | null;
  try {
    raw = window.localStorage.getItem(APPEARANCE_STORAGE_KEY);
  } catch {
    return DEFAULT_APPEARANCE;
  }
  if (!raw) return DEFAULT_APPEARANCE;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) return DEFAULT_APPEARANCE;
    return {
      themeFamily: isThemeFamily(parsed.themeFamily) ? parsed.themeFamily : DEFAULT_APPEARANCE.themeFamily,
      colorScheme: normalizeColorScheme(parsed),
      material: isMaterial(parsed.material) ? parsed.material : DEFAULT_APPEARANCE.material,
      glassIntensity: clampIntensity(parsed.glassIntensity),
      reduceMotion: parsed.reduceMotion === true,
      reduceTransparency: parsed.reduceTransparency === true,
      highContrast: normalizeContrast(parsed.highContrast),
      locale: isLocale(parsed.locale) ? parsed.locale : DEFAULT_APPEARANCE.locale,
    };
  } catch {
    return DEFAULT_APPEARANCE;
  }
}

export function saveAppearance(preferences: AppearancePreferences): void {
  try {
    window.localStorage.setItem(APPEARANCE_STORAGE_KEY, JSON.stringify(preferences));
  } catch {
    // Browser appearance preferences remain functional in memory when storage is blocked or full.
  }
}

function supportsBackdropFilter(): boolean {
  if (typeof CSS === 'undefined' || typeof CSS.supports !== 'function') return false;
  return CSS.supports('backdrop-filter: blur(1px)') || CSS.supports('-webkit-backdrop-filter: blur(1px)');
}

/**
 * Writes the appearance switches onto the document root. `data-theme` receives
 * the applied `<family>-<scheme>` id, resolved against the host scheme when the
 * preference is `system`; `data-color-scheme` keeps the raw preference, which is
 * why no stylesheet may gate on it (scripts/check-theme-selectors.mjs).
 */
export function applyAppearance(root: HTMLElement, preferences: AppearancePreferences, systemScheme: ColorScheme = getSystemColorScheme()): void {
  const theme = resolveThemeSelection(preferences.themeFamily, preferences.colorScheme, systemScheme);
  root.dataset.theme = theme.id;
  root.dataset.themeFamily = theme.family;
  root.dataset.colorScheme = theme.preference;
  root.dataset.material = preferences.reduceTransparency ? 'opaque' : preferences.material;
  root.dataset.glassIntensity = String(clampIntensity(preferences.glassIntensity));
  root.dataset.reduceMotion = String(preferences.reduceMotion);
  root.dataset.reduceTransparency = String(preferences.reduceTransparency);
  root.dataset.highContrast = preferences.highContrast;
  root.dataset.backdropFilter = supportsBackdropFilter() ? 'supported' : 'unsupported';
  root.lang = preferences.locale;
}
