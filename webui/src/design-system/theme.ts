// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Theme identity for the WebUI, in the `@lablup/ui-common` convention: a theme
// is a block of `--token-*` values keyed on one `data-theme` value, and that
// value is always `<family>-<scheme>`. The stored preference (a family plus a
// color scheme that may be `system`) is kept apart from the applied id, so
// `data-theme` never holds `system`, `light` or `dark` on its own.
//
// This module is dependency-free on purpose: `scripts/check-theme-selectors.mjs`
// imports it directly under Node to learn which ids exist, and
// `public/theme-bootstrap.js` mirrors `resolveThemeSelection` before the app
// mounts (a parity test keeps the two in step).

export const THEME_FAMILIES = ['mlxcel', 'glass'] as const;
export type ThemeFamily = (typeof THEME_FAMILIES)[number];

export const COLOR_SCHEMES = ['light', 'dark'] as const;
export type ColorScheme = (typeof COLOR_SCHEMES)[number];

export const COLOR_SCHEME_PREFERENCES = ['system', 'light', 'dark'] as const;
export type ColorSchemePreference = (typeof COLOR_SCHEME_PREFERENCES)[number];

export type ThemeId = `${ThemeFamily}-${ColorScheme}`;

export const DEFAULT_THEME_FAMILY: ThemeFamily = 'mlxcel';
export const DEFAULT_COLOR_SCHEME: ColorSchemePreference = 'system';

/** Every value `data-theme` can hold. */
export const THEME_IDS: readonly ThemeId[] = THEME_FAMILIES.flatMap((family) => COLOR_SCHEMES.map((scheme): ThemeId => `${family}-${scheme}`));

/** The one media query the app consults for the host scheme; nothing else may gate on it. */
export const SYSTEM_DARK_QUERY = '(prefers-color-scheme: dark)';

export function isThemeFamily(value: unknown): value is ThemeFamily {
  return typeof value === 'string' && (THEME_FAMILIES as readonly string[]).includes(value);
}

export function isColorSchemePreference(value: unknown): value is ColorSchemePreference {
  return typeof value === 'string' && (COLOR_SCHEME_PREFERENCES as readonly string[]).includes(value);
}

export function isThemeId(value: unknown): value is ThemeId {
  return typeof value === 'string' && (THEME_IDS as readonly string[]).includes(value);
}

export function themeIdFor(family: ThemeFamily, scheme: ColorScheme): ThemeId {
  return `${family}-${scheme}`;
}

/** Reads the host scheme; a browser without `matchMedia` (or one that throws) counts as light. */
export function getSystemColorScheme(): ColorScheme {
  try {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return 'light';
    return window.matchMedia(SYSTEM_DARK_QUERY).matches ? 'dark' : 'light';
  } catch {
    return 'light';
  }
}

export function resolveColorScheme(preference: ColorSchemePreference, system: ColorScheme = getSystemColorScheme()): ColorScheme {
  return preference === 'system' ? system : preference;
}

export type ThemeSelection = {
  id: ThemeId;
  family: ThemeFamily;
  scheme: ColorScheme;
  preference: ColorSchemePreference;
};

export function resolveThemeSelection(family: ThemeFamily, preference: ColorSchemePreference, system: ColorScheme = getSystemColorScheme()): ThemeSelection {
  const scheme = resolveColorScheme(preference, system);
  return { id: themeIdFor(family, scheme), family, scheme, preference };
}

/** Calls back when the host scheme flips; returns the unsubscribe function. */
export function onSystemColorSchemeChange(callback: (scheme: ColorScheme) => void): () => void {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return () => undefined;
  let query: MediaQueryList;
  try {
    query = window.matchMedia(SYSTEM_DARK_QUERY);
  } catch {
    return () => undefined;
  }
  const handler = (event: MediaQueryListEvent): void => callback(event.matches ? 'dark' : 'light');
  query.addEventListener('change', handler);
  return () => query.removeEventListener('change', handler);
}
