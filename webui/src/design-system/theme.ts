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

// Theme identity for the WebUI. Each product family is a pre-built Astryx theme
// (themes/*.theme.ts, compiled by `astryx theme build` into themes/built/), and
// the document root carries three attributes that all agree with what the root
// Astryx <Theme> writes, so nothing rewrites another's value:
//
//   data-theme          the resolved color scheme, `light` or `dark` (Astryx's
//                       own attribute: its theme CSS sets color-scheme from it)
//   data-astryx-theme   the Astryx theme name of the family (`mlxcel`,
//                       `mlxcel-glass`), which scopes the built theme CSS
//   data-theme-family   the product family (`mlxcel`, `glass`), for the
//                       product-only rules in themes/*.css
//
// The stored preference (a family plus a color scheme that may be `system`)
// stays apart from the applied values, so `data-theme` never holds `system`.
//
// This module is dependency-free on purpose: `scripts/check-theme-selectors.mjs`
// imports it directly under Node to learn which values exist, and
// `public/theme-bootstrap.js` mirrors `resolveThemeSelection` before the app
// mounts (a parity test keeps the two in step).

export const THEME_FAMILIES = ['mlxcel', 'glass'] as const;
export type ThemeFamily = (typeof THEME_FAMILIES)[number];

export const COLOR_SCHEMES = ['light', 'dark'] as const;
export type ColorScheme = (typeof COLOR_SCHEMES)[number];

export const COLOR_SCHEME_PREFERENCES = ['system', 'light', 'dark'] as const;
export type ColorSchemePreference = (typeof COLOR_SCHEME_PREFERENCES)[number];

/**
 * The Astryx theme name of each family: the `name` its themes/*.theme.ts source
 * passes to `astryx theme build`, and so the `data-astryx-theme` value its built
 * stylesheet is scoped to. Names are unique across every theme the app registers.
 */
export const ASTRYX_THEME_NAMES = { mlxcel: 'mlxcel', glass: 'mlxcel-glass' } as const satisfies Record<ThemeFamily, string>;
export type AstryxThemeName = (typeof ASTRYX_THEME_NAMES)[ThemeFamily];

export const DEFAULT_THEME_FAMILY: ThemeFamily = 'mlxcel';
export const DEFAULT_COLOR_SCHEME: ColorSchemePreference = 'system';

/** The one media query the app consults for the host scheme; nothing else may gate on it. */
export const SYSTEM_DARK_QUERY = '(prefers-color-scheme: dark)';

export function isThemeFamily(value: unknown): value is ThemeFamily {
  return typeof value === 'string' && (THEME_FAMILIES as readonly string[]).includes(value);
}

export function isColorSchemePreference(value: unknown): value is ColorSchemePreference {
  return typeof value === 'string' && (COLOR_SCHEME_PREFERENCES as readonly string[]).includes(value);
}

export function isColorScheme(value: unknown): value is ColorScheme {
  return typeof value === 'string' && (COLOR_SCHEMES as readonly string[]).includes(value);
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
  family: ThemeFamily;
  astryxTheme: AstryxThemeName;
  scheme: ColorScheme;
  preference: ColorSchemePreference;
};

export function resolveThemeSelection(family: ThemeFamily, preference: ColorSchemePreference, system: ColorScheme = getSystemColorScheme()): ThemeSelection {
  return { family, astryxTheme: ASTRYX_THEME_NAMES[family], scheme: resolveColorScheme(preference, system), preference };
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
