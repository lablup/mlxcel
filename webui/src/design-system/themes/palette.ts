// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// The product palette as Astryx tokens, shared by the two theme sources
// (mlxcel.theme.ts, glass.theme.ts). Read only by `astryx theme build`; the app
// imports the built modules under ./built/, never this file.
import type { DefineThemeInput } from '@lablup/ui-common/theme';

/** A `[light, dark]` pair; `astryx theme build` turns it into `light-dark(light, dark)`. */
type Pair = [light: string, dark: string];

/** The values a product family chooses. Every Astryx role below derives from them. */
export interface ProductPalette {
  /** The page canvas behind the shell. */
  canvas: Pair;
  /** Cards, tables, popovers, dialogs, form fields. */
  surface: Pair;
  /** Recessed fills: tab strips, progress tracks. */
  recessed: Pair;
  text: Pair;
  textSecondary: Pair;
  border: Pair;
  /** Links, focus rings, accent text and icons. */
  accentText: Pair;
  /** The solid primary fill (white text on it in both schemes). */
  accentFill: Pair;
  /** Hover and selected-row tint. */
  selection: Pair;
  warning: Pair;
  warningMuted: Pair;
  error: Pair;
  errorMuted: Pair;
  success: Pair;
  successMuted: Pair;
  /** The modal and drawer scrim. */
  scrim: Pair;
}

/** System fonts only: the WebUI loads nothing from another origin. */
const SYSTEM_FONT = { family: '-apple-system', fallbacks: 'BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif' };

/** The neutral theme's typography with the system font; scale, weights and code font are neutral's. */
export const productTypography: DefineThemeInput['typography'] = {
  scale: { base: 14, ratio: 1.2 },
  body: SYSTEM_FONT,
  heading: { ...SYSTEM_FONT, weights: { 3: 'bold', 4: 'bold' } },
  code: { family: 'ui-monospace', fallbacks: '"SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace' },
};

const ON_DARK_TEXT: Pair = ['#ffffff', '#111111'];

/** Astryx's semantic roles, each set from the product role that means the same thing. */
export function productTokens(palette: ProductPalette): NonNullable<DefineThemeInput['tokens']> {
  return {
    '--color-background-body': palette.canvas,
    '--color-background-surface': palette.surface,
    '--color-background-card': palette.surface,
    '--color-background-popover': palette.surface,
    '--color-background-muted': palette.recessed,
    '--color-text-primary': palette.text,
    '--color-text-secondary': palette.textSecondary,
    '--color-icon-primary': palette.text,
    '--color-icon-secondary': palette.textSecondary,
    '--color-border': palette.border,
    '--color-accent': palette.accentFill,
    '--color-on-accent': ['#ffffff', '#ffffff'],
    '--color-accent-muted': palette.selection,
    '--color-text-accent': palette.accentText,
    '--color-icon-accent': palette.accentText,
    '--focus-outline-color': palette.accentText,
    '--color-overlay-hover': palette.selection,
    '--color-overlay': palette.scrim,
    '--color-warning': palette.warning,
    '--color-warning-muted': palette.warningMuted,
    '--color-error': palette.error,
    '--color-error-muted': palette.errorMuted,
    '--color-on-error': ON_DARK_TEXT,
    '--color-success': palette.success,
    '--color-success-muted': palette.successMuted,
    '--color-on-success': ON_DARK_TEXT,
  };
}

/**
 * The neutral theme's own status fills (theme-local tokens it enrolls), which
 * Astryx components use for solid accent, success and error fills.
 */
export function productLocalTokens(palette: ProductPalette): NonNullable<DefineThemeInput['localTokens']> {
  return {
    '--astryx-theme-neutral-color-status-fill-accent': palette.accentFill,
    '--astryx-theme-neutral-color-status-fill-success': palette.success,
    '--astryx-theme-neutral-color-status-fill-error': palette.error,
    '--astryx-theme-neutral-color-status-muted-accent': palette.selection,
  };
}
