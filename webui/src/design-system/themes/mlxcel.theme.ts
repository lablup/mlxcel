// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// The mlxcel family (the default): the approved 4185 baseline as an Astryx theme
// over the pre-built neutral theme. `astryx theme build` compiles this file into
// ./built/mlxcel.{css,js,d.ts}; `pnpm run check:themes` fails when those are stale.
// The default export is a plain config, not a defineTheme() call: the CLI resolves
// it, and its icon resolver only recognizes Astryx core's own defineTheme import.
import type { DefineThemeInput } from '@lablup/ui-common/theme';
import { neutralTheme } from '@lablup/ui-common/theme/neutral/built';
import { productLocalTokens, productTokens, productTypography, type ProductPalette } from './palette';

export const mlxcelPalette: ProductPalette = {
  canvas: ['#f3f4f6', '#0e1117'],
  surface: ['#ffffff', '#181d27'],
  recessed: ['#f8f8fa', '#111722'],
  text: ['#151a24', '#f5f7fb'],
  textSecondary: ['#59647a', '#aeb8c9'],
  border: ['rgba(33, 42, 61, 0.12)', 'rgba(255, 255, 255, 0.12)'],
  accentText: ['#245bd8', '#9dbdff'],
  accentFill: ['#245bd8', '#245bd8'],
  selection: ['rgba(36, 42, 55, 0.08)', 'rgba(255, 255, 255, 0.10)'],
  warning: ['#8a5200', '#ffd28a'],
  warningMuted: ['#fff4d8', 'rgba(139, 92, 0, 0.22)'],
  error: ['#b42318', '#ffb4a8'],
  errorMuted: ['#fff0ed', 'rgba(152, 43, 28, 0.24)'],
  success: ['#137333', '#9be6b0'],
  successMuted: ['#eaf7ee', 'rgba(25, 108, 55, 0.22)'],
  scrim: ['rgba(8, 10, 14, 0.42)', 'rgba(8, 10, 14, 0.42)'],
};

export default {
  name: 'mlxcel',
  extends: neutralTheme,
  typography: productTypography,
  tokens: productTokens(mlxcelPalette),
  localTokens: productLocalTokens(mlxcelPalette),
} satisfies DefineThemeInput;
