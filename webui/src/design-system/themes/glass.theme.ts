// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Portions of this file are derived from backend.ai-go
// (https://github.com/lablup/backend.ai-go), Copyright 2025-2026 Lablup Inc.,
// licensed under the Apache License, Version 2.0. See the top-level NOTICE file.
//
// The glass family as an Astryx theme over the pre-built neutral theme. The
// palette follows backend.ai-go's glass-light and glass-dark tokens, with text,
// accent and status colors adjusted to clear WCAG 2.2 AA on this product's
// surfaces. Translucent materials, highlights and their accessibility overrides
// are product CSS (glass.css). Built into ./built/mlxcel-glass.{css,js,d.ts}.
import type { DefineThemeInput } from '@lablup/ui-common/theme';
import { neutralTheme } from '@lablup/ui-common/theme/neutral/built';
import { productLocalTokens, productTokens, productTypography, type ProductPalette } from './palette';

export const glassPalette: ProductPalette = {
  canvas: ['#e6ebf3', '#0b0d12'],
  surface: ['#fbfbfd', '#1c1c1f'],
  recessed: ['#f1f2f6', '#232327'],
  text: ['#1d1d1f', '#f5f5f7'],
  textSecondary: ['#545458', '#b8b8bc'],
  border: ['rgba(60, 60, 67, 0.14)', 'rgba(255, 255, 255, 0.12)'],
  accentText: ['#0058b9', '#6cb2ff'],
  accentFill: ['#0058b9', '#0a5bc0'],
  selection: ['rgba(0, 88, 185, 0.10)', 'rgba(64, 156, 255, 0.18)'],
  warning: ['#8a4f00', '#ffb340'],
  warningMuted: ['rgba(255, 149, 0, 0.12)', 'rgba(255, 159, 10, 0.14)'],
  error: ['#b3261e', '#ff7b72'],
  errorMuted: ['rgba(255, 59, 48, 0.10)', 'rgba(255, 69, 58, 0.16)'],
  success: ['#1b6f3c', '#4cd964'],
  successMuted: ['rgba(52, 199, 89, 0.12)', 'rgba(50, 215, 75, 0.14)'],
  scrim: ['rgba(8, 10, 14, 0.32)', 'rgba(8, 10, 14, 0.32)'],
};

export default {
  name: 'mlxcel-glass',
  extends: neutralTheme,
  typography: productTypography,
  tokens: productTokens(glassPalette),
  localTokens: productLocalTokens(glassPalette),
} satisfies DefineThemeInput;
