// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import { Theme } from '@lablup/ui-common/theme';
import { mlxcelTheme } from './themes/built/mlxcel.js';
import { mlxcelGlassTheme } from './themes/built/mlxcel-glass.js';
import { LibraryStrings } from './common-i18n';
import type { Locale } from '../i18n/catalog';
import type { ColorScheme, ThemeFamily } from './theme';

// The pre-built Astryx theme of each product family (themes/*.theme.ts, compiled by
// `astryx theme build`). `__built: true` makes <Theme> skip its runtime <style>
// injection: the matching CSS ships in themes/index.css, which the strict CSP needs.
const ASTRYX_THEMES = { mlxcel: mlxcelTheme, glass: mlxcelGlassTheme } as const satisfies Record<ThemeFamily, unknown>;

// The root Astryx providers: the theme and, through LibraryStrings (common-i18n.tsx), the strings
// Astryx and ui-common components render themselves.
//
// The root <Theme> writes <html data-theme> (the scheme) and <html data-astryx-theme> (the theme
// name) in a layout effect. applyAppearance() and public/theme-bootstrap.js write the same two
// values, so the attributes never flip between them.
export function ProductTheme(props: { family: ThemeFamily; scheme: ColorScheme; locale: Locale; children: React.ReactNode }): React.JSX.Element {
  return <Theme theme={ASTRYX_THEMES[props.family]} mode={props.scheme}>
    <LibraryStrings locale={props.locale}>{props.children}</LibraryStrings>
  </Theme>;
}
