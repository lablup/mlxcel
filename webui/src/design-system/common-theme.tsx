// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useEffect, useMemo, useState } from 'react';
import { Theme } from '@lablup/ui-common/theme';
import { mlxcelTheme } from './themes/built/mlxcel.js';
import { mlxcelGlassTheme } from './themes/built/mlxcel-glass.js';
import { InternationalizationProvider } from '@lablup/ui-common/i18n';
import type { Locale } from '../i18n/catalog';
import type { ColorScheme, ThemeFamily } from './theme';

// Astryx's locale names for the product locales. English is the built-in catalog of both
// Astryx and ui-common; Korean is loaded on demand so English sessions do not carry it.
const ASTRYX_LOCALES: Record<Locale, string> = { en: 'en', ko: 'ko-KR' };
type Messages = Record<string, Readonly<Record<string, unknown>>>;
let koreanMessages: Promise<Messages> | null = null;
function loadKorean(): Promise<Messages> {
  koreanMessages ??= Promise.all([
    import('@lablup/ui-common/locales/ko-KR.generated.js'),
    import('@lablup/ui-common/ui-common-locales/ko-KR.json'),
  ]).then(([astryx, uiCommon]) => ({ 'ko-KR': { ...astryx.default, ...uiCommon.default } }));
  return koreanMessages;
}
// Locales whose Astryx and ui-common catalogs load on demand; English needs none.
const CATALOG_LOADERS: Partial<Record<Locale, () => Promise<Messages>>> = { ko: loadKorean };

// The pre-built Astryx theme of each product family (themes/*.theme.ts, compiled by
// `astryx theme build`). `__built: true` makes <Theme> skip its runtime <style>
// injection: the matching CSS ships in themes/index.css, which the strict CSP needs.
const ASTRYX_THEMES = { mlxcel: mlxcelTheme, glass: mlxcelGlassTheme } as const satisfies Record<ThemeFamily, unknown>;

// The root Astryx providers: the theme and the strings Astryx and ui-common components render
// themselves (loading announcements, sort and close labels).
//
// The root <Theme> writes <html data-theme> (the scheme) and <html data-astryx-theme> (the theme
// name) in a layout effect. applyAppearance() and public/theme-bootstrap.js write the same two
// values, so the attributes never flip between them.
export function ProductTheme(props: { family: ThemeFamily; scheme: ColorScheme; locale: Locale; children: React.ReactNode }): React.JSX.Element {
  const [messages, setMessages] = useState<Messages | null>(null);
  useEffect(() => {
    const load = CATALOG_LOADERS[props.locale];
    if (!load || messages?.[ASTRYX_LOCALES[props.locale]]) return;
    let live = true;
    void load().then((loaded) => { if (live) setMessages(loaded); }, () => undefined);
    return () => { live = false; };
  }, [props.locale, messages]);
  const providerMessages = useMemo(() => (messages ?? {}) as React.ComponentProps<typeof InternationalizationProvider>['messages'], [messages]);
  return <Theme theme={ASTRYX_THEMES[props.family]} mode={props.scheme}>
    <InternationalizationProvider locale={ASTRYX_LOCALES[props.locale]} messages={providerMessages}>{props.children}</InternationalizationProvider>
  </Theme>;
}
