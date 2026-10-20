// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useEffect, useMemo, useState } from 'react';
import { Theme } from '@lablup/ui-common/theme';
import { neutralTheme } from '@lablup/ui-common/theme/neutral/built';
import { InternationalizationProvider } from '@lablup/ui-common/i18n';
import type { Locale } from '../i18n/catalog';
import type { ColorScheme } from './theme';

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

// The root Astryx providers: the theme and the strings Astryx and ui-common components render
// themselves (loading announcements, sort and close labels).
//
// The pre-built neutral theme is the interim base until the product families (mlxcel, glass)
// are rebuilt as defineTheme themes; until then the product palettes stay the [data-theme]
// blocks in themes/*.css and override Astryx's tokens.
//
// The root <Theme> writes <html data-theme> to the bare mode ("light"/"dark") in a layout
// effect, while the product keeps its applied `<family>-<scheme>` id there. The caller
// renders this inside the component that applies the appearance and applies it in its own
// layout effect, which runs after this one in the same commit, so the product id is what the
// first paint sees after every mode change.
export function ProductTheme(props: { scheme: ColorScheme; locale: Locale; children: React.ReactNode }): React.JSX.Element {
  const [messages, setMessages] = useState<Messages | null>(null);
  useEffect(() => {
    const load = CATALOG_LOADERS[props.locale];
    if (!load || messages?.[ASTRYX_LOCALES[props.locale]]) return;
    let live = true;
    void load().then((loaded) => { if (live) setMessages(loaded); }, () => undefined);
    return () => { live = false; };
  }, [props.locale, messages]);
  const providerMessages = useMemo(() => (messages ?? {}) as React.ComponentProps<typeof InternationalizationProvider>['messages'], [messages]);
  return <Theme theme={neutralTheme} mode={props.scheme}>
    <InternationalizationProvider locale={ASTRYX_LOCALES[props.locale]} messages={providerMessages}>{props.children}</InternationalizationProvider>
  </Theme>;
}
