// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useEffect, useState } from 'react';
import { InternationalizationProvider, useTranslator, type MessagesByLocale } from '@lablup/ui-common/i18n';
import type { Locale } from '../i18n/catalog';

// Astryx's locale tag for each product locale.
export const LIBRARY_LOCALES: Record<Locale, string> = { en: 'en', ko: 'ko-KR' };

// English is the built-in catalog of both Astryx and ui-common. Any other locale's catalogs are
// one chunk fetched on demand, so an English session never downloads them.
const LOADERS: Partial<Record<Locale, () => Promise<MessagesByLocale>>> = {
  ko: () => import('./common-i18n-ko').then((module) => module.koreanMessages),
};
const pending = new Map<Locale, Promise<MessagesByLocale>>();

/** The Astryx and ui-common messages for `locale`, fetched once; a failed fetch is retried next time. */
export function loadLibraryMessages(locale: Locale): Promise<MessagesByLocale> {
  const load = LOADERS[locale];
  if (!load) return Promise.resolve({});
  let request = pending.get(locale);
  if (!request) {
    request = load();
    request.catch(() => pending.delete(locale));
    pending.set(locale, request);
  }
  return request;
}

/**
 * The strings Astryx and ui-common components render themselves (loading announcements, sort,
 * close, navigation and tab-strip labels), in the appearance locale. The provider follows the
 * locale at once; until a fetched catalog arrives the library falls back to its English text.
 */
export function LibraryStrings(props: { locale: Locale; children: React.ReactNode }): React.JSX.Element {
  const tag = LIBRARY_LOCALES[props.locale];
  const [messages, setMessages] = useState<MessagesByLocale>({});
  const ready = LOADERS[props.locale] === undefined || tag in messages;
  useEffect(() => {
    if (ready) return;
    let live = true;
    void loadLibraryMessages(props.locale).then((loaded) => { if (live) setMessages((current) => ({ ...current, ...loaded })); }, () => undefined);
    return () => { live = false; };
  }, [props.locale, ready]);
  return <InternationalizationProvider locale={tag} messages={messages}>{props.children}</InternationalizationProvider>;
}

/** An Astryx built-in string, in the provider's locale; for product controls that show the library's own label. */
export function useLibraryString(key: '@astryx.dialog.close'): string {
  return useTranslator()(key);
}
