// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// The Korean catalogs of Astryx and ui-common, one chunk fetched only when the appearance locale
// is Korean. Not `mergeMessages` / `uiCommonMessages` from `@lablup/ui-common/i18n-catalog`: that
// module builds every ui-common locale at evaluation, which the bundler cannot drop (a 349 KB
// chunk instead of 38 KB), so this takes the same Korean entries from the per-locale files and
// merges them the way mergeMessages does, a later catalog winning on the same key.
import astryxKorean from '@lablup/ui-common/locales/ko-KR.generated.js';
import uiCommonKorean from '@lablup/ui-common/ui-common-locales/ko-KR.json';
import type { Catalog, MessagesByLocale } from '@lablup/ui-common/i18n';

// Astryx ships its runtime catalog as plain strings; ui-common's entries are message objects.
const astryx: Catalog = Object.fromEntries(Object.entries(astryxKorean).map(([key, defaultMessage]) => [key, { defaultMessage }]));

export const koreanMessages: MessagesByLocale = { 'ko-KR': { ...astryx, ...(uiCommonKorean as Catalog) } };
