// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Catalog enums as user copy: the raw values (models_dir, audio_transcription, running) are
// contract identifiers, not words, so every place that shows one goes through these.
import type { CatalogSourceKind, Operation, RootSummary, TaskKind } from '../../api/types';
import type { LifecycleState } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';

export function sourceLabel(locale: Locale, source: CatalogSourceKind | RootSummary['kind']): string {
  return t(locale, `models.source.${source}`);
}

export function taskLabel(locale: Locale, task: TaskKind): string {
  return t(locale, `models.task.${task}`);
}

export function downloadStateLabel(locale: Locale, state: Operation['state']): string {
  return t(locale, `models.download.${state}`);
}

/** A download drawn with the lifecycle badge colors: in flight reads as a transition, failed as failed. */
export function downloadBadgeState(state: Operation['state']): LifecycleState {
  if (state === 'failed') return 'failed';
  if (state === 'cancelling') return 'unloading';
  if (state === 'cancelled') return 'unloaded';
  if (state === 'succeeded') return 'ready';
  return 'loading';
}

/**
 * Wraps a name in Unicode first-strong isolates before it is interpolated into a sentence, so a
 * name holding bidi controls (U+202E) cannot reorder the words around it. Markup can use `<bdi>`;
 * a translated string cannot.
 */
export function isolate(name: string): string {
  return `⁨${name}⁩`;
}
