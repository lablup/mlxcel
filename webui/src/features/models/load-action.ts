// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// The one explicit model load request, shared by the Models library and Chat so both
// send the same body and treat a capacity conflict the same way.
import type { CatalogEntry, LoadProfile, WebUiSnapshot } from '../../api/types';
import { WebUiHttpError } from '../../api/client';
import { t, type Locale } from '../../i18n/catalog';
import type { WebUiActions } from '../../state';
import { canLoad, errorMessage, evictionCandidates } from './policy';

/** The snapshot no longer allows this load: the entry or the eviction target changed, or loading is not allowed. No request was sent. */
export class LoadRefusedError extends Error {
  constructor() {
    super('The model can no longer be loaded as confirmed.');
    this.name = 'LoadRefusedError';
  }
}

/**
 * Sends one `load` model action for `entry`. `profile` is the pending next-load profile,
 * attached only when it sets a field. On a `conflict` envelope (capacity or a conflicting
 * operation) `onCapacity` runs once, so the caller can offer the eviction dialog, and the
 * error is rethrown for the caller's own error handling.
 */
export async function submitLoad(
  actions: Pick<WebUiActions, 'loadModel'>,
  state: WebUiSnapshot,
  entry: CatalogEntry,
  profile: LoadProfile,
  onCapacity: () => void,
  evictionTarget?: { id: string; revision: number },
): Promise<void> {
  if (!canLoad(state, entry)) throw new LoadRefusedError();
  if (
    evictionTarget &&
    !evictionCandidates(state, entry.identity.id).some(
      (candidate) => candidate.identity.id === evictionTarget.id && candidate.identity.revision === evictionTarget.revision,
    )
  )
    throw new LoadRefusedError();
  try {
    await actions.loadModel({
      action: 'load',
      ...(Object.keys(profile).length ? { load_profile: { ...profile } } : {}),
      model_id: entry.identity.id,
      expected_revision: entry.identity.revision,
      idempotency_key: crypto.randomUUID(),
      ...(evictionTarget
        ? {
            eviction_target_id: evictionTarget.id,
            eviction_target_expected_revision: evictionTarget.revision,
          }
        : {}),
    });
  } catch (failure) {
    if (failure instanceof WebUiHttpError && failure.envelope?.error.code === 'conflict') onCapacity();
    throw failure;
  }
}

/** The load error copy both screens show, through the Models library keys. */
export function loadErrorMessage(failure: unknown, locale: Locale): string {
  if (failure instanceof LoadRefusedError) return t(locale, 'models.library.stale');
  return errorMessage(failure, locale);
}
