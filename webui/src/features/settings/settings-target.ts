// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import type { CatalogEntry } from '../../api/types';
import { useWebUi, useWebUiActions } from '../../state';

/** The model every server-scoped Settings tab reads: the app-wide selection, observed only once it is Ready. */
export interface SettingsTarget {
  readonly snapshot: ReturnType<typeof useWebUi>;
  readonly actions: ReturnType<typeof useWebUiActions>;
  readonly selected: CatalogEntry | null;
  readonly connected: boolean;
  /** Selection never loads: a model that is not Ready is never observed. */
  readonly readyId: string | null;
  readonly revision: number | null;
  /** The operator started the server with --settings. */
  readonly liveEnabled: boolean;
  readonly single: boolean;
}

export function useSettingsTarget(): SettingsTarget {
  const snapshot = useWebUi();
  const actions = useWebUiActions();
  const selected = snapshot.catalog.find((entry) => entry.identity.id === snapshot.selectedModelId) ?? null;
  const connected = snapshot.auth.status === 'authenticated' && ['ready', 'streaming', 'polling'].includes(snapshot.connection);
  const readyId = connected && selected?.lifecycle.state === 'ready' ? selected.identity.id : null;
  return {
    snapshot,
    actions,
    selected,
    connected,
    readyId,
    revision: readyId === null || selected === null ? null : selected.identity.revision,
    liveEnabled: snapshot.bootstrap?.features.includes('settings') === true,
    single: snapshot.bootstrap?.server.mode === 'single_model',
  };
}
