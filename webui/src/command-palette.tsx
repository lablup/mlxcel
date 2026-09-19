// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useId, useState } from 'react';
import type { CatalogEntry, ModelId } from './api/types';
import { Button, Dialog, Field } from './design-system/primitives';
import type { RouteId } from './design-system/shell';
import type { Locale, StringKey } from './i18n/catalog';
import { t, testId } from './i18n/catalog';
import { compareByLoadedThenName, lifecycleLabel } from './provider-surfaces';

/** Model hits past this many are dropped; the palette is a jump list, not a library. */
export const PALETTE_MODEL_LIMIT = 20;

// Gallery is not listed: it stays a direct, hash-only `#gallery` route.
type PaletteCommand = { readonly id: Exclude<RouteId, 'gallery'> | 'new-chat'; readonly key: StringKey; readonly testId: string };
const COMMANDS: ReadonlyArray<PaletteCommand> = [
  { id: 'models', key: 'nav.models', testId: 'command-route-models' },
  { id: 'chat', key: 'nav.chat', testId: 'command-route-chat' },
  { id: 'activity', key: 'nav.activity', testId: 'command-route-activity' },
  { id: 'settings', key: 'nav.settings', testId: 'command-route-settings' },
  { id: 'new-chat', key: 'chat.new_conversation', testId: 'command-new-chat' },
];

/** Catalog entries whose display name or id contains `query` (case-insensitive), loaded first, then by name. */
export function paletteModelMatches(catalog: ReadonlyArray<CatalogEntry>, query: string, limit = PALETTE_MODEL_LIMIT): CatalogEntry[] {
  const needle = query.trim().toLowerCase();
  if (needle === '') return [];
  return catalog
    .filter((entry) => entry.identity.display_name.toLowerCase().includes(needle) || entry.identity.id.toLowerCase().includes(needle))
    .sort(compareByLoadedThenName)
    .slice(0, Math.max(0, limit));
}

export interface CommandPaletteProps {
  readonly open: boolean;
  readonly locale: Locale;
  readonly catalog: ReadonlyArray<CatalogEntry>;
  /** The toolbar's loaded models, listed while the query is empty. */
  readonly loaded: ReadonlyArray<CatalogEntry>;
  readonly onClose: () => void;
  readonly onNavigate: (route: Exclude<RouteId, 'gallery'>) => void;
  readonly onNewChat: () => void;
  /** Selects the model and opens its inspector on Models; never loads it. */
  readonly onOpenModel: (id: ModelId) => void;
}

export function CommandPalette(props: CommandPaletteProps): React.JSX.Element {
  const [query, setQuery] = useState('');
  const [wasOpen, setWasOpen] = useState(props.open);
  const commandsId = useId();
  const modelsId = useId();
  // Every opening starts from an empty query, which lists the loaded models. Closing clears it
  // too, so a query typed before closing does not keep matching in the background and holding
  // up to PALETTE_MODEL_LIMIT model buttons in a dialog nothing can see.
  if (props.open !== wasOpen) {
    setWasOpen(props.open);
    setQuery('');
  }
  const needle = query.trim().toLowerCase();
  const commands = COMMANDS.filter((command) => needle === '' || command.id.includes(needle) || t(props.locale, command.key).toLowerCase().includes(needle));
  const models = needle === '' ? props.loaded : paletteModelMatches(props.catalog, needle);
  const run = (command: PaletteCommand): void => {
    if (command.id === 'new-chat') props.onNewChat();
    else props.onNavigate(command.id);
  };
  return (
    <Dialog open={props.open} title={t(props.locale, 'command.title')} onClose={props.onClose} testId="command-dialog" closeLabel={t(props.locale, 'common.close')}>
      <Field label={t(props.locale, 'command.search')} value={query} onChange={setQuery} testId={testId('command.search')} />
      {commands.length === 0 && models.length === 0 ? <p data-testid={testId('command.no_results')}>{t(props.locale, 'command.no_results')}</p> : null}
      {commands.length > 0 ? (
        <div className="command-section" role="group" aria-labelledby={commandsId}>
          <h3 id={commandsId}>{t(props.locale, 'command.section.commands')}</h3>
          <div className="command-list">{commands.map((command) => <Button key={command.id} data-testid={command.testId} onClick={() => run(command)}>{t(props.locale, command.key)}</Button>)}</div>
        </div>
      ) : null}
      {models.length > 0 ? (
        <div className="command-section" role="group" aria-labelledby={modelsId}>
          <h3 id={modelsId}>{t(props.locale, 'command.section.models')}</h3>
          <div className="command-list">{models.map((entry) => (
            <Button key={entry.identity.id} className="command-model" title={entry.identity.display_name} data-testid="command-model" onClick={() => props.onOpenModel(entry.identity.id)}>
              {`${entry.identity.display_name} · ${lifecycleLabel(props.locale, entry.lifecycle.state)}`}
            </Button>
          ))}</div>
        </div>
      ) : null}
    </Dialog>
  );
}

