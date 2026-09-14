// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import type { CatalogEntry, WebUiSnapshot } from '../../api/types';
import { Button, Inspector, StatusBadge } from '../../design-system/primitives';
import { lifecycleLabel } from '../../provider-surfaces';
import { t, type Locale } from '../../i18n/catalog';
import { bytes, canChat, canDelete, canLoad, canUnload } from './policy';

export function ModelInspector({
  entry,
  state,
  locale,
  busy,
  onAction,
  onChat,
}: {
  entry: CatalogEntry;
  state: WebUiSnapshot;
  locale: Locale;
  busy: boolean;
  onAction: (kind: 'load' | 'unload' | 'delete') => void;
  onChat: () => void;
}): React.JSX.Element {
  const m = entry.metadata;
  const runtime = state.runtimes.get(entry.identity.id);
  const context = runtime?.revision === entry.identity.revision ? runtime.settings.effective.ctx_size : null;
  const known = (value: string | number | null | undefined): string =>
    value === null || value === undefined ? t(locale, 'models.library.unknown') : String(value);
  const bool = (value: boolean): string => t(locale, value ? 'models.library.yes' : 'models.library.no');
  const details = [
    [t(locale, 'models.library.source'), entry.identity.source],
    [t(locale, 'models.library.architecture'), known(m.architecture)],
    [t(locale, 'models.library.backend'), bool(m.support.runnable_on_backend)],
    [t(locale, 'models.library.tested'), bool(m.support.tested_checkpoint)],
    [t(locale, 'models.library.disk'), bytes(m.disk_bytes, locale)],
    [t(locale, 'models.library.memory'), bytes(m.memory_estimate_bytes, locale)],
    [t(locale, 'models.library.context'), known(typeof context === 'number' ? context : null)],
    [t(locale, 'models.library.profile'), t(locale, 'models.library.defaults')],
    [t(locale, 'models.library.active'), String(entry.lifecycle.active_requests)],
    [t(locale, 'models.library.worker'), bool(entry.lifecycle.worker_exit_observed)],
  ];
  const reasons = [
    m.support.reason,
    m.support.architecturally_supported_reason,
    m.support.runnable_on_backend_reason,
    m.support.complete_reason,
    m.support.tested_checkpoint_reason,
    ...Object.values(m.unknown_reasons),
    ...entry.capabilities.map((cap) => cap.reason),
  ].filter((reason): reason is string => !!reason);
  return (
    <Inspector title={t(locale, 'models.library.details')}>
      <h3>{entry.identity.display_name}</h3>
      <code className="models-wrap">{entry.identity.id}</code>
      <StatusBadge state={entry.lifecycle.state}>{lifecycleLabel(locale, entry.lifecycle.state)}</StatusBadge>
      <dl>
        {details.map(([label, value]) => (
          <React.Fragment key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </React.Fragment>
        ))}
      </dl>
      <p>{t(locale, 'models.library.tested_help')}</p>
      {[...new Set(reasons)].map((reason) => (
        <p key={reason}>{reason}</p>
      ))}
      {entry.lifecycle.last_error ? (
        <p role="alert">
          {t(locale, 'models.library.last_error')}: {entry.lifecycle.last_error}
        </p>
      ) : null}
      <div className="button-row">
        <Button data-testid="models-load" disabled={busy || !canLoad(state, entry)} onClick={() => onAction('load')}>
          {t(locale, 'models.load')}
        </Button>
        <Button data-testid="models-use-chat" disabled={busy || !canChat(state, entry)} onClick={onChat}>
          {t(locale, 'models.library.chat')}
        </Button>
        <Button
          data-testid="models-unload"
          disabled={busy || !canUnload(state, entry)}
          onClick={() => onAction('unload')}
        >
          {t(locale, 'models.unload')}
        </Button>
        {entry.identity.source === 'cache' ? (
          <Button
            tone="danger"
            data-testid="models-delete"
            disabled={busy || !canDelete(state, entry)}
            onClick={() => onAction('delete')}
          >
            {t(locale, 'models.library.delete')}
          </Button>
        ) : null}
      </div>
      {!canChat(state, entry) ? <p>{t(locale, 'models.library.chat_reason')}</p> : null}
      {!entry.removal.eligible ? (
        <p>
          {entry.removal.reason} {entry.removal.instructions}
        </p>
      ) : null}
      {Object.entries(state.bootstrap?.actions ?? {})
        .filter(([, action]) => action.state !== 'enabled')
        .map(([name, action]) => (
          <p key={name}>
            {name}: {action.reason} {action.instructions}
          </p>
        ))}
      <a href="https://github.com/lablup/mlxcel/blob/main/docs/llama-server-compat.md" target="_blank" rel="noreferrer">
        {t(locale, 'models.library.api')}
      </a>
      <ul>
        {entry.capabilities.map((cap) => (
          <li key={`${cap.task}-${cap.phase}`}>
            {cap.task} · {cap.phase} · {bool(cap.available)}
          </li>
        ))}
      </ul>
    </Inspector>
  );
}
