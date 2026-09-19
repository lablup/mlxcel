// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useState } from 'react';
import type { CatalogEntry, LoadProfile, WebUiSnapshot } from '../../api/types';
import { Badge, Button, Drawer, Inspector, StatusBadge } from '../../design-system/primitives';
import { lifecycleLabel } from '../../provider-surfaces';
import { t, type Locale } from '../../i18n/catalog';
import { sourceLabel, taskLabel } from './labels';
import { NextLoadProfile } from './next-load-profile';
import { bytes, canChat, canDelete, canLoad, canUnload, quantizationValue } from './policy';
import { RootsHelp, RootsList } from './roots';

export type ModelAction = 'load' | 'unload' | 'delete';

type InspectorProps = {
  entry: CatalogEntry;
  profile: LoadProfile;
  state: WebUiSnapshot;
  locale: Locale;
  busy: boolean;
  onAction: (kind: ModelAction, entry: CatalogEntry) => void;
  onChat: (entry: CatalogEntry) => void;
};

/**
 * The selected model: a right pane at 1100 px and wider, a drawer below it. Everything that is an
 * opaque identifier, a raw server reason or an internal counter sits in the closed Details
 * disclosure, and is not rendered at all until it is opened.
 */
export function ModelInspector(props: InspectorProps & { variant: 'pane' } | InspectorProps & { variant: 'drawer'; open: boolean; onClose: () => void }): React.JSX.Element {
  const title = t(props.locale, 'models.library.details');
  const body = <InspectorBody {...props} />;
  if (props.variant === 'pane') return <Inspector title={title}>{body}</Inspector>;
  return (
    <Drawer open={props.open} onClose={props.onClose} title={title} closeLabel={t(props.locale, 'common.close')} testId="models-inspector-drawer" width="medium" side="end">
      {body}
    </Drawer>
  );
}

function InspectorBody({ entry, profile, state, locale, busy, onAction, onChat }: InspectorProps): React.JSX.Element {
  const m = entry.metadata;
  const runtime = state.runtimes.get(entry.identity.id);
  const context = runtime?.revision === entry.identity.revision ? runtime.settings.effective.ctx_size : null;
  const bool = (value: boolean): string => t(locale, value ? 'models.library.yes' : 'models.library.no');
  const overview: [string, string][] = [
    [t(locale, 'models.library.source'), sourceLabel(locale, entry.identity.source)],
    [t(locale, 'models.library.architecture'), m.architecture ?? t(locale, 'models.library.unknown')],
    [t(locale, 'models.library.size'), bytes(m.disk_bytes, locale)],
    [t(locale, 'models.library.quantization'), quantizationValue(entry) ?? t(locale, 'models.library.unknown')],
    ...(typeof context === 'number' ? [[t(locale, 'models.library.context'), context.toLocaleString(locale)] as [string, string]] : []),
    ...(m.memory_estimate_bytes !== null ? [[t(locale, 'models.library.memory'), bytes(m.memory_estimate_bytes, locale)] as [string, string]] : []),
    [t(locale, 'models.library.support'), t(locale, m.support.architecturally_supported ? 'models.library.supported' : 'models.library.unsupported')],
    [t(locale, 'models.library.files'), t(locale, entry.complete ? 'models.library.complete' : 'models.library.incomplete')],
    [t(locale, 'models.library.backend'), bool(m.support.runnable_on_backend)],
  ];
  const tasks = [...new Map(entry.capabilities.map((cap) => [cap.task, cap])).values()];
  return (
    <div className="models-inspector">
      <h3 className="models-inspector-name">{entry.identity.display_name}</h3>
      <StatusBadge state={entry.lifecycle.state}>{lifecycleLabel(locale, entry.lifecycle.state)}</StatusBadge>
      <div className="button-row">
        <Button data-testid="models-load" disabled={busy || !canLoad(state, entry)} onClick={() => onAction('load', entry)}>
          {t(locale, 'models.load')}
        </Button>
        <Button data-testid="models-use-chat" disabled={busy || !canChat(state, entry)} onClick={() => onChat(entry)}>
          {t(locale, 'models.library.chat')}
        </Button>
        <Button data-testid="models-unload" disabled={busy || !canUnload(state, entry)} onClick={() => onAction('unload', entry)}>
          {t(locale, 'models.unload')}
        </Button>
        {entry.identity.source === 'cache' ? (
          <Button tone="danger" data-testid="models-delete" disabled={busy || !canDelete(state, entry)} onClick={() => onAction('delete', entry)}>
            {t(locale, 'models.library.delete')}
          </Button>
        ) : null}
      </div>
      <h4>{t(locale, 'models.library.overview')}</h4>
      <dl className="models-overview">
        {overview.map(([label, value]) => (
          <React.Fragment key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </React.Fragment>
        ))}
      </dl>
      <h4>{t(locale, 'models.library.capabilities')}</h4>
      <ul className="models-capabilities">
        {tasks.map((cap) => (
          <li key={cap.task}>
            <Badge tone={cap.available ? 'neutral' : 'warning'}>{taskLabel(locale, cap.task)}</Badge>
          </li>
        ))}
      </ul>
      <InspectorDetails key={entry.identity.id} entry={entry} profile={profile} state={state} locale={locale} />
    </div>
  );
}

function InspectorDetails({ entry, profile, state, locale }: { entry: CatalogEntry; profile: LoadProfile; state: WebUiSnapshot; locale: Locale }): React.JSX.Element {
  // Controlled so the content mounts only while open: ids and reasons are not in the document otherwise.
  const [open, setOpen] = useState(false);
  const m = entry.metadata;
  const bool = (value: boolean): string => t(locale, value ? 'models.library.yes' : 'models.library.no');
  const identity: [string, string][] = [
    [t(locale, 'models.library.model_id'), entry.identity.id],
    [t(locale, 'models.library.inference_id'), entry.identity.inference_id],
    [t(locale, 'models.library.catalog_revision'), String(entry.identity.revision)],
    [t(locale, 'models.library.active'), String(entry.lifecycle.active_requests)],
    [t(locale, 'models.library.worker'), bool(entry.lifecycle.worker_exit_observed)],
    [t(locale, 'models.library.tested'), bool(m.support.tested_checkpoint)],
    [t(locale, 'models.library.profile'), Object.keys(profile).length ? JSON.stringify(profile) : t(locale, 'models.library.defaults')],
  ];
  const reasons = [
    ...new Set(
      [
        m.support.reason,
        m.support.architecturally_supported_reason,
        m.support.runnable_on_backend_reason,
        m.support.complete_reason,
        m.support.tested_checkpoint_reason,
        ...Object.values(m.unknown_reasons),
        ...entry.capabilities.map((cap) => cap.reason),
      ].filter((reason): reason is string => !!reason),
    ),
  ];
  const unavailable = Object.entries(state.bootstrap?.actions ?? {}).filter(([, action]) => action.state !== 'enabled');
  return (
    <details className="models-details" data-testid="models-details" open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>{t(locale, 'models.library.disclosure')}</summary>
      {open ? (
        <div className="models-details-body">
          <dl className="models-overview">
            {identity.map(([label, value]) => (
              <React.Fragment key={label}>
                <dt>{label}</dt>
                <dd>
                  <code className="models-wrap">{value}</code>
                </dd>
              </React.Fragment>
            ))}
          </dl>
          {entry.lifecycle.last_error ? (
            <p>
              {t(locale, 'models.library.last_error')}: <span className="models-wrap">{entry.lifecycle.last_error}</span>
            </p>
          ) : null}
          {Object.keys(profile).length ? <NextLoadProfile locale={locale} /> : null}
          {!canChat(state, entry) ? <p>{t(locale, 'models.library.chat_reason')}</p> : null}
          <h5>{t(locale, 'models.library.reasons')}</h5>
          <p>{t(locale, 'models.library.tested_help')}</p>
          <ul>
            {reasons.map((reason) => (
              <li key={reason} className="models-wrap">
                {reason}
              </li>
            ))}
          </ul>
          {!entry.removal.eligible && (entry.removal.reason || entry.removal.instructions) ? (
            <>
              <h5>{t(locale, 'models.library.removal')}</h5>
              <p className="models-wrap">
                {entry.removal.reason} {entry.removal.instructions}
              </p>
            </>
          ) : null}
          {unavailable.length ? (
            <>
              <h5>{t(locale, 'models.library.server_actions')}</h5>
              <ul>
                {unavailable.map(([name, action]) => (
                  <li key={name} className="models-wrap">
                    <code>{name}</code>: {action.reason} {action.instructions}
                  </li>
                ))}
              </ul>
            </>
          ) : null}
          <h5>{t(locale, 'models.library.roots')}</h5>
          <RootsList state={state} locale={locale} />
          <RootsHelp locale={locale} />
          <p>
            <a href="https://github.com/lablup/mlxcel/blob/main/docs/llama-server-compat.md" target="_blank" rel="noreferrer">
              {t(locale, 'models.library.api')}
            </a>
          </p>
        </div>
      ) : null}
    </details>
  );
}
