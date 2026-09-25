// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useId, useMemo, useState } from 'react';
import type { ErrorCode, Operation, OperationKind, OperationState } from '../../api/types';
import { Button, EmptyState, ErrorBanner, IconButton, ProgressBar, SmoothHeight, StatusBadge, Tooltip, type LifecycleState } from '../../design-system/primitives';
import { formatBytes } from '../../design-system/format';
import { useWebUiActions } from '../../state';
import { isTerminal } from '../../state/observation';
import { t, type Locale, type StringKey } from '../../i18n/catalog';
import { Disclosure } from './disclosure';
import { dateTime, operationProgress, relativeTime } from './format';

const KIND_KEYS: Record<OperationKind, StringKey> = {
  catalog_refresh: 'activity.kind.catalog_refresh',
  model_load: 'activity.kind.model_load',
  model_unload: 'activity.kind.model_unload',
  download: 'activity.kind.download',
  model_removal: 'activity.kind.model_removal',
  settings_patch: 'activity.kind.settings_patch',
};
const STATE_KEYS: Record<OperationState, StringKey> = {
  queued: 'activity.state.queued',
  running: 'activity.state.running',
  cancelling: 'activity.state.cancelling',
  succeeded: 'activity.state.succeeded',
  failed: 'activity.state.failed',
  cancelled: 'activity.state.cancelled',
};
const ERROR_KEYS: Record<ErrorCode, StringKey> = {
  invalid_request: 'activity.error.invalid_request',
  unauthorized: 'activity.error.unauthorized',
  forbidden: 'activity.error.forbidden',
  not_found: 'activity.error.not_found',
  stale_revision: 'activity.error.stale_revision',
  conflict: 'activity.error.conflict',
  unsupported: 'activity.error.unsupported',
  rate_limited: 'activity.error.rate_limited',
  unavailable: 'activity.error.unavailable',
  payload_too_large: 'activity.error.payload_too_large',
  server_restarted: 'activity.error.server_restarted',
  event_gap: 'activity.error.event_gap',
  partial_success: 'activity.error.partial_success',
};

export const operationKindLabel = (kind: OperationKind, locale: Locale): string => t(locale, KIND_KEYS[kind]);
export const operationStateLabel = (state: OperationState, locale: Locale): string => t(locale, STATE_KEYS[state]);

// A failed operation reads as a localized reason, never the enum word. The runtime value
// can be a code newer than this build, so an unmapped one gets the generic reason; the
// raw code stays in Operation details and the server's message is never rendered.
function errorLabel(error: Operation['error'], locale: Locale): string {
  if (error === null) return t(locale, 'activity.failure_no_code');
  return t(locale, Object.hasOwn(ERROR_KEYS, error.code) ? ERROR_KEYS[error.code] : 'activity.error.other');
}

function badgeState(operation: Operation): LifecycleState {
  if (operation.state === 'failed') return 'failed';
  if (isTerminal(operation)) return 'unloaded';
  return operation.state === 'cancelling' ? 'draining' : 'loading';
}

// What the operation acts on, in user terms: a model's display name or a download's
// repository. Catalog and settings operations add nothing the kind label does not say,
// and an opaque model id never becomes the label.
function targetLabel(operation: Operation, names: ReadonlyMap<string, string>, locale: Locale): string | null {
  const target = operation.target;
  if (target.target_kind === 'model') return names.get(target.model_id) ?? t(locale, 'activity.target.unknown_model');
  if (target.target_kind === 'download') return target.repo_id;
  return null;
}

type OperationsProps = { operations: ReadonlyMap<string, Operation>; names: ReadonlyMap<string, string>; locale: Locale; stale: boolean; now: number };

export function Operations({ operations, names, locale, stale, now }: OperationsProps): React.JSX.Element {
  const headingId = useId();
  // Every 2 s poll re-renders with a new `now`; sort and count only when the operations or locale change.
  const { entries, counts } = useMemo(() => {
    const sorted = [...operations.values()].sort((a, b) => Number(isTerminal(a)) - Number(isTerminal(b)) || b.updated_at.localeCompare(a.updated_at));
    return { entries: sorted, counts: t(locale, 'activity.operations.counts', { active: String(sorted.filter((operation) => !isTerminal(operation)).length), failed: String(sorted.filter((operation) => operation.state === 'failed').length) }) };
  }, [operations, locale]);
  return <section className="activity-operations-section" aria-labelledby={headingId}>
    <div className="activity-section-head">
      <h2 id={headingId}>{t(locale, 'activity.operations')}</h2>
      <p className="activity-counts" role="status" aria-live="polite">{counts}</p>
      {entries.length > 0 ? <Tooltip content={t(locale, 'activity.session')}><IconButton icon="help" label={t(locale, 'activity.session_help')} /></Tooltip> : null}
    </div>
    <SmoothHeight animate={entries.some((operation) => !isTerminal(operation))}>{entries.length === 0
      ? <EmptyState title={t(locale, 'activity.operations.empty.title')} body={t(locale, 'activity.session')} />
      : <ol className="activity-operations">{entries.map((operation) => <OperationRow key={operation.operation_id} operation={operation} target={targetLabel(operation, names, locale)} locale={locale} stale={stale} now={now} />)}</ol>}
    </SmoothHeight>
  </section>;
}

function OperationRow({ operation, target, locale, stale, now }: { operation: Operation; target: string | null; locale: Locale; stale: boolean; now: number }): React.JSX.Element {
  const actions = useWebUiActions();
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const terminal = isTerminal(operation);
  const cancel = async (): Promise<void> => {
    setBusy(true); setFailed(false);
    try { await actions.cancelOperation(operation.operation_id); } catch { setFailed(true); } finally { setBusy(false); }
  };
  const progress = operation.progress;
  const modelId = operation.target.target_kind === 'download' ? null : operation.target.model_id ?? null;
  return <li className="activity-operation">
    <span className="activity-operation__kind">{operationKindLabel(operation.kind, locale)}</span>
    {target === null ? null : <span className="activity-operation__target">{target}</span>}
    <StatusBadge state={badgeState(operation)}>{operationStateLabel(operation.state, locale)}</StatusBadge>
    <time className="activity-muted" dateTime={operation.updated_at}>{relativeTime(operation.updated_at, now, locale)}</time>
    {!terminal && operation.kind === 'download' ? <ProgressBar label={t(locale, 'activity.bytes')} value={operationProgress(operation)} detail={progress.total_bytes === null ? t(locale, 'activity.progress.indeterminate', { bytes: formatBytes(progress.completed_bytes, locale) }) : t(locale, 'activity.download_progress', { completed: formatBytes(progress.completed_bytes, locale), total: formatBytes(progress.total_bytes, locale) })} /> : null}
    {operation.state === 'cancelling' ? <span className="activity-muted" role="status">{t(locale, 'activity.cancelling')}</span> : null}
    {!terminal && operation.state !== 'cancelling' ? operation.cancellable ? <Button disabled={stale} busy={busy} onClick={() => { void cancel(); }}>{t(locale, 'activity.cancel')}</Button> : <span className="activity-muted">{t(locale, 'activity.cancel_unsupported')}</span> : null}
    {/* Identifiers are diagnostics, not copy: they appear only inside this disclosure. */}
    <Disclosure className="activity-disclosure activity-operation__details" summary={t(locale, 'activity.details')}>{() => <ul className="activity-raw-list">
      <li><code>{t(locale, 'activity.details.operation_id', { id: operation.operation_id })}</code></li>
      {modelId === null ? null : <li><code>{t(locale, 'activity.details.model_id', { id: modelId })}</code></li>}
      {operation.error === null ? null : <li><code>{t(locale, 'activity.details.error_code', { code: operation.error.code })}</code></li>}
      <li>{t(locale, 'activity.details.updated', { time: dateTime(operation.updated_at, locale) })}</li>
    </ul>}</Disclosure>
    {operation.state === 'failed' ? <ErrorBanner title={t(locale, 'activity.failure')} body={errorLabel(operation.error, locale)} /> : null}
    {failed ? <ErrorBanner title={t(locale, 'activity.cancel_failed')} body={t(locale, 'activity.refresh')} /> : null}
  </li>;
}
