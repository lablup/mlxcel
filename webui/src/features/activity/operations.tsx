// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useState } from 'react';
import type { Operation } from '../../api/types';
import { Button, EmptyState, ErrorBanner, ProgressBar, StatusBadge } from '../../design-system/primitives';
import { useWebUi, useWebUiActions } from '../../state';
import { isTerminal } from '../../state/observation';
import { operationProgress } from './format';
import { strings } from './strings';
import type { Locale } from '../../i18n/catalog';

export function Operations({ operations, locale, stale }: { operations: ReadonlyMap<string, Operation>; locale: Locale; stale: boolean }): React.JSX.Element {
  const text = strings(locale);
  const entries = [...operations.values()].sort((a, b) => Number(isTerminal(a)) - Number(isTerminal(b)) || b.updated_at.localeCompare(a.updated_at));
  return <section aria-label={text.operations}><h2>{text.operations}</h2><p role="status" aria-live="polite">{text.operations}: {entries.filter((operation) => !isTerminal(operation)).length} {locale === 'ko' ? '진행 중' : 'active'} · {entries.filter((operation) => operation.state === 'failed').length} {locale === 'ko' ? '실패' : 'failed'}</p><p>{text.session}</p>{entries.length === 0 ? <EmptyState title={text.empty} body={text.emptyBody} /> : <ol className="activity-operations">{entries.map((operation) => <li key={operation.operation_id}><OperationRow operation={operation} locale={locale} stale={stale} /></li>)}</ol>}</section>;
}

function OperationRow({ operation, locale, stale }: { operation: Operation; locale: Locale; stale: boolean }): React.JSX.Element {
  const text = strings(locale);
  const actions = useWebUiActions();
  const snapshot = useWebUi();
  const target = operation.target;
  const targetLabel = target.target_kind === 'model' ? snapshot.catalog.find((entry) => entry.identity.id === target.model_id)?.identity.display_name ?? target.model_id : target.target_kind === 'download' ? target.repo_id : target.target_kind;
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const terminal = isTerminal(operation);
  const cancel = async (): Promise<void> => {
    setBusy(true); setFailed(false);
    try { await actions.cancelOperation(operation.operation_id); } catch { setFailed(true); } finally { setBusy(false); }
  };
  return <article className="activity-operation">
    <header><strong>{operation.kind.replaceAll('_', ' ')}</strong><StatusBadge state={operation.state === 'failed' ? 'failed' : terminal ? 'unloaded' : operation.state === 'cancelling' ? 'draining' : 'loading'} label={text.status}>{operation.state}</StatusBadge></header>
    <p>{targetLabel}</p><p><time dateTime={operation.updated_at}>{new Date(operation.updated_at).toLocaleString(locale)}</time></p>
    {!terminal && operation.kind === 'download' ? <ProgressBar label={text.bytes} value={operationProgress(operation)} detail={`${operation.progress.completed_bytes.toLocaleString(locale)} / ${operation.progress.total_bytes?.toLocaleString(locale) ?? 'N/A'} bytes`} /> : null}
    {operation.state === 'cancelling' ? <p role="status">{text.cancelling}</p> : null}
    {operation.state === 'failed' ? <ErrorBanner title={text.failure} body={operation.error?.code ?? 'operation_failed'} /> : null}
    <details><summary>{text.details}</summary><p>{operation.operation_id}</p><p>{operation.target.target_kind === 'model' ? operation.target.model_id : operation.target.target_kind}</p></details>
    {!terminal && operation.state !== 'cancelling' ? operation.cancellable ? <Button disabled={stale} busy={busy} onClick={() => { void cancel(); }}>{text.cancel}</Button> : <p>{text.cancelUnsupported}</p> : null}
    {failed ? <ErrorBanner title={text.cancelFailed} body={text.refresh} /> : null}
  </article>;
}
