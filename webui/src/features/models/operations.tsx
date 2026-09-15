// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import type { Operation, WebUiSnapshot } from '../../api/types';
import { Button, ProgressBar } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { allowed, bytes, current, terminal, canLoad } from './policy';

export function LibraryOperations({
  state,
  locale,
  busy,
  downloadPending,
  onCancel,
  onRetry,
  onCapacity,
}: {
  state: WebUiSnapshot;
  locale: Locale;
  busy: boolean;
  downloadPending: boolean;
  onCancel: (operation: Operation) => void;
  onRetry: (operation: Operation) => void;
  onCapacity: (id: string) => void;
}): React.JSX.Element {
  const operations = [...state.operations.values()]
    .filter((op) => ['download', 'model_load', 'model_unload', 'model_removal', 'catalog_refresh'].includes(op.kind))
    .sort((a, b) => Number(terminal(a)) - Number(terminal(b)) || b.created_at.localeCompare(a.created_at));
  if (!operations.length && !state.pendingReconciliations.size) return <></>;
  return (
    <section className="models-operations" aria-label={t(locale, 'models.library.operations')}>
      <h2>{t(locale, 'models.library.operations')}</h2>
      {state.pendingReconciliations.size ? (
        <p role="status" data-testid="models-pending">
          {t(locale, 'models.library.pending')}
        </p>
      ) : null}
      <ul className="ds-list">
        {operations.slice(0, 64).map((op) => (
          <li key={op.operation_id} data-testid="models-operation">
            <div>
              <strong>{operationLabel(op, state)}</strong>
              <p>
                <code>{op.operation_id}</code>
                {op.target.target_kind === 'model' ? (
                  <span>
                    {' '}
                    · <code>{op.target.model_id}</code>
                  </span>
                ) : null}{' '}
                · {t(locale, 'models.library.operation_state')}: {op.state}
              </p>
              {op.kind === 'download' ? (
                <ProgressBar
                  label={t(locale, 'models.library.progress')}
                  value={
                    !op.progress.indeterminate && op.progress.total_bytes !== null && op.progress.total_bytes > 0
                      ? (op.progress.completed_bytes / op.progress.total_bytes) * 100
                      : undefined
                  }
                  detail={`${bytes(op.progress.completed_bytes, locale)} / ${bytes(op.progress.total_bytes, locale)}`}
                />
              ) : null}
              {op.error ? (
                <p role="alert">
                  {op.error.code}: {op.error.message}
                </p>
              ) : null}
              {op.result && 'lifecycle' in op.result ? (
                <p>
                  {op.result.lifecycle.state} · {t(locale, 'models.library.worker')}:{' '}
                  {t(locale, op.result.lifecycle.worker_exit_observed ? 'models.library.yes' : 'models.library.no')}
                </p>
              ) : null}
              {op.cancel_reason ? <p>{op.cancel_reason}</p> : null}
              <div className="button-row">
                {op.kind === 'model_load' &&
                op.state === 'failed' &&
                op.error?.code === 'conflict' &&
                op.target.target_kind === 'model' ? (
                  <Button
                    disabled={
                      busy ||
                      !state.catalog.some((entry) => entry.identity.id === modelTarget(op) && canLoad(state, entry))
                    }
                    onClick={() => onCapacity(modelTarget(op))}
                  >
                    {t(locale, 'models.library.capacity')}
                  </Button>
                ) : null}
                {op.kind === 'download' && !terminal(op) ? (
                  <Button
                    disabled={busy || !current(state) || !op.cancellable || op.state === 'cancelling'}
                    onClick={() => onCancel(op)}
                  >
                    {t(locale, 'models.library.cancel_download')}
                  </Button>
                ) : null}
                {op.kind === 'download' && (op.state === 'failed' || op.state === 'cancelled') ? (
                  <Button disabled={busy || downloadPending || !allowed(state, 'download')} onClick={() => onRetry(op)}>
                    {t(locale, 'models.library.retry_download')}
                  </Button>
                ) : null}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function modelTarget(op: Operation): string {
  return op.target.target_kind === 'model' ? op.target.model_id : '';
}

function operationLabel(op: Operation, state: WebUiSnapshot): string {
  const target = op.target;
  if (target.target_kind === 'download') return target.repo_id;
  if (target.target_kind === 'model')
    return (
      state.catalog.find((entry) => entry.identity.id === target.model_id)?.identity.display_name ?? target.model_id
    );
  return op.kind;
}
