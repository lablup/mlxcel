// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// The model library's columns: catalog entries and in-flight downloads share one table.
import React from 'react';
import type { CatalogEntry, Operation, WebUiSnapshot } from '../../api/types';
import {
  Badge,
  Button,
  IconButton,
  ProgressBar,
  ROW_PRIMARY_CLASS,
  StatusBadge,
  type DataTableColumn,
} from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { lifecycleLabel } from '../../provider-surfaces';
import type { Confirmation } from './dialogs';
import type { ModelAction } from './inspector';
import { downloadBadgeState, downloadStateLabel, isolate, taskLabel } from './labels';
import {
  allowed,
  bytes,
  canChat,
  canDelete,
  canLoad,
  canUnload,
  capacityFailed,
  compareEntries,
  current,
  entryTasks,
  quantizationValue,
  terminal,
  type InventorySort,
} from './policy';
import type { RegisterRowControl } from './row-focus';

/** A library row: a catalog entry, or a download that has no entry yet (in flight, failed or cancelled). */
export type LibraryRow = { kind: 'model'; entry: CatalogEntry } | { kind: 'download'; op: Operation };

export function downloadRepo(op: Operation): string {
  return op.target.target_kind === 'download' ? op.target.repo_id : op.operation_id;
}

/** The page values and handlers the column renderers close over. */
export type LibraryColumnsContext = {
  state: WebUiSnapshot;
  locale: Locale;
  sort: InventorySort;
  busy: boolean;
  readOnly: boolean;
  downloadPending: boolean;
  register: RegisterRowControl;
  openChat: (entry: CatalogEntry) => void;
  openAction: (kind: ModelAction, entry: CatalogEntry, origin: 'row') => void;
  openCapacity: (entry: CatalogEntry) => void;
  inspect: (entry: CatalogEntry) => void;
  setConfirmation: (confirmation: Confirmation) => void;
  retryDownload: (op: Operation) => void;
};

// Row controls act on their own row only; the row-activation handler must not also open it.
const own = (run: () => void) => (event: React.MouseEvent<HTMLButtonElement>): void => {
  event.stopPropagation();
  run();
};

export function libraryColumns(context: LibraryColumnsContext): DataTableColumn<LibraryRow>[] {
  const {
    state,
    locale,
    sort,
    busy,
    readOnly,
    downloadPending,
    register,
    openChat,
    openAction,
    openCapacity,
    inspect,
    setConfirmation,
    retryDownload,
  } = context;
  const rowComparator = (a: LibraryRow, b: LibraryRow): number => {
    if (a.kind === 'download' || b.kind === 'download') return a.kind === b.kind ? 0 : a.kind === 'download' ? -1 : 1;
    return compareEntries(sort)(a.entry, b.entry);
  };
  // alpha.19 sorts the page itself even in controlled mode, so every sortable column gets the
  // same pinned comparator the pagination used rather than a bare value accessor.
  const sortable: Pick<DataTableColumn<LibraryRow>, 'sortable' | 'sortComparator'> = { sortable: true, sortComparator: rowComparator };
  const modelActions = (entry: CatalogEntry): React.ReactNode => {
    // Isolated: a name holding bidi controls (U+202E) must not reorder the words around it.
    const name = isolate(entry.identity.display_name);
    return (
      <div className="models-row-actions">
        {canChat(state, entry) ? (
          <Button tone="primary" data-testid="models-row-chat" aria-label={t(locale, 'models.library.chat_named', { name })} disabled={busy} onClick={own(() => openChat(entry))} ref={register(entry.identity.id, 'chat')}>
            {t(locale, 'models.library.chat')}
          </Button>
        ) : null}
        {/* Unload follows the lifecycle, not chat: a Ready embedding, rerank or transcription model is
            released from its row too. Every other state keeps Load in place, disabled until the entry
            can load, so a loading, draining or unloading row cannot submit twice. */}
        {readOnly ? null : entry.lifecycle.state === 'ready' ? (
          <Button key="unload" data-testid="models-row-unload" aria-label={t(locale, 'models.library.unload_named', { name })} disabled={busy || !canUnload(state, entry)} onClick={own(() => openAction('unload', entry, 'row'))} ref={register(entry.identity.id, 'unload')}>
            {t(locale, 'models.unload')}
          </Button>
        ) : (
          <Button key="load" data-testid="models-row-load" aria-label={t(locale, 'models.library.load_named', { name })} disabled={busy || !canLoad(state, entry)} onClick={own(() => openAction('load', entry, 'row'))} ref={register(entry.identity.id, 'load')}>
            {t(locale, 'models.load')}
          </Button>
        )}
        {!readOnly && capacityFailed(state, entry.identity.id) ? (
          <Button data-testid="models-row-capacity" aria-label={t(locale, 'models.library.load_evict_named', { name })} disabled={busy || !canLoad(state, entry)} onClick={own(() => openCapacity(entry))}>
            {t(locale, 'models.library.load_evict')}
          </Button>
        ) : null}
        <IconButton
          icon="info"
          label={t(locale, 'models.library.inspect', { name })}
          className={ROW_PRIMARY_CLASS}
          onClick={own(() => inspect(entry))}
          ref={register(entry.identity.id, 'inspect')}
        />
        {!readOnly && entry.identity.source === 'cache' ? (
          <IconButton
            icon="trash"
            tone="danger"
            label={t(locale, 'models.library.delete_named', { name })}
            data-testid="models-row-delete"
            disabled={busy || !canDelete(state, entry)}
            onClick={own(() => openAction('delete', entry, 'row'))}
          />
        ) : null}
      </div>
    );
  };
  // A download row draws its state and progress in two places: its State and Size columns, and a
  // fallback under its name for when the narrowing list hides those columns (it has no inspector).
  const downloadBadge = (op: Operation): React.ReactNode => (
    <StatusBadge state={downloadBadgeState(op.state)}>{downloadStateLabel(locale, op.state)}</StatusBadge>
  );
  const downloadProgress = (op: Operation): React.ReactNode => (
    <ProgressBar
      label={t(locale, 'models.library.progress')}
      value={
        !op.progress.indeterminate && op.progress.total_bytes !== null && op.progress.total_bytes > 0
          ? (op.progress.completed_bytes / op.progress.total_bytes) * 100
          : undefined
      }
      detail={`${bytes(op.progress.completed_bytes, locale)} / ${bytes(op.progress.total_bytes, locale)}`}
    />
  );
  const downloadActions = (op: Operation): React.ReactNode => (
    <div className="models-row-actions">
      {!terminal(op) ? (
        <Button
          disabled={busy || !current(state) || !op.cancellable || op.state === 'cancelling'}
          onClick={() => setConfirmation({ kind: 'cancel', operation: op, instance: state.serverInstanceId })}
        >
          {t(locale, 'models.library.cancel_download')}
        </Button>
      ) : (
        <Button disabled={busy || downloadPending || !allowed(state, 'download')} onClick={() => retryDownload(op)}>
          {t(locale, 'models.library.retry_download')}
        </Button>
      )}
    </div>
  );
  const unknown = t(locale, 'models.library.unknown');
  return [
    {
      id: 'name',
      header: t(locale, 'models.library.name'),
      noResize: true,
      className: 'models-col-name',
      ...sortable,
      render: (row) => {
        if (row.kind === 'download') {
          const repo = downloadRepo(row.op);
          return (
            <span className="models-name" data-testid="models-operation">
              <span className="truncate" title={repo}>{repo}</span>
              {/* Shown only once the State and Size columns have given way to a narrow list. */}
              <span className="models-name-state">{downloadBadge(row.op)}</span>
              <span className="models-name-progress">{downloadProgress(row.op)}</span>
              {row.op.error ? <small className="models-row-note">{row.op.error.message}</small> : null}
            </span>
          );
        }
        const entry = row.entry;
        const flag = !entry.complete
          ? 'models.library.flag_incomplete'
          : !entry.supported || !entry.metadata.support.runnable_on_backend
            ? 'models.library.flag_unsupported'
            : null;
        return (
          <span className="models-name">
            <span className="truncate" title={entry.identity.display_name}>{entry.identity.display_name}</span>
            {flag ? <Badge tone="warning">{t(locale, flag)}</Badge> : null}
            {/* Shown only once the State column has given way to a narrow list. */}
            <span className="models-name-state">
              <StatusBadge state={entry.lifecycle.state}>{lifecycleLabel(locale, entry.lifecycle.state)}</StatusBadge>
            </span>
          </span>
        );
      },
    },
    {
      id: 'size',
      header: t(locale, 'models.library.size'),
      noResize: true,
      align: 'right',
      className: 'models-col-size',
      ...sortable,
      render: (row) => (row.kind === 'download' ? downloadProgress(row.op) : bytes(row.entry.metadata.disk_bytes, locale)),
    },
    {
      id: 'quantization',
      header: t(locale, 'models.library.quantization'),
      noResize: true,
      className: 'models-col-quantization',
      ...sortable,
      render: (row) => (row.kind === 'download' ? null : quantizationValue(row.entry) ?? unknown),
    },
    {
      id: 'tasks',
      header: t(locale, 'models.library.tasks'),
      noResize: true,
      className: 'models-col-tasks',
      render: (row) =>
        row.kind === 'download' ? null : (
          <span className="models-tasks">
            {entryTasks(row.entry).map((task) => (
              <Badge key={task}>{taskLabel(locale, task)}</Badge>
            ))}
          </span>
        ),
    },
    {
      id: 'state',
      header: t(locale, 'models.library.status'),
      noResize: true,
      className: 'models-col-state',
      ...sortable,
      render: (row) =>
        row.kind === 'download' ? downloadBadge(row.op) : <StatusBadge state={row.entry.lifecycle.state}>{lifecycleLabel(locale, row.entry.lifecycle.state)}</StatusBadge>,
    },
    {
      id: 'actions',
      header: t(locale, 'models.library.actions'),
      noResize: true,
      align: 'right',
      className: 'models-col-actions',
      render: (row) => (row.kind === 'download' ? downloadActions(row.op) : modelActions(row.entry)),
    },
  ];
}
