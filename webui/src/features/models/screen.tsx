// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { CatalogEntry, LoadProfile, Operation } from '../../api/types';
import { WebUiHttpError } from '../../api/client';
import {
  Badge,
  Button,
  DataTable,
  EmptyState,
  ErrorBanner,
  Field,
  IconButton,
  LoadingStatus,
  PageHeader,
  ProgressBar,
  ROW_PRIMARY_CLASS,
  Select,
  StatusBadge,
  type DataTableColumn,
  type SortDirection,
} from '../../design-system/primitives';
import { t, testId, type Locale } from '../../i18n/catalog';
import { lifecycleLabel } from '../../provider-surfaces';
import { useWebUi, useWebUiActions } from '../../state';
import { loadProfileFor, useLoadProfile } from '../settings/load-profiles';
import { AddModel, ConfirmAction, type Confirmation } from './dialogs';
import { consumeInspectorRequest, useInspectorRequest, useWideInspector } from './inspect-request';
import { ModelInspector, type ModelAction } from './inspector';
import { downloadBadgeState, downloadStateLabel, sourceLabel, taskLabel } from './labels';
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
  DEFAULT_SORT,
  entryTasks,
  errorMessage,
  evictionCandidates,
  inventory,
  quantizationValue,
  terminal,
  visibleDownloads,
  type InventoryFilter,
  type InventorySort,
  type SortColumn,
} from './policy';
import { RootsDialog } from './roots';
import './models.css';

const PAGE_SIZE = 25;
const LIFECYCLE_FILTERS = ['unloaded', 'loading', 'ready', 'draining', 'unloading', 'failed'] as const;

/** A library row: a catalog entry, or a download that has no entry yet (in flight, failed or cancelled). */
type LibraryRow = { kind: 'model'; entry: CatalogEntry } | { kind: 'download'; op: Operation };
/** Where an action started, so focus can follow it once the lifecycle settles. */
type Origin = 'row' | 'inspector';
/** A row control that focus can follow an action to. */
type RowTarget = 'load' | 'chat' | 'unload';
/**
 * After a row action, the row controls that should take focus once one can, in order of
 * preference: the first that exists and is enabled wins. `from` is the control the action started
 * from, which the action disables.
 */
type RowFocus = { id: string; want: readonly RowTarget[]; from: Element | null };
/** After a Load: Use in Chat for a chat model; Unload for one without chat (embedding, rerank, transcription). */
const AFTER_LOAD: readonly RowTarget[] = ['chat', 'unload'];
const AFTER_UNLOAD: readonly RowTarget[] = ['load'];

function downloadRepo(op: Operation): string {
  return op.target.target_kind === 'download' ? op.target.repo_id : op.operation_id;
}

export function ModelsLibrary({ locale }: { locale: Locale }): React.JSX.Element {
  const state = useWebUi();
  const actions = useWebUiActions();
  const wide = useWideInspector();
  const inspectRequest = useInspectorRequest();
  const [filter, setFilter] = useState<InventoryFilter>({ query: '', source: '', task: '', status: '' });
  const [sort, setSort] = useState<InventorySort>(DEFAULT_SORT);
  const [page, setPage] = useState(0);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const confirmationOrigin = useRef<Origin>('inspector');
  const [add, setAdd] = useState<{ repo: string; revision: string } | null>(null);
  const [rootsOpen, setRootsOpen] = useState(false);
  // Below 1100 px the inspector is a modal drawer: it opens on an explicit Inspect (or a request
  // from the toolbar or palette), not merely because a selection is remembered.
  const [drawerOpen, setDrawerOpen] = useState(false);
  const rowControls = useRef(new Map<string, HTMLButtonElement>());
  const rowFocus = useRef<RowFocus | null>(null);
  const selected = state.catalog.find((entry) => entry.identity.id === state.selectedModelId);
  const { profile: selectedProfile } = useLoadProfile(selected?.identity.id ?? null);
  // Runtime and operation events re-render the page without touching the catalog; skip the re-sort.
  const rows = useMemo(() => inventory(state.catalog, filter, sort), [state.catalog, filter, sort]);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const visiblePage = Math.min(page, pages - 1);
  const readOnly = state.bootstrap?.server.mode === 'single_model';
  const downloads = visibleDownloads(state);
  const downloadPending =
    [...state.operations.values()].some((op) => op.kind === 'download' && !terminal(op)) ||
    [...state.pendingReconciliations.values()].some((item) => item.kind === 'download');
  const rescanPending =
    [...state.operations.values()].some((op) => op.kind === 'catalog_refresh' && !terminal(op)) ||
    [...state.pendingReconciliations.values()].some((item) => item.kind === 'catalog-refresh');

  useEffect(() => {
    if (consumeInspectorRequest() && !wide) setDrawerOpen(true);
  }, [inspectRequest]);

  // Focus follows a row action to the control that replaces the one used, but only while the
  // user has not moved on: focus is still on that row's Inspect button (where it waits) or nowhere.
  useEffect(() => {
    const pending = rowFocus.current;
    if (!pending) return;
    const holder = rowControls.current.get(`${pending.id}:inspect`);
    const target = pending.want
      .map((slot) => rowControls.current.get(`${pending.id}:${slot}`))
      .find((control) => control !== undefined && !control.disabled);
    const active = document.activeElement;
    const waiting =
      active === null || active === document.body || active === holder || active === pending.from || !!active.closest('[data-dialog-focus-fallback]');
    if (!waiting) {
      rowFocus.current = null;
      return;
    }
    if (target) {
      target.focus();
      rowFocus.current = null;
    } else if (holder && active !== holder && !document.querySelector('dialog[open]')) holder.focus();
  });

  const set = (patch: Partial<InventoryFilter>): void => {
    setFilter({ ...filter, ...patch });
    setPage(0);
  };
  const execute = async (run: () => Promise<void>, onFailure?: (failure: unknown) => void): Promise<boolean> => {
    if (busyRef.current || !current(state)) return false;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      await run();
      return true;
    } catch (failure) {
      setError(errorMessage(failure, locale));
      onFailure?.(failure);
      return false;
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  };
  const load = (
    entry: CatalogEntry,
    evictionTarget?: { id: string; revision: number },
  ): void => {
    // The entry's own profile, read now: a row loads entries other than the selected one.
    const profile: LoadProfile = loadProfileFor(entry.identity.id);
    if (!canLoad(state, entry)) {
      setError(t(locale, 'models.library.stale'));
      return;
    }
    if (
      evictionTarget &&
      !evictionCandidates(state, entry.identity.id).some(
        (candidate) => candidate.identity.id === evictionTarget.id && candidate.identity.revision === evictionTarget.revision,
      )
    ) {
      setError(t(locale, 'models.library.stale'));
      return;
    }
    void execute(
      () =>
        actions.loadModel({
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
        }),
      (failure) => {
        if (failure instanceof WebUiHttpError && failure.envelope?.error.code === 'conflict')
          setConfirmation({ kind: 'capacity', entry, instance: state.serverInstanceId });
      },
    );
  };
  // The one path from a row or the inspector into a lifecycle action: Load runs at once (it
  // never evicts), Unload and Delete always go through their confirmation.
  const openAction = (kind: ModelAction, entry: CatalogEntry, origin: Origin): void => {
    confirmationOrigin.current = origin;
    if (kind === 'load') {
      if (origin === 'row') rowFocus.current = { id: entry.identity.id, want: AFTER_LOAD, from: document.activeElement };
      load(entry);
    } else setConfirmation({ kind, entry, instance: state.serverInstanceId });
  };
  const openCapacity = (entry: CatalogEntry): void => {
    confirmationOrigin.current = 'row';
    if (canLoad(state, entry)) setConfirmation({ kind: 'capacity', entry, instance: state.serverInstanceId });
  };
  const confirm = (evictionTarget?: string, evictionRevision?: number): void => {
    const value = confirmation;
    if (!value || busyRef.current) return;
    if (value.instance !== state.serverInstanceId) {
      setError(t(locale, 'models.library.stale'));
      setConfirmation(null);
      return;
    }
    if (value.kind === 'cancel') {
      const op = state.operations.get(value.operation.operation_id);
      if (!op || !op.cancellable || terminal(op) || op.state === 'cancelling') {
        setConfirmation(null);
        return;
      }
      void execute(() => actions.cancelOperation(op.operation_id));
    } else {
      const entry = state.catalog.find((candidate) => candidate.identity.id === value.entry.identity.id);
      if (!entry || entry.identity.revision !== value.entry.identity.revision) {
        setError(t(locale, 'models.library.stale'));
        setConfirmation(null);
        return;
      }
      if (value.kind === 'capacity') {
        if (!evictionCandidates(state, entry.identity.id).some((candidate) => candidate.identity.id === evictionTarget && candidate.identity.revision === evictionRevision)) {
          setError(t(locale, 'models.library.stale'));
          return;
        }
        if (confirmationOrigin.current === 'row') rowFocus.current = { id: entry.identity.id, want: AFTER_LOAD, from: null };
        load(
          entry,
          evictionTarget && evictionRevision !== undefined
            ? { id: evictionTarget, revision: evictionRevision }
            : undefined,
        );
      } else if (value.kind === 'unload' && canUnload(state, entry)) {
        if (confirmationOrigin.current === 'row') rowFocus.current = { id: entry.identity.id, want: AFTER_UNLOAD, from: null };
        void execute(() =>
          actions.unloadModel({
            action: 'unload',
            model_id: entry.identity.id,
            expected_revision: entry.identity.revision,
            idempotency_key: crypto.randomUUID(),
          }),
        );
      } else if (value.kind === 'delete' && canDelete(state, entry))
        void execute(() =>
          actions.removeModel({
            model_id: entry.identity.id,
            expected_revision: entry.identity.revision,
            idempotency_key: crypto.randomUUID(),
          }),
        );
      else setError(t(locale, 'models.library.stale'));
    }
    setConfirmation(null);
    // A confirmation opened from the inspector drawer returns focus to the drawer, not to the
    // page heading behind it, when the control that opened it is now disabled.
    if (confirmationOrigin.current === 'inspector' && !wide && drawerOpen)
      window.setTimeout(() => {
        const panel = document.querySelector<HTMLElement>('[data-testid="models-inspector-drawer"]');
        if (panel && !panel.contains(document.activeElement)) panel.querySelector<HTMLElement>('.drawer__close-btn')?.focus();
      }, 0);
  };
  const retryDownload = (op: Operation): void => {
    if (op.target.target_kind === 'download') setAdd({ repo: op.target.repo_id, revision: op.target.revision ?? '' });
  };
  const inspect = (entry: CatalogEntry): void => {
    // Re-selecting the already-open row is a no-op: selectModel always aborts the live stream,
    // forces a full snapshot refetch and clears runtimeHistory, even for the same id.
    if (entry.identity.id !== state.selectedModelId) actions.selectModel(entry.identity.id);
    // At 1100 px and wider the pane follows the selection; the drawer state is for narrow windows only.
    if (!wide) setDrawerOpen(true);
  };
  const openChat = (entry: CatalogEntry): void => {
    if (!canChat(state, entry)) return;
    if (entry.identity.id !== state.selectedModelId) actions.selectModel(entry.identity.id);
    window.location.hash = 'chat';
  };
  const register = (id: string, slot: 'inspect' | RowTarget) => (element: HTMLButtonElement | null): void => {
    if (element) rowControls.current.set(`${id}:${slot}`, element);
    else rowControls.current.delete(`${id}:${slot}`);
  };
  // Row controls act on their own row only; the row-activation handler must not also open it.
  const own = (run: () => void) => (event: React.MouseEvent<HTMLButtonElement>): void => {
    event.stopPropagation();
    run();
  };

  const rowComparator = (a: LibraryRow, b: LibraryRow): number => {
    if (a.kind === 'download' || b.kind === 'download') return a.kind === b.kind ? 0 : a.kind === 'download' ? -1 : 1;
    return compareEntries(sort)(a.entry, b.entry);
  };
  // alpha.19 sorts the page itself even in controlled mode, so every sortable column gets the
  // same pinned comparator the pagination used rather than a bare value accessor.
  const sortable: Pick<DataTableColumn<LibraryRow>, 'sortable' | 'sortComparator'> = { sortable: true, sortComparator: rowComparator };
  const modelActions = (entry: CatalogEntry): React.ReactNode => {
    const name = entry.identity.display_name;
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
          icon="details"
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
  const columns: DataTableColumn<LibraryRow>[] = [
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
  const pageRows: LibraryRow[] = [
    ...downloads.map((op) => ({ kind: 'download' as const, op })),
    ...rows.slice(visiblePage * PAGE_SIZE, (visiblePage + 1) * PAGE_SIZE).map((entry) => ({ kind: 'model' as const, entry })),
  ];
  const inspectorProps = {
    profile: selectedProfile,
    state,
    locale,
    busy,
    onAction: (kind: ModelAction, entry: CatalogEntry) => openAction(kind, entry, 'inspector'),
    onChat: openChat,
  };
  return (
    <div className="screen-stack models-library" data-testid="models-library">
      <PageHeader
        title={t(locale, 'models.title')}
        titleTestId={testId('models.title')}
        description={t(locale, 'models.library.subtitle')}
        actions={
          <>
            <Button
              tone="primary"
              onClick={() => setAdd({ repo: '', revision: '' })}
              disabled={busy || downloadPending || !allowed(state, 'download')}
              data-testid="models-add"
            >
              {t(locale, 'models.library.add')}
            </Button>
            <Button
              disabled={busy || rescanPending || !current(state) || readOnly}
              onClick={() => {
                void execute(() => actions.refreshCatalog(crypto.randomUUID()));
              }}
              data-testid="models-rescan"
            >
              {t(locale, 'models.library.rescan')}
            </Button>
            <Button
              onClick={() => {
                void actions.refresh();
              }}
              disabled={busy}
            >
              {t(locale, 'models.library.refresh')}
            </Button>
          </>
        }
        // A stale snapshot takes precedence over a local action error: actions are refused
        // until it is refreshed. Retry (shown only with an error) clears it and refreshes.
        error={!current(state) ? t(locale, 'models.library.stale') : error === null ? null : t(locale, 'models.library.error')}
        errorDetail={!current(state) ? (state.error?.message ?? t(locale, 'models.library.waiting')) : error}
        errorTestId={current(state) ? 'models-action-error' : 'connection-error-title'}
        onRetry={() => {
          setError(null);
          void actions.refresh();
        }}
        retryLabel={t(locale, 'models.library.refresh')}
      />
      {readOnly ? (
        <ErrorBanner tone="info" title={t(locale, 'models.library.single')} body="mlxcel-server --webui" testId="models-read-only" />
      ) : null}
      <div className="models-toolbar">
        <Field label={t(locale, 'models.library.search')} value={filter.query} onChange={(query) => set({ query })} testId="models-search" />
        <Select
          locale={locale}
          label={t(locale, 'models.library.source')}
          value={filter.source}
          onChange={(source) => set({ source })}
          options={[
            { value: '', label: t(locale, 'models.library.all') },
            ...[...new Set(state.catalog.map((entry) => entry.identity.source))].sort().map((value) => ({ value, label: sourceLabel(locale, value) })),
          ]}
        />
        <Select
          locale={locale}
          label={t(locale, 'models.library.task')}
          value={filter.task}
          onChange={(task) => set({ task })}
          options={[
            { value: '', label: t(locale, 'models.library.all') },
            ...[...new Set(state.catalog.flatMap((entry) => entry.capabilities.map((cap) => cap.task)))]
              .sort()
              .map((value) => ({ value, label: taskLabel(locale, value) })),
          ]}
        />
        <Select
          locale={locale}
          label={t(locale, 'models.library.status')}
          value={filter.status}
          onChange={(status) => set({ status })}
          options={[
            { value: '', label: t(locale, 'models.library.all') },
            ...LIFECYCLE_FILTERS.map((value) => ({ value, label: lifecycleLabel(locale, value) })),
          ]}
        />
      </div>
      {state.pendingReconciliations.size ? (
        <p role="status" data-testid="models-pending" className="models-pending">
          {t(locale, 'models.library.pending')}
        </p>
      ) : null}
      <div className="models-layout">
        <section className="models-list">
          <DataTable
            columns={columns}
            rows={pageRows}
            getRowKey={(row) => (row.kind === 'download' ? `op:${row.op.operation_id}` : row.entry.identity.id)}
            ariaLabel={t(locale, 'models.title')}
            testId="models-table"
            className="models-table"
            activateRowPrimary
            overflowRegionLabel={t(locale, 'models.title')}
            sortColumnId={sort.column}
            sortDirection={sort.direction}
            onSortChange={(column, direction) => {
              // The third header activation clears the sort in alpha.19; the library always has
              // one, so a cleared sort returns to the default name order.
              setSort(column && direction ? { column: column as SortColumn, direction: direction as SortDirection } : DEFAULT_SORT);
              setPage(0);
            }}
            rowClassName={(row) =>
              row.kind === 'download' ? 'models-download-row' : row.entry.identity.id === state.selectedModelId ? 'models-selected' : undefined
            }
            loading={state.catalogSequence === null}
            loadingState={<LoadingStatus label={t(locale, 'models.library.waiting')} rows={8} />}
            emptyState={
              <EmptyState
                title={t(locale, state.catalog.length === 0 ? 'models.empty.title' : 'models.library.filtered')}
                body={t(locale, state.catalog.length === 0 ? 'models.empty.body' : 'models.library.filtered_body')}
                testId="models-empty"
                action={
                  state.catalog.length === 0 ? (
                    <Button onClick={() => setRootsOpen(true)} data-testid="models-roots">
                      {t(locale, 'models.library.roots')}
                    </Button>
                  ) : undefined
                }
              />
            }
          />
          <nav className="models-pagination" aria-label={t(locale, 'models.title')}>
            <Button disabled={visiblePage === 0} onClick={() => setPage(visiblePage - 1)}>
              {t(locale, 'models.library.previous')}
            </Button>
            <span role="status">
              {t(locale, 'models.library.page', {
                page: String(visiblePage + 1),
                pages: String(pages),
                count: String(rows.length),
              })}
            </span>
            <Button disabled={visiblePage >= pages - 1} onClick={() => setPage(visiblePage + 1)}>
              {t(locale, 'models.library.next')}
            </Button>
          </nav>
        </section>
        {selected && wide ? <ModelInspector variant="pane" entry={selected} {...inspectorProps} /> : null}
      </div>
      {selected && !wide ? (
        <ModelInspector variant="drawer" entry={selected} open={drawerOpen} onClose={() => setDrawerOpen(false)} {...inspectorProps} />
      ) : null}
      {confirmation ? (
        <ConfirmAction
          key={confirmation.kind}
          value={confirmation}
          state={state}
          locale={locale}
          busy={busy}
          onClose={() => setConfirmation(null)}
          onConfirm={confirm}
        />
      ) : null}
      {rootsOpen ? <RootsDialog state={state} locale={locale} onClose={() => setRootsOpen(false)} /> : null}
      {add ? (
        <AddModel
          locale={locale}
          state={state}
          initial={add}
          error={error}
          busy={busy || downloadPending || !allowed(state, 'download')}
          onClose={() => setAdd(null)}
          onDownload={(repo, revision) => {
            if (!allowed(state, 'download') || downloadPending) return;
            void execute(() =>
              actions.downloadModel({
                repo_id: repo,
                revision: revision || null,
                idempotency_key: crypto.randomUUID(),
              }),
            ).then((accepted) => {
              if (accepted) setAdd(null);
            });
          }}
        />
      ) : null}
    </div>
  );
}
