// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { CatalogEntry, LoadProfile, Operation } from '../../api/types';
import {
  Button,
  DataTable,
  EmptyState,
  ErrorBanner,
  LoadingStatus,
  PageHeader,
  type SortDirection,
} from '../../design-system/primitives';
import { t, testId, type Locale } from '../../i18n/catalog';
import { useWebUi, useWebUiActions } from '../../state';
import { loadProfileFor, useLoadProfile } from '../settings/load-profiles';
import { libraryColumns, type LibraryRow } from './columns';
import { AddModel, ConfirmAction, type Confirmation } from './dialogs';
import { consumeInspectorRequest, useInspectorRequest, useWideInspector } from './inspect-request';
import { ModelInspector, type ModelAction } from './inspector';
import { setLibraryFilter, setLibrarySort, useLibraryView } from './library-view';
import { submitLoad } from './load-action';
import {
  allowed,
  canChat,
  canDelete,
  canLoad,
  canUnload,
  current,
  DEFAULT_SORT,
  errorMessage,
  evictionCandidates,
  inventory,
  terminal,
  visibleDownloads,
  type InventoryFilter,
  type SortColumn,
} from './policy';
import { RootsDialog } from './roots';
import { AFTER_LOAD, AFTER_UNLOAD, useRowFocus } from './row-focus';
import { LibraryToolbar } from './toolbar';
import './models.css';

const PAGE_SIZE = 25;
/** Where an action started, so focus can follow it once the lifecycle settles. */
type Origin = 'row' | 'inspector';

export function ModelsLibrary({ locale }: { locale: Locale }): React.JSX.Element {
  const state = useWebUi();
  const actions = useWebUiActions();
  const wide = useWideInspector();
  const inspectRequest = useInspectorRequest();
  // Search, filters and sort outlive this component: the router remounts it on every route change.
  const { filter, sort } = useLibraryView(state.serverInstanceId);
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
  }, [inspectRequest, wide]);
  // The drawer unmounts at the wide breakpoint; forget it was open so narrowing the window
  // again does not reopen it over the list without a request.
  useEffect(() => {
    if (wide) setDrawerOpen(false);
  }, [wide]);

  const { rowFocus, register } = useRowFocus({ rows, pageSize: PAGE_SIZE, visiblePage, setPage });

  const set = (patch: Partial<InventoryFilter>): void => {
    setLibraryFilter(state.serverInstanceId, { ...filter, ...patch });
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
    // A load that does not start or fails ends the row's focus intent: armed, it would pull
    // focus into that row whenever the row next became ready by any other path.
    const dropFocus = (): void => {
      rowFocus.current = null;
    };
    const returnFocus = (): void => {
      if (rowFocus.current) rowFocus.current = { id: rowFocus.current.id, want: ['load'], from: null, once: true };
    };
    if (!canLoad(state, entry)) {
      dropFocus();
      setError(t(locale, 'models.library.stale'));
      return;
    }
    if (
      evictionTarget &&
      !evictionCandidates(state, entry.identity.id).some(
        (candidate) => candidate.identity.id === evictionTarget.id && candidate.identity.revision === evictionTarget.revision,
      )
    ) {
      dropFocus();
      setError(t(locale, 'models.library.stale'));
      return;
    }
    void execute(() =>
      submitLoad(
        actions,
        state,
        entry,
        profile,
        () => setConfirmation({ kind: 'capacity', entry, instance: state.serverInstanceId }),
        evictionTarget,
      ),
      returnFocus,
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
          () => {
            if (rowFocus.current) rowFocus.current = { id: rowFocus.current.id, want: ['unload'], from: null, once: true };
          },
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
  const columns = libraryColumns({
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
  });
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
      {/* Below 1100 px the open inspector drawer is modal: everything behind it is inert. The
          wrapper adds no box (display: contents), so the page keeps its .screen-stack rhythm. */}
      <div className="models-page" inert={!wide && drawerOpen && selected !== undefined}>
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
        <LibraryToolbar locale={locale} catalog={state.catalog} filter={filter} onChange={set} />
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
                setLibrarySort(state.serverInstanceId, column && direction ? { column: column as SortColumn, direction: direction as SortDirection } : DEFAULT_SORT);
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
          onClose={() => {
            // Cancelled: a capacity confirmation after a row Load conflict leaves nothing to follow.
            rowFocus.current = null;
            setConfirmation(null);
          }}
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
