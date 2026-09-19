// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useRef, useState } from 'react';
import type { CatalogEntry, LoadProfile, Operation } from '../../api/types';
import { WebUiHttpError } from '../../api/client';
import {
  Badge,
  Button,
  DataTable,
  EmptyState,
  ErrorBanner,
  Field,
  LoadingStatus,
  ROW_PRIMARY_CLASS,
  Select,
  StatusBadge,
  type DataTableColumn,
} from '../../design-system/primitives';
import { t, testId, type Locale } from '../../i18n/catalog';
import { connectedDetail, lifecycleLabel } from '../../provider-surfaces';
import { useWebUi, useWebUiActions } from '../../state';
import { useLoadProfile } from '../settings/load-profiles';
import { AddModel, ConfirmAction, type Confirmation } from './dialogs';
import { ModelInspector } from './inspector';
import { LibraryOperations } from './operations';
import {
  allowed,
  canChat,
  canDelete,
  canLoad,
  canUnload,
  current,
  errorMessage,
  evictionCandidates,
  inventory,
  terminal,
  type InventoryFilter,
} from './policy';
import './models.css';

const PAGE_SIZE = 25;
export function ModelsLibrary({ locale }: { locale: Locale }): React.JSX.Element {
  const state = useWebUi();
  const actions = useWebUiActions();
  const [filter, setFilter] = useState<InventoryFilter>({ query: '', source: '', task: '', status: '', sort: 'name' });
  const [page, setPage] = useState(0);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const [add, setAdd] = useState<{ repo: string; revision: string } | null>(null);
  const [copyState, setCopyState] = useState<'copy' | 'copied' | 'copy_failed'>('copy');
  const selected = state.catalog.find((entry) => entry.identity.id === state.selectedModelId);
  const { profile: selectedProfile } = useLoadProfile(selected?.identity.id ?? null);
  const { profile: capacityProfile } = useLoadProfile(confirmation?.kind === 'capacity' ? confirmation.entry.identity.id : null);
  const rows = inventory(state.catalog, filter);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const visiblePage = Math.min(page, pages - 1);
  const downloadPending =
    [...state.operations.values()].some((op) => op.kind === 'download' && !terminal(op)) ||
    [...state.pendingReconciliations.values()].some((item) => item.kind === 'download');
  const rescanPending =
    [...state.operations.values()].some((op) => op.kind === 'catalog_refresh' && !terminal(op)) ||
    [...state.pendingReconciliations.values()].some((item) => item.kind === 'catalog-refresh');
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
    profile: LoadProfile = selectedProfile,
  ): void => {
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
  const openAction = (kind: 'load' | 'unload' | 'delete'): void => {
    if (!selected) return;
    if (kind === 'load') load(selected);
    else setConfirmation({ kind, entry: selected, instance: state.serverInstanceId });
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
        load(
          entry,
          evictionTarget && evictionRevision !== undefined
            ? { id: evictionTarget, revision: evictionRevision }
            : undefined,
          capacityProfile,
        );
      } else if (value.kind === 'unload' && canUnload(state, entry))
        void execute(() =>
          actions.unloadModel({
            action: 'unload',
            model_id: entry.identity.id,
            expected_revision: entry.identity.revision,
            idempotency_key: crypto.randomUUID(),
          }),
        );
      else if (value.kind === 'delete' && canDelete(state, entry))
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
  };
  const retryDownload = (op: Operation): void => {
    if (op.target.target_kind === 'download') setAdd({ repo: op.target.repo_id, revision: op.target.revision ?? '' });
  };
  const command = 'mlxcel-server --webui --models-dir /path/to/models --no-models-autoload';
  const columns: DataTableColumn<CatalogEntry>[] = [
    {
      id: 'name',
      header: t(locale, 'models.library.name'),
      noResize: true,
      render: (entry) => (
        <>
          <Button
            tone="ghost"
            className={ROW_PRIMARY_CLASS}
            aria-label={t(locale, 'models.library.inspect', { name: entry.identity.display_name })}
            onClick={() => actions.selectModel(entry.identity.id)}
          >
            {entry.identity.display_name}
          </Button>
          <span className="models-row-meta">
            {entry.identity.id === state.selectedModelId ? <Badge tone="accent">{t(locale, 'models.library.selected')}</Badge> : null}{' '}
            <Badge>{entry.identity.source}</Badge> <Badge>{entry.metadata.quantization ?? t(locale, 'models.library.unknown')}</Badge>
          </span>
        </>
      ),
    },
    {
      id: 'task',
      header: t(locale, 'models.library.task'),
      noResize: true,
      render: (entry) =>
        entry.capabilities
          .map((cap) => cap.task)
          .filter((value, index, all) => all.indexOf(value) === index)
          .join(', ') || t(locale, 'models.library.unknown'),
    },
    {
      id: 'status',
      header: t(locale, 'models.library.status'),
      noResize: true,
      render: (entry) => (
        <StatusBadge state={entry.lifecycle.state}>{lifecycleLabel(locale, entry.lifecycle.state)}</StatusBadge>
      ),
    },
    {
      id: 'support',
      header: t(locale, 'models.library.support'),
      noResize: true,
      render: (entry) => (
        <>
          {t(
            locale,
            entry.metadata.support.architecturally_supported
              ? 'models.library.supported'
              : 'models.library.unsupported',
          )}
          <br />
          {t(locale, entry.complete ? 'models.library.complete' : 'models.library.incomplete')}
        </>
      ),
    },
  ];
  return (
    <div className="screen-stack models-library" data-testid="models-library">
      <section className="screen-heading">
        <p className="eyebrow">{t(locale, 'routes.models.eyebrow')}</p>
        <h1 tabIndex={-1} data-dialog-focus-fallback data-testid={testId('models.title')}>
          {t(locale, 'models.title')}
        </h1>
        <p>{t(locale, 'models.library.subtitle')}</p>
      </section>
      {!current(state) ? (
        <ErrorBanner
          title={t(locale, 'models.library.stale')}
          body={state.error?.message ?? t(locale, 'models.library.waiting')}
          action={
            <Button
              onClick={() => {
                void actions.refresh();
              }}
            >
              {t(locale, 'models.library.refresh')}
            </Button>
          }
          testId="connection-error-title"
        />
      ) : null}
      {error ? (
        <ErrorBanner
          title={t(locale, 'models.library.error')}
          body={error}
          action={
            <Button
              onClick={() => {
                setError(null);
                void actions.refresh();
              }}
            >
              {t(locale, 'models.library.refresh')}
            </Button>
          }
          testId="models-action-error"
        />
      ) : null}
      {state.bootstrap?.server.mode === 'single_model' ? (
        <ErrorBanner
          tone="info"
          title={t(locale, 'models.library.single')}
          body="mlxcel-server --webui"
          testId="models-read-only"
        />
      ) : null}
      <div className="button-row">
        <Button
          tone="primary"
          onClick={() => setAdd({ repo: '', revision: '' })}
          disabled={busy || downloadPending || !allowed(state, 'download')}
          data-testid="models-add"
        >
          {t(locale, 'models.library.add')}
        </Button>
        <Button
          disabled={busy || rescanPending || !current(state) || state.bootstrap?.server.mode === 'single_model'}
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
      </div>
      {Object.entries(state.bootstrap?.actions ?? {})
        .filter(([, action]) => action.state !== 'enabled')
        .map(([name, action]) => (
          <p key={name}>
            {name}: {action.reason} {action.instructions}
          </p>
        ))}
      <p data-testid="connection-authenticated-detail">{connectedDetail(locale, state)}</p>
      <details open={state.catalog.length === 0}>
        <summary>{t(locale, 'models.library.roots')}</summary>
        <ul>
          {state.bootstrap?.roots.map((root, index) => (
            <li key={index}>
              <strong>{root.display_name}</strong> · {root.kind}
              {root.error ? (
                <p role="alert">
                  {t(locale, 'models.library.permission')} {root.error}
                </p>
              ) : null}
            </li>
          ))}
        </ul>
        <p>{t(locale, 'models.library.roots_help')}</p>
        <pre className="models-wrap">{command}</pre>
        <Button
          onClick={() => {
            void navigator.clipboard
              ?.writeText(command)
              .then(() => setCopyState('copied'))
              .catch(() => setCopyState('copy_failed'));
            if (!navigator.clipboard) setCopyState('copy_failed');
          }}
        >
          {t(locale, `models.library.${copyState}`)}
        </Button>
        <span role="status">{copyState !== 'copy' ? t(locale, `models.library.${copyState}`) : ''}</span>
      </details>
      <div className="models-toolbar">
        <Field
          label={t(locale, 'models.library.search')}
          value={filter.query}
          onChange={(query) => set({ query })}
          testId="models-search"
        />
        <Select
          locale={locale}
          label={t(locale, 'models.library.source')}
          value={filter.source}
          onChange={(source) => set({ source })}
          options={[
            { value: '', label: t(locale, 'models.library.all') },
            ...[...new Set(state.catalog.map((entry) => entry.identity.source))]
              .sort()
              .map((value) => ({ value, label: value })),
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
              .map((value) => ({ value, label: value })),
          ]}
        />
        <Select
          locale={locale}
          label={t(locale, 'models.library.status')}
          value={filter.status}
          onChange={(status) => set({ status })}
          options={[
            { value: '', label: t(locale, 'models.library.all') },
            ...(['unloaded', 'loading', 'ready', 'draining', 'unloading', 'failed'] as const).map((value) => ({
              value,
              label: lifecycleLabel(locale, value),
            })),
          ]}
        />
        <Select
          locale={locale}
          label={t(locale, 'models.library.sort')}
          value={filter.sort}
          onChange={(sort) => set({ sort })}
          options={(['name', 'status', 'source', 'task'] as const).map((value) => ({
            value,
            label: t(locale, `models.library.sort_${value}`),
          }))}
        />
      </div>
      <div className="models-layout">
        <section>
          <DataTable
            columns={columns}
            rows={rows.slice(visiblePage * PAGE_SIZE, (visiblePage + 1) * PAGE_SIZE)}
            getRowKey={(entry) => entry.identity.id}
            ariaLabel={t(locale, 'models.title')}
            testId="models-table"
            activateRowPrimary
            rowClassName={(entry) => (entry.identity.id === state.selectedModelId ? 'models-selected' : undefined)}
            loading={state.catalogSequence === null}
            loadingState={<LoadingStatus label={t(locale, 'models.library.waiting')} />}
            emptyState={
              <EmptyState
                title={t(locale, state.catalog.length === 0 ? 'models.empty.title' : 'models.library.filtered')}
                body={t(locale, state.catalog.length === 0 ? 'models.empty.body' : 'models.library.filtered_body')}
                testId="models-empty"
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
        {selected ? (
          <ModelInspector
            entry={selected}
            profile={selectedProfile}
            state={state}
            locale={locale}
            busy={busy}
            onAction={openAction}
            onChat={() => {
              if (canChat(state, selected)) {
                actions.selectModel(selected.identity.id);
                window.location.hash = 'chat';
              }
            }}
          />
        ) : null}
      </div>
      <LibraryOperations
        state={state}
        locale={locale}
        busy={busy}
        downloadPending={downloadPending}
        onCancel={(operation) => setConfirmation({ kind: 'cancel', operation, instance: state.serverInstanceId })}
        onRetry={retryDownload}
        onCapacity={(id) => {
          const entry = state.catalog.find((candidate) => candidate.identity.id === id);
          if (entry && canLoad(state, entry))
            setConfirmation({ kind: 'capacity', entry, instance: state.serverInstanceId });
        }}
      />
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
