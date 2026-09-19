// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useState } from 'react';
import type { CatalogEntry, Operation, WebUiSnapshot } from '../../api/types';
import { Button, Dialog, Field, Select } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { evictionCandidates, validRepo, validRevision } from './policy';

export type Confirmation =
  | { kind: 'delete' | 'unload' | 'capacity'; entry: CatalogEntry; instance: string | null }
  | { kind: 'cancel'; operation: Operation; instance: string | null };
export function ConfirmAction({
  value,
  state,
  locale,
  busy,
  onClose,
  onConfirm,
}: {
  value: Confirmation;
  state: WebUiSnapshot;
  locale: Locale;
  busy: boolean;
  onClose: () => void;
  onConfirm: (target?: string, revision?: number) => void;
}): React.JSX.Element {
  const [token, setToken] = useState('');
  const [victimRevision, setVictimRevision] = useState<number>();
  const candidates = value.kind === 'capacity' ? evictionCandidates(state, value.entry.identity.id) : [];
  const name =
    value.kind === 'cancel'
      ? value.operation.target.target_kind === 'download'
        ? value.operation.target.repo_id
        : value.operation.operation_id
      : value.entry.identity.display_name;
  const title = t(
    locale,
    value.kind === 'delete'
      ? 'models.library.delete'
      : value.kind === 'unload'
        ? 'models.unload'
        : value.kind === 'cancel'
          ? 'models.library.cancel_download'
          : 'models.library.capacity',
  );
  const body =
    value.kind === 'capacity'
      ? t(locale, 'models.library.capacity_body')
      : value.kind === 'cancel'
        ? t(locale, 'models.library.cancel_body', { name })
        : t(locale, value.kind === 'delete' ? 'models.library.delete_body' : 'models.library.unload_body', {
            name,
            source: value.entry.identity.source,
            count: String(value.entry.lifecycle.active_requests),
          });
  const same =
    state.serverInstanceId === value.instance &&
    (value.kind === 'cancel' ||
      state.catalog.some(
        (entry) =>
          entry.identity.id === value.entry.identity.id && entry.identity.revision === value.entry.identity.revision,
      ));
  const valid =
    same &&
    (value.kind === 'delete'
      ? token === value.entry.identity.id
      : value.kind !== 'capacity' || candidates.some((entry) => entry.identity.id === token && entry.identity.revision === victimRevision));
  return (
    <Dialog open title={title} onClose={onClose} closeLabel={t(locale, 'common.close')} testId="models-confirm">
      <p className="models-wrap">{body}</p>
      {value.kind === 'delete' ? <code className="models-wrap">{value.entry.identity.id}</code> : null}
      {value.kind === 'capacity' && token && !valid ? <p role="alert">{t(locale, 'models.library.stale')}</p> : null}
      {!same ? <p role="alert">{t(locale, 'models.library.stale')}</p> : null}
      {value.kind === 'delete' ? (
        <Field
          label={t(locale, 'models.library.confirm_name')}
          value={token}
          onChange={setToken}
          testId="models-confirm-name"
        />
      ) : null}
      {value.kind === 'capacity' ? (
        <>
          <ul>
            {state.catalog
              .filter((entry) => ['ready', 'loading', 'draining', 'unloading'].includes(entry.lifecycle.state))
              .map((entry) => (
                <li key={entry.identity.id} className="models-wrap">
                  {entry.identity.display_name} · {entry.lifecycle.state} · {t(locale, 'models.library.active')}:{' '}
                  {entry.lifecycle.active_requests}
                </li>
              ))}
          </ul>
          <Select
            locale={locale}
            label={t(locale, 'models.library.eviction')}
            value={token}
            onChange={(id) => { setToken(id); setVictimRevision(candidates.find((entry) => entry.identity.id === id)?.identity.revision); }}
            options={[
              { value: '', label: t(locale, 'models.library.choose') },
              ...candidates.map((entry) => ({ value: entry.identity.id, label: entry.identity.display_name })),
            ]}
            testId="models-eviction-target"
          />
        </>
      ) : null}
      <div className="dialog-actions">
        <Button onClick={onClose}>{t(locale, 'models.library.cancel')}</Button>
        <Button
          tone="danger"
          data-testid="models-confirm-submit"
          busy={busy}
          disabled={!valid || busy}
          onClick={() => onConfirm(value.kind === 'capacity' ? token : undefined, victimRevision)}
        >
          {t(locale, value.kind === 'capacity' ? 'models.library.evict_load' : 'models.library.confirm')}
        </Button>
      </div>
    </Dialog>
  );
}

export function AddModel({
  locale,
  state,
  busy,
  onClose,
  onDownload,
  initial,
  error,
}: {
  locale: Locale;
  state: WebUiSnapshot;
  busy: boolean;
  onClose: () => void;
  onDownload: (repo: string, revision: string) => void;
  initial?: { repo: string; revision: string };
  error?: string | null;
}): React.JSX.Element {
  const [repo, setRepo] = useState(initial?.repo ?? '');
  const [revision, setRevision] = useState(initial?.revision ?? '');
  const [publicRepo, setPublicRepo] = useState(false);
  const valid = validRepo(repo.trim()) && validRevision(revision.trim());
  return (
    <Dialog
      open
      title={t(locale, 'models.library.add')}
      onClose={onClose}
      closeLabel={t(locale, 'common.close')}
      testId="models-add-dialog"
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (valid && publicRepo && !busy) onDownload(repo.trim(), revision.trim());
        }}
      >
        {error ? <p role="alert">{error}</p> : null}
        <p>{t(locale, 'models.library.network')}</p>
        <ul>
          {state.bootstrap?.roots
            .filter((item) => item.kind === 'cache')
            .map((item, index) => (
              <li key={index}>{item.display_name}</li>
            ))}
        </ul>
        <Field
          label={t(locale, 'models.library.repo')}
          value={repo}
          onChange={setRepo}
          testId="models-repo"
          error={repo && !validRepo(repo.trim()) ? t(locale, 'models.library.repo_invalid') : undefined}
        />
        <Field
          label={t(locale, 'models.library.revision')}
          value={revision}
          onChange={setRevision}
          testId="models-revision"
        />
        <label className="toggle">
          <input
            type="checkbox"
            checked={publicRepo}
            onChange={(event) => setPublicRepo(event.currentTarget.checked)}
            data-testid="models-public-repo"
          />
          <span>{t(locale, 'models.library.public')}</span>
        </label>
        <p>{t(locale, 'models.library.private')}</p>
        <pre className="models-wrap">hf download OWNER/REPO --local-dir /path/to/models/checkpoint</pre>
        <div className="dialog-actions">
          <Button onClick={onClose}>{t(locale, 'models.library.cancel')}</Button>
          <Button
            type="submit"
            tone="primary"
            busy={busy}
            disabled={!valid || !publicRepo || busy}
            data-testid="models-download-submit"
          >
            {t(locale, 'models.library.add')}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
