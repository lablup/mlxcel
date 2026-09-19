// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import runtimeJson from '../../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import type { WebUiSnapshot, Operation, RuntimeSnapshot } from '../../api/types';
import { validateRuntime } from '../../api/validation';
import { WebUiHttpError } from '../../api/client';
import { t } from '../../i18n/catalog';
import { ModelsLibrary } from './screen';
import { model, snapshot } from './test-fixtures';

let state: WebUiSnapshot;
const actions = {
  selectModel: vi.fn(),
  loadModel: vi.fn(),
  unloadModel: vi.fn(),
  removeModel: vi.fn(),
  downloadModel: vi.fn(),
  refreshCatalog: vi.fn(),
  refresh: vi.fn(),
  cancelOperation: vi.fn(),
};
vi.mock('../../state', () => ({ useWebUi: () => state, useWebUiActions: () => actions }));
let root: Root;
let host: HTMLDivElement;
beforeEach(() => {
  state = snapshot();
  vi.resetAllMocks();
  HTMLDialogElement.prototype.showModal = function () {
    this.setAttribute('open', '');
  };
  HTMLDialogElement.prototype.close = function () {
    this.removeAttribute('open');
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});
const render = (): void => {
  act(() => root.render(<ModelsLibrary locale="en" />));
};
const button = (id: string): HTMLButtonElement =>
  requireValue(host.querySelector<HTMLButtonElement>(`[data-testid="${id}"]`));
async function click(id: string): Promise<void> {
  await act(async () => button(id)?.click());
}
async function input(id: string, value: string): Promise<void> {
  await act(async () => {
    const node = requireValue(host.querySelector<HTMLInputElement>(`[data-testid="${id}"]`));
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(node, value);
    node.dispatchEvent(new Event('input', { bubbles: true }));
  });
}
const runtimeFixture = validateRuntime(Object.fromEntries(Object.entries(runtimeJson).filter(([key]) => key !== '$schemaName')));
const download: Operation = {
  operation_id: 'op_download',
  kind: 'download',
  state: 'running',
  created_at: '2026-09-14T00:00:00Z',
  updated_at: '2026-09-14T00:00:00Z',
  idempotency_scope: 'server_instance',
  target: { target_kind: 'download', repo_id: 'owner/repo', revision: null },
  progress: { completed_bytes: 15, total_bytes: null, indeterminate: true },
  result: null,
  error: null,
  cancellable: true,
  cancel_reason: null,
};

describe('Models workflows', () => {
  it('renders empty-store guidance and root permission recovery without POSTs', async () => {
    state = {
      ...state,
      catalog: [],
      selectedModelId: null,
      bootstrap: {
        ...requireValue(state.bootstrap),
        roots: [{ kind: 'models_dir', display_name: 'Local root', redacted: true, error: 'permission denied' }],
      },
    };
    render();
    expect(host.querySelector('[data-testid="models-empty"]')).not.toBeNull();
    // Roots are the empty state's action, not a permanent disclosure on the page.
    expect(host.textContent).not.toContain('--models-dir /path/to/models');
    await click('models-roots');
    const dialog = requireValue(host.querySelector('[data-testid="models-roots-dialog"]'));
    expect(dialog.textContent).toContain('permission denied');
    expect(dialog.textContent).toContain(t('en', 'models.source.models_dir'));
    expect(dialog.textContent).toContain('--models-dir /path/to/models');
    expect(actions.loadModel).not.toHaveBeenCalled();
  });
  it('selection never loads, and duplicate clicks do not submit twice or assume readiness from 202', async () => {
    let finish: (() => void) | undefined;
    actions.loadModel.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    );
    state = { ...state, selectedModelId: null };
    render();
    await act(async () => host.querySelector<HTMLButtonElement>('[aria-label^="Inspect"]')?.click());
    expect(actions.selectModel).toHaveBeenCalledWith(model().identity.id);
    expect(actions.loadModel).not.toHaveBeenCalled();
    state = { ...state, selectedModelId: model().identity.id };
    render();
    await click('models-load');
    await click('models-load');
    expect(actions.loadModel).toHaveBeenCalledTimes(1);
    expect(actions.loadModel.mock.calls[0][0]).toMatchObject({
      model_id: model().identity.id,
      expected_revision: 4,
      action: 'load',
    });
    expect(button('models-use-chat').disabled).toBe(true);
    await act(async () => finish?.());
    expect(button('models-use-chat').disabled).toBe(true);
  });
  it('rejects a stale delete confirmation after another tab changes the revision', async () => {
    render();
    await click('models-delete');
    await input('models-confirm-name', model().identity.id);
    state = { ...state, catalog: [{ ...model(), identity: { ...model().identity, revision: 5 } }] };
    render();
    expect(button('models-confirm-submit').disabled).toBe(true);
    await click('models-confirm-submit');
    expect(actions.removeModel).not.toHaveBeenCalled();
    expect(host.textContent).toContain('State changed');
  });
  it('tells the user to type the opaque model ID shown in the delete dialog, not the model name', async () => {
    render();
    await click('models-delete');
    const dialog = requireValue(host.querySelector('[data-testid="models-confirm"]'));
    const body = requireValue(dialog.querySelector('p'));
    expect(body.textContent).toContain('type the model ID shown below, not the model name');
    expect(body.classList.contains('models-wrap')).toBe(true);
    expect(dialog.querySelector('code')?.textContent).toBe(model().identity.id);
    const field = requireValue(dialog.querySelector<HTMLInputElement>('[data-testid="models-confirm-name"]'));
    const label = document.getElementById(requireValue(field.getAttribute('aria-labelledby')))?.textContent ?? '';
    expect(label).toContain('model ID');
    expect(label).toContain('mdl_');
    expect(label).not.toMatch(/model name/i);
  });
  it('requires exact typed managed-cache identity and distinguishes unload from disk deletion', async () => {
    render();
    await click('models-delete');
    await input('models-confirm-name', 'DELETE');
    expect(button('models-confirm-submit').disabled).toBe(true);
    await input('models-confirm-name', model().identity.id);
    await click('models-confirm-submit');
    expect(actions.removeModel).toHaveBeenCalledWith(
      expect.objectContaining({ model_id: model().identity.id, expected_revision: 4 }),
    );
    expect(actions.unloadModel).not.toHaveBeenCalled();
  });
  it('shows capacity recovery without repeated load attempts', async () => {
    actions.loadModel.mockRejectedValue(
      new WebUiHttpError(409, {
        request_id: 'req_conflict',
        error: { code: 'conflict', message: 'capacity full', retryable: false },
      }),
    );
    render();
    await click('models-load');
    expect(host.querySelector('[data-testid="models-confirm"]')).not.toBeNull();
    expect(host.textContent).toContain('Do not repeatedly load');
    expect(button('models-confirm-submit').disabled).toBe(true);
    expect(actions.loadModel).toHaveBeenCalledTimes(1);
  });
  it('shows indeterminate progress and explicit cancellation without claiming completion', async () => {
    state = { ...state, operations: new Map([[download.operation_id, download]]) };
    render();
    const progress = host.querySelector('[role="progressbar"]');
    expect(progress?.hasAttribute('aria-valuenow')).toBe(false);
    const cancel = Array.from(host.querySelectorAll('button')).find((node) => node.textContent === 'Cancel download');
    await act(async () => cancel?.click());
    await click('models-confirm-submit');
    expect(actions.cancelOperation).toHaveBeenCalledWith(download.operation_id);
    const name = requireValue(host.querySelector('[data-testid="models-operation"]'));
    expect(name.querySelector('.truncate')?.textContent).toBe('owner/repo');
    // The name cell carries the state and progress the narrowing list hides with the State and
    // Size columns; jsdom does not evaluate container queries, so which one shows is the browser's.
    expect(name.querySelector('.models-name-state')?.textContent).toBe(t('en', 'models.download.running'));
    const fallback = requireValue(name.querySelector('.models-name-progress [role="progressbar"]'));
    expect(fallback.getAttribute('aria-label')).toBe(t('en', 'models.library.progress'));
    expect(fallback.hasAttribute('aria-valuenow')).toBe(false);
    expect(host.textContent).toContain(t('en', 'models.download.running'));
    expect(host.textContent).not.toContain('op_download');
  });
  it.each(['disk full', 'invalid config', 'unsupported backend', 'permission denied'])(
    'retains actionable server failure: %s',
    (message) => {
      state = {
        ...state,
        operations: new Map([
          [
            download.operation_id,
            {
              ...download,
              state: 'failed',
              cancellable: false,
              error: { code: 'unavailable', message, retryable: true },
            },
          ],
        ]),
      };
      render();
      expect(host.textContent).toContain(message);
      expect(host.textContent).toContain('Retry download');
    },
  );
  it('freezes confirmations across a server restart and disables stale mutations', async () => {
    render();
    await click('models-delete');
    await input('models-confirm-name', model().identity.id);
    state = { ...state, serverInstanceId: 'srv_new', connection: 'stale' };
    render();
    expect(button('models-confirm-submit').disabled).toBe(true);
    expect(button('models-load').disabled).toBe(true);
    expect(host.textContent).toContain('Refresh server state');
  });
  it('pages a large inventory and retains selected opaque identity while filtering', async () => {
    const entries = Array.from({ length: 100 }, (_, index) => ({
      ...model(),
      identity: { ...model().identity, id: `id_${index}`, display_name: `模型-long-model-${index}` },
    }));
    state = { ...state, catalog: entries, selectedModelId: 'id_99' };
    render();
    expect(host.querySelectorAll('[data-testid="models-table"] tbody tr')).toHaveLength(25);
    await input('models-search', 'model-1');
    expect(host.querySelectorAll('[data-testid="models-table"] tbody tr')).toHaveLength(11);
    expect(state.selectedModelId).toBe('id_99');
    expect(actions.downloadModel).not.toHaveBeenCalled();
  });
});

function requireValue<T>(value: T | null): T {
  if (value === null) throw new Error('Expected fixture element');
  return value;
}

describe('reviewed asynchronous recovery paths', () => {
  it('offers explicit capacity recovery after an accepted load fails, without auto retry', async () => {
    const ready = {
      ...model(),
      identity: { ...model().identity, id: 'idle-target', display_name: 'idle-target' },
      lifecycle: { ...model().lifecycle, state: 'ready' as const },
    };
    const busy = {
      ...ready,
      identity: { ...ready.identity, id: 'busy-target', display_name: 'busy-target' },
      lifecycle: { ...ready.lifecycle, active_requests: 3, busy: true },
    };
    state = { ...state, catalog: [model(), ready, busy] };
    render();
    await click('models-load');
    expect(actions.loadModel).toHaveBeenCalledTimes(1);
    const failed: Operation = {
      ...download,
      kind: 'model_load',
      state: 'failed',
      cancellable: false,
      target: { target_kind: 'model', model_id: model().identity.id, requested_revision: 4 },
      error: { code: 'conflict', message: 'capacity full', retryable: false },
    };
    state = { ...state, operations: new Map([[failed.operation_id, failed]]) };
    render();
    const recovery = button('models-row-capacity');
    expect(recovery.textContent).toBe(t('en', 'models.library.load_evict'));
    await act(async () => recovery.click());
    const listed = Array.from(host.querySelectorAll('[data-testid="models-confirm"] li'));
    expect(listed.map((item) => item.textContent?.split(' · ')[0])).toEqual(['idle-target', 'busy-target']);
    expect(listed.every((item) => item.classList.contains('models-wrap'))).toBe(true);
    const select = host.querySelector<HTMLSelectElement>('[data-testid="models-eviction-target"]');
    expect(select?.textContent).toContain('idle-target');
    expect(select?.textContent).not.toContain('busy-target');
    await act(async () => {
      if (select) {
        select.value = 'idle-target';
        select.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
    await click('models-confirm-submit');
    expect(actions.loadModel).toHaveBeenCalledTimes(2);
    expect(actions.loadModel.mock.calls[1][0]).toMatchObject({
      eviction_target_id: 'idle-target',
      eviction_target_expected_revision: 4,
      expected_revision: 4,
    });
  });
  it('retains rejected download fields and exposes the server error inside the modal', async () => {
    actions.downloadModel.mockRejectedValue(
      new WebUiHttpError(400, {
        request_id: 'req_invalid',
        error: { code: 'invalid_request', message: 'revision not found', retryable: false },
      }),
    );
    render();
    await click('models-add');
    await input('models-repo', 'owner/repo');
    await input('models-revision', 'main');
    await act(async () => host.querySelector<HTMLInputElement>('[data-testid="models-public-repo"]')?.click());
    await click('models-download-submit');
    expect(host.querySelector('[data-testid="models-add-dialog"]')?.textContent).toContain('revision not found');
    expect(host.querySelector<HTMLInputElement>('[data-testid="models-repo"]')?.value).toBe('owner/repo');
  });
  it('keeps the library actions in the PageHeader and the stale and action errors in its error slot', async () => {
    actions.refreshCatalog.mockRejectedValue(
      new WebUiHttpError(409, {
        request_id: 'req_busy',
        error: { code: 'conflict', message: 'catalog refresh already running', retryable: true },
      }),
    );
    render();
    const header = requireValue(host.querySelector<HTMLElement>('.ds-page-header'));
    expect(header.querySelector('h1')?.dataset.testid).toBe('models-title');
    expect(host.querySelectorAll('[data-dialog-focus-fallback]')).toHaveLength(1);
    for (const id of ['models-add', 'models-rescan'])
      expect(header.querySelector(`.page-header__actions [data-testid="${id}"]`)).not.toBeNull();
    expect(Array.from(header.querySelectorAll('.page-header__actions button')).map((item) => item.textContent)).toEqual([
      t('en', 'models.library.add'),
      t('en', 'models.library.rescan'),
      'Refresh server state',
    ]);
    expect(header.querySelector('.page-header__error')).toBeNull();
    await click('models-rescan');
    const actionError = requireValue(host.querySelector<HTMLElement>('[data-testid="models-action-error"]'));
    expect(actionError.matches('.ds-page-header .page-header__error[role="alert"]')).toBe(true);
    expect(actionError.querySelector('.page-header__error-text')?.textContent).toBe(t('en', 'models.library.error'));
    const retry = requireValue(actionError.querySelector<HTMLButtonElement>('button'));
    expect(retry.textContent).toBe('Refresh server state');
    await act(async () => retry.click());
    expect(actions.refresh).toHaveBeenCalledOnce();
    expect(host.querySelector('[data-testid="models-action-error"]')).toBeNull();
    state = { ...state, connection: 'stale', error: { code: 'stale', message: 'server restarted', retryable: true } };
    render();
    const stale = requireValue(host.querySelector<HTMLElement>('[data-testid="connection-error-title"]'));
    expect(stale.matches('.ds-page-header .page-header__error[role="alert"]')).toBe(true);
    expect(stale.querySelector('.page-header__error-text')?.textContent).toBe(t('en', 'models.library.stale'));
    expect(stale.querySelector('.page-header__error-detail')?.textContent).toBe('server restarted');
    expect(button('models-rescan').disabled).toBe(true);
    await act(async () => requireValue(stale.querySelector<HTMLButtonElement>('button')).click());
    expect(actions.refresh).toHaveBeenCalledTimes(2);
    expect(host.querySelectorAll('.page-header__error')).toHaveLength(1);
  });
  it('keeps rescan and lifecycle read-only in single-model mode', () => {
    if (!state.bootstrap) throw new Error('Missing fixture');
    state = {
      ...state,
      bootstrap: {
        ...state.bootstrap,
        server: { ...state.bootstrap.server, mode: 'single_model' },
        actions: {
          load: { state: 'read_only', reason: 'single-model' },
          unload: { state: 'read_only', reason: 'single-model' },
          cache_delete: { state: 'read_only', reason: 'single-model' },
          download: { state: 'read_only', reason: 'single-model' },
        },
      },
    };
    render();
    expect(button('models-rescan').disabled).toBe(true);
    expect(button('models-load').disabled).toBe(true);
    expect(button('models-add').disabled).toBe(true);
  });
});

describe('reviewed destructive identity fences', () => {
  it('requires the exact cache ID rather than a duplicate display name or another entry ID', async () => {
    const duplicate = { ...model(), identity: { ...model().identity, id: `mdl_${'d'.repeat(43)}` } };
    state = { ...state, catalog: [model(), duplicate] }; render();
    await click('models-delete');
    expect(host.querySelector('[data-testid="models-confirm"]')?.textContent).toContain(model().identity.id);
    await input('models-confirm-name', model().identity.display_name);
    expect(button('models-confirm-submit').disabled).toBe(true);
    await input('models-confirm-name', duplicate.identity.id);
    expect(button('models-confirm-submit').disabled).toBe(true);
    await input('models-confirm-name', model().identity.id);
    await click('models-confirm-submit');
    expect(actions.removeModel).toHaveBeenCalledWith(expect.objectContaining({ model_id: model().identity.id, expected_revision: 4 }));
  });

  it('rejects a selected eviction victim whose revision changes while still ready and idle', async () => {
    const idle = { ...model(), identity: { ...model().identity, id: 'idle-target' }, lifecycle: { ...model().lifecycle, state: 'ready' as const } };
    state = { ...state, catalog: [model(), idle] };
    actions.loadModel.mockRejectedValueOnce(new WebUiHttpError(409, { request_id: 'req_full', error: { code: 'conflict', message: 'capacity full', retryable: false } }));
    render(); await click('models-load');
    await act(async () => {
      const select = requireValue(host.querySelector<HTMLSelectElement>('[data-testid="models-eviction-target"]'));
      select.value = idle.identity.id; select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(button('models-confirm-submit').disabled).toBe(false);
    state = { ...state, catalog: [model(), { ...idle, identity: { ...idle.identity, revision: idle.identity.revision + 1 } }] }; render();
    expect(button('models-confirm-submit').disabled).toBe(true);
    await click('models-confirm-submit');
    expect(actions.loadModel).toHaveBeenCalledTimes(1);
  });
});

describe('library row activation', () => {
  const entries = ['alpha', 'bravo', 'charlie'].map((name, index) => ({
    ...model(),
    identity: { ...model().identity, id: `mdl_row_${index}`, display_name: `row-${name}` },
  }));
  const rowCell = (row: number, column: number): HTMLElement =>
    requireValue(host.querySelectorAll('[data-testid="models-table"] tbody tr')[row]?.querySelectorAll<HTMLElement>('td')[column] ?? null);

  it('opens the inspector from a non-name cell without loading, exactly once per click', async () => {
    state = { ...state, catalog: entries, selectedModelId: null };
    render();
    await act(async () => rowCell(1, 1).click());
    expect(actions.selectModel.mock.calls).toEqual([[entries[1].identity.id]]);
    await act(async () => rowCell(2, 2).click());
    expect(actions.selectModel.mock.calls).toEqual([[entries[1].identity.id], [entries[2].identity.id]]);
    expect(actions.loadModel).not.toHaveBeenCalled();
  });

  it('keeps the name button a single activation, including for a busy entry', async () => {
    const busy = { ...entries[0], lifecycle: { ...entries[0].lifecycle, state: 'loading' as const, busy: true } };
    state = { ...state, catalog: [busy, entries[1]], selectedModelId: null };
    render();
    const name = requireValue(host.querySelector<HTMLButtonElement>(`[aria-label="Inspect ${busy.identity.display_name}"]`));
    expect(name.disabled).toBe(false);
    await act(async () => name.click());
    expect(actions.selectModel.mock.calls).toEqual([[busy.identity.id]]);
    await act(async () => rowCell(0, 3).click());
    expect(actions.selectModel.mock.calls).toEqual([[busy.identity.id], [busy.identity.id]]);
    expect(actions.loadModel).not.toHaveBeenCalled();
  });

  it('treats re-selecting the already-selected row as a no-op, but still opens a different row', async () => {
    state = { ...state, catalog: entries, selectedModelId: entries[0].identity.id };
    render();
    await act(async () => rowCell(0, 1).click());
    expect(actions.selectModel).not.toHaveBeenCalled();
    const name = requireValue(host.querySelector<HTMLButtonElement>(`[aria-label="Inspect ${entries[0].identity.display_name}"]`));
    await act(async () => name.click());
    expect(actions.selectModel).not.toHaveBeenCalled();
    await act(async () => rowCell(1, 1).click());
    expect(actions.selectModel.mock.calls).toEqual([[entries[1].identity.id]]);
  });
});

// #1918: the row is the unit of work.
describe('library row actions', () => {
  const ready = (entry = model()) => ({
    ...entry,
    lifecycle: { ...entry.lifecycle, state: 'ready' as const, worker_exit_observed: false },
    capabilities: [{ task: 'chat' as const, phase: 'provider_ready' as const, available: true, reason: null }],
  });
  const row = (name: string): HTMLTableRowElement =>
    requireValue(
      [...host.querySelectorAll<HTMLTableRowElement>('[data-testid="models-table"] tbody tr')].find(
        (item) => item.querySelector('.truncate')?.textContent === name,
      ) ?? null,
    );
  const inRow = (name: string, id: string): HTMLButtonElement | null => row(name).querySelector<HTMLButtonElement>(`[data-testid="${id}"]`);

  it('loads, opens Chat and unloads from the row, without selecting or opening the inspector first', async () => {
    const other = { ...model(), identity: { ...model().identity, id: `mdl_${'e'.repeat(43)}`, display_name: 'bravo' } };
    state = { ...state, catalog: [model(), other], selectedModelId: null };
    render();
    expect(host.querySelector('aside[aria-label="Model details"]')).toBeNull();
    expect(inRow('bravo', 'models-row-chat')).toBeNull();
    await act(async () => inRow('bravo', 'models-row-load')?.click());
    expect(actions.loadModel).toHaveBeenCalledTimes(1);
    expect(actions.loadModel.mock.calls[0][0]).toMatchObject({ action: 'load', model_id: other.identity.id, expected_revision: 4 });
    expect(actions.selectModel).not.toHaveBeenCalled();
    state = { ...state, catalog: [model(), ready(other)] };
    render();
    expect(inRow('bravo', 'models-row-load')).toBeNull();
    expect(inRow('bravo', 'models-row-chat')?.getAttribute('aria-label')).toBe(t('en', 'models.library.chat_named', { name: 'bravo' }));
    await act(async () => inRow('bravo', 'models-row-unload')?.click());
    expect(actions.unloadModel).not.toHaveBeenCalled();
    await click('models-confirm-submit');
    expect(actions.unloadModel).toHaveBeenCalledWith(expect.objectContaining({ action: 'unload', model_id: other.identity.id }));
    window.location.hash = '';
    await act(async () => inRow('bravo', 'models-row-chat')?.click());
    expect(actions.selectModel).toHaveBeenCalledWith(other.identity.id);
    expect(window.location.hash).toBe('#chat');
    expect(host.querySelector('aside[aria-label="Model details"]')).toBeNull();
  });

  it('keeps a loading row in place with Load disabled, and hides lifecycle controls in single-model mode', () => {
    const loading = { ...model(), lifecycle: { ...model().lifecycle, state: 'loading' as const, busy: true } };
    state = { ...state, catalog: [loading], selectedModelId: null };
    render();
    expect(inRow('alpha', 'models-row-load')?.disabled).toBe(true);
    expect(inRow('alpha', 'models-row-delete')?.disabled).toBe(true);
    if (!state.bootstrap) throw new Error('Missing fixture');
    state = { ...state, catalog: [model(), { ...ready(), identity: { ...model().identity, id: 'id_ready', display_name: 'ready-one' } }], bootstrap: { ...state.bootstrap, server: { ...state.bootstrap.server, mode: 'single_model' } } };
    render();
    for (const id of ['models-row-load', 'models-row-unload', 'models-row-delete']) expect(host.querySelector(`[data-testid="${id}"]`)).toBeNull();
    expect(inRow('ready-one', 'models-row-chat')).not.toBeNull();
    expect(host.querySelectorAll('[aria-label^="Inspect"]')).toHaveLength(2);
  });

  it('moves focus to the row\'s Load button once an Unload started from that row settles', async () => {
    state = { ...state, catalog: [ready()], selectedModelId: null };
    render();
    const unload = requireValue(inRow('alpha', 'models-row-unload'));
    act(() => unload.focus());
    await act(async () => unload.click());
    await click('models-confirm-submit');
    expect(actions.unloadModel).toHaveBeenCalledTimes(1);
    const inspect = requireValue(row('alpha').querySelector<HTMLButtonElement>('[aria-label^="Inspect"]'));
    expect(document.activeElement).toBe(inspect);
    state = { ...state, catalog: [{ ...ready(), lifecycle: { ...ready().lifecycle, state: 'draining' as const, busy: true } }] };
    render();
    expect(document.activeElement).toBe(inspect);
    state = { ...state, catalog: [{ ...model(), identity: { ...model().identity, revision: 6 } }] };
    render();
    expect(document.activeElement).toBe(inRow('alpha', 'models-row-load'));
  });

  it('releases a Ready model without chat from its row: Unload, not a disabled Load', async () => {
    const embedding = { ...ready(), capabilities: [{ task: 'embedding' as const, phase: 'provider_ready' as const, available: true, reason: null }] };
    state = { ...state, catalog: [embedding], selectedModelId: null };
    render();
    expect(inRow('alpha', 'models-row-chat')).toBeNull();
    expect(inRow('alpha', 'models-row-load')).toBeNull();
    const unload = requireValue(inRow('alpha', 'models-row-unload'));
    expect(unload.disabled).toBe(false);
    expect(unload.getAttribute('aria-label')).toBe(t('en', 'models.library.unload_named', { name: 'alpha' }));
    await act(async () => unload.click());
    expect(host.querySelector('[data-testid="models-confirm"]')).not.toBeNull();
    expect(actions.unloadModel).not.toHaveBeenCalled();
    await click('models-confirm-submit');
    expect(actions.unloadModel).toHaveBeenCalledTimes(1);
    expect(actions.unloadModel).toHaveBeenCalledWith(expect.objectContaining({ action: 'unload', model_id: embedding.identity.id, expected_revision: 4 }));
  });

  it('moves focus to the row\'s Unload button once a Load of a model without chat started from that row is Ready', async () => {
    const embedding = { ...model(), capabilities: [{ task: 'embedding' as const, phase: 'pre_load' as const, available: true, reason: null }] };
    state = { ...state, catalog: [embedding], selectedModelId: null };
    render();
    const load = requireValue(inRow('alpha', 'models-row-load'));
    act(() => load.focus());
    await act(async () => load.click());
    expect(actions.loadModel).toHaveBeenCalledTimes(1);
    const inspect = requireValue(row('alpha').querySelector<HTMLButtonElement>('[aria-label^="Inspect"]'));
    state = { ...state, catalog: [{ ...embedding, identity: { ...embedding.identity, revision: 5 }, lifecycle: { ...embedding.lifecycle, state: 'loading' as const, busy: true } }] };
    render();
    expect(inRow('alpha', 'models-row-load')?.disabled).toBe(true);
    expect(document.activeElement).toBe(inspect);
    state = { ...state, catalog: [{ ...ready(embedding), identity: { ...embedding.identity, revision: 6 }, capabilities: [{ task: 'embedding' as const, phase: 'provider_ready' as const, available: true, reason: null }] }] };
    render();
    expect(inRow('alpha', 'models-row-chat')).toBeNull();
    expect(document.activeElement).toBe(inRow('alpha', 'models-row-unload'));
  });

  it('sorts from the column headers inside the lifecycle pin, with unknown sizes last in both directions', async () => {
    const make = (name: string, bytes: number | null, lifecycle: 'ready' | 'unloaded' | 'loading') => ({
      ...(lifecycle === 'ready' ? ready() : model()),
      identity: { ...model().identity, id: `id_${name}`, display_name: name },
      metadata: { ...model().metadata, disk_bytes: bytes },
      ...(lifecycle === 'loading' ? { lifecycle: { ...model().lifecycle, state: 'loading' as const } } : {}),
    });
    state = {
      ...state,
      selectedModelId: null,
      catalog: [make('a-small', 10, 'unloaded'), make('b-unknown', null, 'unloaded'), make('c-big', 999, 'unloaded'), make('d-ready', 1, 'ready'), make('e-loading', 5, 'loading')],
    };
    render();
    const names = (): string[] => [...host.querySelectorAll('[data-testid="models-table"] tbody tr .truncate')].map((node) => node.textContent ?? '');
    const header = (id: string): HTMLElement => requireValue(host.querySelector<HTMLElement>(`[data-testid="models-table"] th.models-col-${id}`));
    expect(names()).toEqual(['d-ready', 'e-loading', 'a-small', 'b-unknown', 'c-big']);
    expect(header('name').getAttribute('aria-sort')).toBe('ascending');
    await act(async () => header('size').click());
    expect(header('size').getAttribute('aria-sort')).toBe('ascending');
    expect(names()).toEqual(['d-ready', 'e-loading', 'a-small', 'c-big', 'b-unknown']);
    await act(async () => header('size').click());
    expect(names()).toEqual(['d-ready', 'e-loading', 'c-big', 'a-small', 'b-unknown']);
    // A third activation clears the table's sort; the library falls back to name order.
    await act(async () => header('size').click());
    expect(header('name').getAttribute('aria-sort')).toBe('ascending');
    expect(names()).toEqual(['d-ready', 'e-loading', 'a-small', 'b-unknown', 'c-big']);
    // The separate Sort select is gone; the headers are the only sort control.
    expect(host.querySelectorAll('.models-toolbar .ds-common-select')).toHaveLength(3);
  });

  it('labels the source and task filters instead of printing enum values', async () => {
    const speech = { ...model(), identity: { ...model().identity, id: 'id_speech', display_name: 'speech', source: 'models_dir' as const }, capabilities: [{ task: 'audio_transcription' as const, phase: 'pre_load' as const, available: true, reason: null }] };
    state = { ...state, catalog: [model(), speech] };
    // jsdom has no scrollIntoView; the open listbox scrolls its active option into view.
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() });
    render();
    const optionsOf = async (index: number): Promise<string[]> => {
      await act(async () => requireValue(host.querySelectorAll<HTMLElement>('.models-toolbar .select__trigger')[index] ?? null).click());
      const labels = [...document.querySelectorAll('[role="option"]')].map((node) => node.textContent?.trim() ?? '');
      await act(async () => requireValue(host.querySelectorAll<HTMLElement>('.models-toolbar .select__trigger')[index] ?? null).click());
      return labels;
    };
    expect(await optionsOf(0)).toEqual(['All', t('en', 'models.source.cache'), t('en', 'models.source.models_dir')]);
    const tasks = await optionsOf(1);
    expect(tasks).toContain(t('en', 'models.task.audio_transcription'));
    expect(tasks.join(' ')).not.toMatch(/audio_transcription|models_dir/);
    Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView');
  });

  it('shows download pseudo-rows for in-flight and failed downloads only, keyed apart from catalog rows', () => {
    const failed: Operation = { ...download, operation_id: 'op_failed', state: 'failed', cancellable: false, created_at: '2026-09-13T00:00:00Z', error: { code: 'unavailable', message: 'disk full', retryable: true } };
    const done: Operation = { ...download, operation_id: 'op_done', state: 'succeeded', cancellable: false, target: { target_kind: 'download', repo_id: 'owner/done', revision: null } };
    const loadOp: Operation = { ...download, operation_id: 'op_load', kind: 'model_load', state: 'succeeded', target: { target_kind: 'model', model_id: model().identity.id, requested_revision: 4 } };
    state = { ...state, operations: new Map([[download.operation_id, download], [failed.operation_id, failed], [done.operation_id, done], [loadOp.operation_id, loadOp]]) };
    render();
    const pseudo = [...host.querySelectorAll('[data-testid="models-operation"]')];
    expect(pseudo.map((node) => node.querySelector('.truncate')?.textContent)).toEqual(['owner/repo', 'owner/repo']);
    expect(host.querySelectorAll('[data-testid="models-table"] tbody tr')).toHaveLength(3);
    expect(host.textContent).toContain('disk full');
    expect(host.textContent).not.toContain('owner/done');
    expect(host.textContent).not.toMatch(/op_[a-z]/);
    expect(host.querySelector('.models-download-row')?.textContent).toContain(t('en', 'models.library.cancel_download'));
  });

  it('keeps the reconciliation notice above the table while a request outcome is unknown', () => {
    state = { ...state, pendingReconciliations: new Map([['idem', { kind: 'model-action' as const, idempotencyKey: 'idem', operationId: null, modelId: model().identity.id, createdAt: 1 }]]) };
    render();
    const pending = requireValue(host.querySelector('[data-testid="models-pending"]'));
    expect(pending.getAttribute('role')).toBe('status');
    expect(pending.compareDocumentPosition(requireValue(host.querySelector('[data-testid="models-table"]'))) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('isolates the model name inside dialog sentences so bidi controls cannot reorder them', async () => {
    const hostile = { ...model(), identity: { ...model().identity, display_name: 'evil‮ledom' } };
    state = { ...state, catalog: [hostile] };
    render();
    await act(async () => inRow('evil‮ledom', 'models-row-delete')?.click());
    const body = host.querySelector('[data-testid="models-confirm"] p')?.textContent ?? '';
    expect(body).toContain('⁨evil‮ledom⁩');
    expect(body).toContain(t('en', 'models.source.cache'));
  });
});

// #1918: internal identifiers and raw reasons are not user copy outside the Details disclosure.
describe('inspector Details disclosure', () => {
  it('renders no opaque id, operation id or raw reason until Details is opened', async () => {
    const entry = model();
    const reasons = {
      support: 'support reason sentinel',
      arch: 'architecture reason sentinel',
      runnable: 'runnable reason sentinel',
      complete: 'complete reason sentinel',
      tested: 'tested reason sentinel',
      unknown: 'parameter count is not measured during metadata-only catalog scans',
      capability: 'capability reason sentinel',
      removal: 'removal reason sentinel',
      error: 'last error sentinel',
      action: 'bootstrap action reason sentinel',
    };
    const readyEntry = {
      ...entry,
      lifecycle: { ...entry.lifecycle, state: 'ready' as const, last_error: reasons.error },
      capabilities: [{ task: 'chat' as const, phase: 'provider_ready' as const, available: true, reason: reasons.capability }],
      removal: { eligible: false, reason: reasons.removal, instructions: 'removal instructions sentinel' },
      metadata: {
        ...entry.metadata,
        support: { ...entry.metadata.support, reason: reasons.support, architecturally_supported_reason: reasons.arch, runnable_on_backend_reason: reasons.runnable, complete_reason: reasons.complete, tested_checkpoint_reason: reasons.tested },
        unknown_reasons: { ...entry.metadata.unknown_reasons, parameter_count: reasons.unknown },
      },
    };
    const loadOp: Operation = { ...download, operation_id: 'op_model_load_000001', kind: 'model_load', state: 'succeeded', target: { target_kind: 'model', model_id: entry.identity.id, requested_revision: 4 } };
    const bootstrap = requireValue(state.bootstrap);
    state = {
      ...state,
      catalog: [readyEntry],
      operations: new Map([[loadOp.operation_id, loadOp]]),
      bootstrap: { ...bootstrap, actions: { ...bootstrap.actions, download: { state: 'disabled', reason: reasons.action, instructions: null } } },
    };
    render();
    const page = (): string => document.body.textContent ?? '';
    expect(host.querySelector('aside[aria-label="Model details"]')).not.toBeNull();
    for (const value of [entry.identity.id, loadOp.operation_id, ...Object.values(reasons)]) expect(page()).not.toContain(value);
    expect(page()).not.toMatch(/mdl_|op_/);
    await act(async () => {
      requireValue(host.querySelector<HTMLElement>('[data-testid="models-details"] summary')).click();
      // The toggle event is queued as a task after the attribute flips.
      await new Promise((resolve) => window.setTimeout(resolve, 0));
    });
    expect(host.querySelector<HTMLDetailsElement>('[data-testid="models-details"]')?.open).toBe(true);
    for (const value of [entry.identity.id, ...Object.values(reasons)]) expect(page()).toContain(value);
    // Operation history belongs to Activity: the library never prints operation ids.
    expect(page()).not.toContain(loadOp.operation_id);
  });

  it('shows context only for the runtime revision that matches, and memory only when estimated', () => {
    const entry = { ...model(), metadata: { ...model().metadata, memory_estimate_bytes: null } };
    const runtime = (revision: number): RuntimeSnapshot => ({ ...runtimeFixture, model_id: entry.identity.id, revision, settings: { ...runtimeFixture.settings, effective: { ctx_size: 40960 } } });
    state = { ...state, catalog: [entry], runtimes: new Map([[entry.identity.id, runtime(entry.identity.revision - 1)]]) };
    render();
    const overview = (): string => host.querySelector('.models-inspector > .models-overview')?.textContent ?? '';
    expect(overview()).not.toContain(t('en', 'models.library.context'));
    expect(overview()).not.toContain(t('en', 'models.library.memory'));
    state = { ...state, catalog: [{ ...entry, metadata: { ...entry.metadata, memory_estimate_bytes: 2048 } }], runtimes: new Map([[entry.identity.id, runtime(entry.identity.revision)]]) };
    render();
    expect(overview()).toContain(`${t('en', 'models.library.context')}40,960`);
    expect(overview()).toContain(`${t('en', 'models.library.memory')}2 KiB`);
  });
});

describe('inspector below 1100 px', () => {
  afterEach(() => vi.unstubAllGlobals());
  it('opens as a drawer dialog from Inspect, never from a remembered selection, and closing keeps the selection', async () => {
    vi.stubGlobal('matchMedia', (query: string) => ({ matches: !query.includes('min-width: 1100px'), media: query, addEventListener: () => undefined, removeEventListener: () => undefined }));
    render();
    expect(host.querySelector('aside[aria-label="Model details"]')).toBeNull();
    const drawer = (): HTMLElement => requireValue(host.querySelector<HTMLElement>('aside.drawer'));
    expect(drawer().getAttribute('role')).toBe('dialog');
    expect(drawer().classList.contains('drawer--open')).toBe(false);
    await act(async () => requireValue(host.querySelector<HTMLButtonElement>('[aria-label^="Inspect"]')).click());
    await act(async () => { await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))); });
    expect(drawer().classList.contains('drawer--open')).toBe(true);
    expect(drawer().textContent).toContain(t('en', 'models.library.details'));
    expect(actions.selectModel).not.toHaveBeenCalled();
    await act(async () => requireValue(drawer().querySelector<HTMLButtonElement>('.drawer__close-btn')).click());
    expect(drawer().classList.contains('drawer--open')).toBe(false);
    expect(actions.selectModel).not.toHaveBeenCalled();
  });
});
