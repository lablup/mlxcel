// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { WebUiSnapshot, Operation } from '../../api/types';
import { WebUiHttpError } from '../../api/client';
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
  it('renders empty-store guidance and root permission recovery without POSTs', () => {
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
    expect(host.textContent).toContain('permission denied');
    expect(host.textContent).toContain('--models-dir /path/to/models');
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
    expect(host.textContent).toContain('running');
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
    const recovery = Array.from(host.querySelectorAll('button')).find(
      (node) => node.textContent === 'Capacity / conflicting operation',
    );
    await act(async () => recovery?.click());
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

it('prefers the catalog display name for operation titles and retains opaque identity', () => {
  const op: Operation = {
    ...download,
    kind: 'model_load',
    target: { target_kind: 'model', model_id: model().identity.id, requested_revision: 4 },
  };
  state = { ...state, operations: new Map([[op.operation_id, op]]) };
  render();
  expect(host.querySelector('[data-testid="models-operation"] strong')?.textContent).toBe(
    model().identity.display_name,
  );
  expect(host.querySelector('[data-testid="models-operation"]')?.textContent).toContain(model().identity.id);
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
