// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { CatalogEntry, WebUiSnapshot } from '../../api/types';
import { WebUiHttpError } from '../../api/client';
import { useLoadProfile } from '../settings/load-profiles';
import { ModelsLibrary } from './screen';
import { model, snapshot } from './test-fixtures';

let state: WebUiSnapshot;
const actions = { selectModel: vi.fn(), loadModel: vi.fn(), refresh: vi.fn() };
vi.mock('../../state', () => ({ useWebUi: () => state, useWebUiActions: () => actions }));
let host: HTMLDivElement;
let root: Root;
let store: ReturnType<typeof useLoadProfile>;
let editorId: string;
const target = model();
const other = { ...target, identity: { ...target.identity, id: `mdl_${'b'.repeat(43)}`, display_name: 'other-model' } };
function Screen(): React.JSX.Element {
  store = useLoadProfile(editorId);
  return <ModelsLibrary locale="en" />;
}
function render(): void { act(() => root.render(<Screen />)); }
function node<T extends Element>(selector: string): T {
  const found = host.querySelector<T>(selector);
  if (!found) throw new Error(`Missing ${selector}`);
  return found;
}
async function click(id: string): Promise<void> {
  await act(async () => node<HTMLButtonElement>(`[data-testid="${id}"]`).click());
}
beforeEach(() => {
  vi.resetAllMocks();
  actions.loadModel.mockResolvedValue(undefined);
  state = snapshot(); editorId = target.identity.id;
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); };
  render();
  act(() => store.importJson('{"version":1,"reusable":{},"models":{}}'));
});
afterEach(() => { act(() => root.unmount()); host.remove(); });

it('uses canonical reusable defaults only on explicit load and freezes the submitted values', async () => {
  act(() => store.save({ ctx_size: 2048, n_parallel: 2, kv_cache_mode: 'int8' }, 'reusable'));
  expect(host.textContent).toContain('Pending browser profile; not applied yet.');
  expect(node<HTMLAnchorElement>('[data-testid="models-pending-profile"] a').getAttribute('href')).toBe('#settings');
  expect(actions.loadModel).not.toHaveBeenCalled();
  state = { ...state, selectedModelId: null };
  render();
  await act(async () => node<HTMLButtonElement>('[aria-label^="Inspect"]').click());
  expect(actions.selectModel).toHaveBeenCalledWith(target.identity.id);
  expect(actions.loadModel).not.toHaveBeenCalled();
  state = { ...state, selectedModelId: target.identity.id };
  render();
  await click('models-load');
  const request = actions.loadModel.mock.calls[0][0];
  expect(request.load_profile).toEqual({ ctx_size: 2048, n_parallel: 2, kv_cache_mode: 'int8' });
  act(() => store.save({ ctx_size: 4096 }, 'reusable'));
  expect(request.load_profile.ctx_size).toBe(2048);
  expect(host.textContent).toContain('4096');
});

it('replaces rather than merges reusable defaults and omits an explicitly empty model profile', async () => {
  act(() => store.save({ ctx_size: 2048, n_parallel: 2, kv_cache_mode: 'int8' }, 'reusable'));
  act(() => store.save({ ctx_size: 1024 }, 'model'));
  await click('models-load');
  expect(actions.loadModel.mock.calls[0][0].load_profile).toEqual({ ctx_size: 1024 });
  act(() => store.save({}, 'model'));
  await click('models-load');
  expect(actions.loadModel.mock.calls[1][0]).not.toHaveProperty('load_profile');
  expect(host.textContent).toContain('Server defaults; see Settings for scope and overrides.');
  act(() => store.reset('model'));
  await click('models-load');
  expect(actions.loadModel.mock.calls[2][0].load_profile).toEqual({ ctx_size: 2048, n_parallel: 2, kv_cache_mode: 'int8' });
});

it('loads the confirmation target profile even after selection changes, using the latest explicit save', async () => {
  const idle: CatalogEntry = { ...other, identity: { ...other.identity, id: `mdl_${'c'.repeat(43)}` }, lifecycle: { ...other.lifecycle, state: 'ready' } };
  state = { ...state, catalog: [target, other, idle] }; render();
  act(() => store.save({ ctx_size: 2048 }, 'model'));
  actions.loadModel.mockRejectedValueOnce(new WebUiHttpError(409, { request_id: 'req_capacity', error: { code: 'conflict', message: 'capacity full', retryable: false } }));
  await click('models-load');
  state = { ...state, selectedModelId: other.identity.id }; editorId = other.identity.id; render();
  act(() => store.save({ ctx_size: 8192 }, 'model'));
  editorId = target.identity.id; render();
  act(() => store.save({ ctx_size: 4096, n_parallel: 2 }, 'model'));
  await act(async () => {
    const select = node<HTMLSelectElement>('[data-testid="models-eviction-target"]');
    select.value = idle.identity.id; select.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await click('models-confirm-submit');
  expect(actions.loadModel.mock.calls[1][0]).toMatchObject({ model_id: target.identity.id, eviction_target_id: idle.identity.id, eviction_target_expected_revision: idle.identity.revision, load_profile: { ctx_size: 4096, n_parallel: 2 } });
  expect(actions.loadModel).toHaveBeenCalledTimes(2);
});
