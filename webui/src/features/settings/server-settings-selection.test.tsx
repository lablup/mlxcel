import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { expect, it, vi } from 'vitest';
import bootstrap from '../../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalog from '../../../../tests/fixtures/webui/examples/catalog.page.json';
import { initialSnapshot } from '../../state/reducer';
import type { BootstrapResponse, CatalogEntry, WebUiSnapshot } from '../../api/types';
import { validateAgainstSchema } from '../../api/jsonSchema';
import { ServerSettings } from './server-settings';
const mocks = vi.hoisted(() => ({ snapshot: null as WebUiSnapshot | null, actions: { selectModel: vi.fn(), loadModel: vi.fn(), unloadModel: vi.fn(), getModelProps: vi.fn(), getSettings: vi.fn() } }));
vi.mock('../../state', () => ({ useWebUi: () => mocks.snapshot, useWebUiActions: () => mocks.actions }));
it('selects an opaque ready model through the sole provider authority without loading it', async () => {
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() });
  const entry = structuredClone(catalog.items[0]) as unknown as CatalogEntry;
  const ready = { ...entry, lifecycle: { ...entry.lifecycle, state: 'ready' as const, worker_exit_observed: false } };
  const connectedBootstrap = {...bootstrap, server: {...bootstrap.server, mode: 'router_pool' as const}};
  validateAgainstSchema('CatalogEntry', ready, '$');
  validateAgainstSchema('BootstrapResponse', connectedBootstrap, '$');
  mocks.snapshot = { ...initialSnapshot(), auth: { status: 'authenticated', tokenPresent: true }, connection: 'ready', bootstrap: connectedBootstrap as unknown as BootstrapResponse, catalog: [ready] };
  mocks.actions.getSettings.mockResolvedValue({schema: [], current: {}, fingerprint: 'a'.repeat(64)});
  mocks.actions.getModelProps.mockResolvedValue({ nCtx: 2048, totalSlots: 1, kvCacheMode: 'fp16', geometry: null });
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    await act(async () => root.render(<ServerSettings locale="en" />));
    expect(mocks.actions.getModelProps).not.toHaveBeenCalled();
    const trigger = host.querySelector('[data-testid="settings-model-selector"] [role="combobox"]');
    expect(trigger).not.toBeNull();
    act(() => { trigger?.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true })); });
    act(() => { trigger?.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true })); });
    act(() => { trigger?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })); });
    expect(mocks.actions.selectModel).toHaveBeenCalledWith(ready.identity.id);
    expect(mocks.actions.loadModel).not.toHaveBeenCalled(); expect(mocks.actions.unloadModel).not.toHaveBeenCalled();
    mocks.snapshot = { ...mocks.snapshot, selectedModelId: ready.identity.id };
    await act(async () => root.render(<ServerSettings locale="en" />));
    expect(mocks.actions.getModelProps).toHaveBeenCalledWith(ready.identity.id, expect.any(AbortSignal));
    expect([...host.querySelectorAll('input')].some((input) => input.value === '2048')).toBe(true);
  } finally { act(() => root.unmount()); host.remove(); }
});
