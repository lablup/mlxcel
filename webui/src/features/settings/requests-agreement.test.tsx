// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Settings > Requests and the Chat "Parameters for next turn" hint resolve the same state to the same value.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import bootstrap from '../../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalog from '../../../../tests/fixtures/webui/examples/catalog.page.json';
import type { BootstrapResponse, CatalogEntry, WebUiSnapshot } from '../../api/types';
import type { SettingsResponse } from '../../api/settings';
import { initialSnapshot } from '../../state/reducer';
import { TurnParameters } from '../chat/parameters';
import { GenerationSettings } from './generation-settings';
import { useGenerationDefaults } from './generation-preferences';

const mocks = vi.hoisted(() => ({ snapshot: null as WebUiSnapshot | null, actions: { getSettings: vi.fn(), selectModel: vi.fn() } }));
vi.mock('../../state', () => ({ useWebUi: () => mocks.snapshot, useWebUiActions: () => mocks.actions }));

const entry = structuredClone(catalog.items[0]) as unknown as CatalogEntry;
const ready: CatalogEntry = { ...entry, lifecycle: { ...entry.lifecycle, state: 'ready', worker_exit_observed: false } };
const settings: SettingsResponse = { schema: [{ name: 'default_temperature', type: 'float', default: 0.800000011920929, mutable: true, allowed: null, help: 'Temperature' }, { name: 'default_seed', type: 'int_or_null', default: null, mutable: true, allowed: null, help: 'Seed' }], current: { default_temperature: 0.800000011920929, default_seed: null }, fingerprint: 'a'.repeat(64) };

let host: HTMLDivElement;
let root: Root;
let generation: ReturnType<typeof useGenerationDefaults> | undefined;
beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() });
  mocks.snapshot = { ...initialSnapshot(), auth: { status: 'authenticated', tokenPresent: true }, connection: 'ready', bootstrap: bootstrap as unknown as BootstrapResponse, catalog: [ready], selectedModelId: ready.identity.id };
  mocks.actions.getSettings.mockReset().mockResolvedValue(settings);
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(() => { act(() => generation?.reset()); act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView'); });

function ChatHint(): React.JSX.Element {
  generation = useGenerationDefaults();
  return <TurnParameters defaults={generation.defaults} draft={{}} onChange={() => undefined} locale="en" />;
}
const hintFor = (selector: string): string => host.querySelector(`${selector}`)?.closest('.ds-field')?.querySelector('small[data-tone="hint"]')?.textContent ?? '';
const settingsHint = (field: string): string => hintFor(`[data-testid="settings-request-${field}"]`);
const chatHint = (field: string): string => [...host.querySelectorAll('.chat-parameters label')].find((label) => label.querySelector('span')?.textContent === `Next turn ${field}`)?.querySelector('small')?.textContent ?? '';

it('shows the same resolved value and source in Settings and the Chat hint', async () => {
  await act(async () => root.render(<><GenerationSettings locale="en" /><ChatHint /></>));
  expect(mocks.actions.getSettings).toHaveBeenCalledWith(ready.identity.id, expect.any(AbortSignal));
  expect(settingsHint('temperature')).toBe('temperature · Next request: 0.8 (server default)');
  expect(chatHint('temperature')).toBe('Inherited: 0.8 (server default)');
  expect(settingsHint('seed')).toContain('not set (server default)');
  expect(chatHint('seed')).toBe('Inherited: not set (server default)');
  // The schema has no default_top_p entry: unknown, in both places.
  expect(settingsHint('top_p')).toContain('server default, not readable');
  expect(chatHint('top_p')).toBe('Inherited: server default, not readable');
  act(() => generation?.setDefaults({ temperature: 0.5 }));
  expect(settingsHint('temperature')).toContain('Next request: 0.5 (session default)');
  expect(chatHint('temperature')).toBe('Inherited: 0.5 (session default)');
});

it('resolves every server default as unknown when no ready model is selected', async () => {
  mocks.snapshot = { ...mocks.snapshot as WebUiSnapshot, selectedModelId: null };
  await act(async () => root.render(<><GenerationSettings locale="en" /><ChatHint /></>));
  expect(mocks.actions.getSettings).not.toHaveBeenCalled();
  expect(host.querySelector('[data-testid="settings-requests-server"]')?.textContent).toBe('No ready model selected, so server defaults are not readable.');
  for (const field of ['max_tokens', 'temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'seed']) {
    expect(settingsHint(field)).toContain('server default, not readable');
    expect(chatHint(field)).toBe('Inherited: server default, not readable');
  }
});

it('does not carry server defaults across a server restart that reuses the model id and revision', async () => {
  await act(async () => root.render(<><GenerationSettings locale="en" /><ChatHint /></>));
  expect(chatHint('temperature')).toBe('Inherited: 0.8 (server default)');
  const restarted = { ...bootstrap, server: { ...bootstrap.server, server_instance_id: `${bootstrap.server.server_instance_id}-restarted` } };
  mocks.snapshot = { ...mocks.snapshot as WebUiSnapshot, bootstrap: restarted as unknown as BootstrapResponse };
  await act(async () => root.render(<ChatHint />));
  expect(chatHint('temperature')).toBe('Inherited: server default, not readable');
});
