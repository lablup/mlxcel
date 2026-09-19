// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { WebUiHttpError } from '../../api/client';
import type { CatalogEntry, ModelActionRequest, WebUiSnapshot } from '../../api/types';
import { t } from '../../i18n/catalog';
import { model, snapshot } from '../models/test-fixtures';
import { useLoadProfile } from '../settings/load-profiles';
import { ModelPicker, pickerOptions } from './model-picker';

const mocked = vi.hoisted(() => ({ state: null as WebUiSnapshot | null, selectModel: vi.fn(), loadModel: vi.fn<(request: ModelActionRequest) => Promise<void>>() }));
vi.mock('../../state', () => ({ useWebUi: () => mocked.state, useWebUiActions: () => ({ selectModel: mocked.selectModel, loadModel: mocked.loadModel }) }));

const chat = (phase: 'pre_load' | 'provider_ready', available = true): CatalogEntry['capabilities'][number] => ({ task: 'chat', phase, available, reason: null });
function entry(name: string, state: CatalogEntry['lifecycle']['state'], capabilities: CatalogEntry['capabilities']): CatalogEntry {
  const base = model();
  return { ...base, identity: { ...base.identity, id: `mdl_${name.padEnd(43, 'x')}`, display_name: name, inference_id: name }, lifecycle: { ...base.lifecycle, state }, capabilities };
}
const zeta = entry('zeta', 'unloaded', [chat('pre_load')]);
const beta = entry('beta', 'ready', [chat('provider_ready')]);
const alpha = entry('alpha', 'ready', [chat('provider_ready')]);
const gamma = entry('gamma', 'loading', [chat('pre_load')]);
const delta = entry('delta', 'failed', [chat('pre_load')]);
const embed = entry('embed', 'ready', [{ task: 'embedding', phase: 'provider_ready', available: true, reason: null }]);
const muted = entry('muted', 'ready', [chat('provider_ready', false)]);
const catalog = [zeta, beta, embed, alpha, gamma, muted, delta];

describe('pickerOptions', () => {
  it('lists Ready chat models first, then loading, then loadable, each by name, and leaves out models without chat', () => {
    const options = pickerOptions(catalog, beta, 'en');
    expect(options.map((option) => option.label)).toEqual(['alpha', 'beta', 'gamma', 'delta', 'zeta']);
    expect(options.map((option) => option.description)).toEqual(['Ready to chat', 'Ready to chat', 'Loading', 'Load failed', 'Not loaded']);
    expect(options.every((option) => option.value !== '')).toBe(true);
  });
  it('keeps a selection that cannot chat, last, and adds the placeholder only when nothing is selected', () => {
    expect(pickerOptions(catalog, embed, 'en').at(-1)).toEqual({ value: embed.identity.id, label: 'embed', description: t('en', 'chat.model.group.unavailable') });
    expect(pickerOptions(catalog, muted, 'en').at(-1)?.label).toBe('muted');
    const unselected = pickerOptions(catalog, undefined, 'en');
    expect(unselected[0]).toEqual({ value: '', label: t('en', 'chat.model.choose') });
    expect(unselected[1].label).toBe('alpha');
  });
  it('never prints lifecycle enum values and localizes every description', () => {
    const descriptions = pickerOptions(catalog, embed, 'ko').map((option) => option.description ?? '');
    for (const description of descriptions.slice(1)) expect(description).not.toMatch(/ready|loading|unloaded|failed|draining|unloading/i);
    expect(descriptions).toContain(t('ko', 'chat.model.group.ready'));
  });
});

let host: HTMLDivElement;
let root: Root;
let profiles: ReturnType<typeof useLoadProfile>;
function Profiles(): null { profiles = useLoadProfile(null); return null; }
function render(): void { act(() => root.render(<><Profiles /><ModelPicker locale="en" /></>)); }
function byTestId<T extends Element>(id: string): T {
  const found = document.querySelector<T>(`[data-testid="${id}"]`);
  if (!found) throw new Error(`Missing ${id}`);
  return found;
}
async function click(element: Element): Promise<void> { await act(async () => { (element as HTMLElement).click(); }); }
const target = model();

beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() });
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); };
  mocked.selectModel.mockReset(); mocked.loadModel.mockReset(); mocked.loadModel.mockResolvedValue(undefined);
  mocked.state = snapshot(target);
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  render();
  act(() => profiles.importJson('{"version":1,"reusable":{},"models":{}}'));
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView'); });

describe('ModelPicker', () => {
  it('selects without loading: choosing a model only calls selectModel', async () => {
    mocked.state = { ...snapshot(target), catalog: [target, alpha], selectedModelId: null };
    render();
    expect(host.textContent).toContain(t('en', 'chat.model.hint.choose'));
    await click(host.querySelector('.select__trigger') as Element);
    const options = [...document.querySelectorAll('[role="option"]')];
    expect(options[1].getAttribute('aria-label')).toBe(`alpha. ${t('en', 'chat.model.group.ready')}`);
    await click(options[1]);
    expect(mocked.selectModel).toHaveBeenCalledExactlyOnceWith(alpha.identity.id);
    expect(mocked.loadModel).not.toHaveBeenCalled();
  });
  it('offers Load for an unloaded selection, confirms, then sends the Models request once', async () => {
    expect(host.textContent).toContain(t('en', 'chat.model.hint.load'));
    await click(byTestId('chat-load'));
    expect(mocked.loadModel).not.toHaveBeenCalled();
    const dialog = byTestId<HTMLDialogElement>('chat-load-dialog');
    expect(dialog.open).toBe(true);
    expect(dialog.textContent).toContain(t('en', 'chat.load.confirm.title', { model: target.identity.display_name }));
    expect(dialog.textContent).not.toContain(t('en', 'models.next_profile.body'));
    await click(byTestId('chat-load-dialog-confirm'));
    expect(mocked.loadModel).toHaveBeenCalledOnce();
    const request = mocked.loadModel.mock.calls[0][0];
    expect(request).toEqual({ action: 'load', model_id: target.identity.id, expected_revision: target.identity.revision, idempotency_key: request.idempotency_key });
  });
  it('carries the pending next-load profile and says so in the confirmation', async () => {
    act(() => profiles.save({ ctx_size: 4096 }, 'reusable'));
    await click(byTestId('chat-load'));
    expect(byTestId('chat-load-dialog').textContent).toContain(t('en', 'models.next_profile.body'));
    await click(byTestId('chat-load-dialog-confirm'));
    expect(mocked.loadModel.mock.calls[0][0].load_profile).toEqual({ ctx_size: 4096 });
  });
  it('sends nothing when the model changed while the confirmation was open', async () => {
    await click(byTestId('chat-load'));
    mocked.state = snapshot({ ...target, identity: { ...target.identity, revision: target.identity.revision + 1 } });
    render();
    await click(byTestId('chat-load-dialog-confirm'));
    expect(mocked.loadModel).not.toHaveBeenCalled();
    expect(host.querySelector('[role="alert"]')?.textContent).toBe(t('en', 'models.library.stale'));
  });
  it('opens the capacity dialog on a conflict and reports the error in the header', async () => {
    mocked.loadModel.mockRejectedValueOnce(new WebUiHttpError(409, { request_id: 'req_capacity', error: { code: 'conflict', message: 'capacity full', retryable: false } }));
    await click(byTestId('chat-load'));
    await click(byTestId('chat-load-dialog-confirm'));
    expect(byTestId<HTMLDialogElement>('models-confirm').open).toBe(true);
    expect(byTestId('models-confirm').textContent).toContain(t('en', 'models.library.capacity'));
    expect(host.querySelector('[role="alert"]')?.textContent).toBe('capacity full');
  });
  it('disables Load when the model cannot be loaded and explains where to look', () => {
    mocked.state = snapshot({ ...target, supported: false });
    render();
    expect(byTestId<HTMLButtonElement>('chat-load').disabled).toBe(true);
    expect(host.textContent).toContain(t('en', 'chat.model.hint.cannot_load'));
    expect(host.querySelector('a[href="#models"]')?.textContent).toBe(t('en', 'chat.no_model.action'));
  });
  it('shows no Load and no hint once the selection is Ready for chat', () => {
    mocked.state = snapshot(alpha);
    render();
    expect(host.querySelector('[data-testid="chat-load"]')).toBeNull();
    expect(byTestId('chat-model-status').textContent).toBe('');
  });
});
