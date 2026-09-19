// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { WebUiSnapshot } from '../api/types';
import { App } from '../app';
import { t } from '../i18n/catalog';
import { ProductConnectionSurface } from '../provider-surfaces';
import { WebUiProvider } from '../state';
import { initialSnapshot } from '../state/reducer';
import { PageHeader } from './primitives';

let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() });
  HTMLDialogElement.prototype.showModal = vi.fn(function showModal(this: HTMLDialogElement) { this.setAttribute('open', ''); });
  HTMLDialogElement.prototype.close = vi.fn(function close(this: HTMLDialogElement) { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); });
  localStorage.clear();
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals(); Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView'); });

const settle = async (): Promise<void> => { await act(async () => { await new Promise((resolve) => window.setTimeout(resolve, 10)); }); };
const byTestId = (id: string): HTMLElement => {
  const element = document.querySelector<HTMLElement>(`[data-testid="${id}"]`);
  if (!element) throw new Error(`Missing ${id}`);
  return element;
};
function renderRoute(hash: string): void {
  window.history.replaceState(null, '', `/webui/#${hash}`);
  act(() => root.render(<WebUiProvider fetchImpl={(() => { throw new Error('Unexpected API request'); }) as typeof fetch}><App /></WebUiProvider>));
}
function expectPageTitle(titleId: string, title: string, descriptionId?: string, description?: string): void {
  const heading = byTestId(titleId);
  expect(heading.tagName).toBe('H1');
  expect(heading.textContent).toBe(title);
  expect(document.querySelectorAll('h1')).toHaveLength(1);
  if (descriptionId) expect(byTestId(descriptionId).textContent).toBe(description);
}
const expectSharedHeader = (titleId: string, descriptionId?: string): void => {
  expect(byTestId(titleId).matches('.page-header .page-header__title')).toBe(true);
  if (descriptionId) expect(byTestId(descriptionId).matches('.page-header .page-header__description')).toBe(true);
};

describe('PageHeader adoption', () => {
  it('keeps the Settings title as the single focus fallback with its test ids', () => {
    renderRoute('settings');
    expectPageTitle('settings-title', t('en', 'settings.title'), 'settings-appearance', t('en', 'settings.browser_only'));
    expect(document.querySelectorAll('[data-dialog-focus-fallback]')).toEqual(document.querySelectorAll('[data-testid="settings-title"]'));
    expect(byTestId('settings-title').tabIndex).toBe(-1);
    expectSharedHeader('settings-title', 'settings-appearance');
  });

  it('lands focus on the Settings title when a dialog closes after its trigger was detached', async () => {
    renderRoute('settings');
    const trigger = document.createElement('button');
    trigger.textContent = 'Temporary trigger';
    document.body.append(trigger);
    trigger.focus();
    act(() => { trigger.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true })); });
    await settle();
    expect(byTestId('help-dialog').hasAttribute('open')).toBe(true);
    trigger.remove();
    await act(async () => byTestId('help-dialog').querySelector<HTMLButtonElement>('[data-testid="dialog-close"]')?.click());
    await settle();
    expect(document.activeElement).toBe(byTestId('settings-title'));
    expectSharedHeader('settings-title');
  });

  it('keeps the gallery title and subtitle test ids', () => {
    renderRoute('gallery');
    expectPageTitle('gallery-title', t('en', 'gallery.title'), 'gallery-subtitle', t('en', 'gallery.subtitle'));
    expectSharedHeader('gallery-title', 'gallery-subtitle');
  });

  it('keeps the signed-out connection surface title and prompt', () => {
    renderRoute('models');
    expectPageTitle('models-title', t('en', 'models.title'), 'connection-prompt-body', t('en', 'connection.prompt.body'));
    expectSharedHeader('models-title', 'connection-prompt-body');
  });

  it.each([['offline', 'connection-authenticated-body'], ['schema-mismatch', undefined]] as const)('keeps the %s connection surface title', (connection, descriptionId) => {
    const snapshot: WebUiSnapshot = { ...initialSnapshot(), auth: { status: 'authenticated', tokenPresent: true }, connection };
    act(() => root.render(<ProductConnectionSurface locale="en" title={t('en', 'activity.title')} titleTestId="activity-title" snapshot={snapshot} authFailure={null} onLogin={() => undefined} onLogout={() => undefined} onRetry={() => undefined} onRecoverSchema={() => undefined} />));
    expectPageTitle('activity-title', t('en', 'activity.title'), descriptionId, descriptionId ? t('en', 'connection.authenticated.body') : undefined);
    expectSharedHeader('activity-title', descriptionId);
  });

  it('always makes the title the focus fallback and re-applies the error test id each time the error mounts', () => {
    const retry = vi.fn();
    const header = (error: string | null): React.JSX.Element => <PageHeader title="Route" titleTestId="route-title" actions={<button type="button">Act</button>} error={error} errorDetail="detail" errorTestId="route-error" onRetry={retry} retryLabel="Refresh route" />;
    act(() => root.render(header(null)));
    expect(byTestId('route-title').hasAttribute('data-dialog-focus-fallback')).toBe(true);
    expect(byTestId('route-title').tabIndex).toBe(-1);
    expect(document.querySelector('.page-header__actions')?.textContent).toBe('Act');
    expect(document.querySelector('.page-header__error')).toBeNull();
    for (const message of ['First failure', 'Second failure']) {
      act(() => root.render(header(message)));
      expect(byTestId('route-error').matches('.page-header__error[role="alert"]')).toBe(true);
      expect(byTestId('route-error').textContent).toContain(message);
      act(() => byTestId('route-error').querySelector<HTMLButtonElement>('button')?.click());
      act(() => root.render(header(null)));
      expect(document.querySelector('[data-testid="route-error"]')).toBeNull();
    }
    expect(retry).toHaveBeenCalledTimes(2);
  });
});
