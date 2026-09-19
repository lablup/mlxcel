// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App, routeFromHash } from '../../app';
import { WebUiProvider } from '../../state';
import { t } from '../../i18n/catalog';

let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() });
  localStorage.clear();
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView'); window.history.replaceState(null, '', '/webui/#models'); });

function renderAt(hash: string): void {
  window.history.replaceState(null, '', `/webui/${hash}`);
  act(() => root.render(<WebUiProvider fetchImpl={(() => { throw new Error('Unexpected API request'); }) as typeof fetch}><App /></WebUiProvider>));
}
const activeTab = (): string | null | undefined => host.querySelector('[role="tab"][aria-selected="true"]')?.textContent;
const description = (): string | null | undefined => host.querySelector('.page-header__description')?.textContent;

describe('Settings route sections', () => {
  it('parses #settings/<section> into the settings route with that section', () => {
    expect(routeFromHash('#settings/model')).toEqual({ route: 'settings', section: 'model' });
    expect(routeFromHash('#settings/server')).toEqual({ route: 'settings', section: 'server' });
    expect(routeFromHash('#/settings/requests')).toEqual({ route: 'settings', section: 'requests' });
  });

  it('opens Appearance for a bare #settings or a section this build does not know', () => {
    expect(routeFromHash('#settings')).toEqual({ route: 'settings', section: 'appearance' });
    expect(routeFromHash('#settings/')).toEqual({ route: 'settings', section: 'appearance' });
    expect(routeFromHash('#settings/bogus')).toEqual({ route: 'settings', section: 'appearance' });
    expect(routeFromHash('#settings/model/extra')).toEqual({ route: 'settings', section: 'appearance' });
  });

  it('leaves every other route as before', () => {
    expect(routeFromHash('#chat')).toEqual({ route: 'chat', section: 'appearance' });
    expect(routeFromHash('#chat/model')).toEqual({ route: 'models', section: 'appearance' });
    expect(routeFromHash('')).toEqual({ route: 'models', section: 'appearance' });
  });

  it('renders the four tabs, opens #settings/server directly and names the section in the one PageHeader', () => {
    renderAt('#settings/server');
    expect([...host.querySelectorAll('[role="tab"]')].map((tab) => tab.textContent)).toEqual(['Appearance', 'Requests', 'Model', 'Server']);
    expect(activeTab()).toBe('Server');
    expect(host.querySelectorAll('h1')).toHaveLength(1);
    expect(description()).toBe(t('en', 'settings.section.server.description'));
    // Signed out: the server tab explains itself instead of reading anything.
    expect(host.textContent).toContain(t('en', 'settings.server.unavailable.title'));
  });

  it('switches tabs by replacing the hash, so Back leaves Settings instead of stepping through tabs', () => {
    renderAt('#settings');
    expect(activeTab()).toBe('Appearance');
    expect(description()).toBe(t('en', 'settings.browser_only'));
    const depth = window.history.length;
    const push = vi.spyOn(window.history, 'pushState');
    act(() => [...host.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((tab) => tab.textContent === 'Model')?.click());
    expect(window.location.hash).toBe('#settings/model');
    expect(activeTab()).toBe('Model');
    act(() => [...host.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((tab) => tab.textContent === 'Requests')?.click());
    expect(window.location.hash).toBe('#settings/requests');
    expect(window.history.length).toBe(depth);
    expect(push).not.toHaveBeenCalled();
  });

  it('follows a hash change to another section, as a link to #settings/model does', () => {
    renderAt('#settings');
    act(() => { window.history.replaceState(null, '', '#settings/model'); window.dispatchEvent(new HashChangeEvent('hashchange')); });
    expect(activeTab()).toBe('Model');
  });

  it('keeps the appearance controls on the Appearance tab', () => {
    renderAt('#settings/appearance');
    for (const id of ['settings-theme', 'settings-color-scheme', 'settings-material', 'settings-locale', 'settings-high-contrast', 'settings-glass-intensity', 'settings-reduce-motion', 'settings-reduce-transparency']) expect(host.querySelector(`[data-testid="${id}"]`), id).not.toBeNull();
    act(() => host.querySelector<HTMLInputElement>('[data-testid="settings-reduce-motion"]')?.click());
    expect(document.documentElement.dataset.reduceMotion).toBe('true');
  });
});
