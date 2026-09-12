import React from 'react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import stringsFixture from '../../tests/fixtures/webui/strings.json';
import { App } from './app';
import { LoginView, SchemaMismatchView } from './design-system/primitives';
import { DEFAULT_APPEARANCE, applyAppearance, loadAppearance, saveAppearance } from './design-system/preferences';
import { entries } from './i18n/catalog';
import { WebUiProvider } from './state';

let root: Root | null = null;
let host: HTMLDivElement | null = null;

beforeEach(() => {
  HTMLDialogElement.prototype.showModal = vi.fn(function showModal(this: HTMLDialogElement) { this.setAttribute('open', ''); });
  HTMLDialogElement.prototype.close = vi.fn(function close(this: HTMLDialogElement) { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); });
  localStorage.clear();
  window.history.replaceState(null, '', '/webui/#models');
  host = document.createElement('div');
  host.id = 'root';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  vi.restoreAllMocks();
  act(() => root?.unmount());
  host?.remove();
  document.documentElement.removeAttribute('style');
  document.documentElement.dataset.theme = '';
  root = null;
  host = null;
});

function neverFetch(): Promise<Response> {
  throw new Error('Unexpected WebUI API request before explicit login.');
}

function renderApp(): void {
  act(() => root?.render(<WebUiProvider fetchImpl={neverFetch as typeof fetch}><App /></WebUiProvider>));
}

function keydown(key: string, init: KeyboardEventInit = {}): void {
  const target = document.activeElement instanceof HTMLElement ? document.activeElement : window;
  target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, ...init }));
}

describe('mlxcel WebUI shell', () => {
  it('keeps hash navigation client-side', () => {
    const url = new URL('http://127.0.0.1:8080/private/webui/#models');
    expect(url.pathname).toBe('/private/webui/');
    expect(url.hash).toBe('#models');
  });

  it('opens the command palette with Cmd/Ctrl+K and restores focus on Escape', async () => {
    renderApp();
    const command = document.querySelector<HTMLButtonElement>('[data-testid="toolbar-command"]');
    command?.focus();
    act(() => keydown('k', { metaKey: true }));
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(true);
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="command-dialog"] [data-testid="dialog-close"]')?.click());
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    expect(document.activeElement).toBe(command);
  });

  it('keeps overlays mutually exclusive and suppresses shortcuts inside modals', async () => {
    renderApp();
    act(() => keydown('k', { metaKey: true }));
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(true);
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    act(() => keydown('?'));
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(true);
    expect(document.querySelector('[data-testid="help-dialog"]')?.hasAttribute('open')).toBe(false);
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="command-dialog"] [data-testid="dialog-close"]')?.click());
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    document.body.focus();
    act(() => keydown('?'));
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    expect(document.querySelector('[data-testid="help-dialog"]')?.hasAttribute('open')).toBe(true);
    act(() => keydown('k', { metaKey: true }));
    expect(document.querySelector('[data-testid="help-dialog"]')?.hasAttribute('open')).toBe(true);
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(false);
  });

  it('moves navigation with brackets only when the sidebar owns focus', () => {
    renderApp();
    const navModels = document.querySelector<HTMLAnchorElement>('[data-testid="nav-models"]');
    navModels?.focus();
    act(() => navModels?.dispatchEvent(new KeyboardEvent('keydown', { key: ']', bubbles: true })));
    expect(window.location.hash).toBe('#chat');
    document.querySelector<HTMLButtonElement>('[data-testid="toolbar-command"]')?.focus();
    act(() => keydown(']'));
    expect(window.location.hash).toBe('#chat');
  });

  it('does not open global overlays from editable fields, IME composition, or Alt chords', () => {
    renderApp();
    act(() => document.querySelector<HTMLAnchorElement>('[data-testid="nav-settings"]')?.click());
    const select = document.querySelector<HTMLSelectElement>('[data-testid="settings-theme"]');
    select?.focus();
    act(() => keydown('k', { metaKey: true }));
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(false);
    act(() => keydown('?', { isComposing: true }));
    act(() => keydown('?', { altKey: true }));
    expect(document.querySelector('[data-testid="help-dialog"]')?.hasAttribute('open')).toBe(false);
  });

  it('applies appearance through data attributes without inline style', () => {
    applyAppearance(document.documentElement, { ...DEFAULT_APPEARANCE, glassIntensity: 71, reduceTransparency: true, highContrast: 'on' });
    expect(document.documentElement.dataset.glassIntensity).toBe('71');
    expect(document.documentElement.dataset.material).toBe('opaque');
    expect(document.documentElement.dataset.highContrast).toBe('on');
    expect(document.documentElement.getAttribute('style')).toBeNull();
  });

  it('keeps appearance usable when storage is invalid or unavailable', () => {
    localStorage.setItem('mlxcel.webui.appearance', '[]');
    expect(loadAppearance()).toEqual(DEFAULT_APPEARANCE);
    localStorage.setItem('mlxcel.webui.appearance', '{');
    expect(loadAppearance()).toEqual(DEFAULT_APPEARANCE);
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new DOMException('blocked', 'SecurityError'); });
    expect(loadAppearance()).toEqual(DEFAULT_APPEARANCE);
    vi.restoreAllMocks();
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new DOMException('full', 'QuotaExceededError'); });
    expect(() => saveAppearance(DEFAULT_APPEARANCE)).not.toThrow();
  });


  it('submits LoginView tokens without persisting them and clears recovery state', () => {
    const submitted: string[] = [];
    act(() => root?.render(<><LoginView title="Login" body="Use local key" tokenLabel="Session key" tokenHelp="Memory only" submitLabel="Connect" logoutLabel="Clear" onSubmit={(token) => submitted.push(token)} onLogout={() => submitted.push('logout')} testId="login-view" /><SchemaMismatchView title="Mismatch" body="Update required" actionLabel="Reload" onRecover={() => submitted.push('recover')} /></>));
    const form = document.querySelector<HTMLFormElement>('[data-testid="login-view"]');
    const input = document.querySelector<HTMLInputElement>('input[type="password"]');
    expect(form).not.toBeNull();
    expect(input).not.toBeNull();
    if (!input || !form) return;
    expect(form.getAttribute('autocomplete')).toBe('off');
    expect(input.getAttribute('autocomplete')).toBe('off');
    expect(input.getAttribute('spellcheck')).toBe('false');
    expect(input.getAttribute('autocapitalize')).toBe('none');
    expect(input.getAttribute('autocorrect')).toBe('off');
    const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    act(() => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
    expect(submitted).toEqual([]);
    act(() => { valueSetter?.call(input, 'secret-token'); input.dispatchEvent(new Event('input', { bubbles: true })); });
    act(() => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
    expect(submitted).toContain('secret-token');
    expect(input?.value).toBe('');
    expect(localStorage.getItem('secret-token')).toBeNull();
    act(() => document.querySelectorAll('button')[1]?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(submitted).toContain('logout');
    act(() => document.querySelectorAll('button')[2]?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(submitted).toContain('recover');
    act(() => root?.render(<LoginView title="Login" body="Use local key" tokenLabel="Session key" tokenHelp="Memory only" submitLabel="Connect" error="Denied" busy onSubmit={(token) => submitted.push(token)} testId="login-view" />));
    const busyInput = document.querySelector<HTMLInputElement>('input[type="password"]');
    expect(busyInput?.disabled).toBe(true);
    expect(busyInput?.getAttribute('aria-invalid')).toBe('true');
    act(() => document.querySelector<HTMLFormElement>('[data-testid="login-view"]')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
    expect(submitted.filter((value) => value === '')).toHaveLength(0);
    expect(submitted.filter((value) => value === 'secret-token')).toHaveLength(1);
    expect(document.body.textContent).toContain('Denied');
  });

  it('blocks programmatic LoginView submits while busy even with a buffered token', () => {
    const submitted: string[] = [];
    const renderLogin = (busy: boolean) => root?.render(<LoginView title="Login" body="Use local key" tokenLabel="Session key" tokenHelp="Memory only" submitLabel="Connect" busy={busy} onSubmit={(token) => submitted.push(token)} testId="login-view" />);
    act(() => renderLogin(false));
    const input = document.querySelector<HTMLInputElement>('input[type="password"]');
    const form = document.querySelector<HTMLFormElement>('[data-testid="login-view"]');
    expect(input).not.toBeNull();
    expect(form).not.toBeNull();
    if (!input || !form) return;
    const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    act(() => { valueSetter?.call(input, 'busy-token'); input.dispatchEvent(new Event('input', { bubbles: true })); });
    act(() => renderLogin(true));
    act(() => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
    expect(submitted).toEqual([]);
  });

  it('keeps the checked string fixture synchronized with typed keys', () => {
    const fixture = stringsFixture.strings.map((entry) => entry.key).sort();
    const typed = entries.map((entry) => entry.key).sort();
    expect(fixture).toEqual(typed);
    for (const entry of stringsFixture.strings) {
      expect(entry.en).toBeTruthy();
      expect(entry.ko).toBeTruthy();
      expect(entry.test_id).toMatch(/^[a-z0-9-]+$/);
    }
  });
});
