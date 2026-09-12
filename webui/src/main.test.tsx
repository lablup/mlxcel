import React from 'react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import stringsFixture from '../../tests/fixtures/webui/strings.json';
import { App } from './app';
import { DEFAULT_APPEARANCE, applyAppearance, loadAppearance, saveAppearance } from './design-system/preferences';
import { entries } from './i18n/catalog';

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

function renderApp(): void {
  act(() => root?.render(<App />));
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
    act(() => document.querySelector<HTMLAnchorElement>('[data-testid="nav-gallery"]')?.click());
    const input = document.querySelector<HTMLInputElement>('[data-testid="gallery-field"]');
    input?.focus();
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
