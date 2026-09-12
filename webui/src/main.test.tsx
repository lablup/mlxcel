import React from 'react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import stringsFixture from '../../tests/fixtures/webui/strings.json';
import { App } from './app';
import { applyAppearance, DEFAULT_APPEARANCE } from './design-system/preferences';
import { entries } from './i18n/catalog';

let root: Root | null = null;
let host: HTMLDivElement | null = null;

beforeEach(() => {
  HTMLDialogElement.prototype.showModal = vi.fn(function showModal(this: HTMLDialogElement) { this.open = true; });
  HTMLDialogElement.prototype.close = vi.fn(function close(this: HTMLDialogElement) { this.open = false; this.dispatchEvent(new Event('close')); });
  localStorage.clear();
  window.history.replaceState(null, '', '/webui/#models');
  host = document.createElement('div');
  host.id = 'root';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

function renderApp(): void {
  act(() => root?.render(<App />));
}

function keydown(key: string, init: KeyboardEventInit = {}): void {
  window.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, ...init }));
}

describe('mlxcel WebUI shell', () => {
  it('keeps hash navigation client-side', () => {
    const url = new URL('http://127.0.0.1:8080/private/webui/#models');
    expect(url.pathname).toBe('/private/webui/');
    expect(url.hash).toBe('#models');
  });

  it('opens the command palette with Cmd/Ctrl+K and restores focus on Escape', () => {
    renderApp();
    const command = document.querySelector<HTMLButtonElement>('[data-testid="toolbar-command"]');
    command?.focus();
    act(() => keydown('k', { metaKey: true }));
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(true);
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="dialog-close"]')?.click());
    expect(document.activeElement).toBe(command);
  });

  it('moves navigation with brackets only when the sidebar owns focus', () => {
    renderApp();
    const navModels = document.querySelector<HTMLAnchorElement>('[data-testid="nav-models"]');
    navModels?.focus();
    navModels?.dispatchEvent(new KeyboardEvent('keydown', { key: ']', bubbles: true }));
    expect(window.location.hash).toBe('#chat');
    act(() => keydown(']'));
    expect(window.location.hash).toBe('#chat');
  });

  it('does not open global overlays from editable fields or IME composition', () => {
    renderApp();
    act(() => document.querySelector<HTMLAnchorElement>('[data-testid="nav-chat"]')?.click());
    const textarea = document.querySelector('textarea');
    textarea?.focus();
    textarea?.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', metaKey: true, bubbles: true }));
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(false);
    act(() => keydown('?', { isComposing: true }));
    expect(document.querySelector('[data-testid="help-dialog"]')?.hasAttribute('open')).toBe(false);
  });

  it('applies appearance through data attributes without inline style', () => {
    applyAppearance(document.documentElement, { ...DEFAULT_APPEARANCE, glassIntensity: 71, reduceTransparency: true, highContrast: true });
    expect(document.documentElement.dataset.glassIntensity).toBe('71');
    expect(document.documentElement.dataset.material).toBe('opaque');
    expect(document.documentElement.getAttribute('style')).toBeNull();
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
