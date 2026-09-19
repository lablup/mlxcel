// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../app';
import { WebUiProvider } from '../state';
import { AppShell } from './shell';

let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  HTMLDialogElement.prototype.showModal = vi.fn(function showModal(this: HTMLDialogElement) { this.setAttribute('open', ''); });
  HTMLDialogElement.prototype.close = vi.fn(function close(this: HTMLDialogElement) { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); });
  localStorage.clear();
  window.history.replaceState(null, '', '/webui/#models');
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

const frames = async (): Promise<void> => { await act(async () => { await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => window.setTimeout(resolve, 0)))); }); };
const panel = (): HTMLElement => {
  const element = document.querySelector<HTMLElement>('[data-testid="mobile-nav-sheet"]');
  if (!element) throw new Error('Missing navigation drawer');
  return element;
};
const navLink = (id: string): HTMLAnchorElement => {
  const link = panel().querySelector<HTMLAnchorElement>(`[data-testid="${id}"]`);
  if (!link) throw new Error(`Missing drawer link ${id}`);
  return link;
};
async function openNav(): Promise<void> {
  await act(async () => document.querySelector<HTMLButtonElement>('[data-testid="toolbar-menu"]')?.click());
  await frames();
}
const expectSharedDrawer = (): void => { expect(panel().matches('aside.drawer[role="dialog"][aria-modal="true"]')).toBe(true); };

function Shell({ connectionLabel }: { connectionLabel: string }): React.JSX.Element {
  return <AppShell locale="en" route="models" onRouteChange={() => undefined} onCommand={() => undefined} onHelp={() => undefined} selectedModel="none" connectionLabel={connectionLabel} connectionState="streaming"><p>Route content</p></AppShell>;
}

describe('navigation drawer adoption', () => {
  it('keeps focus on a nav link when the shell re-renders while the drawer is open', async () => {
    act(() => root.render(<Shell connectionLabel="snapshot 1" />));
    await openNav();
    const link = navLink('nav-chat');
    act(() => link.focus());
    expect(document.activeElement).toBe(link);
    // The SSE snapshot re-renders AppShell continuously while the drawer is open.
    act(() => root.render(<Shell connectionLabel="snapshot 2" />));
    await frames();
    act(() => root.render(<Shell connectionLabel="snapshot 3" />));
    await frames();
    expect(document.activeElement).toBe(link);
    expectSharedDrawer();
  });

  it('suppresses Cmd/Ctrl+K and ? while focus is inside the open drawer', async () => {
    act(() => root.render(<WebUiProvider fetchImpl={(() => { throw new Error('Unexpected API request'); }) as typeof fetch}><App /></WebUiProvider>));
    await openNav();
    const link = navLink('nav-activity');
    act(() => link.focus());
    act(() => { link.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', metaKey: true, bubbles: true, cancelable: true })); });
    act(() => { link.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true })); });
    act(() => { link.dispatchEvent(new KeyboardEvent('keydown', { key: '?', bubbles: true, cancelable: true })); });
    await frames();
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(false);
    expect(document.querySelector('[data-testid="help-dialog"]')?.hasAttribute('open')).toBe(false);
    expectSharedDrawer();
  });
});
