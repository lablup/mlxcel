// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from '../app';
import { WebUiProvider } from '../state';
import { Drawer } from './primitives';
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
const isPanelOpen = (): boolean => panel().classList.contains('drawer--open');

function Shell({ connectionLabel }: { connectionLabel: string }): React.JSX.Element {
  return <AppShell locale="en" route="models" onRouteChange={() => undefined} onCommand={() => undefined} onHelp={() => undefined} onNewChat={() => undefined} onOpenModel={() => undefined} loadedModels={[]} connection={{ label: connectionLabel, state: 'streaming', details: null }}><p>Route content</p></AppShell>;
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

  it('leaves the drawer open when an Escape reaching it was already handled elsewhere', async () => {
    act(() => root.render(<Shell connectionLabel="snapshot 1" />));
    await openNav();
    expect(isPanelOpen()).toBe(true);
    // Stands in for a future popup that portals outside `.drawer` (like the Tooltip
    // content or a nested dialog) and already handles Escape for itself: the drawer's
    // own outside-the-panel handler must back off once defaultPrevented is set, rather
    // than also closing the drawer underneath the popup.
    const popup = document.createElement('div');
    popup.tabIndex = -1;
    popup.addEventListener('keydown', (event) => { if (event.key === 'Escape') event.preventDefault(); });
    document.body.append(popup);
    act(() => popup.focus());
    act(() => { popup.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })); });
    await frames();
    expect(isPanelOpen()).toBe(true);
    expectSharedDrawer();
    popup.remove();
  });
});

describe('Drawer variants and Escape ownership', () => {
  function Sheet({ open, onClose, width, side }: { open: boolean; onClose: () => void; width?: 'narrow' | 'medium'; side?: 'start' | 'end' }): React.JSX.Element {
    return <Drawer open={open} onClose={onClose} title="Sheet" closeLabel="Close" testId="variant-sheet" width={width} side={side}><label>Field<input data-testid="variant-field" /></label></Drawer>;
  }
  const sheet = (): HTMLElement => { const element = document.querySelector<HTMLElement>('[data-testid="variant-sheet"]'); if (!element) throw new Error('Missing sheet'); return element; };
  it('adds only modifier classes and keeps the constant inline sheet width', () => {
    act(() => root.render(<Sheet open={false} onClose={() => undefined} />));
    expect(sheet().className).not.toMatch(/ds-drawer--/);
    act(() => root.render(<Sheet open={false} onClose={() => undefined} width="medium" side="end" />));
    expect(sheet().classList.contains('ds-drawer--medium')).toBe(true);
    expect(sheet().classList.contains('ds-drawer--end')).toBe(true);
    expect(sheet().style.width).toMatch(/^min\(320px/);
  });
  it('leaves an Escape inside an open native dialog to that dialog', async () => {
    const onClose = vi.fn();
    act(() => root.render(<Sheet open onClose={onClose} />));
    await frames();
    const dialog = document.createElement('dialog'); dialog.setAttribute('open', '');
    const inside = document.createElement('button'); dialog.append(inside); document.body.append(dialog);
    act(() => { inside.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })); });
    expect(onClose).not.toHaveBeenCalled();
    dialog.remove();
    act(() => { document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })); });
    expect(onClose).toHaveBeenCalledOnce();
  });
  it('does not close on an Escape that only ends an IME composition inside the panel', async () => {
    const onClose = vi.fn();
    act(() => root.render(<Sheet open onClose={onClose} />));
    await frames();
    const field = document.querySelector<HTMLInputElement>('[data-testid="variant-field"]');
    if (!field) throw new Error('Missing field');
    act(() => field.focus());
    act(() => { field.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', isComposing: true, bubbles: true, cancelable: true })); });
    expect(onClose).not.toHaveBeenCalled();
    act(() => { field.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })); });
    expect(onClose).toHaveBeenCalledOnce();
  });
  it('wraps Tab at the controls usable now, not the ones it saw when it opened', async () => {
    const Panel = ({ locked }: { locked: boolean }): React.JSX.Element => <Drawer open onClose={() => undefined} title="Sheet" closeLabel="Close" testId="variant-sheet"><button type="button" data-testid="first-control">First</button><button type="button" data-testid="middle-control">Middle</button><button type="button" data-testid="last-control" disabled={locked}>Last</button></Drawer>;
    act(() => root.render(<Panel locked={false} />));
    await frames();
    // Disabled after opening, as Chat's drawer controls are while a response streams.
    act(() => root.render(<Panel locked />));
    const control = (id: string): HTMLElement => { const element = sheet().querySelector<HTMLElement>(`[data-testid="${id}"]`); if (!element) throw new Error(`Missing ${id}`); return element; };
    const close = sheet().querySelector<HTMLElement>('.drawer__close-btn');
    const tab = (target: HTMLElement, shiftKey = false): KeyboardEvent => { const event = new KeyboardEvent('keydown', { key: 'Tab', shiftKey, bubbles: true, cancelable: true }); act(() => { target.dispatchEvent(event); }); return event; };
    act(() => control('first-control').focus());
    expect(tab(control('first-control')).defaultPrevented).toBe(false);
    act(() => control('middle-control').focus());
    expect(tab(control('middle-control')).defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(close);
    expect(tab(close as HTMLElement, true).defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(control('middle-control'));
  });
});
