// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { beforeAll, afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App, preloadRoutes } from '../app';
import { WebUiProvider } from '../state';
import { Dialog, Drawer } from './primitives';
import { AppShell } from './shell';

// Split routes (Chat, Activity, Settings, the gallery) render without suspending once loaded.
beforeAll(async () => { await preloadRoutes(); });

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
// ui-common's lab Drawer is a native modal <dialog> (showModal), named by its title.
const expectSharedDrawer = (): void => { expect(panel().matches('dialog.ds-drawer.astryx-drawer[aria-modal="true"]')).toBe(true); };
const isPanelOpen = (): boolean => panel().hasAttribute('open');

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
    // Stands in for a popup outside the panel that already handles Escape for itself:
    // Astryx's layer stack backs off once defaultPrevented is set, rather than also
    // closing the drawer underneath the popup.
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

// jsdom lays nothing out, so getClientRects() is empty for every element. Stand in for layout
// instead of loosening the adapter's rendered-only filter: `hidden` elements and the contents
// of a closed <details> (everything but its own summary) have no boxes, the rest have one.
function emulateLayout(): void {
  vi.spyOn(Element.prototype, 'getClientRects').mockImplementation(function getClientRects(this: Element) {
    const closed = this.closest('details:not([open])');
    const collapsed = closed !== null && !(this.tagName === 'SUMMARY' && this.parentElement === closed);
    const rects = this.closest('[hidden]') || collapsed ? [] : [{ x: 0, y: 0, top: 0, left: 0, right: 10, bottom: 10, width: 10, height: 10 }];
    return Object.assign(rects, { item: (index: number) => rects[index] ?? null }) as unknown as DOMRectList;
  });
}
const tab = (target: Element, shiftKey = false): KeyboardEvent => {
  const event = new KeyboardEvent('keydown', { key: 'Tab', shiftKey, bubbles: true, cancelable: true });
  act(() => { target.dispatchEvent(event); });
  return event;
};
const byText = (text: string): HTMLElement => {
  const element = [...document.querySelectorAll<HTMLElement>('[data-testid="test-drawer"] button, [data-testid="test-drawer"] summary')].find((item) => item.textContent === text);
  if (!element) throw new Error(`Missing drawer control ${text}`);
  return element;
};

describe('drawer adapter keyboard contract', () => {
  it('computes the Tab cycle at key-press time, including a summary and only rendered controls', async () => {
    emulateLayout();
    act(() => root.render(
      <Drawer open title="Panel" closeLabel="Close" onClose={() => undefined} testId="test-drawer" width="medium" side="end">
        <button type="button">First</button>
        <details open><summary>More</summary><button type="button">Inner</button></details>
        <button type="button" hidden>Hidden</button>
      </Drawer>,
    ));
    await frames();
    const close = document.querySelector<HTMLElement>('[data-testid="test-drawer"] .ds-drawer__close');
    if (!close) throw new Error('Missing drawer close button');
    const inner = byText('Inner');
    act(() => inner.focus());
    // The last rendered control wraps to the close button; the hidden button after it does not count.
    expect(tab(inner).defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(close);
    expect(tab(close, true).defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(inner);
    // Between the ends the browser moves focus itself, so the adapter leaves the key alone.
    const first = byText('First');
    act(() => first.focus());
    expect(tab(first).defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(first);
    // Collapsing the disclosure while open makes its summary the last stop, read at key-press time.
    const more = byText('More');
    act(() => { (more.parentElement as HTMLDetailsElement).open = false; });
    act(() => more.focus());
    expect(tab(more).defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(close);
    expect(tab(close, true).defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(more);
  });

  it('leaves the drawer open for an Escape inside a native dialog above it, and still closes on one outside', async () => {
    const onClose = vi.fn();
    act(() => root.render(
      <>
        <Drawer open title="Panel" closeLabel="Close" onClose={onClose} testId="test-drawer" width="medium" side="end">
          <button type="button">First</button>
        </Drawer>
        <Dialog open title="Confirm" onClose={() => undefined} testId="test-confirm"><button type="button">Cancel</button></Dialog>
      </>,
    ));
    await frames();
    const cancel = [...document.querySelectorAll<HTMLElement>('[data-testid="test-confirm"] button')].find((item) => item.textContent === 'Cancel');
    if (!cancel) throw new Error('Missing dialog button');
    act(() => cancel.focus());
    // The native dialog closes itself on this Escape and does not preventDefault it.
    act(() => { cancel.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })); });
    await frames();
    expect(onClose).not.toHaveBeenCalled();
    // jsdom does not close the native dialog on Escape; once it is gone, an Escape that starts
    // outside both the panel and any dialog still closes the drawer.
    act(() => root.render(
      <Drawer open title="Panel" closeLabel="Close" onClose={onClose} testId="test-drawer" width="medium" side="end">
        <button type="button">First</button>
      </Drawer>,
    ));
    act(() => { document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })); });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('leaves the drawer open for an Escape or Tab from <body> while a native dialog is open above it', async () => {
    const onClose = vi.fn();
    act(() => root.render(
      <>
        <Drawer open title="Panel" closeLabel="Close" onClose={onClose} testId="test-drawer" width="medium" side="end">
          <button type="button">First</button>
        </Drawer>
        <Dialog open title="Confirm" onClose={() => undefined} testId="test-confirm"><p>Body text</p></Dialog>
      </>,
    ));
    await frames();
    // A pointer press on the dialog's non-focusable text leaves focus on <body>.
    act(() => (document.activeElement as HTMLElement | null)?.blur());
    const escape = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true });
    act(() => { document.body.dispatchEvent(escape); });
    expect(onClose).not.toHaveBeenCalled();
    expect(tab(document.body).defaultPrevented).toBe(false);
  });
  // A Tab from <body> while the drawer is open needs no adapter code: the drawer is a native modal
  // dialog, so the browser keeps sequential focus inside it (tests/ui-common-shell.spec.ts drives it).
});

describe('Drawer variants and Escape ownership', () => {
  function Sheet({ open, onClose, width, side }: { open: boolean; onClose: () => void; width?: 'narrow' | 'medium'; side?: 'start' | 'end' }): React.JSX.Element {
    return <Drawer open={open} onClose={onClose} title="Sheet" closeLabel="Close" testId="variant-sheet" width={width} side={side}><label>Field<input data-testid="variant-field" /></label></Drawer>;
  }
  const sheet = (): HTMLElement => { const element = document.querySelector<HTMLElement>('[data-testid="variant-sheet"]'); if (!element) throw new Error('Missing sheet'); return element; };
  it('adds only modifier classes for the wider and end-anchored sheets', () => {
    act(() => root.render(<Sheet open={false} onClose={() => undefined} />));
    expect(sheet().className).not.toMatch(/ds-drawer--/);
    act(() => root.render(<Sheet open={false} onClose={() => undefined} width="medium" side="end" />));
    expect(sheet().classList.contains('ds-drawer--medium')).toBe(true);
    expect(sheet().classList.contains('ds-drawer--end')).toBe(true);
    act(() => root.render(<Sheet open onClose={() => undefined} width="medium" side="end" />));
    expect(sheet().getAttribute('data-side')).toBe('end');
  });
  it('leaves an Escape inside an open product dialog to that dialog', async () => {
    const onClose = vi.fn();
    const onDialogClose = vi.fn();
    const Both = ({ dialog }: { dialog: boolean }): React.JSX.Element => <><Sheet open onClose={onClose} /><Dialog open={dialog} title="Confirm" onClose={onDialogClose} testId="variant-confirm"><button type="button">Inside</button></Dialog></>;
    act(() => root.render(<Both dialog />));
    await frames();
    const inside = [...document.querySelectorAll<HTMLElement>('[data-testid="variant-confirm"] button')].find((item) => item.textContent === 'Inside');
    if (!inside) throw new Error('Missing dialog button');
    act(() => { inside.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })); });
    expect(onDialogClose).toHaveBeenCalledOnce();
    expect(onClose).not.toHaveBeenCalled();
    act(() => root.render(<Both dialog={false} />));
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
  it('lets a Tab between the edges reach a descendant handler, including from a control before a trailing summary', async () => {
    const seen = vi.fn();
    act(() => root.render(<Drawer open onClose={() => undefined} title="Sheet" closeLabel="Close" testId="variant-sheet"><button type="button" data-testid="first-control">First</button><button type="button" data-testid="middle-control" onKeyDown={(event) => seen(event.key)}>Middle</button><button type="button" data-testid="last-control" onKeyDown={(event) => seen(event.key)}>Last</button><details open><summary>More</summary></details></Drawer>));
    await frames();
    const control = (id: string): HTMLElement => { const element = sheet().querySelector<HTMLElement>(`[data-testid="${id}"]`); if (!element) throw new Error(`Missing ${id}`); return element; };
    const tab = (target: HTMLElement): KeyboardEvent => { const event = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true }); act(() => { target.dispatchEvent(event); }); return event; };
    act(() => control('middle-control').focus());
    // Neither the adapter nor the package wraps here: the browser moves focus, and the control's own handler sees the key.
    expect(tab(control('middle-control')).defaultPrevented).toBe(false);
    expect(seen).toHaveBeenCalledWith('Tab');
    seen.mockClear();
    // Last is not the last stop (More's summary is): nothing wraps early to the close button.
    act(() => control('last-control').focus());
    expect(tab(control('last-control')).defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(control('last-control'));
    expect(seen).toHaveBeenCalledWith('Tab');
  });
  it('wraps Tab at the controls usable now, not the ones it saw when it opened', async () => {
    const Panel = ({ locked }: { locked: boolean }): React.JSX.Element => <Drawer open onClose={() => undefined} title="Sheet" closeLabel="Close" testId="variant-sheet"><button type="button" data-testid="first-control">First</button><button type="button" data-testid="middle-control">Middle</button><button type="button" data-testid="last-control" disabled={locked}>Last</button></Drawer>;
    act(() => root.render(<Panel locked={false} />));
    await frames();
    // Disabled after opening, as Chat's drawer controls are while a response streams.
    act(() => root.render(<Panel locked />));
    const control = (id: string): HTMLElement => { const element = sheet().querySelector<HTMLElement>(`[data-testid="${id}"]`); if (!element) throw new Error(`Missing ${id}`); return element; };
    const close = sheet().querySelector<HTMLElement>('.ds-drawer__close');
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
