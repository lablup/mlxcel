import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { ProfileSettings } from './profile-settings';

let host: HTMLDivElement; let root: Root;
beforeEach(() => {
  localStorage.clear();
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); };
});
afterEach(() => { act(() => root.unmount()); host.remove(); Reflect.deleteProperty(navigator, 'clipboard'); });
async function render(): Promise<void> { await act(async () => { root.render(<ProfileSettings model={null} locale="en" single={false} />); }); }
const button = (label: string): HTMLButtonElement | undefined => [...host.querySelectorAll('button')].find((item) => item.textContent === label);
const status = (): string | undefined => host.querySelector('[role="status"]')?.textContent ?? undefined;

// "Show as CLI flags" holds the button behind a <details> disclosure; its content stays in the DOM
// (only visually hidden) so it does not need to be opened first to be queried or clicked here.
describe('profile "Show as CLI flags" copy button', () => {
  it('reports success once the Clipboard API resolves', async () => {
    const writeText = (): Promise<void> => Promise.resolve();
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    await render();
    expect(status()).toBe('');
    await act(async () => { button('Copy flags')?.click(); });
    expect(status()).toBe('CLI flags copied.');
  });
  it('reports failure when the Clipboard API is unavailable', async () => {
    Reflect.deleteProperty(navigator, 'clipboard');
    await render();
    await act(async () => { button('Copy flags')?.click(); });
    expect(status()).toBe('Copy failed. Select the flags and copy them instead.');
  });
  it('reports failure when the Clipboard API rejects', async () => {
    const writeText = (): Promise<void> => Promise.reject(new Error('denied'));
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    await render();
    await act(async () => { button('Copy flags')?.click(); });
    expect(status()).toBe('Copy failed. Select the flags and copy them instead.');
  });
});
