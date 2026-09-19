import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ConfirmDialog, type ConfirmDialogProps } from './primitives';

let host: HTMLDivElement;
let root: Root;
const onConfirm = vi.fn();
const onClose = vi.fn();
beforeEach(() => {
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); };
  vi.resetAllMocks();
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(() => { act(() => root.unmount()); host.remove(); });
function render(overrides: Partial<ConfirmDialogProps> = {}): void {
  act(() => root.render(<ConfirmDialog open title="Discard?" body="Later turns are discarded." confirmLabel="Discard" cancelLabel="Keep" closeLabel="Dismiss" tone="danger" testId="probe" onConfirm={onConfirm} onClose={onClose} {...overrides} />));
}
const node = <T extends HTMLElement>(testId: string): T => { const element = host.querySelector<T>(`[data-testid="${testId}"]`); if (!element) throw new Error(`Missing ${testId}`); return element; };

describe('ConfirmDialog', () => {
  it('renders the body, derived test IDs, localized close label and danger tone', () => {
    render();
    const dialog = node<HTMLDialogElement>('probe');
    expect(dialog.open).toBe(true);
    expect(dialog.querySelector('p')?.textContent).toBe('Later turns are discarded.');
    expect(node('dialog-close').getAttribute('aria-label')).toBe('Dismiss');
    expect(node('probe-cancel').textContent).toBe('Keep');
    expect(node('probe-confirm').textContent).toBe('Discard');
    expect(node('probe-confirm').className).toContain('ds-button-danger');
  });
  it('routes confirm, cancel and the native close event to their callbacks', async () => {
    render();
    act(() => node('probe-confirm').click()); expect(onConfirm).toHaveBeenCalledOnce(); expect(onClose).not.toHaveBeenCalled();
    act(() => node('probe-cancel').click()); expect(onClose).toHaveBeenCalledOnce();
    await act(async () => node<HTMLDialogElement>('probe').close()); expect(onClose).toHaveBeenCalledTimes(2); expect(onConfirm).toHaveBeenCalledOnce();
  });
  it('disables confirm while busy instead of confirming or closing', () => {
    render({ busy: true });
    const confirm = node<HTMLButtonElement>('probe-confirm');
    expect(confirm.disabled).toBe(true);
    act(() => confirm.click());
    expect(onConfirm).not.toHaveBeenCalled(); expect(onClose).not.toHaveBeenCalled(); expect(node<HTMLDialogElement>('probe').open).toBe(true);
  });
  it('closes without calling onClose when the parent sets open to false', () => {
    render(); render({ open: false });
    expect(node<HTMLDialogElement>('probe').open).toBe(false); expect(onClose).not.toHaveBeenCalled();
  });
});
