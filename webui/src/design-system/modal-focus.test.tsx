import React, { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { Button, Dialog } from './primitives';

let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); };
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(() => { act(() => root.unmount()); host.remove(); });
function Fixture({ mode = 'normal' }: { mode?: 'normal' | 'disabled' | 'detached' | 'new-focus' }): React.JSX.Element {
  const [open, setOpen] = useState(false);
  const [finished, setFinished] = useState(false);
  const dismiss = (): void => { setOpen(false); setFinished(true); };
  return <main><h1 tabIndex={-1} data-dialog-focus-fallback>Models</h1>{!(finished && mode === 'detached') ? <Button disabled={finished && mode === 'disabled'} data-testid="trigger" onClick={() => setOpen(true)}>Open</Button> : null}<Button data-testid="other">Other</Button>{open ? <Dialog open title="Confirm" onClose={dismiss}><Button data-testid="cancel" onClick={dismiss}>Cancel</Button><Button data-testid="confirm" onClick={dismiss}>Confirm</Button></Dialog> : null}</main>;
}
async function click(id: string): Promise<void> { await act(async () => { const node = host.querySelector<HTMLButtonElement>(`[data-testid="${id}"]`); node?.focus(); node?.click(); }); }
const settle = async (): Promise<void> => { await act(async () => { await new Promise((resolve) => setTimeout(resolve, 10)); }); };
describe('conditional Dialog focus restoration', () => {
  it.each(['cancel', 'confirm', 'dialog-close'])('restores the trigger after %s unmount', async (action) => {
    act(() => root.render(<Fixture />)); await click('trigger'); await click(action); await settle();
    expect(document.activeElement).toBe(host.querySelector('[data-testid="trigger"]'));
  });
  it('restores after native Escape close event', async () => {
    act(() => root.render(<Fixture />)); await click('trigger');
    await act(async () => host.querySelector('dialog')?.close()); await settle();
    expect(document.activeElement).toBe(host.querySelector('[data-testid="trigger"]'));
  });
  it.each(['disabled', 'detached'] as const)('uses a stable fallback when trigger is %s', async (mode) => {
    act(() => root.render(<Fixture mode={mode} />)); await click('trigger'); await click('confirm'); await settle();
    expect(document.activeElement).toBe(host.querySelector('h1'));
  });
  it('does not steal focus intentionally moved to another control', async () => {
    act(() => root.render(<Fixture />)); await click('trigger'); await click('confirm');
    host.querySelector<HTMLButtonElement>('[data-testid="other"]')?.focus(); await settle();
    expect(document.activeElement).toBe(host.querySelector('[data-testid="other"]'));
  });
  it('does not steal focus from a newly opened modal', async () => {
    act(() => root.render(<Fixture />)); await click('trigger');
    const modal = document.createElement('dialog');
    act(() => { host.querySelector<HTMLButtonElement>('[data-testid="confirm"]')?.click(); modal.setAttribute('open', ''); document.body.append(modal); });
    await settle(); expect(document.activeElement).not.toBe(host.querySelector('[data-testid="trigger"]')); modal.remove();
  });
});
