// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HistoryControls } from './privacy';
import { exportConversations, type ChatConversation } from './history';
import type { MediaImageLimits } from './images';
const repository = vi.hoisted(() => ({ setEnabled: vi.fn(), load: vi.fn(), save: vi.fn(), clear: vi.fn(), close: vi.fn() }));
vi.mock('./history', async () => ({ ...await vi.importActual<typeof import('./history')>('./history'), createHistoryRepository: () => repository }));
const limits: MediaImageLimits = { max_images: 4, max_image_bytes: 8 * 1024 * 1024, max_width: 4096, max_height: 4096, max_decoded_bytes: 64 * 1024 * 1024, max_body_bytes: 16 * 1024 * 1024 };
const empty = (): ChatConversation[] => [{ id: 'imported', title: 'Imported', systemPrompt: '', turns: [], updatedAt: 1 }];
let root: Root | null;
let host: HTMLDivElement;
let replace = vi.fn<(next: ChatConversation[]) => void>();
function deferred<T>() {
  let resolve: (value: T) => void = () => { throw new Error('Promise not initialized'); };
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
function Harness({ busy = false }: { busy?: boolean }): React.JSX.Element {
  const [pending, setPending] = useState(false);
  return <><button data-testid="send" disabled={busy || pending}>Send</button><HistoryControls conversations={[]} busy={busy} onPending={setPending} limits={limits} onReplace={replace} /></>;
}
function input(): HTMLInputElement {
  const element = host.querySelector<HTMLInputElement>('input[type="file"]');
  if (!element) throw new Error('Import field missing');
  return element;
}
function button(text: string): HTMLButtonElement {
  const element = [...host.querySelectorAll('button')].find((item) => item.textContent === text);
  if (!element) throw new Error(`Button missing: ${text}`);
  return element;
}
function importFile(text: Promise<string>): void {
  const file = new File(['pending'], 'history.json', { type: 'application/json' });
  Object.defineProperty(file, 'text', { value: () => text });
  Object.defineProperty(input(), 'files', { configurable: true, value: [file] });
  act(() => input().dispatchEvent(new Event('change', { bubbles: true })));
}
beforeEach(() => {
  vi.resetAllMocks(); repository.load.mockResolvedValue([]); repository.save.mockResolvedValue(undefined); repository.clear.mockResolvedValue(undefined);
  vi.spyOn(window, 'confirm').mockReturnValue(true);
  replace = vi.fn(); host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  act(() => root?.render(<Harness />));
});
afterEach(() => { act(() => root?.unmount()); root = null; host.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });
describe('history async privacy boundaries', () => {
  it('disables sending and Clear All while import is pending, then restores controls', async () => {
    const reading = deferred<string>(); importFile(reading.promise);
    expect(button('Send').disabled).toBe(true); expect(button('Clear All').disabled).toBe(true);
    act(() => button('Clear All').click()); expect(repository.clear).not.toHaveBeenCalled();
    await act(async () => { reading.resolve(exportConversations(empty())); await reading.promise; });
    expect(replace).toHaveBeenCalledWith(empty()); expect(button('Send').disabled).toBe(false);
  });
  it('does not restore imported history after unmount invalidates its epoch', async () => {
    const reading = deferred<string>(); importFile(reading.promise);
    act(() => { root?.unmount(); root = null; });
    await act(async () => { reading.resolve(exportConversations(empty())); await reading.promise; });
    expect(replace).not.toHaveBeenCalled(); expect(window.confirm).not.toHaveBeenCalled();
  });
  it('does not replace history if generation becomes busy while a read is pending', async () => {
    const reading = deferred<string>(); importFile(reading.promise);
    act(() => root?.render(<Harness busy />));
    await act(async () => { reading.resolve(exportConversations(empty())); await reading.promise; });
    expect(replace).not.toHaveBeenCalled();
  });
  it('disables sending during saved-history loading and rejects late unmounted results', async () => {
    const loading = deferred<ChatConversation[]>(); repository.load.mockReturnValue(loading.promise);
    const checkbox = host.querySelector<HTMLInputElement>('input[type="checkbox"]');
    act(() => checkbox?.click()); expect(button('Send').disabled).toBe(true);
    act(() => { root?.unmount(); root = null; });
    await act(async () => { loading.resolve(empty()); await loading.promise; });
    expect(replace).not.toHaveBeenCalled();
  });
  it('marks Clear All pending and does not apply a late clear after unmount', async () => {
    const clearing = deferred<undefined>(); repository.clear.mockReturnValue(clearing.promise);
    act(() => button('Clear All').click()); expect(button('Send').disabled).toBe(true); expect(input().disabled).toBe(true);
    act(() => { root?.unmount(); root = null; });
    await act(async () => { clearing.resolve(undefined); await clearing.promise; });
    expect(replace).not.toHaveBeenCalled();
  });
  it('clears once and releases pending controls after deletion completes', async () => {
    const clearing = deferred<undefined>(); repository.clear.mockReturnValue(clearing.promise);
    act(() => button('Clear All').click());
    await act(async () => { clearing.resolve(undefined); await clearing.promise; });
    expect(replace).toHaveBeenCalledExactlyOnceWith([]); expect(button('Send').disabled).toBe(false);
  });
  it('rejects imported giant-dimension images before invoking a browser decoder', async () => {
    const bytes = Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jB1kAAAAASUVORK5CYII='), (char) => char.charCodeAt(0));
    new DataView(bytes.buffer).setUint32(16, 1_000_000);
    const image = { name: 'bomb.png', type: 'image/png', dataUrl: `data:image/png;base64,${btoa(String.fromCharCode(...bytes))}` };
    const conversations = empty(); conversations[0].turns = [{ id: 'turn', modelId: 'model', modelRevision: 1, inferenceId: 'model', modelName: 'Model', prompt: 'Image', content: 'Earlier response', reasoning: '', tools: [], status: 'complete', finishReason: 'stop', usage: null, ttftMs: null, elapsedMs: null, error: null, parameters: {}, images: [image] }];
    const decoder = vi.fn(); vi.stubGlobal('createImageBitmap', decoder);
    act(() => host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')[1].click());
    importFile(Promise.resolve(exportConversations(conversations, { includeImages: true })));
    await vi.waitFor(async () => {
      await act(async () => { await Promise.resolve(); });
      expect(host.textContent).toContain('Import rejected');
    });
    expect(replace).not.toHaveBeenCalled(); expect(decoder).not.toHaveBeenCalled(); expect(button('Send').disabled).toBe(false);
  });
});
