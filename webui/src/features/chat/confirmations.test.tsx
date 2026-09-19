// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
// The four Chat confirmations (edit a turn, replace with saved history, Clear All,
// replace with imported history) render through the design-system Dialog.
import React, { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import bootstrap from '../../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalog from '../../../../tests/fixtures/webui/examples/catalog.page.json';
import type { ChatStreamHandlers } from '../../api/client';
import type { WebUiSnapshot } from '../../api/types';
import { t, type Locale } from '../../i18n/catalog';
import { initialSnapshot } from '../../state/reducer';
import { Chat } from './chat';
import { exportConversations, type ChatConversation } from './history';
import type { MediaImageLimits } from './images';
import { HistoryControls } from './privacy';
import { replaceConversations, updateConversation, useConversations } from './session';

const repository = vi.hoisted(() => ({ setEnabled: vi.fn(), load: vi.fn(), save: vi.fn(), clear: vi.fn(), close: vi.fn() }));
vi.mock('./history', async () => ({ ...await vi.importActual<typeof import('./history')>('./history'), createHistoryRepository: () => repository }));
const mocked = vi.hoisted(() => ({ snapshot: null as WebUiSnapshot | null, stream: vi.fn(), select: vi.fn(), runtime: vi.fn() }));
vi.mock('../../state', () => ({ useWebUi: () => mocked.snapshot, useWebUiActions: () => ({ streamChatCompletions: mocked.stream, selectModel: mocked.select, refreshRuntime: mocked.runtime }) }));

const limits: MediaImageLimits = { max_images: 4, max_image_bytes: 8 * 1024 * 1024, max_width: 4096, max_height: 4096, max_decoded_bytes: 64 * 1024 * 1024, max_body_bytes: 16 * 1024 * 1024 };
const conversation = (id: string): ChatConversation => ({ id, title: id, systemPrompt: '', turns: [], updatedAt: 1 });
let host: HTMLDivElement;
let root: Root | null;
let replace = vi.fn<(next: ChatConversation[]) => void>();
let conversations: ChatConversation[] = [];

function Probe(): null { conversations = useConversations(); return null; }
function Harness({ busy = false, current = [], locale = 'en' }: { busy?: boolean; current?: ChatConversation[]; locale?: Locale }): React.JSX.Element {
  const [pending, setPending] = useState(false);
  return <main><button data-testid="send" disabled={busy || pending}>Send</button><HistoryControls conversations={current} busy={busy} onPending={setPending} limits={limits} onReplace={replace} locale={locale} /></main>;
}
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve: (value: T) => void = () => { throw new Error('Promise not initialized'); };
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
const settle = async (): Promise<void> => { await act(async () => { await new Promise((resolve) => setTimeout(resolve, 10)); }); };
const dialog = (testId: string): HTMLDialogElement | null => document.querySelector<HTMLDialogElement>(`dialog[data-testid="${testId}"]`);
function button(text: string): HTMLButtonElement {
  const element = [...host.querySelectorAll('button')].find((item) => item.textContent === text);
  if (!element) throw new Error(`Button missing: ${text}`);
  return element;
}
async function press(element: HTMLElement): Promise<void> { await act(async () => { element.focus(); element.click(); }); }
async function confirm(testId: string): Promise<void> {
  const element = document.querySelector<HTMLButtonElement>(`[data-testid="${testId}-confirm"]`);
  if (!element) throw new Error(`Confirm missing: ${testId}`);
  await act(async () => element.click()); await settle();
}
// jsdom has no Escape-to-cancel; close() dispatches the same native close event.
async function escape(testId: string): Promise<void> { await act(async () => dialog(testId)?.close()); await settle(); }
function checkbox(): HTMLInputElement {
  const element = host.querySelector<HTMLInputElement>('input[type="checkbox"]');
  if (!element) throw new Error('History checkbox missing');
  return element;
}
function fileInput(): HTMLInputElement {
  const element = host.querySelector<HTMLInputElement>('input[type="file"]');
  if (!element) throw new Error('Import field missing');
  return element;
}
async function importFile(text: Promise<string>): Promise<void> {
  const file = new File(['pending'], 'history.json', { type: 'application/json' });
  Object.defineProperty(file, 'text', { value: () => text });
  const input = fileInput(); input.focus();
  Object.defineProperty(input, 'files', { configurable: true, value: [file] });
  await act(async () => { input.dispatchEvent(new Event('change', { bubbles: true })); await Promise.resolve(); });
}
// Browsers move focus to <body> once the focused control becomes disabled; jsdom keeps
// it there and ignores blur() on a disabled element, so park focus on a removed sink.
function focusFixup(): void {
  if (!document.activeElement?.matches(':disabled')) throw new Error('Trigger is not a disabled focused control');
  const sink = document.createElement('button'); document.body.append(sink); sink.focus(); sink.remove();
  expect(document.activeElement).toBe(document.body);
}
function render(node: React.ReactNode): void { act(() => root?.render(node)); }

beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); };
  vi.resetAllMocks(); repository.load.mockResolvedValue([]); repository.save.mockResolvedValue(undefined); repository.clear.mockResolvedValue(undefined);
  replace = vi.fn(); replaceConversations([]);
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(() => { act(() => root?.unmount()); root = null; host.remove(); replaceConversations([]); vi.unstubAllGlobals(); });

describe('edit turn confirmation', () => {
  const completion = async (_id: string, _body: unknown, handlers: ChatStreamHandlers): Promise<void> => { handlers.onFrame({ data: JSON.stringify({ choices: [{ index: 0, delta: { content: 'Answer' }, finish_reason: 'stop' }] }), event: 'message', id: null, retry: null }); };
  const composer = (): HTMLTextAreaElement => { const field = host.querySelector<HTMLTextAreaElement>('textarea[aria-label="Message"]'); if (!field) throw new Error('Missing composer'); return field; };
  async function send(text: string): Promise<void> {
    act(() => { Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set?.call(composer(), text); composer().dispatchEvent(new Event('input', { bubbles: true })); });
    await act(async () => button('Send').click());
  }
  const edits = (): HTMLButtonElement[] => [...host.querySelectorAll('button')].filter((item) => item.textContent === t('en', 'chat.transcript.edit'));
  beforeEach(async () => {
    const entry = { ...catalog.items[0], lifecycle: { ...catalog.items[0].lifecycle, state: 'ready' }, capabilities: [{ task: 'chat', phase: 'provider_ready', available: true, reason: null }] };
    mocked.snapshot = { ...initialSnapshot(), auth: { status: 'authenticated', tokenPresent: true }, bootstrap, connection: 'ready', catalog: [entry], selectedModelId: entry.identity.id } as WebUiSnapshot;
    mocked.stream.mockImplementation(completion); mocked.runtime.mockResolvedValue({ measurements: {} });
    render(<><Probe /><Chat locale="en" /></>);
    await send('First'); await send('Second');
    expect(host.querySelectorAll('article.chat-turn')).toHaveLength(2);
  });

  it('keeps every turn on Escape and returns focus to the Edit button', async () => {
    const trigger = edits()[0];
    await press(trigger);
    expect(dialog('chat-edit-dialog')?.open).toBe(true);
    expect(dialog('chat-edit-dialog')?.textContent).toContain(t('en', 'chat.transcript.edit.confirm.body'));
    await escape('chat-edit-dialog');
    expect(dialog('chat-edit-dialog')).toBeNull();
    expect(host.querySelectorAll('article.chat-turn')).toHaveLength(2); expect(composer().value).toBe('');
    expect(document.activeElement).toBe(trigger);
  });
  it('restores the prompt, discards later turns and focuses the composer on confirm', async () => {
    await press(edits()[0]); await confirm('chat-edit-dialog');
    expect(host.querySelectorAll('article.chat-turn')).toHaveLength(0);
    expect(composer().value).toBe('First'); expect(document.activeElement).toBe(composer());
  });
  it('drops a stale answer when the edited turn no longer exists', async () => {
    await press(edits()[1]);
    const current = conversations[0];
    act(() => { updateConversation({ ...current, turns: current.turns.slice(0, 1) }); });
    await confirm('chat-edit-dialog');
    expect(host.querySelectorAll('article.chat-turn')).toHaveLength(1); expect(composer().value).toBe('');
  });
  it('drops the answer while a local history import is pending', async () => {
    await press(edits()[0]);
    await importFile(deferred<string>().promise);
    await confirm('chat-edit-dialog');
    expect(host.querySelectorAll('article.chat-turn')).toHaveLength(2); expect(composer().value).toBe('');
  });
  it('drops a stale answer when a different conversation now holds a turn at the same index', async () => {
    await press(edits()[0]);
    await press(button(t('en', 'chat.new_conversation')));
    await send('Other conversation prompt');
    expect(host.querySelectorAll('article.chat-turn')).toHaveLength(1);
    await confirm('chat-edit-dialog');
    expect(host.querySelectorAll('article.chat-turn')).toHaveLength(1); expect(composer().value).toBe('');
  });
});

describe('replace with saved history confirmation', () => {
  const saved = [conversation('saved')];
  beforeEach(() => { render(<Harness current={[conversation('memory')]} />); });
  async function open(): Promise<HTMLInputElement> {
    const loading = deferred<ChatConversation[]>(); repository.load.mockReturnValue(loading.promise);
    const trigger = checkbox(); await press(trigger);
    expect(trigger.disabled).toBe(true); focusFixup();
    await act(async () => { loading.resolve(saved); await loading.promise; }); await settle();
    expect(dialog('chat-replace-saved-dialog')?.open).toBe(true);
    return trigger;
  }
  it('keeps memory, disables saving and returns focus on Escape', async () => {
    const trigger = await open();
    await escape('chat-replace-saved-dialog');
    expect(replace).not.toHaveBeenCalled(); expect(repository.setEnabled).toHaveBeenLastCalledWith(false);
    expect(checkbox().checked).toBe(false); expect(document.activeElement).toBe(trigger);
  });
  it('replaces memory with saved history and enables saving on confirm', async () => {
    await open(); await confirm('chat-replace-saved-dialog');
    expect(replace).toHaveBeenCalledExactlyOnceWith(saved); expect(checkbox().checked).toBe(true);
  });
  it('drops the answer once a generation has started', async () => {
    await open(); render(<Harness busy current={[conversation('memory')]} />);
    await confirm('chat-replace-saved-dialog');
    expect(replace).not.toHaveBeenCalled(); expect(checkbox().checked).toBe(false);
  });
});

describe('Clear All confirmation', () => {
  it('renders the required clear-history body and clears nothing on Escape', async () => {
    render(<Harness />);
    const trigger = button(t('en', 'chat.privacy.clear'));
    await press(trigger);
    expect(dialog('chat-clear-history-dialog')?.open).toBe(true);
    expect(dialog('chat-clear-history-dialog')?.textContent).toContain(t('en', 'settings.clear_history.confirm.body'));
    expect(dialog('chat-clear-history-dialog')?.textContent).toContain(t('en', 'chat.privacy.clear.confirm.title'));
    await escape('chat-clear-history-dialog');
    expect(repository.clear).not.toHaveBeenCalled(); expect(replace).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(trigger);
  });
  it('clears once on confirm and localizes the dialog', async () => {
    render(<Harness locale="ko" />);
    await press(button(t('ko', 'chat.privacy.clear')));
    expect(dialog('chat-clear-history-dialog')?.textContent).toContain(t('ko', 'settings.clear_history.confirm.body'));
    await confirm('chat-clear-history-dialog');
    expect(repository.clear).toHaveBeenCalledOnce(); expect(replace).toHaveBeenCalledExactlyOnceWith([]);
  });
  it('drops the answer once a generation has started', async () => {
    render(<Harness />);
    await press(button(t('en', 'chat.privacy.clear'))); render(<Harness busy />);
    await confirm('chat-clear-history-dialog');
    expect(repository.clear).not.toHaveBeenCalled();
  });
  it('refocuses the Clear All trigger once clearing settles if focus is still on body', async () => {
    render(<Harness />);
    const trigger = button(t('en', 'chat.privacy.clear'));
    const clearing = deferred<undefined>();
    repository.clear.mockReturnValue(clearing.promise);
    await press(trigger);
    // Every Chat control is disabled while clearing is in flight, so the dialog's own
    // focus restoration finds nothing usable and Chromium leaves focus on <body>.
    await confirm('chat-clear-history-dialog');
    expect(document.activeElement).toBe(document.body);
    await act(async () => { clearing.resolve(undefined); await clearing.promise; });
    await settle();
    expect(document.activeElement).toBe(trigger);
  });
});

describe('replace with imported history confirmation', () => {
  const imported = [conversation('imported')];
  beforeEach(() => { render(<Harness />); });
  it('keeps memory and returns focus to the import field on Escape', async () => {
    const reading = deferred<string>(); await importFile(reading.promise);
    expect(fileInput().disabled).toBe(true); focusFixup();
    await act(async () => { reading.resolve(exportConversations(imported)); await reading.promise; }); await settle();
    expect(dialog('chat-replace-import-dialog')?.open).toBe(true);
    await escape('chat-replace-import-dialog');
    expect(replace).not.toHaveBeenCalled(); expect(document.activeElement).toBe(fileInput());
  });
  it('replaces memory with the validated import on confirm', async () => {
    await importFile(Promise.resolve(exportConversations(imported))); await settle();
    await confirm('chat-replace-import-dialog');
    expect(replace).toHaveBeenCalledExactlyOnceWith(imported);
  });
  it('drops an answer superseded by a newer import', async () => {
    await importFile(Promise.resolve(exportConversations(imported))); await settle();
    await importFile(deferred<string>().promise);
    await confirm('chat-replace-import-dialog');
    expect(replace).not.toHaveBeenCalled();
  });
});
