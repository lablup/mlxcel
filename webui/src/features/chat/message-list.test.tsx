// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { chatStrings } from './strings';
import { t } from '../../i18n/catalog';
import type { ChatTurn } from './history';
import { MessageList } from './message-list';

const STARTED = Date.UTC(2026, 8, 19, 5, 30, 0);
const turn = (id: string, patch: Partial<ChatTurn> = {}): ChatTurn => ({ id, modelId: 'opaque', modelRevision: 1, inferenceId: 'model', modelName: 'qwen3-0.6b-4bit', prompt: `Prompt ${id}`, content: `Answer ${id}`, reasoning: '', tools: [], status: 'complete', finishReason: 'stop', usage: null, ttftMs: null, elapsedMs: 1500, error: null, parameters: {}, images: [], startedAt: STARTED, ...patch });
let host: HTMLDivElement;
let root: Root;
const onEdit = vi.fn();
const onRetry = vi.fn();
function render(turns: ChatTurn[], busy = false, locale: 'en' | 'ko' = 'en', canSend = true): void { act(() => root.render(<MessageList turns={turns} onEdit={onEdit} onRetry={onRetry} busy={busy} canSend={canSend} locale={locale} />)); }
const named = (label: string): HTMLButtonElement[] => [...host.querySelectorAll<HTMLButtonElement>('button')].filter((button) => button.getAttribute('aria-label') === label);

beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); onEdit.mockReset(); onRetry.mockReset(); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('MessageList', () => {
  it('renders each turn as a user message and an assistant message with distinct role markup', () => {
    render([turn('a')]);
    const article = host.querySelector('article.chat-turn');
    expect(article?.getAttribute('aria-label')).toBe(t('en', 'chat.transcript.turn_label', { model: 'qwen3-0.6b-4bit' }));
    const user = article?.querySelector('.chat-message--user[data-role="user"]');
    const assistant = article?.querySelector('.chat-message--assistant[data-role="assistant"]');
    expect(user?.querySelector('.chat-role')?.textContent).toBe(t('en', 'chat.message.you'));
    expect(user?.querySelector('.chat-prompt')?.textContent).toBe('Prompt a');
    expect(assistant?.querySelector('.badge')?.textContent).toBe('qwen3-0.6b-4bit');
    expect(assistant?.querySelector('.chat-markdown')?.textContent).toContain('Answer a');
    // Send time on the prompt, completion time on the answer.
    expect(user?.querySelector('time')?.getAttribute('dateTime')).toBe(new Date(STARTED).toISOString());
    expect(assistant?.querySelector('time')?.getAttribute('dateTime')).toBe(new Date(STARTED + 1500).toISOString());
  });
  it('omits timestamps for turns imported from version-1 history', () => {
    render([turn('a', { startedAt: 0 })]);
    expect(host.querySelector('time')).toBeNull();
  });
  it.each(['complete', 'cancelled', 'interrupted', 'error'] as const)('labels a %s turn without printing the status value', (status) => {
    render([turn('a', { status, finishReason: status === 'complete' ? 'stop' : null })]);
    const label = host.querySelector(`.chat-status[data-status="${status}"]`)?.textContent ?? '';
    expect(label).toBe(t('en', `chat.transcript.status.${status}`));
    expect(label).not.toMatch(/streaming|complete|cancelled|interrupted|error/i);
  });
  it('keeps every status label and the completion announcement free of raw status words in both locales', () => {
    const keys = ['chat.transcript.status.streaming', 'chat.transcript.status.complete', 'chat.transcript.status.cancelled', 'chat.transcript.status.interrupted', 'chat.transcript.status.error', 'chat.announce.complete', 'chat.message.elapsed'];
    for (const entry of chatStrings.filter((item) => keys.includes(item.key))) for (const value of [entry.en, entry.ko]) expect(value, entry.key).not.toMatch(/streaming|complete|cancelled|interrupted|error/i);
  });
  it('counts elapsed seconds once a second while a turn streams, and stops when it ends', () => {
    vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] });
    vi.setSystemTime(STARTED + 2_400);
    render([turn('a', { status: 'streaming', content: '', finishReason: null, elapsedMs: null })]);
    const status = (): string => host.querySelector('.chat-status')?.textContent ?? '';
    expect(status()).toBe(t('en', 'chat.message.elapsed', { status: t('en', 'chat.transcript.status.streaming'), seconds: '2' }));
    expect(host.textContent).toContain(t('en', 'chat.transcript.waiting'));
    act(() => { vi.advanceTimersByTime(1000); });
    expect(status()).toContain('3');
    act(() => { vi.advanceTimersByTime(2000); });
    expect(status()).toContain('5');
    render([turn('a')]);
    expect(status()).toBe(t('en', 'chat.transcript.status.complete'));
    expect(vi.getTimerCount()).toBe(0);
  });
  it('offers Retry only on the last turn when it failed or did not finish', () => {
    const retry = t('en', 'chat.message.retry');
    render([turn('a', { status: 'error' }), turn('b')]);
    expect(named(retry)).toHaveLength(0);
    render([turn('a'), turn('b', { status: 'cancelled' })]);
    expect(named(retry)).toHaveLength(0);
    render([turn('a'), turn('b', { status: 'interrupted' })]);
    expect(named(retry)).toHaveLength(1);
    render([turn('a'), turn('b', { status: 'error', error: 'Generation failed' })]);
    act(() => named(retry)[0].click());
    expect(onRetry).toHaveBeenCalledExactlyOnceWith('b');
    render([turn('a'), turn('b', { status: 'error' })], true);
    expect(named(retry)[0].disabled).toBe(true);
    // Retry sends, so without a model Send would accept it is disabled like Send.
    render([turn('a'), turn('b', { status: 'error' })], false, 'en', false);
    expect(named(retry)[0].disabled).toBe(true);
    expect(named(t('en', 'chat.transcript.edit'))[1].disabled).toBe(false);
  });
  it('keeps the edit, copy and details actions per message, named by what they act on', () => {
    render([turn('a'), turn('b')]);
    expect(named(t('en', 'chat.transcript.edit'))).toHaveLength(2);
    expect(named(t('en', 'chat.message.copy_prompt'))).toHaveLength(2);
    expect(named(t('en', 'chat.transcript.copy'))).toHaveLength(2);
    act(() => named(t('en', 'chat.transcript.edit'))[1].click());
    expect(onEdit).toHaveBeenCalledExactlyOnceWith(1);
    const details = named(t('en', 'chat.transcript.details'))[0];
    const panel = document.getElementById(details.getAttribute('aria-controls') ?? '');
    expect(details.getAttribute('aria-expanded')).toBe('false'); expect(panel?.hidden).toBe(true);
    act(() => details.click());
    expect(details.getAttribute('aria-expanded')).toBe('true'); expect(panel?.hidden).toBe(false);
    expect(panel?.querySelector('dl.chat-metrics')).not.toBeNull();
  });
  it('keeps reasoning collapsed as plain text and shows tool calls as inert cards', () => {
    render([turn('a', { reasoning: '<b>thinking</b>', tools: [{ index: 0, id: 'call', name: '', arguments: '{"x":1}' }] })]);
    const reasoning = host.querySelector<HTMLDetailsElement>('details.chat-reasoning');
    expect(reasoning?.open).toBe(false);
    expect(reasoning?.querySelector('.chat-reasoning-text')?.textContent).toBe('<b>thinking</b>');
    expect(reasoning?.querySelector('b, pre')).toBeNull();
    const tool = host.querySelector('.chat-tool');
    expect(tool?.getAttribute('aria-label')).toBe(t('en', 'chat.transcript.tool_call', { name: t('en', 'chat.transcript.tool_name_pending') }));
    expect(tool?.textContent).toContain(t('en', 'chat.tool_call'));
    expect(tool?.textContent).toContain(t('en', 'chat.message.not_executed'));
    expect(tool?.querySelector('pre')?.textContent).toBe('{"x":1}');
  });
  it('announces copy results and keeps the select-the-text fallback', async () => {
    const writeText = vi.fn().mockRejectedValueOnce(new Error('denied')).mockResolvedValue(undefined);
    vi.stubGlobal('navigator', { ...navigator, clipboard: { writeText } });
    render([turn('a')]);
    await act(async () => { named(t('en', 'chat.transcript.copy'))[0].click(); });
    expect(host.querySelector('.chat-message--assistant [role="status"]')?.textContent).toBe(t('en', 'chat.transcript.copy_failed'));
    await act(async () => { named(t('en', 'chat.message.copy_prompt'))[0].click(); });
    expect(writeText).toHaveBeenLastCalledWith('Prompt a');
    expect(host.querySelector('.chat-message--user [role="status"]')?.textContent).toBe(t('en', 'chat.message.copied_prompt'));
  });
  it('shows the empty copy when a conversation has no turns', () => {
    render([], false, 'ko');
    expect(host.querySelector('section[aria-label]')?.getAttribute('aria-label')).toBe(t('ko', 'chat.transcript.label'));
    expect(host.querySelector('.chat-empty')?.textContent).toBe(t('ko', 'chat.transcript.empty'));
  });
});
