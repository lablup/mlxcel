// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { consumeNewConversationRequest, newConversation, replaceConversations, requestNewConversation, sessionGeneration, updateConversation, useNewConversationRequest } from './session';
import type { ChatTurn } from './history';
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); replaceConversations([]); });
afterEach(() => { replaceConversations([]); vi.unstubAllGlobals(); });
describe('bounded memory-only chat session', () => {
  it('admits fifty conversations but refuses a fifty-first without invalidating active generation', () => {
    const generation = sessionGeneration();
    for (let index = 0; index < 50; index++) expect(updateConversation(newConversation('Test conversation'))).toBe(true);
    expect(updateConversation(newConversation('Test conversation'))).toBe(false);
    expect(sessionGeneration()).toBe(generation);
  });
  it('increments replacement generation for imports, clear and logout but not normal turn updates', () => {
    const initial = sessionGeneration(); const conversation = newConversation('Test conversation');
    expect(updateConversation(conversation)).toBe(true); expect(sessionGeneration()).toBe(initial);
    replaceConversations([conversation]); expect(sessionGeneration()).toBe(initial + 1);
    updateConversation({ ...conversation, title: 'Updated' }); expect(sessionGeneration()).toBe(initial + 1);
    replaceConversations([]); expect(sessionGeneration()).toBe(initial + 2);
  });
  it('refuses oversized replacements atomically without changing the active generation', () => {
    const initial = sessionGeneration();
    expect(() => replaceConversations(Array.from({ length: 51 }, () => newConversation('Test conversation')))).toThrow('memory limit');
    expect(sessionGeneration()).toBe(initial);
    const conversation = newConversation('Test conversation'); conversation.systemPrompt = 'a'.repeat(16 * 1024 * 1024);
    expect(updateConversation(conversation)).toBe(false); expect(sessionGeneration()).toBe(initial);
  });
  it('bounds aggregate turns independently from the number of conversations', () => {
    const turn: ChatTurn = { id: 'turn', modelId: 'model', modelRevision: 1, inferenceId: 'model', modelName: 'Model', prompt: '', content: '', reasoning: '', tools: [], status: 'complete', finishReason: 'stop', usage: null, ttftMs: null, elapsedMs: null, error: null, parameters: {}, images: [], startedAt: 1 };
    const conversations = Array.from({ length: 10 }, () => ({ ...newConversation('Test conversation'), turns: Array.from({ length: 100 }, (_, index) => ({ ...turn, id: String(index) })) }));
    replaceConversations(conversations);
    expect(updateConversation({ ...newConversation('Test conversation'), turns: [turn] })).toBe(false);
  });
});
describe('new-conversation requests', () => {
  it('holds one pending request until it is consumed once', () => {
    expect(consumeNewConversationRequest()).toBe(false);
    requestNewConversation();
    requestNewConversation();
    expect(consumeNewConversationRequest()).toBe(true);
    expect(consumeNewConversationRequest()).toBe(false);
  });
  it('is cleared by replacing the conversations, including logout', () => {
    requestNewConversation();
    replaceConversations([]);
    expect(consumeNewConversationRequest()).toBe(false);
    requestNewConversation();
    replaceConversations([newConversation('Imported')]);
    expect(consumeNewConversationRequest()).toBe(false);
  });
  it('notifies subscribers with a fresh token for every request and 0 for none', () => {
    const seen: number[] = [];
    function Probe(): null { seen.push(useNewConversationRequest()); return null; }
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(createElement(Probe)));
    expect(seen.at(-1)).toBe(0);
    act(() => requestNewConversation());
    const first = seen.at(-1);
    expect(first).toBeGreaterThan(0);
    act(() => requestNewConversation());
    expect(seen.at(-1)).toBeGreaterThan(first ?? 0);
    act(() => replaceConversations([]));
    expect(seen.at(-1)).toBe(0);
    act(() => root.unmount());
  });
});
