// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import { beforeEach, describe, expect, it } from 'vitest';
import { newConversation, replaceConversations, sessionGeneration, updateConversation } from './session';
import type { ChatTurn } from './history';
beforeEach(() => replaceConversations([]));
describe('bounded memory-only chat session', () => {
  it('admits fifty conversations but refuses a fifty-first without invalidating active generation', () => {
    const generation = sessionGeneration();
    for (let index = 0; index < 50; index++) expect(updateConversation(newConversation())).toBe(true);
    expect(updateConversation(newConversation())).toBe(false);
    expect(sessionGeneration()).toBe(generation);
  });
  it('increments replacement generation for imports, clear and logout but not normal turn updates', () => {
    const initial = sessionGeneration(); const conversation = newConversation();
    expect(updateConversation(conversation)).toBe(true); expect(sessionGeneration()).toBe(initial);
    replaceConversations([conversation]); expect(sessionGeneration()).toBe(initial + 1);
    updateConversation({ ...conversation, title: 'Updated' }); expect(sessionGeneration()).toBe(initial + 1);
    replaceConversations([]); expect(sessionGeneration()).toBe(initial + 2);
  });
  it('refuses oversized replacements atomically without changing the active generation', () => {
    const initial = sessionGeneration();
    expect(() => replaceConversations(Array.from({ length: 51 }, () => newConversation()))).toThrow('memory limit');
    expect(sessionGeneration()).toBe(initial);
    const conversation = newConversation(); conversation.systemPrompt = 'a'.repeat(16 * 1024 * 1024);
    expect(updateConversation(conversation)).toBe(false); expect(sessionGeneration()).toBe(initial);
  });
  it('bounds aggregate turns independently from the number of conversations', () => {
    const turn: ChatTurn = { id: 'turn', modelId: 'model', modelRevision: 1, inferenceId: 'model', modelName: 'Model', prompt: '', content: '', reasoning: '', tools: [], status: 'complete', finishReason: 'stop', usage: null, ttftMs: null, elapsedMs: null, error: null, parameters: {}, images: [] };
    const conversations = Array.from({ length: 10 }, () => ({ ...newConversation(), turns: Array.from({ length: 100 }, (_, index) => ({ ...turn, id: String(index) })) }));
    replaceConversations(conversations);
    expect(updateConversation({ ...newConversation(), turns: [turn] })).toBe(false);
  });
});
