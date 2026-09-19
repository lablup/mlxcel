// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import { useSyncExternalStore } from 'react';
import type { ChatConversation, ChatTurn } from './history';
import { HISTORY_LIMITS } from './history';
let conversations: ChatConversation[] = [];
let generation = 0;
// A pending "start a new conversation" request from the global shortcut or the palette.
// Chat consumes it on mount or on change; 0 means none. It is published through the same
// listener set as the conversations, so a mounted Chat sees it without a second store.
let newConversationRequest = 0;
let requestCounter = 0;
const listeners = new Set<() => void>();
function subscribe(listener: () => void): () => void { listeners.add(listener); return () => { listeners.delete(listener); }; }
export function sessionGeneration(): number { return generation; }
/** The title is always supplied by the caller, localized; there is no English default. */
export function newConversation(title: string): ChatConversation {
  return { id: crypto.randomUUID(), title, systemPrompt: '', turns: [], updatedAt: Date.now() };
}
const turnBytes = new WeakMap<ChatTurn, number>();
function byteSize(turn: ChatTurn): number {
  const cached = turnBytes.get(turn);
  if (cached !== undefined) return cached;
  const size = new TextEncoder().encode(JSON.stringify(turn)).byteLength;
  turnBytes.set(turn, size); return size;
}
function bounded(next: ChatConversation[]): boolean {
  if (next.length > HISTORY_LIMITS.conversations || next.reduce((count, item) => count + item.turns.length, 0) > HISTORY_LIMITS.totalTurns) return false;
  // Exact JSON byte accounting, reusing immutable historical turn sizes rather than
  // serializing an entire origin's 16 MiB history on each 50 ms stream flush.
  const size = 2 + Math.max(0, next.length - 1) + next.reduce((sum, conversation) => sum + new TextEncoder().encode(JSON.stringify({ ...conversation, turns: [] })).byteLength + conversation.turns.reduce((count, turn) => count + byteSize(turn), 0) + Math.max(0, conversation.turns.length - 1), 0);
  return size <= HISTORY_LIMITS.jsonBytes;
}
function notify(): void { for (const listener of listeners) listener(); }
function publish(next: ChatConversation[]): void { conversations = next; notify(); }
/** Imports, Clear All and logout (an empty list) also drop a pending new-conversation request. */
export function replaceConversations(next: ChatConversation[]): void {
  if (!bounded(next)) throw new Error('Conversation memory limit exceeded.');
  generation++;
  newConversationRequest = 0;
  publish(next);
}
/** Asks the next mounted (or already mounted) Chat to start an empty conversation once. */
export function requestNewConversation(): void {
  requestCounter++;
  newConversationRequest = requestCounter;
  notify();
}
/** The pending request token, 0 when none; changes on every new request. */
export function useNewConversationRequest(): number {
  return useSyncExternalStore(subscribe, () => newConversationRequest);
}
/** Clears the pending request and reports whether there was one. Consume, then act: a second call finds nothing. */
export function consumeNewConversationRequest(): boolean {
  const pending = newConversationRequest !== 0;
  newConversationRequest = 0;
  return pending;
}
export function updateConversation(value: ChatConversation): boolean {
  const next = conversations.some((entry) => entry.id === value.id) ? conversations.map((entry) => entry.id === value.id ? value : entry) : [...conversations, value];
  if (!bounded(next)) return false;
  publish(next); return true;
}
export function useConversations(): ChatConversation[] {
  return useSyncExternalStore(subscribe, () => conversations);
}
