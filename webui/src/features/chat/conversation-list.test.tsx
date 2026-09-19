// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ConversationList } from './conversation-list';
import { importConversations } from './history';

let host: HTMLDivElement;
let root: Root;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

describe('ConversationList', () => {
  it('renders an imported update time a Date cannot hold instead of failing to render', () => {
    // History accepts any safe integer here, which reaches past the largest Date (8.64e15 ms).
    const conversations = importConversations(JSON.stringify({ version: 2, conversations: [{ id: 'far', title: 'Far future', systemPrompt: '', turns: [], updatedAt: Number.MAX_SAFE_INTEGER }] }));
    act(() => root.render(<ConversationList locale="en" conversations={conversations} currentId={null} disabled={false} atLimit={false} onSelect={vi.fn()} onRename={vi.fn()} onDelete={vi.fn()} />));
    expect(host.querySelector('.chat-row-title')?.textContent).toBe('Far future');
    expect(host.querySelector('.chat-row-meta time')?.hasAttribute('dateTime')).toBe(false);
  });
});
