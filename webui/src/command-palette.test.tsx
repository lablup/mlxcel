// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { CatalogEntry } from './api/types';
import { CommandPalette } from './command-palette';
import { model } from './features/models/test-fixtures';

function entry(id: string, name: string): CatalogEntry {
  const base = model();
  return { ...base, identity: { ...base.identity, id, display_name: name } };
}

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  HTMLDialogElement.prototype.showModal = vi.fn(function showModal(this: HTMLDialogElement) { this.setAttribute('open', ''); });
  HTMLDialogElement.prototype.close = vi.fn(function close(this: HTMLDialogElement) { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.restoreAllMocks();
});

describe('CommandPalette', () => {
  it('clears a typed query on close, so the closed dialog holds no stale model matches', () => {
    const catalog = [entry('mdl_llama', 'llama-3'), entry('mdl_qwen', 'qwen-2.5')];
    let open = true;
    const render = (): void => act(() => root.render(
      <CommandPalette open={open} locale="en" catalog={catalog} loaded={[]} onClose={() => {}} onNavigate={() => {}} onNewChat={() => {}} onOpenModel={() => {}} />,
    ));
    render();
    const search = host.querySelector<HTMLInputElement>('[data-testid="command-search"]');
    if (!search) throw new Error('Missing search field');
    act(() => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(search, 'llama');
      search.dispatchEvent(new Event('input', { bubbles: true }));
    });
    expect(host.querySelectorAll('[data-testid="command-model"]')).toHaveLength(1);
    open = false;
    render();
    expect(search.value).toBe('');
    expect(host.querySelectorAll('[data-testid="command-model"]')).toHaveLength(0);
    open = true;
    render();
    expect(search.value).toBe('');
    expect(host.querySelectorAll('[data-testid="command-model"]')).toHaveLength(0);
  });
});
