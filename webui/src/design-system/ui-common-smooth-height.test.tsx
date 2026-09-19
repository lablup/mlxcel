// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import operationsFixture from '../../../tests/fixtures/webui/examples/operations.list.json';
import type { Operation, WebUiSnapshot } from '../api/types';
import { validateOperationsList } from '../api/validation';
import { ActivityPage } from '../features/activity';
import { initialSnapshot } from '../state/reducer';
import { SmoothHeight } from './common-layout';

let snapshot: WebUiSnapshot = initialSnapshot();
const actions = { refresh: vi.fn(async () => undefined), cancelOperation: vi.fn(async () => undefined), selectModel: vi.fn() };
vi.mock('../state', () => ({ useWebUi: () => snapshot, useWebUiActions: () => actions }));
const operation = validateOperationsList(JSON.parse(JSON.stringify(operationsFixture), (key, value: unknown) => key === '$schemaName' ? undefined : value)).items[0];
let host: HTMLDivElement;
let root: Root;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.clearAllMocks(); vi.unstubAllGlobals(); });

function show(state: Operation['state']): HTMLOListElement {
  snapshot = { ...initialSnapshot(), connection: 'ready', operations: new Map([[operation.operation_id, { ...operation, state }]]) };
  act(() => root.render(<ActivityPage locale="en" />));
  const list = host.querySelector<HTMLOListElement>('ol.activity-operations');
  if (!list) throw new Error('Missing operations list');
  return list;
}
const pinned = (list: Element): boolean => Boolean(list.closest('.smooth-height--active'));

describe('Activity operations list height animation', () => {
  it('never pins or clips the list once every operation has settled', async () => {
    const list = show('succeeded');
    expect(list.textContent).toContain('succeeded');
    expect(pinned(list)).toBe(false);
    await act(async () => { await new Promise((resolve) => window.setTimeout(resolve, 600)); });
    expect(pinned(show('succeeded'))).toBe(false);
    expect(list.closest('.smooth-height')).not.toBeNull();
  });

  it('animates while an operation is in flight and releases after the last one settles', async () => {
    const list = show('running');
    expect(list.textContent).toContain('running');
    expect(pinned(list)).toBe(true);
    expect(pinned(show('succeeded'))).toBe(true);
    await act(async () => { await new Promise((resolve) => window.setTimeout(resolve, 600)); });
    expect(pinned(show('succeeded'))).toBe(false);
  });
});

describe('SmoothHeight settle window', () => {
  it('restarts the full 400ms settle window from the latest transition to false', () => {
    vi.useFakeTimers();
    try {
      const active = (): boolean => Boolean(host.querySelector('.smooth-height--active'));
      const render = (animate: boolean): void => { act(() => root.render(<SmoothHeight animate={animate}><p>content</p></SmoothHeight>)); };
      render(false);
      render(true);
      render(false); // opens the settle window; a naive implementation ends it 400ms from here
      act(() => { vi.advanceTimersByTime(300); });
      render(true);
      render(false); // a second transition to false inside the first window must restart it from here
      act(() => { vi.advanceTimersByTime(300); }); // 600ms since the window opened, 300ms since the latest transition to false
      expect(active()).toBe(true);
      act(() => { vi.advanceTimersByTime(100); }); // 400ms since the latest transition to false
      expect(active()).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
});
