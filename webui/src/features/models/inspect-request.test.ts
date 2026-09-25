// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { consumeInspectorRequest, requestInspector, useInspectorRequest } from './inspect-request';

function Token(): React.JSX.Element {
  return React.createElement('output', null, String(useInspectorRequest()));
}

let host: HTMLDivElement;
beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  host = document.createElement('div');
  document.body.append(host);
});
afterEach(() => {
  host.remove();
  vi.unstubAllGlobals();
});

describe('inspector request', () => {
  it('tells subscribers when a pending request is consumed', () => {
    const root = createRoot(host);
    act(() => root.render(React.createElement(Token)));
    expect(host.textContent).toBe('0');
    act(() => requestInspector());
    expect(host.textContent).not.toBe('0');
    let consumed = false;
    act(() => { consumed = consumeInspectorRequest(); });
    expect(consumed).toBe(true);
    expect(host.textContent).toBe('0');
    // Nothing pending: consuming again reports nothing and changes nothing.
    expect(consumeInspectorRequest()).toBe(false);
    act(() => root.unmount());
  });
});
