// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { WebUiSnapshot } from '../api/types';
import { ModelsLibrary } from '../features/models/screen';
import { model, snapshot } from '../features/models/test-fixtures';
import { t } from '../i18n/catalog';

let state: WebUiSnapshot;
const actions = { selectModel: vi.fn(), loadModel: vi.fn(), unloadModel: vi.fn(), removeModel: vi.fn(), downloadModel: vi.fn(), refreshCatalog: vi.fn(), refresh: vi.fn(), cancelOperation: vi.fn() };
vi.mock('../state', () => ({ useWebUi: () => state, useWebUiActions: () => actions }));
let host: HTMLDivElement;
let root: Root;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() }); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView'); });

describe('Models library row metadata', () => {
  it('keeps the selected state, source and quantization readable beside a single row control', () => {
    state = snapshot();
    act(() => root.render(<ModelsLibrary locale="en" />));
    const entry = model();
    const quantization = entry.metadata.quantization ?? t('en', 'models.library.unknown');
    const cell = host.querySelector('[data-testid="models-table"] tbody tr td');
    if (!cell) throw new Error('Missing name cell');
    for (const value of [entry.identity.display_name, t('en', 'models.library.selected'), entry.identity.source, quantization]) expect(cell.textContent).toContain(value);
    expect(cell.textContent).not.toContain(`${entry.identity.source}${quantization}`);
    const focusable = [...cell.querySelectorAll('button, a[href], input, select, textarea, [tabindex]')];
    expect(focusable.map((node) => node.getAttribute('aria-label'))).toEqual([t('en', 'models.library.inspect', { name: entry.identity.display_name })]);
    // Delivered by the shared component: asserted last so a pre-swap run fails here.
    expect([...cell.querySelectorAll('.badge:not(.status-tag)')].map((node) => node.textContent)).toEqual([t('en', 'models.library.selected'), entry.identity.source, quantization]);
  });
});
