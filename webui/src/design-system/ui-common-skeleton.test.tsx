// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { WebUiSnapshot } from '../api/types';
import { ModelsLibrary } from '../features/models/screen';
import { snapshot } from '../features/models/test-fixtures';
import { t } from '../i18n/catalog';

let state: WebUiSnapshot;
const actions = { selectModel: vi.fn(), loadModel: vi.fn(), unloadModel: vi.fn(), removeModel: vi.fn(), downloadModel: vi.fn(), refreshCatalog: vi.fn(), refresh: vi.fn(), cancelOperation: vi.fn() };
vi.mock('../state', () => ({ useWebUi: () => state, useWebUiActions: () => actions }));
let host: HTMLDivElement;
let root: Root;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() }); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView'); });

describe('Models catalog loading slot', () => {
  it.each(['en', 'ko'] as const)('announces one localized %s waiting status while the catalog is pending', (locale) => {
    state = { ...snapshot(), catalog: [], catalogSequence: null };
    act(() => root.render(<ModelsLibrary locale={locale} />));
    const table = host.querySelector('[data-testid="models-table"]');
    if (!table) throw new Error('Missing models table');
    const statuses = [...table.querySelectorAll('[role="status"], [aria-live]')];
    expect(statuses).toHaveLength(1);
    expect(statuses[0].textContent?.trim()).toBe(t(locale, 'models.library.waiting'));
    expect(statuses[0].closest('[aria-hidden="true"], [hidden], .sr-only')).toBeNull();
    expect(host.querySelector('[aria-label="Loading"]')).toBeNull();
    expect(host.textContent).not.toContain('Loading');
    // Delivered by the shared component: asserted last so a pre-swap run fails here.
    const shapes = [...statuses[0].querySelectorAll('.skeleton')];
    expect(shapes.length).toBeGreaterThan(0);
    for (const shape of shapes) expect(shape.closest('[aria-hidden="true"]')).not.toBeNull();
  });
});
