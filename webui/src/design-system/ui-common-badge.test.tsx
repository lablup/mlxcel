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
  const cells = (): HTMLElement[] => [...host.querySelectorAll<HTMLElement>('[data-testid="models-table"] tbody tr:first-child td')];

  it('prints the name as text, the quantization in its own column and one task badge per task in declared order', () => {
    state = snapshot();
    act(() => root.render(<ModelsLibrary locale="en" />));
    const entry = model();
    const [name, , quantization, tasks] = cells();
    const label = name.querySelector('.truncate');
    expect(label?.textContent).toBe(entry.identity.display_name);
    expect(label?.getAttribute('title')).toBe(entry.identity.display_name);
    // No pill, no Selected label, no source or quantization line in the name cell. The lifecycle
    // badge there is the State column's, shown only once a narrow list hides that column.
    expect(name.querySelectorAll('button, a[href], input, select, textarea, [tabindex], .badge:not(.status-tag)')).toHaveLength(0);
    expect(name.querySelector('.models-name-state .status-tag')?.textContent).toBe(t('en', 'models.status.unloaded'));
    expect(quantization.textContent).toBe(entry.metadata.quantization);
    // Delivered by the shared component: asserted last so a pre-swap run fails here.
    expect([...tasks.querySelectorAll('.badge:not(.status-tag)')].map((node) => node.textContent)).toEqual(
      entry.metadata.output_tasks.map((task) => t('en', `models.task.${task}`)),
    );
  });

  it('falls back to the weight dtype for an unquantized checkpoint and flags an unsupported one with one warning badge', () => {
    const entry = model();
    state = snapshot({ ...entry, supported: false, metadata: { ...entry.metadata, quantization: null, dtype: 'bf16' } });
    act(() => root.render(<ModelsLibrary locale="en" />));
    const [name, , quantization] = cells();
    expect(quantization.textContent).toBe('bf16');
    const flags = [...name.querySelectorAll('.badge:not(.status-tag)')];
    expect(flags.map((node) => node.textContent)).toEqual([t('en', 'models.library.flag_unsupported')]);
    expect(flags[0].classList.contains('badge--warning')).toBe(true);
  });
});
