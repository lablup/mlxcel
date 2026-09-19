// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import runtimeFixture from '../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import type { MeasuredValue, WebUiSnapshot } from '../api/types';
import { validateRuntime } from '../api/validation';
import { ActivityPage } from '../features/activity';
import { initialSnapshot } from '../state/reducer';

let snapshot: WebUiSnapshot = initialSnapshot();
const actions = { refresh: vi.fn(async () => undefined), cancelOperation: vi.fn(async () => undefined), selectModel: vi.fn() };
vi.mock('../state', () => ({ useWebUi: () => snapshot, useWebUiActions: () => actions }));
let cleanup = (): void => undefined;
afterEach(() => { cleanup(); vi.clearAllMocks(); });

function mount(): HTMLDivElement {
  const element = document.createElement('div');
  document.body.append(element);
  const root = createRoot(element);
  act(() => root.render(<ActivityPage locale="en" />));
  cleanup = () => { act(() => root.unmount()); element.remove(); };
  return element;
}
const measured = (value: number | null, unit: string, reason: string | null): MeasuredValue => ({ value, unit, scope: value === null ? 'unknown' : 'model', measured_at: value === null ? null : '2026-09-15T00:00:00Z', reason });

describe('runtime tiles through the shared StatCard', () => {
  it('keeps every label, value, scope, observation and reason on non-interactive tiles', () => {
    const runtime = validateRuntime(runtimeFixture);
    const measurements = { active_requests: measured(2, 'requests', 'Authoritative route completions'), completion_tokens_total: measured(0, 'tokens', null), gpu_utilization: measured(null, 'percent', 'not measured by mlxcel') };
    snapshot = { ...initialSnapshot(), connection: 'ready', selectedModelId: runtime.model_id, runtimes: new Map([[runtime.model_id, { ...runtime, measurements }]]) };
    const element = mount();
    const summary = [...element.querySelectorAll('[data-testid="runtime-summary"] .activity-metric')];
    expect(summary).toHaveLength(2);
    expect(summary[0].textContent).toContain('Active requests');
    expect(summary[0].textContent).toContain('2 requests');
    expect(summary[1].textContent).toContain('Total completion tokens');
    expect(summary[1].textContent).toContain('0 tokens');
    const details = [...element.querySelectorAll('.activity-measurement-details .activity-metric')];
    expect(details).toHaveLength(3);
    expect(details[0].textContent).toContain('Authoritative route completions');
    expect(details[0].textContent).toContain('Scope: model');
    expect(details[2].textContent).toContain('GPU utilization');
    expect(details[2].textContent).toContain('N/A');
    expect(details[2].textContent).toContain('Scope: unknown');
    expect(details[2].textContent).toContain('not measured by mlxcel');
    for (const tile of [...summary, ...details]) expect(tile.matches('[tabindex], button, [role="button"]')).toBe(false);
    // Delivered by the shared component: asserted last so a pre-swap run fails here.
    expect(summary.map((tile) => tile.matches('.stat-card[role="group"]') ? tile.getAttribute('aria-label') : null)).toEqual(['Active requests: 2 requests', 'Total completion tokens: 0 tokens']);
  });
});
