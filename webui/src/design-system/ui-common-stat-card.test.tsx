// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import runtimeFixture from '../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import type { MeasuredValue, WebUiSnapshot } from '../api/types';
import { validateRuntime } from '../api/validation';
import { ActivityPage } from '../features/activity';
import { initialSnapshot } from '../state/reducer';
import { StatCard } from './primitives';

let snapshot: WebUiSnapshot = initialSnapshot();
const actions = { refresh: vi.fn(async () => undefined), cancelOperation: vi.fn(async () => undefined), selectModel: vi.fn() };
vi.mock('../state', () => ({ useWebUi: () => snapshot, useWebUiActions: () => actions }));
let cleanup = (): void => undefined;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); });
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.unstubAllGlobals(); });

function mount(node: React.ReactNode): HTMLDivElement {
  const element = document.createElement('div');
  document.body.append(element);
  const root = createRoot(element);
  act(() => root.render(node));
  cleanup = () => { act(() => root.unmount()); element.remove(); };
  return element;
}
const measured = (value: number | null, unit: string, reason: string | null): MeasuredValue => ({ value, unit, scope: value === null ? 'unknown' : 'model', measured_at: value === null ? null : '2026-09-15T00:00:00Z', reason });

describe('runtime tiles through the shared StatCard', () => {
  it('keeps every label, value and localized provenance on non-interactive tiles', async () => {
    const runtime = validateRuntime(runtimeFixture);
    const measurements = { active_requests: measured(2, 'requests', 'Authoritative route completions'), completion_tokens_total: measured(0, 'tokens', null), gpu_utilization: measured(null, 'percent', 'not measured by mlxcel') };
    snapshot = { ...initialSnapshot(), connection: 'ready', selectedModelId: runtime.model_id, runtimes: new Map([[runtime.model_id, { ...runtime, measurements }]]) };
    const element = mount(<ActivityPage locale="en" />);
    const summary = [...element.querySelectorAll('[data-testid="runtime-summary"] .activity-metric')];
    expect(summary).toHaveLength(4);
    expect(summary[0].textContent).toContain('Active requests');
    expect(summary[0].textContent).toContain('2 requests');
    expect(summary[0].querySelector('.stat-card__hint')?.textContent).toMatch(/^Observed at /);
    expect(summary[2].textContent).toContain('Total completion tokens');
    expect(summary[2].textContent).toContain('0 tokens');
    for (const tile of [summary[1], summary[3]]) expect(tile.querySelector('.stat-card__value')?.textContent).toBe('unknown');
    const details = element.querySelector<HTMLDetailsElement>('.activity-measurement-details');
    await act(async () => { details?.querySelector('summary')?.click(); await new Promise((resolve) => setTimeout(resolve, 0)); });
    const rows = [...element.querySelectorAll('.activity-measurement-details .activity-measurements li')];
    expect(rows).toHaveLength(3);
    expect(rows[0].textContent).toContain('Authoritative route completions');
    expect(rows[0].textContent).toContain('Scope: Model');
    expect(rows[2].textContent).toContain('GPU utilization');
    expect(rows[2].textContent).toContain('unknown');
    expect(rows[2].textContent).toContain('Scope: Unknown');
    expect(rows[2].textContent).toContain('not measured by mlxcel');
    for (const tile of summary) expect(tile.matches('[tabindex], button, [role="button"]')).toBe(false);
    // Delivered by the shared component: asserted last so a pre-swap run fails here.
    expect(summary.map((tile) => tile.matches('.stat-card[role="group"]') ? tile.getAttribute('aria-label') : null)).toEqual(['Active requests: 2 requests', 'Total completed requests: unknown', 'Total completion tokens: 0 tokens', 'Queued requests: unknown']);
  });
});

describe('StatCard adapter', () => {
  it('renders loading skeletons as decorative shapes, leaving the announcement to the caller', () => {
    const element = mount(<StatCard label="Active requests" value="" hint=" " loading className="probe" />);
    const card = element.querySelector('.stat-card.probe');
    expect(card?.getAttribute('aria-label')).toBe('Active requests');
    const shapes = [...element.querySelectorAll('.skeleton')];
    expect(shapes).toHaveLength(2);
    for (const shape of shapes) {
      expect(shape.getAttribute('aria-hidden')).toBe('true');
      expect(shape.hasAttribute('role')).toBe(false);
      expect(shape.hasAttribute('aria-label')).toBe(false);
    }
    expect(element.querySelector('[role="status"]')).toBeNull();
    expect(element.textContent).not.toContain('Loading');
  });
  it('forwards the sparkline slot beside the value', () => {
    const element = mount(<StatCard label="Active requests" value="2 requests" sparkline={<svg className="probe-line" role="img" aria-label="trend" />} />);
    expect(element.querySelector('.stat-card__sparkline .probe-line')).not.toBeNull();
    expect(element.querySelector('.stat-card__value')?.textContent).toBe('2 requests');
    expect(element.querySelector('.skeleton')).toBeNull();
  });
});
