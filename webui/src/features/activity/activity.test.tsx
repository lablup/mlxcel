import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { initialSnapshot } from '../../state/reducer';
import type { MeasuredValue, WebUiSnapshot } from '../../api/types';
import operationsFixture from '../../../../tests/fixtures/webui/examples/operations.list.json';
import runtimeFixture from '../../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import { validateOperationsList, validateRuntime } from '../../api/validation';
import { ActivityPage } from './index';

let snapshot: WebUiSnapshot = initialSnapshot();
const actions = { refresh: vi.fn(async () => undefined), cancelOperation: vi.fn(async () => undefined), selectModel: vi.fn() };
vi.mock('../../state', () => ({ useWebUi: () => snapshot, useWebUiActions: () => actions }));
const operation = validateOperationsList(JSON.parse(JSON.stringify(operationsFixture), (key, value: unknown) => key === '$schemaName' ? undefined : value)).items[0];
let cleanup = (): void => undefined;
afterEach(() => { cleanup(); vi.clearAllMocks(); });
function mount(locale: 'en' | 'ko' = 'en'): HTMLDivElement {
  const element = document.createElement('div'); document.body.append(element);
  const root = createRoot(element);
  act(() => root.render(<ActivityPage locale={locale} />));
  cleanup = () => { act(() => root.unmount()); element.remove(); };
  return element;
}

describe('Activity page', () => {
  it('provides semantic empty state and shared model selection without loading', () => {
    snapshot = { ...initialSnapshot(), auth: { status: 'authenticated', tokenPresent: true }, connection: 'ready' };
    const element = mount();
    expect(element.querySelector('h1')?.textContent).toBe('Activity');
    expect(element.textContent).toContain('No operations in this session');
    expect(element.querySelector('[role="combobox"]')).not.toBeNull();
    expect(actions.refresh).not.toHaveBeenCalled();
    expect(actions.selectModel).not.toHaveBeenCalled();
    expect(element.querySelector('svg.activity-chart')).toBeNull();
  });
  it('disables stale cancellation and never exposes debug error text', () => {
    snapshot = { ...initialSnapshot(), connection: 'offline', operations: new Map([['op', { ...operation, state: 'running', cancellable: true, error: { code: 'unavailable', message: '/Users/private prompt output SECRET', retryable: true } }]]) };
    const element = mount();
    expect(element.textContent).toContain('Observations are stale');
    expect(element.textContent).not.toContain('SECRET');
    const cancel = [...element.querySelectorAll('button')].find((button) => button.textContent?.includes('Request cancellation'));
    expect(cancel?.disabled).toBe(true);
  });
  it('uses shared cancellation and does not mark accepted cancellation completed', async () => {
    snapshot = { ...initialSnapshot(), connection: 'ready', operations: new Map([['op', { ...operation, state: 'running', cancellable: true }]]) };
    const element = mount();
    const cancel = [...element.querySelectorAll('button')].find((button) => button.textContent?.includes('Request cancellation'));
    await act(async () => cancel?.click());
    expect(actions.cancelOperation).toHaveBeenCalledWith(operation.operation_id);
    expect(snapshot.operations.get('op')?.state).toBe('running');
  });
  it('reports unknown context as N/A rather than drawing invented occupancy', () => {
    const runtime = validateRuntime(runtimeFixture);
    snapshot = { ...initialSnapshot(), connection: 'ready', selectedModelId: runtime.model_id, runtimes: new Map([[runtime.model_id, { ...runtime, slots: { ...runtime.slots, available: true, request_context_tokens: null, items: [{ id: 0, processing: true, prompt_tokens: 100, cached_prompt_tokens: 0, decoded_tokens: 10 }] } }]]) };
    const element = mount();
    expect(element.textContent).toContain('context denominator is unknown');
    expect(element.querySelector('[role="progressbar"]')).toBeNull();
  });
  it('discloses a partial slot snapshot even when observed slots are available', () => {
    const runtime = validateRuntime(runtimeFixture);
    snapshot = { ...initialSnapshot(), connection: 'ready', selectedModelId: runtime.model_id, runtimes: new Map([[runtime.model_id, { ...runtime, slots: { ...runtime.slots, available: true, reason: 'showing the first 256 observational slots' } }]]) };
    const element = mount();
    expect(element.textContent).toContain('showing the first 256 observational slots');
  });

});

describe('Runtime progressive disclosure', () => {
  it.each(['en', 'ko'] as const)('keeps measured zero and slots in the %s summary with honest details', (locale) => {
    const runtime = validateRuntime(runtimeFixture);
    const measured: MeasuredValue = { value: 0, unit: 'requests', scope: 'model', measured_at: '2026-09-15T00:00:00Z', reason: 'Authoritative route completions' };
    const projected = { ...runtime, measurements: { active_requests: measured, gpu_utilization: { ...measured, value: null, unit: 'percent', measured_at: null, reason: 'not measured by mlxcel' } } };
    snapshot = { ...initialSnapshot(), connection: 'ready', selectedModelId: runtime.model_id, runtimes: new Map([[runtime.model_id, projected]]) };
    const element = mount(locale);
    const summary = element.querySelector('[data-testid="runtime-summary"]');
    expect(summary?.textContent).toContain('0 requests');
    expect(summary?.textContent).toContain(locale === 'en' ? 'Active requests' : '활성 요청');
    expect(summary?.textContent).not.toContain('GPU');
    const details = element.querySelector<HTMLDetailsElement>('details.activity-measurement-details');
    expect(details?.open).toBe(false);
    expect(details?.querySelector('summary')?.textContent).toBe(locale === 'en' ? 'All measurements and sources' : '전체 측정값과 출처');
    expect(details?.parentElement?.querySelector('h3')?.textContent).toBe(locale === 'en' ? 'Request slots' : '요청 슬롯');
    expect(element.textContent).toContain(locale === 'en' ? 'Unavailable measurements: 1' : '사용할 수 없는 측정값: 1');
    act(() => details?.querySelector('summary')?.click());
    expect(details?.open).toBe(true);
    expect(details?.textContent).toContain('N/A');
    expect(details?.textContent).toContain('not measured by mlxcel');
    expect(details?.textContent).toContain('Authoritative route completions');
    expect(details?.textContent).toContain(locale === 'en' ? 'Scope: model' : '범위: model');
  });
});
