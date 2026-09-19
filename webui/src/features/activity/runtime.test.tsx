import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { initialSnapshot } from '../../state/reducer';
import type { MeasuredValue, RuntimeSnapshot, WebUiSnapshot } from '../../api/types';
import runtimeFixture from '../../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import { validateRuntime } from '../../api/validation';
import { t } from '../../i18n/catalog';
import { ActivityPage } from './index';

let snapshot: WebUiSnapshot = initialSnapshot();
const actions = { refresh: vi.fn(async () => undefined), cancelOperation: vi.fn(async () => undefined), selectModel: vi.fn() };
vi.mock('../../state', () => ({ useWebUi: () => snapshot, useWebUiActions: () => actions }));
let cleanup = (): void => undefined;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); });
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.unstubAllGlobals(); });
function mount(locale: 'en' | 'ko' = 'en'): HTMLDivElement {
  const element = document.createElement('div'); document.body.append(element);
  const root = createRoot(element);
  act(() => root.render(<ActivityPage locale={locale} />));
  cleanup = () => { act(() => root.unmount()); element.remove(); };
  return element;
}
// jsdom queues the native toggle event, so wait one task after activating a summary.
async function activate(target: Element | null | undefined): Promise<void> {
  await act(async () => { target?.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })); await new Promise((resolve) => setTimeout(resolve, 0)); });
}
const at = '2026-09-15T00:00:00Z';
const measured = (value: number, unit: string): MeasuredValue => ({ value, unit, scope: 'model', measured_at: at, reason: 'Authoritative route completions' });
const missing = (unit: string, reason: string | null): MeasuredValue => ({ value: null, unit, scope: 'unknown', measured_at: null, reason });
function observe(measurements: Record<string, MeasuredValue>, extra: Partial<WebUiSnapshot> = {}): RuntimeSnapshot {
  const runtime = { ...validateRuntime(runtimeFixture), measurements };
  snapshot = { ...initialSnapshot(), connection: 'ready', selectedModelId: runtime.model_id, runtimes: new Map([[runtime.model_id, runtime]]), runtimeHistory: [{ receivedAt: Date.parse(at), runtime }], lastUpdatedAt: Date.parse(at), ...extra };
  return runtime;
}
const tiles = (element: Element): Element[] => [...element.querySelectorAll('[data-testid="runtime-summary"] .activity-metric')];

describe('runtime summary tiles', () => {
  it.each(['en', 'ko'] as const)('always shows four %s tiles in order, keeping measured zero and unknown apart', (locale) => {
    observe({ active_requests: measured(0, 'requests'), completion_tokens_total: missing('tokens', 'metrics disabled; restart with --metrics'), queued_requests: missing('requests', 'a reason the page has never seen'), gpu_utilization: missing('percent', 'not measured by mlxcel') });
    const element = mount(locale);
    const [active, completed, tokens, queued] = tiles(element);
    expect(tiles(element).map((tile) => tile.querySelector('.stat-card__label')?.textContent)).toEqual(['active_requests', 'completed_requests_total', 'completion_tokens_total', 'queued_requests'].map((name) => t(locale, `activity.metric.${name}` as 'activity.metric.active_requests')));
    expect(active.textContent).toContain(t(locale, 'activity.unit.requests', { value: '0' }));
    expect(active.textContent).toContain(t(locale, 'activity.observed_at', { time: '' }).trim());
    // Absent from the snapshot and null both read unknown, never 0.
    for (const tile of [completed, tokens, queued]) expect(tile.querySelector('.stat-card__value')?.textContent).toBe(t(locale, 'format.unknown'));
    expect(tokens.textContent).toContain(t(locale, 'activity.reason.metrics_disabled'));
    expect(queued.textContent).toContain(t(locale, 'activity.reason.see_details'));
    expect(completed.textContent).toContain(t(locale, 'activity.reason.see_details'));
    // Raw server reasons stay out of the tiles.
    for (const raw of ['metrics disabled', 'a reason the page has never seen', 'Authoritative route completions']) expect(element.querySelector('[data-testid="runtime-summary"]')?.textContent).not.toContain(raw);
    expect(element.querySelector('[data-testid="runtime-summary"]')?.textContent).not.toContain('GPU');
  });
  it('draws the active-request sparkline as one path once two samples exist', () => {
    const runtime = observe({ active_requests: measured(1, 'requests') });
    let element = mount();
    expect(element.querySelector('.activity-sparkline')).toBeNull();
    cleanup();
    const samples = Array.from({ length: 150 }, (_, index) => ({ receivedAt: Date.parse(at) + index * 2000, runtime: index % 7 === 0 ? { ...runtime, measurements: { active_requests: missing('requests', 'counter unavailable') } } : runtime }));
    snapshot = { ...snapshot, runtimeHistory: samples, lastUpdatedAt: samples.at(-1)?.receivedAt ?? null };
    element = mount();
    const svg = element.querySelector('[data-testid="runtime-summary"] .activity-sparkline');
    expect(svg?.getAttribute('role')).toBe('img');
    expect(svg?.getAttribute('aria-label')).toBe(t('en', 'activity.chart.label'));
    expect(svg?.children).toHaveLength(1);
    // One zero-length subpath per non-null sample; gaps stay blank.
    expect(svg?.querySelector('path')?.getAttribute('d')?.match(/M/g)).toHaveLength(samples.length - Math.ceil(150 / 7));
    expect(svg?.querySelector('circle')).toBeNull();
  });
  it('shows four decorative loading tiles under one localized status until the first sample', () => {
    const runtime = validateRuntime(runtimeFixture);
    snapshot = { ...initialSnapshot(), connection: 'polling', selectedModelId: runtime.model_id };
    const element = mount('ko');
    expect(tiles(element)).toHaveLength(4);
    const statuses = [...element.querySelectorAll('.activity-runtime [role="status"]')];
    expect(statuses.map((status) => status.textContent)).toEqual([t('ko', 'activity.waiting')]);
    expect(element.querySelector('[aria-label="Loading"]')).toBeNull();
    const shapes = [...element.querySelectorAll('[data-testid="runtime-summary"] .skeleton')];
    expect(shapes.length).toBeGreaterThanOrEqual(4);
    for (const shape of shapes) expect(shape.getAttribute('aria-hidden')).toBe('true');
    expect(element.textContent).not.toContain('N/A');
  });
  it('keeps a localized empty state without a sample while observations are stale', () => {
    const runtime = validateRuntime(runtimeFixture);
    snapshot = { ...initialSnapshot(), connection: 'offline', selectedModelId: runtime.model_id };
    const element = mount();
    expect(tiles(element)).toHaveLength(0);
    expect(element.textContent).toContain(t('en', 'activity.unavailable'));
    expect(element.textContent).not.toContain('N/A');
  });
});

describe('All measurements and sources', () => {
  it.each(['en', 'ko'] as const)('puts the %s unavailable count in the summary and renders the list only while open', async (locale) => {
    observe({ active_requests: measured(0, 'requests'), gpu_utilization: missing('percent', 'not measured by mlxcel'), ttft: missing('ms', 'request-send to first output token timing is not available in runtime counters') });
    const element = mount(locale);
    const details = element.querySelector<HTMLDetailsElement>('details.activity-measurement-details');
    const summary = details?.querySelector('summary');
    const badge = summary?.querySelector('.ds-badge');
    expect(badge?.textContent).toBe(t(locale, 'activity.unavailable_badge', { count: '2' }));
    expect(summary?.textContent).toBe(`${t(locale, 'activity.metric_details')} ${t(locale, 'activity.unavailable_badge', { count: '2' })}`);
    expect(element.textContent).not.toContain('Unavailable measurements');
    // Closed: nothing but the summary is built.
    expect(details?.open).toBe(false);
    expect(details?.children).toHaveLength(1);
    await activate(badge);
    expect(details?.open).toBe(true);
    const text = details?.textContent ?? '';
    expect(text).toContain(t(locale, 'activity.memory'));
    expect(text).toContain(t(locale, 'activity.timing'));
    expect(text).toContain('not measured by mlxcel');
    expect(text).toContain('Authoritative route completions');
    expect(text).toContain(t(locale, 'activity.scope', { scope: t(locale, 'activity.scope.model') }));
    expect(text).toContain(t(locale, 'activity.scope', { scope: t(locale, 'activity.scope.unknown') }));
    expect(text).toContain(t(locale, 'activity.history.note'));
    expect(text).toContain(t(locale, 'activity.chart.point', { value: '0' }));
    expect(text).not.toContain('N/A');
  });
  it('omits the badge when every measurement has a value', () => {
    observe({ active_requests: measured(0, 'requests') });
    const element = mount();
    expect(element.querySelector('details.activity-measurement-details .ds-badge')).toBeNull();
  });
});

describe('slot table', () => {
  const slots = (overrides: Partial<RuntimeSnapshot['slots']>): void => {
    const runtime = observe({});
    const next = { ...runtime, slots: { ...runtime.slots, available: true, reason: null, measured_at: at, request_context_tokens: 40960, ...overrides } };
    snapshot = { ...snapshot, runtimes: new Map([[runtime.model_id, next]]) };
  };
  const cells = (element: Element): string[][] => [...element.querySelectorAll('[data-testid="activity-slot-table"] tbody tr')].map((row) => [...row.querySelectorAll('td')].map((cell) => cell.textContent ?? ''));
  it('reads a processing slot without a count as unknown and an idle one as empty', () => {
    slots({ items: [{ id: 0, processing: true, prompt_tokens: 21, cached_prompt_tokens: 3, decoded_tokens: 5 }, { id: 1, processing: true, prompt_tokens: null, cached_prompt_tokens: null, decoded_tokens: null }, { id: 2, processing: false, prompt_tokens: null, cached_prompt_tokens: null, decoded_tokens: null }] });
    const element = mount();
    const unknown = t('en', 'format.unknown');
    expect(cells(element)).toEqual([
      ['0', 'Processing', '21 / 40,960 tokens', '5', '3'],
      ['1', 'Processing', unknown, unknown, unknown],
      ['2', 'Idle', '0 / 40,960 tokens', '', ''],
    ]);
    expect(element.querySelector('.activity-slot-meta')?.textContent).toBe(`${t('en', 'activity.parallel', { effective: unknown, configured: '4' })}${t('en', 'activity.context', { tokens: '40,960 tokens' })}${t('en', 'activity.pool', { tokens: unknown })}`);
    const region = element.querySelector('.activity-slot-scroll');
    expect(region?.getAttribute('role')).toBe('region');
    expect(region?.getAttribute('tabindex')).toBe('0');
    expect(document.getElementById(region?.getAttribute('aria-labelledby') ?? '')?.textContent).toBe(t('en', 'activity.slots'));
  });
  it.each(['en', 'ko'] as const)('localizes the %s server slot reasons and keeps an unrecognized one behind a disclosure', async (locale) => {
    slots({ available: false, reason: 'slots disabled; restart with --slots', items: [] });
    let element = mount(locale);
    expect(element.textContent).toContain(t(locale, 'activity.slots_reason.disabled'));
    expect(element.querySelector('[data-testid="activity-slot-table"]')).toBeNull();
    if (locale === 'ko') expect(element.textContent).not.toContain('slots disabled');
    cleanup();
    slots({ available: false, reason: 'a brand new server reason', items: [] });
    element = mount(locale);
    expect(element.textContent).toContain(t(locale, 'activity.slots_reason.other'));
    const details = [...element.querySelectorAll('details')].find((item) => item.querySelector('summary')?.textContent === t(locale, 'activity.slots_reason.raw'));
    expect(details?.textContent).not.toContain('a brand new server reason');
    await activate(details?.querySelector('summary'));
    expect(details?.textContent).toContain('a brand new server reason');
  });
  it('bounds the scroll region only past eight slots', () => {
    slots({ items: Array.from({ length: 9 }, (_, id) => ({ id, processing: false, prompt_tokens: null, cached_prompt_tokens: null, decoded_tokens: null })) });
    const element = mount();
    expect(element.querySelector('.activity-slot-scroll')?.hasAttribute('data-bounded')).toBe(true);
    cleanup();
    slots({ items: Array.from({ length: 8 }, (_, id) => ({ id, processing: false, prompt_tokens: null, cached_prompt_tokens: null, decoded_tokens: null })) });
    expect(mount().querySelector('.activity-slot-scroll')?.hasAttribute('data-bounded')).toBe(false);
  });
});
