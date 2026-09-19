import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { initialSnapshot } from '../../state/reducer';
import type { CatalogEntry, Operation, OperationKind, OperationState, WebUiSnapshot } from '../../api/types';
import catalogFixture from '../../../../tests/fixtures/webui/examples/catalog.page.json';
import operationsFixture from '../../../../tests/fixtures/webui/examples/operations.list.json';
import runtimeFixture from '../../../../tests/fixtures/webui/examples/runtime.snapshot.json';
import { validateOperationsList, validateRuntime } from '../../api/validation';
import { t } from '../../i18n/catalog';
import { ActivityPage, modelOptions } from './index';

let snapshot: WebUiSnapshot = initialSnapshot();
const actions = { refresh: vi.fn(async () => undefined), cancelOperation: vi.fn(async () => undefined), selectModel: vi.fn() };
vi.mock('../../state', () => ({ useWebUi: () => snapshot, useWebUiActions: () => actions }));
const operation = validateOperationsList(JSON.parse(JSON.stringify(operationsFixture), (key, value: unknown) => key === '$schemaName' ? undefined : value)).items[0];
const entry = catalogFixture.items[0] as CatalogEntry;
let cleanup = (): void => undefined;
const occurrences = (element: Element, text: string): number => (element.textContent ?? '').split(text).length - 1;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); });
afterEach(() => { cleanup(); vi.clearAllMocks(); vi.unstubAllGlobals(); });
function mount(locale: 'en' | 'ko' = 'en'): HTMLDivElement {
  const element = document.createElement('div'); document.body.append(element);
  const root = createRoot(element);
  act(() => root.render(<ActivityPage locale={locale} />));
  cleanup = () => { act(() => root.unmount()); element.remove(); };
  return element;
}
/** Text a reader sees without opening anything: every closed disclosure contributes its summary only. */
function collapsedText(element: Element): string {
  const copy = element.cloneNode(true) as Element;
  for (const details of copy.querySelectorAll('details:not([open])')) for (const child of [...details.children]) if (child.tagName !== 'SUMMARY') child.remove();
  return copy.textContent ?? '';
}
const withRuntime = (slots: Partial<ReturnType<typeof validateRuntime>['slots']>): WebUiSnapshot => {
  const runtime = validateRuntime(runtimeFixture);
  return { ...initialSnapshot(), connection: 'ready', selectedModelId: runtime.model_id, runtimes: new Map([[runtime.model_id, { ...runtime, slots: { ...runtime.slots, ...slots } }]]) };
};

describe('Activity page', () => {
  it('provides semantic empty state and shared model selection without loading', () => {
    snapshot = { ...initialSnapshot(), auth: { status: 'authenticated', tokenPresent: true }, connection: 'ready' };
    const element = mount();
    expect(element.querySelector('h1')?.textContent).toBe('Activity');
    expect(element.textContent).toContain('No operations in this session');
    expect(element.querySelectorAll('[role="combobox"]')).toHaveLength(1);
    // The model picker is a header action, beside Refresh and Export.
    expect(element.querySelector('.page-header__actions [role="combobox"]')).not.toBeNull();
    expect(actions.refresh).not.toHaveBeenCalled();
    expect(actions.selectModel).not.toHaveBeenCalled();
    expect(element.textContent).not.toContain('Show recent history');
  });
  it('puts the last snapshot time in the header description instead of a separate line', () => {
    snapshot = { ...initialSnapshot(), connection: 'ready', lastSuccessfulAt: Date.parse('2026-09-15T01:02:03Z') };
    const element = mount();
    const description = element.querySelector('.page-header__description')?.textContent ?? '';
    expect(description.startsWith(t('en', 'activity.intro'))).toBe(true);
    expect(description).toContain('Last snapshot');
    expect(element.querySelector('.activity-updated')).toBeNull();
  });
  it('offers loaded models first, then catalog order, labelled by the verbatim display name', () => {
    const named = (id: string, state: CatalogEntry['lifecycle']['state']): CatalogEntry => ({ ...entry, identity: { ...entry.identity, id, display_name: `${id}-name` }, lifecycle: { ...entry.lifecycle, state } });
    const options = modelOptions([named('a', 'unloaded'), named('b', 'ready'), named('c', 'unloaded'), named('d', 'ready')], 'en');
    expect(options.map((option) => option.label)).toEqual([t('en', 'activity.select'), 'b-name', 'd-name', 'a-name', 'c-name']);
    expect(options[0].value).toBe('');
  });
  it('disables stale cancellation and never exposes debug error text', () => {
    snapshot = { ...initialSnapshot(), connection: 'offline', operations: new Map([['op', { ...operation, state: 'running', cancellable: true, error: { code: 'unavailable', message: '/Users/private prompt output SECRET', retryable: true } }]]) };
    const element = mount();
    expect(element.textContent).toContain('Observations are stale');
    expect(element.textContent).not.toContain('SECRET');
    const cancel = [...element.querySelectorAll('button')].find((button) => button.textContent?.includes('Request cancellation'));
    expect(cancel?.disabled).toBe(true);
  });
  it('shows a failed operation by its error code only', () => {
    snapshot = { ...initialSnapshot(), connection: 'ready', operations: new Map([['op', { ...operation, state: 'failed', error: { code: 'conflict', message: '/Users/private SECRET', retryable: false } }]]) };
    const element = mount();
    expect(element.textContent).toContain('conflict');
    expect(element.textContent).not.toContain('SECRET');
  });
  it('uses shared cancellation and does not mark accepted cancellation completed', async () => {
    snapshot = { ...initialSnapshot(), connection: 'ready', operations: new Map([['op', { ...operation, state: 'running', cancellable: true }]]) };
    const element = mount();
    const cancel = [...element.querySelectorAll('button')].find((button) => button.textContent?.includes('Request cancellation'));
    await act(async () => cancel?.click());
    expect(actions.cancelOperation).toHaveBeenCalledWith(operation.operation_id);
    expect(snapshot.operations.get('op')?.state).toBe('running');
  });
  it('keeps operation and model identifiers inside the Operation details disclosure', async () => {
    const known: Operation = { ...operation, operation_id: 'op_known_1', target: { target_kind: 'model', model_id: entry.identity.id, requested_revision: 1 } };
    const orphan: Operation = { ...operation, operation_id: 'op_orphan_2', updated_at: '2026-09-12T03:04:04Z' };
    snapshot = { ...initialSnapshot(), connection: 'ready', catalog: [entry], operations: new Map<string, Operation>([[known.operation_id, known], [orphan.operation_id, orphan]]) };
    const element = mount();
    const visible = collapsedText(element);
    for (const id of [known.operation_id, orphan.operation_id, entry.identity.id, operation.target.target_kind === 'model' ? operation.target.model_id : '']) expect(visible).not.toContain(id);
    expect(visible).toContain(entry.identity.display_name);
    expect(visible).toContain(t('en', 'activity.target.unknown_model'));
    expect(visible).not.toContain('model_load');
    const details = element.querySelector<HTMLDetailsElement>('.activity-operation__details');
    await act(async () => { details?.querySelector('summary')?.click(); await new Promise((resolve) => setTimeout(resolve, 0)); });
    expect(details?.open).toBe(true);
    expect(details?.textContent).toContain(known.operation_id);
    expect(details?.textContent).toContain(entry.identity.id);
  });
  it.each(['en', 'ko'] as const)('labels every operation kind and state from the %s catalog', (locale) => {
    const kinds: OperationKind[] = ['catalog_refresh', 'model_load', 'model_unload', 'download', 'model_removal', 'settings_patch'];
    const states: OperationState[] = ['queued', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled'];
    const operations = new Map<string, Operation>(kinds.map((kind, index) => [`op_${kind}`, { ...operation, operation_id: `op_${kind}`, kind, state: states[index], updated_at: `2026-09-12T03:04:0${index}Z`, target: kind === 'download' ? { target_kind: 'download', repo_id: 'org/repo', revision: 'main' } : operation.target }]));
    snapshot = { ...initialSnapshot(), connection: 'ready', operations };
    const element = mount(locale);
    const rows = [...element.querySelectorAll('.activity-operation')];
    expect(rows).toHaveLength(6);
    const text = rows.map((row) => row.textContent ?? '').join('\n');
    for (const kind of kinds) expect(text).toContain(t(locale, `activity.kind.${kind}`));
    for (const state of states) expect(text).toContain(t(locale, `activity.state.${state}`));
    for (const raw of [...kinds, ...states.filter((state) => t(locale, `activity.state.${state}`).toLowerCase() !== state)]) expect(collapsedText(element)).not.toContain(raw);
    expect(text).toContain('org/repo');
    expect(element.querySelector('.activity-counts')?.textContent).toBe(t(locale, 'activity.operations.counts', { active: '3', failed: '1' }));
    expect(element.querySelectorAll('.activity-counts[role="status"]')).toHaveLength(1);
  });
  it('describes retention on the count while operations are listed', () => {
    snapshot = { ...initialSnapshot(), connection: 'ready', operations: new Map([['op', operation]]) };
    const element = mount();
    const help = element.querySelector<HTMLButtonElement>(`button[aria-label="${t('en', 'activity.session_help')}"]`);
    expect(help).not.toBeNull();
    const description = help?.getAttribute('aria-describedby')?.split(' ').map((id) => document.getElementById(id)?.textContent).join(' ');
    expect(description).toContain(t('en', 'activity.session'));
  });
  it('reports unknown context rather than drawing invented occupancy', () => {
    snapshot = withRuntime({ available: true, request_context_tokens: null, items: [{ id: 0, processing: true, prompt_tokens: 100, cached_prompt_tokens: 0, decoded_tokens: 10 }] });
    const element = mount();
    expect(element.textContent).toContain(t('en', 'activity.no_context'));
    expect(element.querySelector('[role="progressbar"]')).toBeNull();
  });
  it('reads an idle slot as 0 of the known context and says unknown only without a denominator', () => {
    // The server reports prompt_tokens null for a slot without a task, not for a failed measurement.
    const idle = [0, 1, 2, 3].map((id) => ({ id, processing: false, prompt_tokens: null, cached_prompt_tokens: null, decoded_tokens: null }));
    const noContext = t('en', 'activity.no_context');
    snapshot = withRuntime({ available: true, reason: null, measured_at: '2026-09-15T00:00:00Z', request_context_tokens: 40960, items: idle });
    let element = mount();
    const occupancy = `0 / ${new Intl.NumberFormat('en-US').format(40960)} tokens`;
    expect(occurrences(element, occupancy), `idle rows reading "${occupancy}"`).toBe(4);
    expect(occurrences(element, noContext), 'idle rows claiming an unknown context').toBe(0);
    expect([...element.querySelectorAll('[role="progressbar"]')].map((bar) => bar.getAttribute('aria-valuenow'))).toEqual(['0', '0', '0', '0']);
    const rows = [...element.querySelectorAll('[data-testid="activity-slot-table"] tbody tr')];
    expect(rows).toHaveLength(4);
    for (const row of rows) expect(row.textContent).not.toContain(t('en', 'format.unknown'));
    cleanup();
    snapshot = withRuntime({ available: true, reason: null, measured_at: '2026-09-15T00:00:00Z', request_context_tokens: null, items: idle });
    element = mount();
    expect(occurrences(element, noContext), 'rows without a denominator').toBe(4);
    expect(element.querySelector('[role="progressbar"]')).toBeNull();
  });
  it('discloses a partial slot snapshot even when observed slots are available', () => {
    snapshot = withRuntime({ available: true, reason: 'showing the first 256 observational slots' });
    const element = mount();
    expect(element.textContent).toContain('showing the first 256 observational slots');
  });
});
