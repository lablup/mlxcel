// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Settings > Server prints the server's read-only startup values as user copy, so no value may be
// Rust Debug output or a raw JSON object (#2329), and no label may be a raw field name (#2334). The
// fixtures are the server's own output for a configured server and its full read-only field list;
// src/server/runtime_settings_values_tests.rs asserts both stay current.
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import bootstrap from '../../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalog from '../../../../tests/fixtures/webui/examples/catalog.page.json';
import startupValues from '../../../../tests/fixtures/settings/startup-values.json';
import readOnlyFields from '../../../../tests/fixtures/settings/read-only-fields.json';
import { initialSnapshot } from '../../state/reducer';
import type { BootstrapResponse, CatalogEntry, WebUiSnapshot } from '../../api/types';
import type { SettingKind, SettingsResponse, SettingSpec } from '../../api/settings';
import { t, type Locale } from '../../i18n/catalog';
import { ServerSettings } from './server-settings';

const mocks = vi.hoisted(() => ({ snapshot: null as WebUiSnapshot | null, actions: { selectModel: vi.fn(), loadModel: vi.fn(), unloadModel: vi.fn(), getModelProps: vi.fn(), getSettings: vi.fn() } }));
vi.mock('../../state', () => ({ useWebUi: () => mocks.snapshot, useWebUiActions: () => mocks.actions }));

/** `Some(`, `None`, braces of a struct or JSON object, or a bare CamelCase enum variant such as `LongestFirst`. */
const DEBUG_SHAPE = /Some\(|\bNone\b|[{}]|^[A-Z][a-z]+(?:[A-Z][a-z0-9]*)*$/;

/** A snake_case identifier such as `tp_size`: what a raw field name looks like. */
const RAW_NAME = /[a-z0-9]_[a-z0-9]/i;

const specs: SettingSpec[] = startupValues.values.map((entry) => ({ name: entry.name, type: entry.type as SettingKind, default: entry.default, mutable: false, allowed: null, help: 'Startup server configuration.', reason: 'restart required' }));

async function renderServerTab(locale: Locale, response: SettingsResponse): Promise<{ host: HTMLDivElement; unmount: () => void }> {
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() });
  const entry = structuredClone(catalog.items[0]) as unknown as CatalogEntry;
  const ready = { ...entry, lifecycle: { ...entry.lifecycle, state: 'ready' as const, worker_exit_observed: false } };
  const settingsBootstrap = { ...bootstrap, features: [...bootstrap.features, 'settings'], server: { ...bootstrap.server, mode: 'router_pool' as const } };
  mocks.snapshot = { ...initialSnapshot(), auth: { status: 'authenticated', tokenPresent: true }, connection: 'ready', bootstrap: settingsBootstrap as unknown as BootstrapResponse, catalog: [ready], selectedModelId: ready.identity.id };
  mocks.actions.getSettings.mockResolvedValue(response);
  mocks.actions.getModelProps.mockResolvedValue({ nCtx: 2048, totalSlots: 1, kvCacheMode: 'fp16', geometry: null });
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  await act(async () => root.render(<ServerSettings locale={locale} />));
  return { host, unmount: () => { act(() => root.unmount()); host.remove(); } };
}

/** Every value cell: the labelled list and the raw-name disclosure for values the catalog has no label for. */
function startupCells(host: HTMLElement): HTMLElement[] {
  const list = host.querySelector('[data-testid="settings-startup-values"]');
  if (list === null) throw new Error('Missing startup values');
  const other = host.querySelector('[data-testid="settings-startup-other"]');
  return [list, ...other === null ? [] : [other]].flatMap((root) => Array.from(root.querySelectorAll<HTMLElement>('dd')));
}

/** Labels a reader sees without opening a disclosure: every name cell outside a details element. */
function visibleLabels(host: HTMLElement): string[] {
  const section = host.querySelector('[data-testid="settings-startup-values"]')?.closest('section');
  if (!section) throw new Error('Missing startup section');
  return Array.from(section.querySelectorAll('dt')).filter((cell) => cell.closest('details') === null).map((cell) => cell.textContent ?? '');
}

/** Each read-only field the server sends, with a value carrying every sub-field it can have. */
const allFields: SettingSpec[] = readOnlyFields.fields.map(({ name, sub_fields }) => {
  const value = sub_fields.length === 0 ? 1 : Object.fromEntries(sub_fields.map((sub) => [sub, 1]));
  return { name, type: sub_fields.length === 0 ? 'int' : 'object', default: value, mutable: false, allowed: null, help: 'Startup server configuration.', reason: 'restart required' };
});

describe('Settings > Server startup values', () => {
  let cleanup: (() => void) | null = null;
  afterEach(() => { cleanup?.(); cleanup = null; });

  it('keeps a list of structured items as sub-rows and joins a list of plain values', async () => {
    const nested: SettingSpec = { name: 'nested_probe', type: 'object', default: { peers: [{ host: 'a.test', port: 9100 }], tags: ['x', 'y'] }, mutable: false, allowed: null, help: 'Startup server configuration.', reason: 'restart required' };
    const rendered = await renderServerTab('en', { schema: [nested], current: {}, fingerprint: 'a'.repeat(64) });
    cleanup = rendered.unmount;
    const texts = startupCells(rendered.host).map((cell) => cell.textContent ?? '');
    expect(texts.filter((text) => DEBUG_SHAPE.test(text))).toEqual([]);
    expect(texts).toContain('x, y');
    expect(texts).toContain('a.test');
    // A field this UI has no label for keeps its raw name, but only inside the disclosure.
    expect(rendered.host.querySelector('[data-testid="settings-startup-other"] > summary')?.textContent).toBe(t('en', 'settings.server.startup.other'));
    expect(visibleLabels(rendered.host)).toEqual([]);
  });

  for (const locale of ['en', 'ko'] as const) {
    it(`prints no Debug or JSON shape in any value cell (${locale})`, async () => {
      const rendered = await renderServerTab(locale, { schema: specs, current: {}, fingerprint: 'a'.repeat(64) });
      cleanup = rendered.unmount;
      const cells = startupCells(rendered.host);
      const offending = cells.map((cell) => cell.textContent ?? '').filter((text) => DEBUG_SHAPE.test(text));
      expect(offending).toEqual([]);
      expect(cells.length).toBeGreaterThan(specs.length);
      // Structured values become sub-rows labelled from the catalog, never by their raw field names.
      const tensorParallel = rendered.host.querySelector('.settings-value-group[data-setting="tensor_parallel"]');
      expect(tensorParallel?.querySelector(':scope > dt')?.textContent).toBe(t(locale, 'settings.startup.tensor_parallel'));
      const tpSize = tensorParallel?.querySelector('.settings-subvalues > [data-setting="tp_size"]');
      expect(tpSize?.querySelector('dt')?.textContent).toBe(t(locale, 'settings.startup.tensor_parallel.tp_size'));
      expect(tpSize?.querySelector('dd')?.textContent).toBe('2');
      expect(tensorParallel?.textContent).not.toContain('tp_size');
      expect(visibleLabels(rendered.host).filter((label) => RAW_NAME.test(label))).toEqual([]);
      expect(rendered.host.querySelector('[data-testid="settings-startup-other"]')).toBeNull();
    });

    it(`labels every read-only field and sub-field the server sends from the catalog (${locale})`, async () => {
      const rendered = await renderServerTab(locale, { schema: allFields, current: {}, fingerprint: 'a'.repeat(64) });
      cleanup = rendered.unmount;
      const labels = visibleLabels(rendered.host);
      expect(labels.length).toBe(readOnlyFields.fields.reduce((total, field) => total + 1 + field.sub_fields.length, 0));
      expect(labels.filter((label) => RAW_NAME.test(label))).toEqual([]);
      expect(rendered.host.querySelector('[data-testid="settings-startup-other"]')).toBeNull();
    });

    it(`labels a top-level list of structured items as sub-rows, not JSON (${locale})`, async () => {
      const listed: SettingSpec = { name: 'tensor_parallel', type: 'array', default: [{ tp_size: 2 }, { tp_size: 4 }], mutable: false, allowed: null, help: 'Startup server configuration.', reason: 'restart required' };
      const rendered = await renderServerTab(locale, { schema: [listed], current: {}, fingerprint: 'a'.repeat(64) });
      cleanup = rendered.unmount;
      const sub = t(locale, 'settings.startup.tensor_parallel.tp_size');
      expect(visibleLabels(rendered.host)).toEqual([t(locale, 'settings.startup.tensor_parallel'), sub, sub]);
      expect(startupCells(rendered.host).map((cell) => cell.textContent ?? '').filter((text) => DEBUG_SHAPE.test(text))).toEqual([]);
      expect(rendered.host.querySelector('[data-setting="tensor_parallel"]')?.classList.contains('settings-value-group')).toBe(true);
    });

    it(`keeps an unlabelled sub-field's raw name inside a disclosure (${locale})`, async () => {
      const future: SettingSpec = { name: 'tensor_parallel', type: 'object', default: { tp_size: 2, future_knob: 3 }, mutable: false, allowed: null, help: 'Startup server configuration.', reason: 'restart required' };
      const rendered = await renderServerTab(locale, { schema: [future], current: {}, fingerprint: 'a'.repeat(64) });
      cleanup = rendered.unmount;
      expect(visibleLabels(rendered.host)).toEqual([t(locale, 'settings.startup.tensor_parallel'), t(locale, 'settings.startup.tensor_parallel.tp_size')]);
      const disclosure = rendered.host.querySelector('[data-setting="tensor_parallel"] details');
      expect(disclosure?.querySelector('summary')?.textContent).toBe(t(locale, 'settings.server.startup.other'));
      expect(disclosure?.querySelector('dt')?.textContent).toBe('future_knob');
      expect(disclosure?.querySelector('dd')?.textContent).toBe('3');
    });

    it(`prints the localized unset text for an absent value (${locale})`, async () => {
      const current = Object.fromEntries(specs.filter((spec) => spec.type.endsWith('_or_null')).map((spec) => [spec.name, null]));
      const rendered = await renderServerTab(locale, { schema: specs, current, fingerprint: 'a'.repeat(64) });
      cleanup = rendered.unmount;
      const unset = t(locale, 'settings.value.unset');
      const topLevel = Array.from(rendered.host.querySelectorAll('[data-testid="settings-startup-values"] > div > dd')).map((cell) => cell.textContent ?? '');
      expect(topLevel.filter((text) => text === unset)).toHaveLength(Object.keys(current).length);
      expect(startupCells(rendered.host).map((cell) => cell.textContent ?? '').filter((text) => DEBUG_SHAPE.test(text))).toEqual([]);
    });
  }
});
