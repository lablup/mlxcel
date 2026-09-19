// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest';
import { bootstrap, loadValidator, model, runtime } from '../../../tests/models-fixtures';

describe('Playwright fixture contract loader (no browser)', () => {
  it('executes the canonical validator and validates the complete CJK catalog', async () => {
    const { validateAgainstSchema } = await loadValidator();
    validateAgainstSchema('BootstrapResponse', bootstrap);
    const items = Array.from({ length: 120 }, (_, index) => ({ ...model(), identity: { ...model().identity, id: `mdl_${String(index).padStart(43, '0')}`, display_name: `模型-${index}` } }));
    const page = { schema_version: 'webui.ui-api.v1', items, pagination: { limit: 200, next_cursor: null, total_known: items.length }, server_instance_id: bootstrap.server.server_instance_id, snapshot_sequence: 1 };
    expect(() => validateAgainstSchema('CatalogListResponse', page)).not.toThrow();
    expect(() => validateAgainstSchema('CatalogListResponse', { ...page, items: [{ ...items[0], identity: { ...items[0].identity, id: 'id_invalid' } }] })).toThrow();
    expect(() => validateAgainstSchema('CatalogListResponse', { ...page, extra: true })).toThrow();
  });
  it('accepts a full-length cache owner/name as display_name and rejects one past inference_id bounds', async () => {
    const { validateAgainstSchema } = await loadValidator();
    const pageWith = (display_name: string) => ({
      schema_version: 'webui.ui-api.v1',
      items: [{ ...model(), identity: { ...model().identity, display_name } }],
      pagination: { limit: 50, next_cursor: null, total_known: 1 },
      server_instance_id: bootstrap.server.server_instance_id,
      snapshot_sequence: 1,
    });
    const cacheId = `${'o'.repeat(96)}/${'n'.repeat(96)}`;
    expect(cacheId).toHaveLength(193);
    expect(() => validateAgainstSchema('CatalogListResponse', pageWith(cacheId))).not.toThrow();
    expect(() => validateAgainstSchema('CatalogListResponse', pageWith('m'.repeat(257)))).toThrow();
  });
});


it('validates the exact Models browser runtime response across lifecycle revisions', async () => {
  const { validateAgainstSchema } = await loadValidator();
  for (const state of ['unloaded', 'loading', 'ready', 'draining', 'unloading', 'failed'] as const) {
    const entry = { ...model(), identity: { ...model().identity, revision: 18 }, lifecycle: { ...model().lifecycle, state } };
    const response = runtime(entry, 71);
    expect(() => validateAgainstSchema('RuntimeSnapshot', response)).not.toThrow();
    expect(response).toMatchObject({ model_id: entry.identity.id, revision: 18, snapshot_sequence: 71, server_instance_id: bootstrap.server.server_instance_id });
    expect(response.measurements).toEqual({});
    expect(response.slots.available).toBe(false);
    expect(response.settings.scope).toBe(state === 'ready' ? 'loaded_model_live' : 'next_load_profile');
    const withoutSlots = Object.fromEntries(Object.entries(response).filter(([key]) => key !== 'slots'));
    expect(() => validateAgainstSchema('RuntimeSnapshot', withoutSlots)).toThrow('$.slots: missing required property');
  }
});
