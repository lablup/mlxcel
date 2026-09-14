// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest';
import { bootstrap, loadValidator, model } from '../../../tests/models-fixtures';

describe('Playwright fixture contract loader (no browser)', () => {
  it('executes the canonical validator and validates the complete CJK catalog', async () => {
    const { validateAgainstSchema } = await loadValidator();
    validateAgainstSchema('BootstrapResponse', bootstrap);
    const items = Array.from({ length: 120 }, (_, index) => ({ ...model(), identity: { ...model().identity, id: `mdl_${String(index).padStart(43, '0')}`, display_name: `模型 ${index}` } }));
    const page = { schema_version: 'webui.ui-api.v1', items, pagination: { limit: 200, next_cursor: null, total_known: items.length }, server_instance_id: bootstrap.server.server_instance_id, snapshot_sequence: 1 };
    expect(() => validateAgainstSchema('CatalogListResponse', page)).not.toThrow();
    expect(() => validateAgainstSchema('CatalogListResponse', { ...page, items: [{ ...items[0], identity: { ...items[0].identity, id: 'id_invalid' } }] })).toThrow();
    expect(() => validateAgainstSchema('CatalogListResponse', { ...page, extra: true })).toThrow();
  });
});
