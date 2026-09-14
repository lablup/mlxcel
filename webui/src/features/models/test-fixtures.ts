// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import bootstrapFixture from '../../../../tests/fixtures/webui/examples/bootstrap.model-free.json';
import catalogFixture from '../../../../tests/fixtures/webui/examples/catalog.page.json';
import { validateBootstrap, validateCatalogList } from '../../api/validation';
import type { CatalogEntry, WebUiSnapshot } from '../../api/types';
import { initialSnapshot } from '../../state/reducer';

export function model(): CatalogEntry {
  const entry = validateCatalogList(catalogFixture).items[0];
  return {
    ...entry,
    identity: { ...entry.identity, source: 'cache', revision: 4 },
    complete: true,
    supported: true,
    removable: true,
    removal: { eligible: true, reason: null, instructions: null },
    metadata: {
      ...entry.metadata,
      support: {
        ...entry.metadata.support,
        runnable_on_backend: true,
        architecturally_supported: true,
        complete: true,
      },
    },
    lifecycle: { ...entry.lifecycle, state: 'unloaded', active_requests: 0, draining_requests: 0, busy: false },
  };
}
export function snapshot(entry = model()): WebUiSnapshot {
  return {
    ...initialSnapshot(),
    auth: { status: 'authenticated', tokenPresent: true },
    connection: 'ready',
    serverInstanceId: bootstrapFixture.server.server_instance_id,
    bootstrap: validateBootstrap(bootstrapFixture),
    catalog: [entry],
    catalogSequence: 1,
    selectedModelId: entry.identity.id,
  };
}
