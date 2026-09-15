// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { ModuleKind, ScriptTarget, transpileModule } from 'typescript';
import type { BootstrapResponse, CatalogEntry, CatalogListResponse, RuntimeSnapshot } from '../src/api/types';

function read(relative: string): string {
  return readFileSync(fileURLToPath(new URL(relative, import.meta.url)), 'utf8');
}
export const bootstrap = JSON.parse(
  read('../../tests/fixtures/webui/examples/bootstrap.model-free.json'),
) as BootstrapResponse;
const catalog = JSON.parse(read('../../tests/fixtures/webui/examples/catalog.page.json')) as CatalogListResponse;
export function model(): CatalogEntry {
  const entry = catalog.items[0];
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

const runtimeFixture = JSON.parse(read('../../tests/fixtures/webui/examples/runtime.snapshot.json')) as RuntimeSnapshot;
export function runtime(entry: CatalogEntry, sequence: number): RuntimeSnapshot {
  return {
    ...runtimeFixture,
    server_instance_id: bootstrap.server.server_instance_id,
    model_id: entry.identity.id,
    revision: entry.identity.revision,
    snapshot_sequence: sequence,
    measurements: {},
    settings: { scope: entry.lifecycle.state === 'ready' ? 'loaded_model_live' : 'next_load_profile', effective: {}, overridden_by_cli: [], partial_errors: [] },
  };
}

// Playwright's Node loader cannot load Vite's ?raw import. Execute the actual
// project validator with only that static import supplied from the canonical file.
// This does not copy its implementation or evaluate any server/user response as code.
let validator: Promise<typeof import('../src/api/jsonSchema')> | null = null;
export function loadValidator(): Promise<typeof import('../src/api/jsonSchema')> {
  if (validator === null) {
    const source = read('../src/api/jsonSchema.ts');
    const rawImport = "import contractText from '../../../docs/webui/api.yaml?raw';";
    if (!source.includes(rawImport))
      throw new Error('Canonical validator import changed; update the test loader explicitly.');
    const supplied = source.replace(
      rawImport,
      `const contractText = ${JSON.stringify(read('../../docs/webui/api.yaml'))};`,
    );
    const compiled = transpileModule(supplied, {
      compilerOptions: { module: ModuleKind.ES2022, target: ScriptTarget.ES2022 },
    }).outputText;
    validator = import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`) as Promise<
      typeof import('../src/api/jsonSchema')
    >;
  }
  return validator;
}
