// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest';
import { validateAgainstSchema } from './jsonSchema';
import { validatePerformanceFixtureSet } from '../../tests/performance-fixtures';

describe('performance Playwright fixtures', () => {
  it('construct schema-valid DTOs for the browser performance harness', () => {
    const checked = validatePerformanceFixtureSet(validateAgainstSchema);
    expect(checked).toEqual([
      'BootstrapResponse:bootstrap',
      'OperationsListResponse:empty-operations',
      'CatalogListResponse:catalog-0',
      'CatalogListResponse:catalog-200',
      'CatalogListResponse:catalog-400',
      'CatalogListResponse:catalog-600',
      'CatalogListResponse:catalog-800',
      'CatalogListResponse:ready-chat-catalog',
      'RuntimeSnapshot:ready-runtime',
    ]);
  });
});
