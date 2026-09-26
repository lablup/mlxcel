// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// The model library's search field and source, task and status filters.
import React from 'react';
import type { CatalogEntry } from '../../api/types';
import { Field, Select } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { lifecycleLabel } from '../../provider-surfaces';
import { sourceLabel, taskLabel } from './labels';
import type { InventoryFilter } from './policy';

const LIFECYCLE_FILTERS = ['unloaded', 'loading', 'ready', 'draining', 'unloading', 'failed'] as const;

export type LibraryToolbarProps = {
  locale: Locale;
  catalog: readonly CatalogEntry[];
  filter: InventoryFilter;
  onChange: (patch: Partial<InventoryFilter>) => void;
};

export function LibraryToolbar({ locale, catalog, filter, onChange }: LibraryToolbarProps): React.JSX.Element {
  return (
    <div className="models-toolbar">
      <Field label={t(locale, 'models.library.search')} value={filter.query} onChange={(query) => onChange({ query })} testId="models-search" />
      <Select
        locale={locale}
        label={t(locale, 'models.library.source')}
        value={filter.source}
        onChange={(source) => onChange({ source })}
        options={[
          { value: '', label: t(locale, 'models.library.all') },
          ...[...new Set(catalog.map((entry) => entry.identity.source))].sort().map((value) => ({ value, label: sourceLabel(locale, value) })),
        ]}
      />
      <Select
        locale={locale}
        label={t(locale, 'models.library.task')}
        value={filter.task}
        onChange={(task) => onChange({ task })}
        options={[
          { value: '', label: t(locale, 'models.library.all') },
          ...[...new Set(catalog.flatMap((entry) => entry.capabilities.map((cap) => cap.task)))]
            .sort()
            .map((value) => ({ value, label: taskLabel(locale, value) })),
        ]}
      />
      <Select
        locale={locale}
        label={t(locale, 'models.library.status')}
        value={filter.status}
        onChange={(status) => onChange({ status })}
        options={[
          { value: '', label: t(locale, 'models.library.all') },
          ...LIFECYCLE_FILTERS.map((value) => ({ value, label: lifecycleLabel(locale, value) })),
        ]}
      />
    </div>
  );
}
