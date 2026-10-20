// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// The design system's one public surface for feature pages: every adapter over ui-common and
// the product's own controls. Feature pages import from here, never from the package.
//
// The first-paint modules (app.tsx, design-system/shell.tsx, provider-surfaces.tsx,
// command-palette.tsx) import the individual seam modules instead: the bundler keeps whatever a
// barrel re-exports in the chunk that imports the barrel, so importing it from the entry would
// pull every screen's controls (Table, Selector, TabList) into the first bundle.
export { Button, IconButton, StatusBadge, ProgressBar, EmptyState } from './common-adapters';
export type { ButtonProps, ButtonTone, LifecycleState } from './common-adapters';
export { Tabs } from './common-tabs';
export { DataTable, ROW_PRIMARY_CLASS, useOverflowRegion } from './common-table';
export type { DataTableColumn, SortDirection } from './common-table';
export { Select } from './common-select';
export { Drawer } from './common-drawer';
export { Tooltip } from './common-overlays';
export { ErrorBanner, LoadingStatus } from './common-feedback';
export { Card, PageHeader, PageLayout, SmoothHeight } from './common-layout';
export { Badge, StatCard } from './common-data';
export { Field, Dialog, ConfirmDialog, Inspector, StaticTable, DenseList, AuthGate, LoginView, SchemaMismatchView, Toggle } from './local-components';
export type { ConfirmDialogProps } from './local-components';
