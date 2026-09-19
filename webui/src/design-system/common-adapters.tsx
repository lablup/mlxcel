// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { forwardRef, useId } from 'react';
import { Button as CommonButton, type ButtonProps as CommonButtonProps } from '@lablup/ui-common/components/Button';
import { StatusTag, type StatusKind } from '@lablup/ui-common/components/StatusTag';
import { ProgressBar as CommonProgressBar } from '@lablup/ui-common/components/ProgressBar';
import { EmptyState as CommonEmptyState } from '@lablup/ui-common/components/EmptyState';
import { Tabs as CommonTabs } from '@lablup/ui-common/components/Tabs';
import { DataTable as CommonDataTable, type DataTableProps } from '@lablup/ui-common/components/DataTable';
import { Icon, type IconName } from './icons';

export type ButtonTone = 'primary' | 'secondary' | 'danger' | 'ghost';
export type LifecycleState = 'unloaded' | 'loading' | 'ready' | 'draining' | 'unloading' | 'failed';
// Deliberately not ButtonHTMLAttributes: alpha.19 does not forward arbitrary native props.
export type ButtonProps = Omit<CommonButtonProps, 'variant' | 'loading' | 'ariaLabel' | 'style' | 'size' | 'shape' | 'fullWidth' | 'inline'> & {
  tone?: ButtonTone; busy?: boolean; 'aria-label'?: string;
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button({ tone = 'secondary', busy = false, className = '', children, 'aria-label': label, ...supported }, ref) {
  return <CommonButton {...supported} ref={ref} inline variant={tone} loading={busy} aria-busy={busy || supported['aria-busy']} ariaLabel={label} className={`ds-button ds-button-${tone} ${className}`.trim()}><span>{children}</span></CommonButton>;
});

export const IconButton = forwardRef<HTMLButtonElement, Omit<ButtonProps, 'children' | 'icon' | 'iconOnly' | 'tone' | 'aria-label'> & { label: string; icon: IconName }>(function IconButton({ label, icon, busy = false, className = '', ...supported }, ref) {
  return <CommonButton {...supported} ref={ref} inline iconOnly icon={<Icon name={icon} />} ariaLabel={label} title={supported.title ?? label} variant="secondary" loading={busy} aria-busy={busy || supported['aria-busy']} className={`ds-icon-button ${className}`.trim()} />;
});

const lifecycleKinds: Record<LifecycleState, StatusKind> = {
  unloaded: 'terminated', loading: 'preparing', ready: 'running', draining: 'stopping', unloading: 'stopping', failed: 'error',
};
export function StatusBadge(props: { state: LifecycleState; children: string; label?: string }): React.JSX.Element {
  return <span className="ds-status-wrap" data-state={props.state} aria-label={props.label}><StatusTag state={lifecycleKinds[props.state]} label={props.children} pulse={false} className="ds-status" /></span>;
}

export function ProgressBar(props: { label: string; value?: number; detail?: string }): React.JSX.Element {
  const value = props.value === undefined || !Number.isFinite(props.value) ? null : Math.max(0, Math.min(100, props.value));
  return <div className="ds-progress"><CommonProgressBar value={value} ariaLabel={props.label} animated={false} />{props.detail ? <small>{props.detail}</small> : null}</div>;
}

export function EmptyState(props: { title: string; body: string; action?: React.ReactNode; testId?: string }): React.JSX.Element {
  return <section className="ds-empty" data-testid={props.testId} ref={(element) => {
    // alpha.19 fixes its visual heading at h3; preserve this product's h2 hierarchy.
    const heading = element?.querySelector('.empty-state__title');
    heading?.setAttribute('role', 'heading');
    heading?.setAttribute('aria-level', '2');
  }}><CommonEmptyState title={props.title} description={props.body} illustration={<Icon name="models" />}>{props.action}</CommonEmptyState></section>;
}

export function Tabs(props: { tabs: { id: string; label: string; panel: React.ReactNode }[]; active: string; onChange: (id: string) => void; label?: string }): React.JSX.Element {
  const namespace = useId();
  if (props.tabs.length === 0) return <section className="ds-tabs" />;
  const tabs = props.tabs.map((tab) => ({ id: `${namespace}-${tab.id}`, label: tab.label, content: tab.panel }));
  const active = props.tabs.find((tab) => tab.id === props.active) ?? props.tabs[0];
  return <section onKeyDownCapture={(event) => { if (event.nativeEvent.isComposing) event.stopPropagation(); }}><CommonTabs className="ds-tabs" tabs={tabs} activeTab={`${namespace}-${active.id}`} onTabChange={(id) => {
    const tab = props.tabs.find((item) => `${namespace}-${item.id}` === id);
    if (tab) props.onChange(tab.id);
  }} ariaLabel={props.label ?? 'Sections'} variant="segmented" fillContainer overflowMode="menu" showOverflowControls={false} showGroupLabels={false} /></section>;
}

/** Marks a row's primary control, the one whole-row activation delegates to. A class, because the common Button forwards `className` but no arbitrary `data-*`. */
export const ROW_PRIMARY_CLASS = 'ds-row-primary';
// Mirrors the common table's own nested-interactive guard: a click on any of these inside the row is theirs, not the row's.
const ROW_INTERACTIVE = 'a[href], button, input, select, textarea, summary, label, [contenteditable="true"], [role="button"], [role="link"], [tabindex]:not([tabindex="-1"])';

function activateRowPrimary(event: React.MouseEvent<HTMLDivElement>): void {
  if (event.defaultPrevented || event.button !== 0 || !(event.target instanceof Element)) return;
  const row = event.target.closest('tbody > tr');
  if (!row || !event.currentTarget.contains(row) || row.classList.contains('data-table__row--state')) return;
  const primary = row.querySelector<HTMLElement>(`.${ROW_PRIMARY_CLASS}`);
  if (!primary || primary.matches(':disabled, [aria-disabled="true"]')) return;
  const nested = event.target.closest(ROW_INTERACTIVE);
  if (nested && row.contains(nested)) return;
  // A drag that selected text in this row is a selection, not an activation.
  const selection = window.getSelection();
  if (selection && !selection.isCollapsed && selection.toString().trim() !== '' && (row.contains(selection.anchorNode) || row.contains(selection.focusNode))) return;
  primary.focus({ preventScroll: true });
  primary.click();
}

// New feature tables use this typed seam, not the gallery's legacy markup table.
// onRowClick/isRowClickable are withheld: alpha.19 turns each body <tr> into role="button" with its own tab stop, which fails axe
// aria-required-children and nested-interactive inside role="table" and gives the row an "Inspect ..." name that collides with its button.
export type DataTableAdapterProps<T> = Omit<DataTableProps<T>, 'ariaLabel' | 'onRowClick' | 'isRowClickable'> & {
  ariaLabel: string;
  /**
   * Whole-row pointer activation. A click anywhere in a body row that is not on another interactive element, and that did
   * not end a text selection, focuses and clicks the row's `ROW_PRIMARY_CLASS` control, so pointer and keyboard run the same
   * handler. Rows keep their native `row` role and gain no tab stop; the primary control stays the keyboard and
   * screen-reader target, and the row draws a focus outline while that control has `:focus-visible`. Rows without an
   * enabled primary control, and the loading and empty rows, stay inert.
   */
  activateRowPrimary?: boolean;
};
export function DataTable<T>({ activateRowPrimary: delegate = false, ...props }: DataTableAdapterProps<T>): React.JSX.Element {
  const table = <CommonDataTable {...props} className={`ds-common-table ${props.className ?? ''}`.trim()} />;
  return delegate ? <div className="ds-row-activation" onClick={activateRowPrimary}>{table}</div> : table;
}
export type { DataTableColumn, DataTablePersistedState, SortDirection } from '@lablup/ui-common/components/DataTable';
