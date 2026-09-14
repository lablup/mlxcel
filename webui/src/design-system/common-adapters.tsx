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

// New feature tables use this typed seam, not the gallery's legacy markup table.
export function DataTable<T>(props: Omit<DataTableProps<T>, 'ariaLabel'> & { ariaLabel: string }): React.JSX.Element {
  return <CommonDataTable {...props} className={`ds-common-table ${props.className ?? ''}`.trim()} />;
}
export type { DataTableColumn, DataTablePersistedState, SortDirection } from '@lablup/ui-common/components/DataTable';
