// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import { Button as CommonButton, type ButtonVariant } from '@lablup/ui-common/Button';
import { StatusDot, type StatusDotVariant } from '@lablup/ui-common/StatusDot';
import { ProgressBar as CommonProgressBar } from '@lablup/ui-common/ProgressBar';
import { EmptyState as CommonEmptyState } from '@lablup/ui-common/EmptyState';
import { Icon, type IconName } from './icons';

export type ButtonTone = 'primary' | 'secondary' | 'danger' | 'ghost';
export type LifecycleState = 'unloaded' | 'loading' | 'ready' | 'draining' | 'unloading' | 'failed';

// The product button surface: the props the feature call sites use, not Astryx's whole prop
// bag. Arbitrary native props (onFocus, name, form) are deliberately not advertised; extend
// this type explicitly when a real consumer needs one.
export type ButtonProps = {
  children?: React.ReactNode;
  tone?: ButtonTone;
  busy?: boolean;
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onDoubleClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onMouseDown?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onMouseEnter?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  onKeyDown?: (event: React.KeyboardEvent<HTMLButtonElement>) => void;
  title?: string;
  id?: string;
  role?: React.AriaRole;
  tabIndex?: number;
  className?: string;
  ref?: React.Ref<HTMLButtonElement>;
  'aria-label'?: string;
  'aria-pressed'?: boolean;
  'aria-expanded'?: boolean;
  'aria-haspopup'?: boolean | 'menu' | 'listbox' | 'tree' | 'grid' | 'dialog';
  'aria-controls'?: string;
  'aria-describedby'?: string;
  'aria-busy'?: boolean;
  'data-testid'?: string;
  /** Other data attributes (`data-autofocus` for a dialog's initial focus). */
  [data: `data-${string}`]: string | undefined;
};

const buttonVariants: Record<ButtonTone, ButtonVariant> = { primary: 'primary', secondary: 'secondary', danger: 'destructive', ghost: 'ghost' };

/**
 * The visible text of a button's children, which Astryx needs as its required `label` (it
 * becomes the accessible name when the children are not that plain string). Separate pieces
 * (a name beside a status) are joined with a space, as a reader would hear them.
 */
function textOf(node: React.ReactNode): string {
  if (node === null || node === undefined || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(textOf).filter((part) => part.trim() !== '').join(' ').replace(/\s+/g, ' ').trim();
  if (React.isValidElement<{ children?: React.ReactNode }>(node)) return textOf(node.props.children);
  return '';
}

// Astryx's BaseProps leave out `title`; the product keeps the native tooltip and test ids.
type NativeExtras = { title?: string; role?: React.AriaRole };

export function Button({ tone = 'secondary', busy = false, disabled = false, className = '', children, 'aria-label': ariaLabel, title, role, ref, ...supported }: ButtonProps): React.JSX.Element {
  const label = ariaLabel ?? textOf(children);
  const extras: NativeExtras = { title, role };
  return <CommonButton {...supported} {...extras} ref={ref} label={label} variant={buttonVariants[tone]} isLoading={busy} isDisabled={disabled} className={`ds-button ds-button-${tone} ${className}`.trim()}>{children}</CommonButton>;
}

export type IconButtonProps = Omit<ButtonProps, 'children' | 'tone' | 'aria-label'> & { label: string; icon: IconName; tone?: 'secondary' | 'danger' };

// `danger` keeps the outlined control and draws the glyph in the error color, so a destructive
// action in a dense row reads as one without a filled red button on every line.
export function IconButton({ label, icon, tone = 'secondary', busy = false, disabled = false, className = '', title, role, ref, ...supported }: IconButtonProps): React.JSX.Element {
  const extras: NativeExtras = { title: title ?? label, role };
  return <CommonButton {...supported} {...extras} ref={ref} isIconOnly icon={<Icon name={icon} />} label={label} variant="secondary" isLoading={busy} isDisabled={disabled} className={`ds-icon-button ${tone === 'danger' ? 'ds-icon-button-danger ' : ''}${className}`.trim()} />;
}

const lifecycleVariants: Record<LifecycleState, StatusDotVariant> = {
  unloaded: 'neutral', loading: 'accent', ready: 'success', draining: 'warning', unloading: 'warning', failed: 'error',
};
// StatusDot renders only the dot; the domain label stays visible beside it and is the name.
// `inline` drops the pill (border, fill, padding) for a dot and label set in running text.
export function StatusBadge(props: { state: LifecycleState; children: string; label?: string; appearance?: 'pill' | 'inline' }): React.JSX.Element {
  return <span className={props.appearance === 'inline' ? 'ds-status-wrap ds-status-wrap--inline' : 'ds-status-wrap'} data-state={props.state} aria-label={props.label}>
    <span className="ds-status"><StatusDot variant={lifecycleVariants[props.state]} label={props.children} aria-hidden="true" className="ds-status__dot" /><span className="ds-status__label">{props.children}</span></span>
  </span>;
}

export function ProgressBar(props: { label: string; value?: number; detail?: string }): React.JSX.Element {
  const value = props.value === undefined || !Number.isFinite(props.value) ? undefined : Math.max(0, Math.min(100, props.value));
  return <div className="ds-progress"><CommonProgressBar value={value} isIndeterminate={value === undefined} label={props.label} isLabelHidden className="ds-progress__bar" />{props.detail ? <small>{props.detail}</small> : null}</div>;
}

// The product outline puts the empty state's title at level two.
export function EmptyState(props: { title: string; body: string; action?: React.ReactNode; testId?: string }): React.JSX.Element {
  return <section className="ds-empty" data-testid={props.testId}><CommonEmptyState title={props.title} description={props.body} icon={<Icon name="models" />} headingLevel={2} actions={props.action} /></section>;
}
