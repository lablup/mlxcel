import React, { useEffect, useId, useRef } from 'react';
import { Icon } from './icons';

import { Button, IconButton } from './common-adapters';
import { ErrorBanner } from './common-feedback';
import { NativeModalContext } from './modal-context';
export { Button, IconButton, StatusBadge, ProgressBar, EmptyState, Tabs, DataTable } from './common-adapters';
export type { ButtonProps, ButtonTone, LifecycleState, DataTableColumn, DataTablePersistedState, SortDirection } from './common-adapters';
export { Select } from './common-select';
export { Drawer, Tooltip } from './common-overlays';
export { ErrorBanner, LoadingStatus } from './common-feedback';
export { Card, PageHeader, PageLayout, SmoothHeight } from './common-layout';
export { Badge, StatCard } from './common-data';

export function Field(props: { label: string; value: string; onChange?: (value: string) => void; placeholder?: string; disabled?: boolean; busy?: boolean; error?: string; hint?: string; testId?: string }): React.JSX.Element {
  const id = useId();
  const labelId = `${id}-label`;
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [props.hint ? hintId : null, props.error ? errorId : null].filter(Boolean).join(' ') || undefined;
  return (
    <label className="ds-field" htmlFor={id} data-disabled={props.disabled || props.busy || undefined}>
      <span id={labelId}>{props.label}</span>
      <input id={id} aria-labelledby={labelId} value={props.value} placeholder={props.placeholder} disabled={props.disabled || props.busy} aria-busy={props.busy || undefined} aria-invalid={props.error ? 'true' : undefined} aria-describedby={describedBy} data-testid={props.testId} onChange={(event) => props.onChange?.(event.currentTarget.value)} />
      {props.hint ? <small id={hintId} data-tone="hint">{props.hint}</small> : null}
      {props.error ? <small id={errorId} data-tone="error">{props.error}</small> : null}
    </label>
  );
}

type ModalProps = { open: boolean; title: string; children: React.ReactNode; onClose: () => void; labelledBy?: string; testId?: string; closeLabel?: string; className?: string; position?: 'center' | 'left' };

function ModalDialog(props: ModalProps): React.JSX.Element {
  const ref = useRef<HTMLDialogElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const generatedTitleId = useId();
  const titleId = props.labelledBy ?? generatedTitleId;
  const closingFromProp = useRef(false);
  const onCloseRef = useRef(props.onClose);
  onCloseRef.current = props.onClose;

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (props.open && !dialog.open) {
      previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      dialog.showModal();
      const focusTarget = dialog.querySelector<HTMLElement>('[data-autofocus], button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])');
      focusTarget?.focus();
    }
    if (!props.open && dialog.open) {
      closingFromProp.current = true;
      dialog.close();
    }
  }, [props.open]);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    const restoreFocus = (): void => {
      const target = previousFocus.current;
      restoreModalFocus(dialog, target);
      previousFocus.current = null;
    };
    const handleClose = (): void => {
      const closedByProp = closingFromProp.current;
      if (closedByProp) closingFromProp.current = false;
      else onCloseRef.current();
      window.setTimeout(restoreFocus, 0);
    };
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (event.key !== 'Tab') return;
      const focusable = Array.from(dialog.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])')).filter((item) => item.offsetParent !== null || item === document.activeElement);
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    dialog.addEventListener('close', handleClose);
    dialog.addEventListener('keydown', handleKeyDown);
    return () => {
      dialog.removeEventListener('close', handleClose);
      dialog.removeEventListener('keydown', handleKeyDown);
      // Feature confirmations may unmount without a native close event.
      const target = previousFocus.current;
      window.setTimeout(() => restoreModalFocus(dialog, target), 0);
    };
  }, []);

  return (
    <dialog className={`ds-dialog ${props.className ?? ''}`.trim()} data-position={props.position ?? 'center'} ref={ref} aria-labelledby={titleId} data-testid={props.testId}>
      <header>
        <h2 id={titleId}>{props.title}</h2>
        <IconButton label={props.closeLabel ?? 'Close'} icon="close" onClick={() => { const target = previousFocus.current; closingFromProp.current = true; onCloseRef.current(); ref.current?.close(); window.setTimeout(() => { const dialog = ref.current; if (dialog) restoreModalFocus(dialog, target); }, 0); }} data-testid="dialog-close" />
      </header>
      <div><NativeModalContext.Provider value={true}>{props.children}</NativeModalContext.Provider></div>
    </dialog>
  );
}

// Do not steal focus from a newly opened modal or an intentionally focused control.
function restoreModalFocus(dialog: HTMLDialogElement, target: HTMLElement | null): void {
  if (dialog.isConnected && dialog.open) return;
  const active = document.activeElement;
  if (active && active !== document.body && active !== document.documentElement && !dialog.contains(active)) return;
  if (document.querySelector('dialog[open]')) return;
  const usable = (element: HTMLElement | null): element is HTMLElement => !!element?.isConnected && element !== document.body && !element.matches(':disabled, [aria-disabled="true"]') && !element.closest('[inert], [hidden]');
  if (usable(target)) { target.focus(); return; }
  const fallback = document.querySelector<HTMLElement>('[data-dialog-focus-fallback]') ?? document.querySelector<HTMLElement>('main button:not(:disabled)');
  if (usable(fallback)) fallback.focus();
}

export function Dialog(props: Omit<ModalProps, 'position' | 'className'>): React.JSX.Element {
  return <ModalDialog {...props} />;
}

export function Inspector(props: { title: string; children: React.ReactNode }): React.JSX.Element {
  return (
    <aside className="ds-inspector" aria-label={props.title}>
      <h2>{props.title}</h2>
      {props.children}
    </aside>
  );
}

export function StaticTable(props: { caption: string; children: React.ReactNode }): React.JSX.Element {
  return <table className="ds-table"><caption>{props.caption}</caption>{props.children}</table>;
}

export function DenseList(props: { label: string; children: React.ReactNode }): React.JSX.Element {
  return <ul className="ds-list" aria-label={props.label}>{props.children}</ul>;
}

export function AuthGate(props: { title: string; body: string; actionLabel: string }): React.JSX.Element {
  return <ErrorBanner tone="warning" title={props.title} body={props.body} action={<Button><Icon name="key" />{props.actionLabel}</Button>} />;
}

export function LoginView(props: { title: string; body: string; tokenLabel: string; tokenHelp: string; submitLabel: string; logoutLabel?: string; error?: string; busy?: boolean; onSubmit: (token: string) => void; onLogout?: () => void; testId?: string }): React.JSX.Element {
  const [token, setToken] = React.useState('');
  const helpId = React.useId();
  React.useEffect(() => () => setToken(''), []);
  const handleSubmit = (event: React.FormEvent): void => {
    event.preventDefault();
    if (props.busy || token.trim().length === 0) return;
    const submitted = token;
    setToken('');
    props.onSubmit(submitted);
  };
  const handleLogout = (): void => {
    setToken('');
    props.onLogout?.();
  };
  return (
    <form className="ds-login" data-testid={props.testId} autoComplete="off" onSubmit={handleSubmit}>
      <h2>{props.title}</h2>
      <p>{props.body}</p>
      <label className="ds-field">
        <span>{props.tokenLabel}</span>
        <input type="password" autoComplete="off" spellCheck={false} autoCapitalize="none" autoCorrect="off" value={token} disabled={props.busy} aria-invalid={props.error ? 'true' : undefined} aria-describedby={helpId} onChange={(event) => setToken(event.currentTarget.value)} />
        <small id={helpId} data-tone={props.error ? 'error' : 'hint'}>{props.error ?? props.tokenHelp}</small>
      </label>
      <div className="dialog-actions"><Button tone="primary" type="submit" busy={props.busy} disabled={token.trim().length === 0}>{props.submitLabel}</Button>{props.onLogout && props.logoutLabel ? <Button type="button" onClick={handleLogout}>{props.logoutLabel}</Button> : null}</div>
    </form>
  );
}

export function SchemaMismatchView(props: { title: string; body: string; actionLabel: string; onRecover: () => void }): React.JSX.Element {
  return <ErrorBanner tone="error" title={props.title} body={props.body} action={<Button onClick={props.onRecover}><Icon name="schema" />{props.actionLabel}</Button>} />;
}
