import React, { useEffect, useId, useRef } from 'react';
import { Icon, type IconName } from './icons';

export type ButtonTone = 'primary' | 'secondary' | 'danger' | 'ghost';
export type LifecycleState = 'unloaded' | 'loading' | 'ready' | 'draining' | 'unloading' | 'failed';


export function Button(props: React.ButtonHTMLAttributes<HTMLButtonElement> & { tone?: ButtonTone; busy?: boolean }): React.JSX.Element {
  const { tone = 'secondary', busy = false, className = '', children, disabled, ...rest } = props;
  return (
    <button className={`ds-button ds-button-${tone} ${className}`.trim()} disabled={disabled || busy} aria-busy={busy || undefined} data-busy={busy} {...rest}>
      {busy ? <span className="ds-spinner" aria-hidden="true" /> : null}
      <span>{children}</span>
    </button>
  );
}

export function IconButton(props: React.ButtonHTMLAttributes<HTMLButtonElement> & { label: string; icon: IconName | string; busy?: boolean }): React.JSX.Element {
  const { label, icon, className = '', busy = false, disabled, ...rest } = props;
  return (
    <button className={`ds-icon-button ${className}`.trim()} aria-label={label} title={label} disabled={disabled || busy} aria-busy={busy || undefined} data-busy={busy} {...rest}>
      {isIconName(icon) ? <Icon name={icon} /> : <span aria-hidden="true">{icon}</span>}
    </button>
  );
}

function isIconName(value: string): value is IconName {
  return ['models', 'chat', 'activity', 'settings', 'gallery', 'command', 'help', 'menu', 'close', 'search', 'warning', 'key', 'schema'].includes(value);
}

export function Field(props: { label: string; value: string; onChange?: (value: string) => void; placeholder?: string; disabled?: boolean; busy?: boolean; error?: string; hint?: string; testId?: string }): React.JSX.Element {
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [props.hint ? hintId : null, props.error ? errorId : null].filter(Boolean).join(' ') || undefined;
  return (
    <label className="ds-field" htmlFor={id} data-disabled={props.disabled || props.busy || undefined}>
      <span>{props.label}</span>
      <input id={id} value={props.value} placeholder={props.placeholder} disabled={props.disabled || props.busy} aria-busy={props.busy || undefined} aria-invalid={props.error ? 'true' : undefined} aria-describedby={describedBy} data-testid={props.testId} onChange={(event) => props.onChange?.(event.currentTarget.value)} />
      {props.hint ? <small id={hintId} data-tone="hint">{props.hint}</small> : null}
      {props.error ? <small id={errorId} data-tone="error">{props.error}</small> : null}
    </label>
  );
}

export function Select(props: { label: string; value: string; options: { value: string; label: string; disabled?: boolean }[]; onChange: (value: string) => void; disabled?: boolean; busy?: boolean; error?: string; hint?: string; testId?: string }): React.JSX.Element {
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [props.hint ? hintId : null, props.error ? errorId : null].filter(Boolean).join(' ') || undefined;
  return (
    <label className="ds-field" htmlFor={id} data-disabled={props.disabled || props.busy || undefined}>
      <span>{props.label}</span>
      <select id={id} value={props.value} disabled={props.disabled || props.busy} aria-busy={props.busy || undefined} aria-invalid={props.error ? 'true' : undefined} aria-describedby={describedBy} data-testid={props.testId} onChange={(event) => props.onChange(event.currentTarget.value)}>
        {props.options.map((option) => <option key={option.value} value={option.value} disabled={option.disabled}>{option.label}</option>)}
      </select>
      {props.hint ? <small id={hintId} data-tone="hint">{props.hint}</small> : null}
      {props.error ? <small id={errorId} data-tone="error">{props.error}</small> : null}
    </label>
  );
}

export function StatusBadge(props: { state: LifecycleState; children: React.ReactNode; label?: string }): React.JSX.Element {
  return <span className="ds-status" data-state={props.state} aria-label={props.label}>{props.children}</span>;
}

export function ProgressBar(props: { label: string; value?: number; detail?: string }): React.JSX.Element {
  const percent = props.value == null ? undefined : Math.max(0, Math.min(100, props.value));
  return (
    <div className="ds-progress">
      <div role="progressbar" aria-label={props.label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} data-indeterminate={percent == null}>
        <span data-progress={percent == null ? 'indeterminate' : String(Math.round(percent / 5) * 5)} />
      </div>
      {props.detail ? <small>{props.detail}</small> : null}
    </div>
  );
}

export function EmptyState(props: { title: string; body: string; action?: React.ReactNode; testId?: string }): React.JSX.Element {
  return (
    <section className="ds-empty" data-testid={props.testId}>
      <div aria-hidden="true"><Icon name="models" /></div>
      <h2>{props.title}</h2>
      <p>{props.body}</p>
      {props.action}
    </section>
  );
}

export function ErrorBanner(props: { title: string; body: string; action?: React.ReactNode; tone?: 'error' | 'warning' | 'info'; testId?: string }): React.JSX.Element {
  return (
    <section className="ds-banner" data-tone={props.tone ?? 'error'} role={props.tone === 'info' ? 'status' : 'alert'} data-testid={props.testId}>
      <strong>{props.title}</strong>
      <p>{props.body}</p>
      {props.action}
    </section>
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
      if (target?.isConnected) target.focus();
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
    };
  }, []);

  return (
    <dialog className={`ds-dialog ${props.className ?? ''}`.trim()} data-position={props.position ?? 'center'} ref={ref} aria-labelledby={titleId} data-testid={props.testId}>
      <header>
        <h2 id={titleId}>{props.title}</h2>
        <IconButton label={props.closeLabel ?? 'Close'} icon="close" onClick={() => { const target = previousFocus.current; closingFromProp.current = true; onCloseRef.current(); ref.current?.close(); window.setTimeout(() => { if (target?.isConnected) target.focus(); }, 0); }} data-testid="dialog-close" />
      </header>
      <div>{props.children}</div>
    </dialog>
  );
}

export function Dialog(props: Omit<ModalProps, 'position' | 'className'>): React.JSX.Element {
  return <ModalDialog {...props} />;
}

export function Sheet(props: Omit<ModalProps, 'position'>): React.JSX.Element {
  return <ModalDialog {...props} position="left" className={`ds-sheet ${props.className ?? ''}`.trim()} />;
}

export function Tooltip(props: { label: string; children: React.ReactElement }): React.JSX.Element {
  const id = useId();
  return (
    <span className="ds-tooltip-wrap">
      {React.cloneElement(props.children, { 'aria-describedby': id } as Partial<HTMLElement>)}
      <span className="ds-tooltip" role="tooltip" id={id}>{props.label}</span>
    </span>
  );
}

export function Tabs(props: { tabs: { id: string; label: string; panel: React.ReactNode }[]; active: string; onChange: (id: string) => void; label?: string }): React.JSX.Element {
  const generated = useId();
  const active = props.tabs.find((tab) => tab.id === props.active) ?? props.tabs[0];
  const setActive = (id: string): void => props.onChange(id);
  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>): void => {
    const focusedIndex = props.tabs.findIndex((tab) => document.activeElement?.id === `${generated}-tab-${tab.id}`);
    const index = focusedIndex >= 0 ? focusedIndex : props.tabs.findIndex((tab) => tab.id === active.id);
    const nextIndex = event.key === 'ArrowRight' ? index + 1 : event.key === 'ArrowLeft' ? index - 1 : event.key === 'Home' ? 0 : event.key === 'End' ? props.tabs.length - 1 : null;
    if (nextIndex === null) return;
    event.preventDefault();
    const next = props.tabs[(nextIndex + props.tabs.length) % props.tabs.length];
    setActive(next.id);
    requestAnimationFrame(() => document.getElementById(`${generated}-tab-${next.id}`)?.focus());
  };
  return (
    <section className="ds-tabs">
      <div role="tablist" aria-label={props.label ?? 'Sections'} onKeyDown={handleKeyDown}>
        {props.tabs.map((tab) => <button id={`${generated}-tab-${tab.id}`} key={tab.id} role="tab" aria-selected={tab.id === active.id} aria-controls={`${generated}-panel-${tab.id}`} tabIndex={tab.id === active.id ? 0 : -1} onClick={() => setActive(tab.id)}>{tab.label}</button>)}
      </div>
      <div id={`${generated}-panel-${active.id}`} role="tabpanel" aria-labelledby={`${generated}-tab-${active.id}`}>{active.panel}</div>
    </section>
  );
}

export function Inspector(props: { title: string; children: React.ReactNode }): React.JSX.Element {
  return (
    <aside className="ds-inspector" aria-label={props.title}>
      <h2>{props.title}</h2>
      {props.children}
    </aside>
  );
}

export function DataTable(props: { caption: string; children: React.ReactNode }): React.JSX.Element {
  return <table className="ds-table"><caption>{props.caption}</caption>{props.children}</table>;
}

export function DenseList(props: { label: string; children: React.ReactNode }): React.JSX.Element {
  return <ul className="ds-list" aria-label={props.label}>{props.children}</ul>;
}

export function AuthGate(props: { title: string; body: string; actionLabel: string }): React.JSX.Element {
  return <ErrorBanner tone="warning" title={props.title} body={props.body} action={<Button><Icon name="key" />{props.actionLabel}</Button>} />;
}

export function SchemaMismatchView(props: { title: string; body: string; actionLabel: string }): React.JSX.Element {
  return <ErrorBanner tone="error" title={props.title} body={props.body} action={<Button><Icon name="schema" />{props.actionLabel}</Button>} />;
}
