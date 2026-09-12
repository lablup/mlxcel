import React, { useEffect, useId, useRef } from 'react';

export type ButtonTone = 'primary' | 'secondary' | 'danger' | 'ghost';

export function Button(props: React.ButtonHTMLAttributes<HTMLButtonElement> & { tone?: ButtonTone; busy?: boolean }): React.JSX.Element {
  const { tone = 'secondary', busy = false, className = '', children, disabled, ...rest } = props;
  return (
    <button className={`ds-button ds-button-${tone} ${className}`.trim()} disabled={disabled || busy} data-busy={busy} {...rest}>
      {busy ? <span className="ds-spinner" aria-hidden="true" /> : null}
      <span>{children}</span>
    </button>
  );
}

export function IconButton(props: React.ButtonHTMLAttributes<HTMLButtonElement> & { label: string; icon: string }): React.JSX.Element {
  const { label, icon, className = '', ...rest } = props;
  return (
    <button className={`ds-icon-button ${className}`.trim()} aria-label={label} title={label} {...rest}>
      <span aria-hidden="true">{icon}</span>
    </button>
  );
}

export function Field(props: { label: string; value: string; onChange?: (value: string) => void; placeholder?: string; error?: string; testId?: string }): React.JSX.Element {
  const id = useId();
  const errorId = `${id}-error`;
  return (
    <label className="ds-field" htmlFor={id}>
      <span>{props.label}</span>
      <input id={id} value={props.value} placeholder={props.placeholder} aria-invalid={props.error ? 'true' : undefined} aria-describedby={props.error ? errorId : undefined} data-testid={props.testId} onChange={(event) => props.onChange?.(event.currentTarget.value)} />
      {props.error ? <small id={errorId}>{props.error}</small> : null}
    </label>
  );
}

export function Select(props: { label: string; value: string; options: { value: string; label: string }[]; onChange: (value: string) => void; testId?: string }): React.JSX.Element {
  const id = useId();
  return (
    <label className="ds-field" htmlFor={id}>
      <span>{props.label}</span>
      <select id={id} value={props.value} role="combobox" data-testid={props.testId} onChange={(event) => props.onChange(event.currentTarget.value)}>
        {props.options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
      </select>
    </label>
  );
}

export function StatusBadge(props: { state: 'ready' | 'unloaded' | 'failed' | 'loading'; children: React.ReactNode }): React.JSX.Element {
  return <span className="ds-status" data-state={props.state}>{props.children}</span>;
}

export function ProgressBar(props: { label: string; value?: number; detail?: string }): React.JSX.Element {
  const percent = props.value == null ? undefined : Math.max(0, Math.min(100, props.value));
  return (
    <div className="ds-progress" aria-label={props.label}>
      <div role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} data-indeterminate={percent == null}>
        <span data-progress={percent == null ? 'indeterminate' : String(Math.round(percent / 5) * 5)} />
      </div>
      {props.detail ? <small>{props.detail}</small> : null}
    </div>
  );
}

export function EmptyState(props: { title: string; body: string; action?: React.ReactNode; testId?: string }): React.JSX.Element {
  return (
    <section className="ds-empty" data-testid={props.testId}>
      <div aria-hidden="true">◇</div>
      <h2>{props.title}</h2>
      <p>{props.body}</p>
      {props.action}
    </section>
  );
}

export function ErrorBanner(props: { title: string; body: string; action?: React.ReactNode; tone?: 'error' | 'warning' | 'info'; testId?: string }): React.JSX.Element {
  return (
    <section className="ds-banner" data-tone={props.tone ?? 'error'} role="alert" data-testid={props.testId}>
      <strong>{props.title}</strong>
      <p>{props.body}</p>
      {props.action}
    </section>
  );
}

export function Dialog(props: { open: boolean; title: string; children: React.ReactNode; onClose: () => void; labelledBy?: string; testId?: string; closeLabel?: string }): React.JSX.Element {
  const ref = useRef<HTMLDialogElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const titleId = props.labelledBy ?? useId();
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (props.open && !dialog.open) {
      previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      dialog.showModal();
      const focusTarget = dialog.querySelector<HTMLElement>('[data-autofocus], button, input, select, textarea, a[href]');
      focusTarget?.focus();
    }
    if (!props.open && dialog.open) dialog.close();
  }, [props.open]);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    const handleClose = (): void => {
      props.onClose();
      previousFocus.current?.focus();
    };
    dialog.addEventListener('close', handleClose);
    return () => dialog.removeEventListener('close', handleClose);
  }, [props]);
  return (
    <dialog className="ds-dialog" ref={ref} aria-labelledby={titleId} data-testid={props.testId}>
      <header>
        <h2 id={titleId}>{props.title}</h2>
        <IconButton label={props.closeLabel ?? 'Close'} icon="×" onClick={() => ref.current?.close()} data-testid="dialog-close" />
      </header>
      <div>{props.children}</div>
    </dialog>
  );
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
  const active = props.tabs.find((tab) => tab.id === props.active) ?? props.tabs[0];
  return (
    <section className="ds-tabs">
      <div role="tablist" aria-label={props.label ?? 'Sections'}>
        {props.tabs.map((tab) => <button key={tab.id} role="tab" aria-selected={tab.id === active.id} onClick={() => props.onChange(tab.id)}>{tab.label}</button>)}
      </div>
      <div role="tabpanel">{active.panel}</div>
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
