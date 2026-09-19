// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// One control per live-setting schema kind. The draft stays text, exactly what the
// person typed; parseSettingInput remains the only validator and runs on Apply.
import React, { useId, useState } from 'react';
import { formatSettingValue, type SettingSpec } from '../../api/settings';
import { Field, IconButton, Select, Toggle, Tooltip } from '../../design-system/primitives';
import { t, type Locale, type StringKey } from '../../i18n/catalog';
import { LIVE_SETTING_KEYS } from './strings';

/** The catalog label for a live setting, with its raw key as secondary text; an uncatalogued name shows the raw key alone. */
export function settingLabel(locale: Locale, name: string): { label: string; key: string | null } {
  const catalogKey = `settings.live.${name}`;
  return LIVE_SETTING_KEYS.has(catalogKey) ? { label: t(locale, catalogKey as StringKey), key: name } : { label: name, key: null };
}

/** The UI's own help for a known setting; otherwise the server's help sentence, as the server wrote it. */
export function settingHelp(locale: Locale, spec: SettingSpec): string {
  const catalogKey = `settings.live.${spec.name}.help`;
  return LIVE_SETTING_KEYS.has(catalogKey) ? t(locale, catalogKey as StringKey) : spec.help;
}

/**
 * A labelled field with the Field primitive's hint and error wiring, built as a div so its label
 * row can also hold a control of its own (the "Leave unset" toggle) without nesting labels.
 */
function FieldShell(props: { id: string; label: string; aside?: React.ReactNode; hint?: string; error?: string; innerDisabled?: boolean; children: React.ReactNode }): React.JSX.Element {
  return (
    <div className="ds-field setting-field" data-inner-disabled={props.innerDisabled || undefined}>
      <div className="setting-label-row"><label htmlFor={props.id} id={`${props.id}-label`}>{props.label}</label>{props.aside}</div>
      {props.children}
      {props.hint ? <small id={`${props.id}-hint`} data-tone="hint">{props.hint}</small> : null}
      {props.error ? <small id={`${props.id}-error`} data-tone="error">{props.error}</small> : null}
    </div>
  );
}
const describedBy = (id: string, hint?: string, error?: string): string | undefined => [hint ? `${id}-hint` : null, error ? `${id}-error` : null].filter(Boolean).join(' ') || undefined;

/** A number input with the Field primitive's label, hint and error wiring. */
export function NumberField(props: { label: string; value: string; onChange: (value: string) => void; step: 'any' | '1'; min?: number; max?: number; hint?: string; error?: string; disabled?: boolean; aside?: React.ReactNode; testId?: string }): React.JSX.Element {
  const id = useId();
  return (
    <FieldShell id={id} label={props.label} aside={props.aside} hint={props.hint} error={props.error} innerDisabled={props.disabled}>
      <input id={id} type="number" step={props.step} min={props.min} max={props.max} aria-labelledby={`${id}-label`} value={props.value} disabled={props.disabled} aria-invalid={props.error ? 'true' : undefined} aria-describedby={describedBy(id, props.hint, props.error)} data-testid={props.testId} onChange={(event) => props.onChange(event.currentTarget.value)} />
    </FieldShell>
  );
}

/** A JSON textarea that reports a parse failure when it loses focus; Apply still validates through parseSettingInput. */
function JsonField(props: { label: string; value: string; onChange: (value: string) => void; hint?: string; error?: string; disabled?: boolean; aside?: React.ReactNode; locale: Locale; testId?: string }): React.JSX.Element {
  const id = useId();
  const [invalid, setInvalid] = useState(false);
  const error = props.error ?? (invalid ? t(props.locale, 'settings.control.invalid_json') : undefined);
  const validate = (text: string): void => {
    if (props.disabled) return;
    try { JSON.parse(text); setInvalid(false); } catch { setInvalid(true); }
  };
  return (
    <FieldShell id={id} label={props.label} aside={props.aside} hint={props.hint} error={error} innerDisabled={props.disabled}>
      <textarea id={id} rows={2} spellCheck={false} aria-labelledby={`${id}-label`} value={props.value} disabled={props.disabled} aria-invalid={error ? 'true' : undefined} aria-describedby={describedBy(id, props.hint, error)} data-testid={props.testId} onChange={(event) => { setInvalid(false); props.onChange(event.currentTarget.value); }} onBlur={(event) => validate(event.currentTarget.value)} />
    </FieldShell>
  );
}

export interface SettingControlProps {
  readonly spec: SettingSpec;
  /** The server's current value. */
  readonly value: unknown;
  /** The pending text for this setting, or undefined when there is none. */
  readonly draft: string | undefined;
  /** New draft text, or undefined to discard the draft and show the current value again. */
  readonly onChange: (draft: string | undefined) => void;
  readonly error?: string;
  readonly disabled?: boolean;
  readonly locale: Locale;
}

/** Maps a schema kind to its control: bool Toggle, enum Select, text, int and float number inputs, JSON textarea, plus a null toggle for `_or_null` kinds. */
export function SettingControl({ spec, value, draft, onChange, error, disabled = false, locale }: SettingControlProps): React.JSX.Element {
  const { label, key } = settingLabel(locale, spec.name);
  const nullable = spec.type.endsWith('_or_null');
  const base = spec.type.replace('_or_null', '');
  // Only an `_or_null` kind has a null state: typed text `null` in any other kind is just a (possibly invalid) draft that stays editable.
  const isNull = nullable && (draft === undefined ? value === null : draft === 'null');
  const shown = draft ?? formatSettingValue(spec, value);
  const serverText = value === null ? t(locale, 'settings.value.unset') : formatSettingValue(spec, value);
  const edited = draft !== undefined && draft !== formatSettingValue(spec, value) && !(draft === 'null' && value === null);
  // The raw key leads so a label never hides it.
  const hint = [key, settingHelp(locale, spec), edited ? t(locale, 'settings.control.server_value', { value: serverText }) : null].filter(Boolean).join(' · ') || undefined;
  const setNull = (on: boolean): void => {
    if (on) onChange('null');
    // Back to a value: restore the current one, or start empty when the server holds none.
    else onChange(value === null ? '' : undefined);
  };
  const unset = nullable ? <Toggle label={t(locale, 'settings.control.unset')} checked={isNull} onChange={setNull} disabled={disabled} testId={`setting-${spec.name}-unset`} /> : null;
  const text = isNull ? '' : shown;
  const innerDisabled = disabled || isNull;
  const testId = `setting-${spec.name}`;
  // A nullable setting is a named group, so "Leave unset" is announced with the setting it belongs to.
  const wrap = (children: React.ReactNode): React.JSX.Element => <div className="setting-control" data-kind={spec.type} role={nullable ? 'group' : undefined} aria-label={nullable ? label : undefined}>{children}</div>;
  // Number and JSON fields carry the unset toggle in their label row; the shared Field and Select
  // take no extra label content, so a nullable string keeps it on its own line below.
  if (base === 'int' || base === 'float') return wrap(<NumberField label={label} value={text} step={base === 'int' ? '1' : 'any'} onChange={(next) => onChange(next)} disabled={innerDisabled} aside={unset} hint={hint} error={error} testId={testId} />);
  if (base === 'array' || base === 'object') return wrap(<JsonField label={label} value={text} onChange={(next) => onChange(next)} disabled={innerDisabled} aside={unset} hint={hint} error={error} locale={locale} testId={testId} />);
  let control: React.JSX.Element;
  if (base === 'bool') control = <Toggle label={label} checked={shown === 'true'} onChange={(checked) => onChange(String(checked))} disabled={innerDisabled} hint={hint} error={error} testId={testId} />;
  else if (spec.allowed !== null) {
    const options: { value: string; label: string; disabled?: boolean }[] = spec.allowed.map((option) => ({ value: option, label: option }));
    // A current value outside `allowed` stays visible, never silently replaced.
    if (text !== '' && !spec.allowed.includes(text)) options.push({ value: text, label: t(locale, 'settings.control.not_allowed', { value: text }), disabled: true });
    control = <Select locale={locale} label={label} value={text} options={options} onChange={(next) => onChange(next)} disabled={innerDisabled} hint={hint} error={error} testId={testId} />;
  } else control = <Field label={label} value={text} onChange={(next) => onChange(next)} disabled={innerDisabled} hint={hint} error={error} testId={testId} />;
  return wrap(<>{control}{unset}</>);
}

/** A help button whose tooltip carries a contract statement that used to be inline prose. */
export function HelpTip({ label, content }: { label: string; content: string }): React.JSX.Element {
  return <Tooltip content={content}><IconButton label={label} icon="help" className="settings-help" /></Tooltip>;
}
