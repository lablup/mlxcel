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

/** A number input with the Field primitive's label, hint and error wiring. */
export function NumberField(props: { label: string; value: string; onChange: (value: string) => void; step: 'any' | '1'; min?: number; max?: number; hint?: string; error?: string; disabled?: boolean; testId?: string }): React.JSX.Element {
  const id = useId();
  const labelId = `${id}-label`;
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [props.hint ? hintId : null, props.error ? errorId : null].filter(Boolean).join(' ') || undefined;
  return (
    <label className="ds-field" htmlFor={id} data-disabled={props.disabled || undefined}>
      <span id={labelId}>{props.label}</span>
      <input id={id} type="number" step={props.step} min={props.min} max={props.max} aria-labelledby={labelId} value={props.value} disabled={props.disabled} aria-invalid={props.error ? 'true' : undefined} aria-describedby={describedBy} data-testid={props.testId} onChange={(event) => props.onChange(event.currentTarget.value)} />
      {props.hint ? <small id={hintId} data-tone="hint">{props.hint}</small> : null}
      {props.error ? <small id={errorId} data-tone="error">{props.error}</small> : null}
    </label>
  );
}

/** A JSON textarea that reports a parse failure when it loses focus; Apply still validates through parseSettingInput. */
function JsonField(props: { label: string; value: string; onChange: (value: string) => void; hint?: string; error?: string; disabled?: boolean; locale: Locale; testId?: string }): React.JSX.Element {
  const id = useId();
  const labelId = `${id}-label`;
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const [invalid, setInvalid] = useState(false);
  const error = props.error ?? (invalid ? t(props.locale, 'settings.control.invalid_json') : undefined);
  const describedBy = [props.hint ? hintId : null, error ? errorId : null].filter(Boolean).join(' ') || undefined;
  const validate = (text: string): void => {
    if (props.disabled) return;
    try { JSON.parse(text); setInvalid(false); } catch { setInvalid(true); }
  };
  return (
    <label className="ds-field" htmlFor={id} data-disabled={props.disabled || undefined}>
      <span id={labelId}>{props.label}</span>
      <textarea id={id} rows={2} spellCheck={false} aria-labelledby={labelId} value={props.value} disabled={props.disabled} aria-invalid={error ? 'true' : undefined} aria-describedby={describedBy} data-testid={props.testId} onChange={(event) => { setInvalid(false); props.onChange(event.currentTarget.value); }} onBlur={(event) => validate(event.currentTarget.value)} />
      {props.hint ? <small id={hintId} data-tone="hint">{props.hint}</small> : null}
      {error ? <small id={errorId} data-tone="error">{error}</small> : null}
    </label>
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
  const isNull = draft === undefined ? value === null : draft === 'null';
  const shown = draft ?? formatSettingValue(spec, value);
  const serverText = value === null ? t(locale, 'settings.value.unset') : formatSettingValue(spec, value);
  const edited = draft !== undefined && draft !== formatSettingValue(spec, value) && !(draft === 'null' && value === null);
  // The raw key leads so a label never hides it.
  const hint = [key, settingHelp(locale, spec), edited ? t(locale, 'settings.control.server_value', { value: serverText }) : null].filter(Boolean).join(' · ') || undefined;
  const inner = (innerDisabled: boolean): React.JSX.Element => {
    const text = isNull ? '' : shown;
    if (base === 'bool') return <Toggle label={label} checked={shown === 'true'} onChange={(checked) => onChange(String(checked))} disabled={innerDisabled} hint={hint} error={error} testId={`setting-${spec.name}`} />;
    if (base === 'str' && spec.allowed !== null) {
      const options: { value: string; label: string; disabled?: boolean }[] = spec.allowed.map((option) => ({ value: option, label: option }));
      // A current value outside `allowed` stays visible, never silently replaced.
      if (text !== '' && !spec.allowed.includes(text)) options.push({ value: text, label: t(locale, 'settings.control.not_allowed', { value: text }), disabled: true });
      return <Select locale={locale} label={label} value={text} options={options} onChange={(next) => onChange(next)} disabled={innerDisabled} hint={hint} error={error} testId={`setting-${spec.name}`} />;
    }
    if (base === 'int' || base === 'float') return <NumberField label={label} value={text} step={base === 'int' ? '1' : 'any'} onChange={(next) => onChange(next)} disabled={innerDisabled} hint={hint} error={error} testId={`setting-${spec.name}`} />;
    if (base === 'array' || base === 'object') return <JsonField label={label} value={text} onChange={(next) => onChange(next)} disabled={innerDisabled} hint={hint} error={error} locale={locale} testId={`setting-${spec.name}`} />;
    return <Field label={label} value={text} onChange={(next) => onChange(next)} disabled={innerDisabled} hint={hint} error={error} testId={`setting-${spec.name}`} />;
  };
  if (!nullable) return <div className="setting-control" data-kind={spec.type}>{inner(disabled)}</div>;
  const setNull = (on: boolean): void => {
    if (on) onChange('null');
    // Back to a value: restore the current one, or start empty when the server holds none.
    else onChange(value === null ? '' : undefined);
  };
  return (
    <div className="setting-control" data-kind={spec.type} role="group" aria-label={label}>
      {inner(disabled || isNull)}
      <Toggle label={t(locale, 'settings.control.unset')} checked={isNull} onChange={setNull} disabled={disabled} testId={`setting-${spec.name}-unset`} />
    </div>
  );
}

/** A help button whose tooltip carries a contract statement that used to be inline prose. */
export function HelpTip({ label, content }: { label: string; content: string }): React.JSX.Element {
  return <Tooltip content={content}><IconButton label={label} icon="help" className="settings-help" /></Tooltip>;
}
