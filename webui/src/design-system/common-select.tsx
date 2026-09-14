// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useContext, useId } from 'react';
import { Select as CommonSelect } from '@lablup/ui-common/components/Select';
import { NativeModalContext } from './modal-context';
import { t, type Locale } from '../i18n/catalog';

export interface SelectProps {
  label: string; value: string; options: { value: string; label: string; disabled?: boolean }[];
  onChange: (value: string) => void; disabled?: boolean; busy?: boolean; error?: string;
  hint?: string; testId?: string; locale?: Locale;
}

export function Select(props: SelectProps): React.JSX.Element {
  const nativeModal = useContext(NativeModalContext);
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [props.hint ? hintId : null, props.error ? errorId : null].filter(Boolean).join(' ') || undefined;
  const disabled = props.disabled || props.busy;
  const details = <>{props.hint ? <small id={hintId} data-tone="hint">{props.hint}</small> : null}{props.error ? <small id={errorId} data-tone="error">{props.error}</small> : null}</>;
  if (nativeModal) return <label className="ds-field" htmlFor={id} data-disabled={disabled || undefined}>
    <span>{props.label}</span>
    <select id={id} value={props.value} disabled={disabled} aria-busy={props.busy || undefined} aria-invalid={props.error ? 'true' : undefined} aria-describedby={describedBy} data-testid={props.testId} onChange={(event) => props.onChange(event.currentTarget.value)}>
      {props.options.map((option) => <option key={option.value} {...option}>{option.label}</option>)}
    </select>{details}
  </label>;
  return <div ref={(element) => {
    // alpha.19 supplies listbox controls/active-descendant on a button, but no role prop.
    // Complete the select-only combobox pattern without changing its keyboard implementation.
    const trigger = element?.querySelector('.select__trigger');
    trigger?.setAttribute('role', 'combobox');
    if (props.busy) trigger?.setAttribute('aria-busy', 'true');
    else trigger?.removeAttribute('aria-busy');
  }} onKeyDownCapture={(event) => { if (event.nativeEvent.isComposing) event.stopPropagation(); }} className="ds-field ds-common-select" data-disabled={disabled || undefined} aria-busy={props.busy || undefined} data-testid={props.testId}>
    <CommonSelect value={props.value} options={props.options} onChange={props.onChange} label={props.label} disabled={disabled} invalid={Boolean(props.error)} aria-describedby={describedBy} fullWidth noOptionsLabel={t(props.locale ?? 'en', 'select.no_options')} searchPlaceholder={t(props.locale ?? 'en', 'select.search')} />
    {details}
  </div>;
}
