// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useContext, useId } from 'react';
import { Selector } from '@lablup/ui-common/Selector';
import { NativeModalContext } from './modal-context';
import { t, type Locale } from '../i18n/catalog';

export interface SelectProps {
  // `description` is a secondary line under the option label; the shared Select folds it
  // into the option's accessible name as "label. description".
  label: string; value: string; options: { value: string; label: string; description?: string; disabled?: boolean }[];
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
      {props.options.map((option) => <option key={option.value} value={option.value} disabled={option.disabled}>{option.description ? `${option.label}. ${option.description}` : option.label}</option>)}
    </select>{details}
  </label>;
  // Selector supplies the select-only combobox pattern, busy and invalid states itself; the
  // product hint and error stay below it and describe its trigger. The list always opens as a
  // popover (never the narrow-viewport bottom sheet), inline next to the trigger.
  return <div onKeyDownCapture={(event) => { if (event.nativeEvent.isComposing) event.stopPropagation(); }} className="ds-field ds-common-select" data-disabled={disabled || undefined} data-testid={props.testId}>
    <Selector value={props.value} options={props.options} onChange={props.onChange} label={props.label} isDisabled={disabled} isLoading={props.busy} status={props.error ? { type: 'error' } : undefined} aria-describedby={describedBy} width="100%" presentation="popover" emptyText={t(props.locale ?? 'en', 'select.no_options')} />
    {details}
  </div>;
}
