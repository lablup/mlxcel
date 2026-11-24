// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useId } from 'react';
import { Selector } from '@lablup/ui-common/Selector';
import { t, type Locale } from '../i18n/catalog';

export interface SelectProps {
  // `description` is a secondary line under the option label; the shared Select folds it
  // into the option's accessible name as "label. description".
  label: string; value: string; options: { value: string; label: string; description?: string; disabled?: boolean }[];
  onChange: (value: string) => void; disabled?: boolean; busy?: boolean; error?: string;
  hint?: string; testId?: string; locale?: Locale;
}

export function Select(props: SelectProps): React.JSX.Element {
  const id = useId();
  const labelId = `${id}-label`;
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [props.hint ? hintId : null, props.error ? errorId : null].filter(Boolean).join(' ') || undefined;
  const disabled = props.disabled || props.busy;
  const details = <>{props.hint ? <small id={hintId} data-tone="hint">{props.hint}</small> : null}{props.error ? <small id={errorId} data-tone="error">{props.error}</small> : null}</>;
  // Selector supplies the select-only combobox pattern, busy and invalid states itself; the
  // product hint and error stay below it and describe its trigger. The list always opens as a
  // popover (never the narrow-viewport bottom sheet), inline next to the trigger, so inside a
  // native modal Dialog it stays in the dialog's subtree. No IME guard: the trigger is not
  // editable, and Astryx guards type-to-select and its search input against composing keys.
  // Since ui-common 0.2.0-alpha.20 the Selector is a Form.Item wrapper and drops `label`: outside
  // an item it renders an empty, visually hidden label. The visible label is rendered here and
  // names the trigger through `aria-labelledby`, which Astryx joins with that empty label.
  return (
    <div className="ds-field ds-common-select" data-disabled={disabled || undefined} data-testid={props.testId}>
      <span id={labelId} className="ds-common-select-label">{props.label}</span>
      <Selector value={props.value} options={props.options} onChange={props.onChange} aria-labelledby={labelId} disabled={disabled} isLoading={props.busy} status={props.error ? 'error' : ''} aria-describedby={describedBy} width="100%" presentation="popover" emptyText={t(props.locale ?? 'en', 'select.no_options')} />
      {details}
    </div>
  );
}
