// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React from 'react';
import { Button, Field } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { describeEffectiveParameter, GENERATION_FIELDS, resolveEffectiveParameter, validateGenerationDefaults, type GenerationDefaults } from '../settings/generation-defaults';
import { useSelectedServerGenerationDefaults } from '../settings/server-defaults';

export type TurnParameterDraft = Partial<Record<typeof GENERATION_FIELDS[number], string>>;
/** Blank overrides inherit the session default; absent defaults remain omitted. */
export function resolveTurnParameters(defaults: GenerationDefaults, draft: TurnParameterDraft): Record<string, number> {
  const overrides = Object.fromEntries(Object.entries(draft).filter(([, value]) => value !== undefined && value.trim() !== '').map(([key, value]) => [key, Number(value)]));
  return { ...validateGenerationDefaults({ ...defaults, ...overrides }) };
}
export function TurnParameters({ defaults, draft, onChange, locale }: { defaults: GenerationDefaults; draft: TurnParameterDraft; onChange: (next: TurnParameterDraft) => void; locale: Locale }): React.JSX.Element {
  // Same resolution and wording as Settings > Requests, against the server defaults Settings last read.
  const server = useSelectedServerGenerationDefaults();
  return <div className="chat-parameters"><p>{t(locale, 'chat.params.body')}</p><div className="chat-parameter-grid">{GENERATION_FIELDS.map((name) => <Field key={name} label={t(locale, 'chat.params.field', { name })} value={draft[name] ?? ''} onChange={(value) => onChange({ ...draft, [name]: value })} hint={t(locale, 'chat.params.inherited', { value: describeEffectiveParameter(locale, resolveEffectiveParameter(name, undefined, defaults[name], server?.[name])) })} />)}</div><Button onClick={() => onChange({})}>{t(locale, 'chat.params.clear')}</Button></div>;
}
