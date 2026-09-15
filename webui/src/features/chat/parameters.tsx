// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React from 'react';
import { Button, Field } from '../../design-system/primitives';
import { GENERATION_FIELDS, validateGenerationDefaults, type GenerationDefaults } from '../settings/generation-defaults';

export type TurnParameterDraft = Partial<Record<typeof GENERATION_FIELDS[number], string>>;
/** Blank overrides inherit the session default; absent defaults remain omitted. */
export function resolveTurnParameters(defaults: GenerationDefaults, draft: TurnParameterDraft): Record<string, number> {
  const overrides = Object.fromEntries(Object.entries(draft).filter(([, value]) => value !== undefined && value.trim() !== '').map(([key, value]) => [key, Number(value)]));
  return { ...validateGenerationDefaults({ ...defaults, ...overrides }) };
}
export function TurnParameters({ defaults, draft, onChange }: { defaults: GenerationDefaults; draft: TurnParameterDraft; onChange: (next: TurnParameterDraft) => void }): React.JSX.Element {
  return <details className="chat-parameters"><summary>Parameters for next turn</summary><p>These overrides apply once, after Send accepts a request. Blank fields inherit Settings; absent Settings fields inherit the server. Changes never affect a running turn.</p><div className="chat-parameter-grid">{GENERATION_FIELDS.map((name) => <Field key={name} label={`Next turn ${name}`} value={draft[name] ?? ''} onChange={(value) => onChange({ ...draft, [name]: value })} hint={`Inherited: ${defaults[name] ?? 'server default'}`} />)}</div><Button onClick={() => onChange({})}>Clear next-turn overrides</Button></details>;
}
