import React, { useState } from 'react';
import { Button, Dialog, ErrorBanner, Field } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { GENERATION_FIELDS, validateGenerationDefaults } from './generation-defaults';
import { useGenerationDefaults } from './generation-preferences';

export function GenerationSettings({ locale }: { locale: Locale }): React.JSX.Element {
  const { defaults, setDefaults, reset } = useGenerationDefaults();
  const [draft, setDraft] = useState<Record<string, string>>(() => Object.fromEntries(Object.entries(defaults).map(([key, value]) => [key, String(value)])));
  const [error, setError] = useState('');
  const [resetOpen, setResetOpen] = useState(false);
  const apply = (): void => {
    try { setDefaults(validateGenerationDefaults(Object.fromEntries(Object.entries(draft).filter(([, value]) => value.trim() !== '').map(([key, value]) => [key, Number(value)])))); setError(''); } catch (cause) { setError(cause instanceof Error ? cause.message : t(locale, 'settings.generation.invalid')); }
  };
  return <section className="screen-stack"><h2>{t(locale, 'settings.generation.title')}</h2><p>{t(locale, 'settings.generation.body')}</p><div className="settings-grid">{GENERATION_FIELDS.map((name) => <Field key={name} label={name} value={draft[name] ?? ''} onChange={(value) => setDraft((old) => ({ ...old, [name]: value }))} hint={t(locale, 'settings.generation.current', { value: defaults[name]?.toString() ?? t(locale, 'settings.generation.inherit') })} />)}</div>{error ? <ErrorBanner title={t(locale, 'settings.generation.error_title')} body={error} /> : null}<div><Button onClick={apply}>{t(locale, 'settings.generation.save')}</Button> <Button onClick={() => setResetOpen(true)}>{t(locale, 'settings.generation.reset_open')}</Button></div><Field label={t(locale, 'settings.generation.reasoning_label')} value={t(locale, 'settings.generation.reasoning_value')} disabled hint={t(locale, 'settings.generation.reasoning_hint')} /><Dialog open={resetOpen} title={t(locale, 'settings.generation.reset.title')} closeLabel={t(locale, 'common.close')} onClose={() => setResetOpen(false)}><p>{t(locale, 'settings.generation.reset.body')}</p><Button onClick={() => { reset(); setDraft({}); setError(''); setResetOpen(false); }}>{t(locale, 'settings.generation.reset.confirm')}</Button></Dialog></section>;
}
