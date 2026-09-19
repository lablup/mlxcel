import React, { useEffect, useRef, useState } from 'react';
import { Button, Field } from '../../design-system/primitives';
import { useWebUiActions } from '../../state';
import { t, type Locale } from '../../i18n/catalog';
import { useGenerationDefaults } from './generation-preferences';
/** A raw tokenizer count is a lower bound, not a templated chat admission oracle. */
export function contextBudget(nCtx: number | null, input: number | null, output: number | undefined): 'unknown' | 'exceeds' | 'raw-fits' {
  if (nCtx === null || nCtx <= 0 || input === null || output === undefined) return 'unknown';
  return input + output > nCtx ? 'exceeds' : 'raw-fits';
}
export function ContextCheck({ modelId, nCtx, locale }: { modelId: string; nCtx: number | null; locale: Locale }): React.JSX.Element {
  const actions = useWebUiActions();
  const { defaults } = useGenerationDefaults();
  const [text, setText] = useState('');
  const [count, setCount] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(false);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  const check = async (): Promise<void> => {
    controller.current?.abort(); const current = new AbortController(); controller.current = current;
    setBusy(true); setError(false); setCount(null);
    try { const next = await actions.getTokenCount(modelId, text, current.signal); if (!current.signal.aborted) setCount(next); } catch { if (!current.signal.aborted) setError(true); } finally { if (!current.signal.aborted) setBusy(false); }
  };
  const budget = contextBudget(nCtx, count, defaults.max_tokens);
  const verdict = budget === 'exceeds' ? t(locale, 'settings.context_check.exceeds') : budget === 'unknown' ? t(locale, 'settings.context_check.unknown') : t(locale, 'settings.context_check.fits');
  return <div><Field label={t(locale, 'settings.context_check.label')} value={text} onChange={(next) => { controller.current?.abort(); setBusy(false); setText(next); setCount(null); }} hint={t(locale, 'settings.context_check.hint')} /><Button disabled={busy || text.length === 0} onClick={() => void check()}>{t(locale, 'settings.context_check.check')}</Button><p role="status">{error ? t(locale, 'settings.context_check.unavailable') : count === null ? t(locale, 'settings.context_check.not_measured') : t(locale, 'settings.context_check.count', { count: String(count), verdict })}</p></div>;
}
