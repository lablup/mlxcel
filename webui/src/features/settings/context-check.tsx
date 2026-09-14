import React, { useEffect, useRef, useState } from 'react';
import { Button, Field } from '../../design-system/primitives';
import { useWebUiActions } from '../../state';
import type { Locale } from '../../i18n/catalog';
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
  const words = (en: string, ko: string): string => locale === 'ko' ? ko : en;
  const check = async (): Promise<void> => {
    controller.current?.abort(); const current = new AbortController(); controller.current = current;
    setBusy(true); setError(false); setCount(null);
    try { const next = await actions.getTokenCount(modelId, text, current.signal); if (!current.signal.aborted) setCount(next); } catch { if (!current.signal.aborted) setError(true); } finally { if (!current.signal.aborted) setBusy(false); }
  };
  const budget = contextBudget(nCtx, count, defaults.max_tokens);
  return <div><Field label={words('Optional raw prompt budget check', '선택적 원시 프롬프트 예산 확인')} value={text} onChange={(next) => { controller.current?.abort(); setBusy(false); setText(next); setCount(null); }} hint={words('Sent only when Check is pressed; never persisted. Raw tokens exclude chat templates, conversation history and media. Server validation is final.', '확인을 누를 때만 전송하며 저장하지 않습니다. 원시 토큰은 채팅 템플릿, 대화 기록, 미디어를 제외합니다. 최종 검증은 서버가 수행합니다.')} /><Button disabled={busy || text.length === 0} onClick={() => void check()}>{words('Check with model tokenizer', '모델 토크나이저로 확인')}</Button><p role="status">{error ? words('Tokenization unavailable; budget unknown.', '토큰화 불가; 예산을 알 수 없습니다.') : count === null ? words('Not measured', '측정하지 않음') : `${count} ${words('raw tokens', '원시 토큰')}. ${budget === 'exceeds' ? words('Raw input + requested output already exceeds observed context.', '원시 입력 + 요청 출력이 관측 컨텍스트를 초과합니다.') : budget === 'unknown' ? words('Context/output is unknown; no fit claim.', '컨텍스트/출력을 알 수 없어 수용 가능 여부를 판단하지 않습니다.') : words('Raw input + output fits, but templated chat may still exceed the context.', '원시 입력 + 출력은 들어가지만 템플릿 적용 채팅은 컨텍스트를 초과할 수 있습니다.')}`}</p></div>;
}
