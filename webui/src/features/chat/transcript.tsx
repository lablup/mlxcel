// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../../design-system/primitives';
import { t, type Locale, type StringKey } from '../../i18n/catalog';
import type { ChatTurn } from './history';
import { SafeMarkdown } from './markdown';
import { decodeRate } from './stream';

const STATUS_KEYS: Record<ChatTurn['status'], StringKey> = {
  streaming: 'chat.transcript.status.streaming',
  complete: 'chat.transcript.status.complete',
  cancelled: 'chat.transcript.status.cancelled',
  interrupted: 'chat.transcript.status.interrupted',
  error: 'chat.transcript.status.error',
};

const Turn = memo(function Turn({ turn, index, onEdit, busy, locale }: { turn: ChatTurn; index: number; onEdit: (index: number) => void; busy: boolean; locale: Locale }): React.JSX.Element {
  const [copyStatus, setCopyStatus] = useState<StringKey | null>(null);
  const rate = decodeRate(turn);
  const unknown = t(locale, 'chat.transcript.unknown');
  return <article className="chat-turn" aria-label={t(locale, 'chat.transcript.turn_label', { model: turn.modelName })}>
    <header><h2>{turn.modelName}</h2><span>{t(locale, STATUS_KEYS[turn.status])}</span></header>
    <p className="chat-prompt">{turn.prompt}</p>
    {turn.images.length > 0 ? <p>{t(locale, 'chat.transcript.images', { count: String(turn.images.length) })}</p> : null}
    <Button disabled={busy} onClick={() => onEdit(index)}>{t(locale, 'chat.transcript.edit')}</Button>
    {turn.reasoning ? <details><summary>{turn.status === 'streaming' && !turn.content ? t(locale, 'chat.transcript.reasoning_thinking') : t(locale, 'chat.reasoning')}</summary><pre>{turn.reasoning}</pre></details> : null}
    {turn.status === 'streaming' && !turn.content ? <p>{turn.reasoning ? t(locale, 'chat.transcript.thinking') : t(locale, 'chat.transcript.waiting')}</p> : null}
    <SafeMarkdown text={turn.content} streaming={turn.status === 'streaming'} />
    {turn.tools.map((tool) => <details key={tool.index}><summary>{t(locale, 'chat.transcript.tool_call', { name: tool.name || t(locale, 'chat.transcript.tool_name_pending') })}</summary><pre>{tool.arguments}</pre></details>)}
    {turn.error ? <p role="alert">{turn.error}</p> : null}
    <div className="chat-toolbar"><Button onClick={() => { void (async () => { try { await navigator.clipboard.writeText(turn.content); setCopyStatus('chat.transcript.copied'); } catch { setCopyStatus('chat.transcript.copy_failed'); } })(); }}>{t(locale, 'chat.transcript.copy')}</Button><span role="status">{copyStatus === null ? '' : t(locale, copyStatus)}</span></div>
    <details><summary>{t(locale, 'chat.transcript.details')}</summary><dl className="chat-metrics"><dt>{t(locale, 'chat.transcript.finish_reason')}</dt><dd>{turn.finishReason ?? unknown}</dd><dt>{t(locale, 'chat.transcript.usage')}</dt><dd>{turn.usage ? t(locale, 'chat.transcript.usage_value', { prompt: String(turn.usage.prompt_tokens), completion: String(turn.usage.completion_tokens), total: String(turn.usage.total_tokens) }) : unknown}</dd><dt>{t(locale, 'chat.transcript.first_delta')}</dt><dd>{turn.ttftMs === null ? unknown : `${Math.round(turn.ttftMs)} ms`}</dd><dt>{t(locale, 'chat.transcript.decode_rate')}</dt><dd>{rate === null ? unknown : `${rate.toFixed(1)} tokens/s`}</dd></dl><p>{t(locale, 'chat.transcript.metrics_note')}</p><pre>{JSON.stringify(turn.parameters, null, 2)}</pre></details>
  </article>;
});

export function Transcript({ turns, onEdit, busy, locale }: { turns: readonly ChatTurn[]; onEdit: (index: number) => void; busy: boolean; locale: Locale }): React.JSX.Element {
  const editRef = useRef(onEdit); editRef.current = onEdit;
  const editAt = useCallback((index: number) => editRef.current(index), []);
  const scroll = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const last = turns.at(-1);
  useEffect(() => {
    if (following.current && scroll.current && window.getSelection()?.isCollapsed !== false) scroll.current.scrollTop = scroll.current.scrollHeight;
  }, [last?.content, last?.reasoning, turns.length]);
  return <section aria-label={t(locale, 'chat.transcript.label')}><div className="chat-transcript" ref={scroll} tabIndex={0} onScroll={() => {
    const node = scroll.current;
    if (!node) return;
    following.current = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
    setShowJump(!following.current);
  }}>{turns.length ? turns.map((turn, index) => <Turn key={turn.id} turn={turn} index={index} busy={busy} locale={locale} onEdit={editAt} />) : <p>{t(locale, 'chat.transcript.empty')}</p>}</div>{showJump ? <Button onClick={() => { following.current = true; if (scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight; setShowJump(false); }}>{t(locale, 'chat.transcript.jump')}</Button> : null}</section>;
}
