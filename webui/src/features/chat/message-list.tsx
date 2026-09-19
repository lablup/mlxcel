// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
// The conversation's messages: each turn is a user message (inline-end) and an assistant
// message (inline-start), in the one scrolling region of the conversation pane.
import React, { memo, useCallback, useEffect, useId, useRef, useState } from 'react';
import { Badge, Button, IconButton } from '../../design-system/primitives';
import { t, type Locale, type StringKey } from '../../i18n/catalog';
import type { ChatTurn } from './history';
import { SafeMarkdown } from './markdown';
import { decodeRate } from './stream';
import { formatClock, isoTime, useNow } from './time';

// Status copy never prints the enum value itself; see the labels in strings.ts.
const STATUS_KEYS: Record<ChatTurn['status'], StringKey> = {
  streaming: 'chat.transcript.status.streaming',
  complete: 'chat.transcript.status.complete',
  cancelled: 'chat.transcript.status.cancelled',
  interrupted: 'chat.transcript.status.interrupted',
  error: 'chat.transcript.status.error',
};

/** Retry is offered only on a conversation's last turn, and only when it did not finish cleanly. */
export function canRetry(turn: ChatTurn, last: boolean): boolean {
  return last && (turn.status === 'error' || turn.status === 'interrupted');
}

function Clock({ at, locale }: { at: number; locale: Locale }): React.JSX.Element | null {
  // startedAt 0 marks a turn imported from a version-1 payload, whose time is unknown.
  return at > 0 ? <time className="chat-time" dateTime={isoTime(at)}>{formatClock(at, locale)}</time> : null;
}

// The only per-second timer in Chat, mounted while its turn streams.
function Elapsed({ startedAt, locale }: { startedAt: number; locale: Locale }): React.JSX.Element {
  const now = useNow(1000);
  const seconds = startedAt > 0 ? Math.max(0, Math.floor((now - startedAt) / 1000)) : 0;
  return <Badge>{t(locale, 'chat.message.elapsed', { status: t(locale, STATUS_KEYS.streaming), seconds: String(seconds) })}</Badge>;
}

function useCopy(): [StringKey | null, (text: string, done: StringKey, failed: StringKey) => void] {
  const [status, setStatus] = useState<StringKey | null>(null);
  const copy = (text: string, done: StringKey, failed: StringKey): void => {
    void (async () => { try { await navigator.clipboard.writeText(text); setStatus(done); } catch { setStatus(failed); } })();
  };
  return [status, copy];
}

function UserMessage({ turn, index, busy, locale, onEdit }: { turn: ChatTurn; index: number; busy: boolean; locale: Locale; onEdit: (index: number) => void }): React.JSX.Element {
  const [copyStatus, copy] = useCopy();
  return <div className="chat-message chat-message--user" data-role="user">
    <div className="chat-message-meta"><span className="chat-role">{t(locale, 'chat.message.you')}</span><Clock at={turn.startedAt} locale={locale} /></div>
    <div className="chat-bubble chat-bubble--user">
      <p className="chat-prompt">{turn.prompt}</p>
      {turn.images.length > 0 ? <p className="chat-attachments">{t(locale, 'chat.transcript.images', { count: String(turn.images.length) })}</p> : null}
    </div>
    <div className="chat-message-tools">
      <div className="chat-message-actions" role="group" aria-label={t(locale, 'chat.message.prompt_actions')}>
        <IconButton label={t(locale, 'chat.message.copy_prompt')} icon="copy" onClick={() => copy(turn.prompt, 'chat.message.copied_prompt', 'chat.message.copy_prompt_failed')} />
        <IconButton label={t(locale, 'chat.transcript.edit')} icon="edit" disabled={busy} onClick={() => onEdit(index)} />
      </div>
      <span className="chat-copy-status" role="status">{copyStatus === null ? '' : t(locale, copyStatus)}</span>
    </div>
  </div>;
}

function AssistantMessage({ turn, last, busy, locale, onRetry }: { turn: ChatTurn; last: boolean; busy: boolean; locale: Locale; onRetry: (turnId: string) => void }): React.JSX.Element {
  const [copyStatus, copy] = useCopy();
  const [details, setDetails] = useState(false);
  const detailsId = useId();
  const streaming = turn.status === 'streaming';
  const rate = decodeRate(turn);
  const unknown = t(locale, 'chat.transcript.unknown');
  const finishedAt = turn.startedAt > 0 && turn.elapsedMs !== null ? turn.startedAt + Math.round(turn.elapsedMs) : turn.startedAt;
  return <div className="chat-message chat-message--assistant" data-role="assistant">
    <div className="chat-message-meta">
      <Badge tone="accent">{turn.modelName}</Badge>
      <span className="chat-status" data-status={turn.status}>{streaming ? <Elapsed startedAt={turn.startedAt} locale={locale} /> : <Badge>{t(locale, STATUS_KEYS[turn.status])}</Badge>}</span>
      <Clock at={finishedAt} locale={locale} />
    </div>
    <div className="chat-bubble chat-bubble--assistant">
      {turn.reasoning ? <details className="chat-reasoning"><summary>{streaming && !turn.content ? t(locale, 'chat.transcript.reasoning_thinking') : t(locale, 'chat.reasoning')}</summary><p className="chat-reasoning-text">{turn.reasoning}</p></details> : null}
      {streaming && !turn.content ? <p className="chat-waiting">{turn.reasoning ? t(locale, 'chat.transcript.thinking') : t(locale, 'chat.transcript.waiting')}</p> : null}
      <SafeMarkdown text={turn.content} streaming={streaming} />
      {turn.tools.map((tool) => {
        const name = tool.name || t(locale, 'chat.transcript.tool_name_pending');
        return <div key={tool.index} className="chat-tool" role="group" aria-label={t(locale, 'chat.transcript.tool_call', { name })}>
          <div className="chat-tool-head"><span className="chat-tool-label">{t(locale, 'chat.tool_call')}</span><code>{name}</code><Badge>{t(locale, 'chat.message.not_executed')}</Badge></div>
          <pre>{tool.arguments}</pre>
        </div>;
      })}
      {turn.error ? <p className="chat-turn-error" role="alert">{turn.error}</p> : null}
    </div>
    <div className="chat-message-tools">
      <div className="chat-message-actions" role="group" aria-label={t(locale, 'chat.message.response_actions')}>
        <IconButton label={t(locale, 'chat.transcript.copy')} icon="copy" onClick={() => copy(turn.content, 'chat.transcript.copied', 'chat.transcript.copy_failed')} />
        {canRetry(turn, last) ? <IconButton label={t(locale, 'chat.message.retry')} icon="retry" disabled={busy} onClick={() => onRetry(turn.id)} /> : null}
        <IconButton label={t(locale, 'chat.transcript.details')} icon="details" aria-expanded={details} aria-controls={detailsId} onClick={() => setDetails(!details)} />
      </div>
      <span className="chat-copy-status" role="status">{copyStatus === null ? '' : t(locale, copyStatus)}</span>
    </div>
    <div className="chat-details" id={detailsId} hidden={!details}>
      <dl className="chat-metrics"><dt>{t(locale, 'chat.transcript.finish_reason')}</dt><dd>{turn.finishReason ?? unknown}</dd><dt>{t(locale, 'chat.transcript.usage')}</dt><dd>{turn.usage ? t(locale, 'chat.transcript.usage_value', { prompt: String(turn.usage.prompt_tokens), completion: String(turn.usage.completion_tokens), total: String(turn.usage.total_tokens) }) : unknown}</dd><dt>{t(locale, 'chat.transcript.first_delta')}</dt><dd>{turn.ttftMs === null ? unknown : `${Math.round(turn.ttftMs)} ms`}</dd><dt>{t(locale, 'chat.transcript.decode_rate')}</dt><dd>{rate === null ? unknown : `${rate.toFixed(1)} tokens/s`}</dd></dl>
      <p>{t(locale, 'chat.transcript.metrics_note')}</p>
      <pre>{JSON.stringify(turn.parameters, null, 2)}</pre>
    </div>
  </div>;
}

// Historical turns are immutable, so memo keeps them out of every 50 ms stream flush.
const Turn = memo(function Turn({ turn, index, last, busy, locale, onEdit, onRetry }: { turn: ChatTurn; index: number; last: boolean; busy: boolean; locale: Locale; onEdit: (index: number) => void; onRetry: (turnId: string) => void }): React.JSX.Element {
  return <article className="chat-turn" aria-label={t(locale, 'chat.transcript.turn_label', { model: turn.modelName })}>
    <UserMessage turn={turn} index={index} busy={busy} locale={locale} onEdit={onEdit} />
    <AssistantMessage turn={turn} last={last} busy={busy} locale={locale} onRetry={onRetry} />
  </article>;
});

export function MessageList({ turns, onEdit, onRetry, busy, locale }: { turns: readonly ChatTurn[]; onEdit: (index: number) => void; onRetry: (turnId: string) => void; busy: boolean; locale: Locale }): React.JSX.Element {
  // Stable callbacks, so a parent re-render does not re-render every memoized turn.
  const editRef = useRef(onEdit); editRef.current = onEdit;
  const retryRef = useRef(onRetry); retryRef.current = onRetry;
  const editAt = useCallback((index: number) => editRef.current(index), []);
  const retryAt = useCallback((turnId: string) => retryRef.current(turnId), []);
  const scroll = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const last = turns.at(-1);
  // Follow new output only while the reader is at the end and not selecting text.
  useEffect(() => {
    if (following.current && scroll.current && window.getSelection()?.isCollapsed !== false) scroll.current.scrollTop = scroll.current.scrollHeight;
  }, [last?.content, last?.reasoning, turns.length]);
  return <section className="chat-transcript" aria-label={t(locale, 'chat.transcript.label')}>
    <div className="chat-messages" ref={scroll} tabIndex={0} onScroll={() => {
      const node = scroll.current;
      if (!node) return;
      following.current = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
      setShowJump(!following.current);
    }}>{turns.length ? turns.map((turn, index) => <Turn key={turn.id} turn={turn} index={index} last={index === turns.length - 1} busy={busy} locale={locale} onEdit={editAt} onRetry={retryAt} />) : <p className="chat-empty">{t(locale, 'chat.transcript.empty')}</p>}</div>
    {showJump ? <Button className="chat-jump" onClick={() => { following.current = true; if (scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight; setShowJump(false); }}>{t(locale, 'chat.transcript.jump')}</Button> : null}
  </section>;
}
