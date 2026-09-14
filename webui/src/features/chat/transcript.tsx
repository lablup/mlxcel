// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '../../design-system/primitives';
import type { ChatTurn } from './history';
import { SafeMarkdown } from './markdown';
import { decodeRate } from './stream';

const Turn = memo(function Turn({ turn, index, onEdit, busy }: { turn: ChatTurn; index: number; onEdit: (index: number) => void; busy: boolean }): React.JSX.Element {
  const [copyStatus, setCopyStatus] = useState('');
  const rate = decodeRate(turn);
  return <article className="chat-turn" aria-label={`Turn with ${turn.modelName}`}>
    <header><h2>{turn.modelName}</h2><span>{turn.status}</span></header>
    <p className="chat-prompt">{turn.prompt}</p>
    {turn.images.length > 0 ? <p>{turn.images.length} local image attachment(s)</p> : null}
    <Button disabled={busy} onClick={() => onEdit(index)}>Edit and regenerate</Button>
    {turn.reasoning ? <details><summary>Reasoning{turn.status === 'streaming' && !turn.content ? ' · Thinking' : ''}</summary><pre>{turn.reasoning}</pre></details> : null}
    {turn.status === 'streaming' && !turn.content ? <p>{turn.reasoning ? 'Thinking…' : 'Waiting for response…'}</p> : null}
    <SafeMarkdown text={turn.content} streaming={turn.status === 'streaming'} />
    {turn.tools.map((tool) => <details key={tool.index}><summary>Tool call: {tool.name || 'Name pending'} (not executed)</summary><pre>{tool.arguments}</pre></details>)}
    {turn.error ? <p role="alert">{turn.error}</p> : null}
    <div className="chat-toolbar"><Button onClick={() => { void (async () => { try { await navigator.clipboard.writeText(turn.content); setCopyStatus('Copied response.'); } catch { setCopyStatus('Copy unavailable; select the response text instead.'); } })(); }}>Copy response</Button><span role="status">{copyStatus}</span></div>
    <details><summary>Response details</summary><dl className="chat-metrics"><dt>Finish reason</dt><dd>{turn.finishReason ?? 'Unknown'}</dd><dt>Usage (server reported)</dt><dd>{turn.usage ? `${turn.usage.prompt_tokens} prompt / ${turn.usage.completion_tokens} completion / ${turn.usage.total_tokens} total tokens` : 'Unknown'}</dd><dt>First delta (client observed)</dt><dd>{turn.ttftMs === null ? 'Unknown' : `${Math.round(turn.ttftMs)} ms`}</dd><dt>Decode rate (client estimate)</dt><dd>{rate === null ? 'Unknown' : `${rate.toFixed(1)} tokens/s`}</dd></dl><p>First delta measures send to first non-empty content, reasoning or tool delta, including transport. Decode rate uses reported completion tokens minus one over the remaining client-observed stream duration; it is not a server kernel metric.</p><pre>{JSON.stringify(turn.parameters, null, 2)}</pre></details>
  </article>;
});

export function Transcript({ turns, onEdit, busy }: { turns: readonly ChatTurn[]; onEdit: (index: number) => void; busy: boolean }): React.JSX.Element {
  const editRef = useRef(onEdit); editRef.current = onEdit;
  const editAt = useCallback((index: number) => editRef.current(index), []);
  const scroll = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const last = turns.at(-1);
  useEffect(() => {
    if (following.current && scroll.current && window.getSelection()?.isCollapsed !== false) scroll.current.scrollTop = scroll.current.scrollHeight;
  }, [last?.content, last?.reasoning, turns.length]);
  return <section aria-label="Conversation transcript"><div className="chat-transcript" ref={scroll} tabIndex={0} onScroll={() => {
    const node = scroll.current;
    if (!node) return;
    following.current = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
    setShowJump(!following.current);
  }}>{turns.length ? turns.map((turn, index) => <Turn key={turn.id} turn={turn} index={index} busy={busy} onEdit={editAt} />) : <p>No messages yet. Start with a prompt after loading a model.</p>}</div>{showJump ? <Button onClick={() => { following.current = true; if (scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight; setShowJump(false); }}>Jump to latest</Button> : null}</section>;
}
