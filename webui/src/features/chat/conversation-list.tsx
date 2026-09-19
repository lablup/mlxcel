// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
// The conversation list: one row per conversation, newest activity first. Selecting,
// renaming and asking to delete are callbacks; Chat owns the state and the guards.
import React, { memo, useCallback, useMemo, useRef, useState } from 'react';
import { EmptyState, IconButton } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import type { ChatConversation } from './history';
import { formatRelative, isoTime, useNow } from './time';

/** The longest title a conversation keeps, here and in the settings drawer. */
export const TITLE_LIMIT = 120;
// Relative times are coarse (minutes and up), so a half-minute refresh is enough.
const REFRESH_MS = 30_000;

export interface ConversationListProps {
  readonly locale: Locale;
  readonly conversations: ReadonlyArray<ChatConversation>;
  readonly currentId: string | null;
  /** A request, history operation or image load is running: rows and actions are disabled. */
  readonly disabled: boolean;
  readonly atLimit: boolean;
  readonly onSelect: (id: string) => void;
  readonly onRename: (id: string, title: string) => void;
  readonly onDelete: (id: string) => void;
  readonly listRef?: React.Ref<HTMLUListElement>;
}

// Memoized with stable callbacks: a stream flush replaces only the streaming conversation,
// so the other rows skip the re-render.
const Row = memo(function Row({ item, current, disabled, now, locale, onSelect, onRename, onDelete }: { item: ChatConversation; current: boolean; disabled: boolean; now: number; locale: Locale; onSelect: (id: string) => void; onRename: (id: string, title: string) => void; onDelete: (id: string) => void }): React.JSX.Element {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState('');
  // The edit ends once, whichever of Enter, Escape or blur comes first; a ref, because the
  // blur a browser may fire while React removes the field still sees `editing` as true.
  const open = useRef(false);
  const composing = useRef(false);
  const selectRef = useRef<HTMLButtonElement>(null);
  const last = item.turns.at(-1);
  const start = (): void => { open.current = true; setValue(item.title); setEditing(true); };
  // Enter and blur commit, Escape cancels. Keyboard endings return focus to the row;
  // a blur leaves it wherever the pointer or Tab moved it.
  const finish = (commit: boolean, refocus: boolean): void => {
    if (!open.current) return;
    open.current = false;
    setEditing(false);
    if (commit) onRename(item.id, value);
    if (refocus) window.setTimeout(() => selectRef.current?.focus(), 0);
  };
  return <li className="chat-row" data-current={current || undefined}>
    {editing
      ? <input className="chat-row-rename" aria-label={t(locale, 'chat.list.rename_field', { title: item.title })} value={value} maxLength={TITLE_LIMIT} autoFocus
        onChange={(event) => setValue(event.currentTarget.value)}
        onCompositionStart={() => { composing.current = true; }} onCompositionEnd={() => { composing.current = false; }}
        onBlur={() => finish(true, false)}
        // Capture phase: an Escape that cancels the rename must not also reach the list
        // drawer's own Escape handler (a native listener on the drawer panel) and close it.
        onKeyDownCapture={(event) => {
          if (event.key !== 'Escape' && event.key !== 'Enter') return;
          if (composing.current || event.nativeEvent.isComposing || event.keyCode === 229) return;
          event.preventDefault();
          event.stopPropagation();
          finish(event.key === 'Enter', true);
        }} />
      : <button ref={selectRef} type="button" className="chat-row-select" aria-current={current ? 'true' : undefined} disabled={disabled} onClick={() => onSelect(item.id)}>
        <span className="chat-row-title">{item.title}</span>
        <span className="chat-row-meta"><span>{last ? last.modelName : t(locale, 'chat.list.no_messages')}</span><time dateTime={isoTime(item.updatedAt)}>{formatRelative(item.updatedAt, now, locale)}</time></span>
      </button>}
    <div className="chat-row-actions">
      <IconButton label={t(locale, 'chat.list.rename', { title: item.title })} icon="edit" disabled={disabled || editing} onClick={start} />
      <IconButton label={t(locale, 'chat.list.delete', { title: item.title })} icon="trash" disabled={disabled} onClick={() => onDelete(item.id)} />
    </div>
  </li>;
});

// Up/Down and Home/End move between the row buttons; Tab still reaches every control.
function moveFocus(event: React.KeyboardEvent<HTMLUListElement>): void {
  if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key) || event.altKey || event.ctrlKey || event.metaKey || event.nativeEvent.isComposing) return;
  const rows = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('button.chat-row-select:not(:disabled)'));
  const index = rows.findIndex((row) => row === event.target);
  if (index === -1) return;
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? rows.length - 1 : Math.min(rows.length - 1, Math.max(0, index + (event.key === 'ArrowDown' ? 1 : -1)));
  event.preventDefault();
  rows[next]?.focus();
}

export function ConversationList(props: ConversationListProps): React.JSX.Element {
  const now = useNow(REFRESH_MS);
  const sorted = useMemo(() => [...props.conversations].sort((a, b) => b.updatedAt - a.updatedAt), [props.conversations]);
  const handlers = useRef(props); handlers.current = props;
  const onSelect = useCallback((id: string) => handlers.current.onSelect(id), []);
  const onRename = useCallback((id: string, title: string) => handlers.current.onRename(id, title), []);
  const onDelete = useCallback((id: string) => handlers.current.onDelete(id), []);
  return <div className="chat-list">
    {props.atLimit ? <p className="chat-list-limit" data-testid="chat-list-limit">{t(props.locale, 'chat.list.limit')}</p> : null}
    {sorted.length === 0
      ? <EmptyState title={t(props.locale, 'chat.list.empty.title')} body={t(props.locale, 'chat.list.empty.body')} testId="chat-list-empty" />
      : <ul className="chat-list-rows" aria-label={t(props.locale, 'chat.list.label')} ref={props.listRef} onKeyDown={moveFocus}>
        {sorted.map((item) => <Row key={item.id} item={item} current={item.id === props.currentId} disabled={props.disabled} now={now} locale={props.locale} onSelect={onSelect} onRename={onRename} onDelete={onDelete} />)}
      </ul>}
  </div>;
}
