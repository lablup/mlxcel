// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
// The pinned composer: attach, message, Send or Stop, then one helper line. Chat owns the
// draft, the images, the keyboard handling and every send guard; this is the markup.
import React, { useId, useLayoutEffect, useRef, useState } from 'react';
import { Button, IconButton, Tooltip } from '../../design-system/primitives';
import { localeTag } from '../../design-system/format';
import { t, type Locale } from '../../i18n/catalog';
import type { ChatTurn } from './history';
import { MAX_PROMPT_CHARACTERS } from './stream';

const MIN_ROWS = 1;
const MAX_ROWS = 8;

export interface ComposerProps {
  readonly locale: Locale;
  readonly textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  readonly draft: string;
  readonly onDraftChange: (value: string) => void;
  readonly images: ChatTurn['images'];
  readonly onRemoveImage: (index: number) => void;
  readonly canImage: boolean;
  readonly onAddImages: (files: FileList) => void;
  /** A request, history operation or image load is running. */
  readonly disabled: boolean;
  /** A request is running: Stop replaces Send. */
  readonly running: boolean;
  readonly canSend: boolean;
  readonly onSend: () => void;
  readonly onStop: () => void;
  readonly onKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  readonly onCompositionStart: () => void;
  readonly onCompositionEnd: () => void;
}

// Grow from one to eight rows with the content, through the rows attribute rather than
// an inline height (CSP). Beyond eight rows the field scrolls internally.
function useAutoRows(field: React.RefObject<HTMLTextAreaElement | null>, value: string): number {
  const [rows, setRows] = useState(MIN_ROWS);
  useLayoutEffect(() => {
    const node = field.current;
    if (!node) return;
    const style = window.getComputedStyle(node);
    const line = Number.parseFloat(style.lineHeight);
    const padding = Number.parseFloat(style.paddingTop) + Number.parseFloat(style.paddingBottom);
    if (!Number.isFinite(line) || line <= 0 || !Number.isFinite(padding)) return;
    // Measure at one row, so scrollHeight reflects the content rather than the current rows.
    const current = node.rows;
    node.rows = MIN_ROWS;
    // scrollHeight is rounded to whole pixels, so round rather than ceil (2 lines of 22.4px read as 45px).
    const needed = Math.round((node.scrollHeight - padding) / line);
    node.rows = current;
    setRows(Math.min(MAX_ROWS, Math.max(MIN_ROWS, needed)));
  }, [field, value]);
  return rows;
}

export function Composer(props: ComposerProps): React.JSX.Element {
  const { locale } = props;
  const fileRef = useRef<HTMLInputElement>(null);
  const countId = useId();
  const rows = useAutoRows(props.textareaRef, props.draft);
  const full = props.draft.length >= MAX_PROMPT_CHARACTERS;
  const number = new Intl.NumberFormat(localeTag(locale));
  return <div className="chat-composer">
    {props.images.length ? <div className="chat-images">{props.images.map((image, index) => <figure key={`${image.name}-${index}`}><img src={image.dataUrl} alt={image.name} /><Button disabled={props.disabled} onClick={() => props.onRemoveImage(index)}>{t(locale, 'chat.images.remove', { name: image.name })}</Button></figure>)}</div> : null}
    <div className="chat-composer-row">
      {props.canImage ? <>
        <IconButton label={t(locale, 'chat.images.attach')} icon="attach" disabled={props.disabled} onClick={() => fileRef.current?.click()} />
        <input ref={fileRef} className="chat-file-input" type="file" aria-label={t(locale, 'chat.images.label')} accept="image/png,image/jpeg,image/webp" multiple disabled={props.disabled} onChange={(event) => {
          const files = event.target.files;
          if (files) props.onAddImages(files);
          event.target.value = '';
        }} />
      </> : null}
      <textarea ref={props.textareaRef} rows={rows} aria-label={t(locale, 'chat.composer.label')} aria-describedby={countId} value={props.draft} maxLength={MAX_PROMPT_CHARACTERS} disabled={props.disabled} onChange={(event) => props.onDraftChange(event.target.value)} onCompositionStart={props.onCompositionStart} onCompositionEnd={props.onCompositionEnd} onKeyDown={props.onKeyDown} />
      {props.running
        ? <Button onClick={props.onStop}>{t(locale, 'chat.stop')}</Button>
        : <Button tone="primary" disabled={!props.canSend} onClick={props.onSend}>{t(locale, 'common.send')}</Button>}
    </div>
    <div className="chat-composer-help">
      <span className="chat-count" id={countId} data-full={full || undefined}>{t(locale, 'chat.composer.count', { count: number.format(props.draft.length), max: number.format(MAX_PROMPT_CHARACTERS) })}{full ? ` · ${t(locale, 'chat.composer.limit')}` : ''}</span>
      <Tooltip content={t(locale, 'chat.composer.hint')}><IconButton label={t(locale, 'chat.composer.keys')} icon="info" /></Tooltip>
    </div>
  </div>;
}
