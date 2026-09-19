// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { useEffect, useRef, useState } from 'react';
import { Button, ConfirmDialog, ErrorBanner } from '../../design-system/primitives';
import { t, type Locale, type StringKey } from '../../i18n/catalog';
import { loadLocalImages, type MediaImageLimits } from './images';
import { createHistoryRepository, exportConversations, importConversations, type ChatConversation } from './history';

// The async flows hold their trigger disabled while pending, so the dialog cannot
// capture it for focus restoration; each such confirmation carries it explicitly.
type Confirmation =
  | { kind: 'replace-saved'; saved: ChatConversation[]; operation: number; trigger: HTMLElement }
  | { kind: 'clear' }
  | { kind: 'replace-import'; imported: ChatConversation[]; operation: number; trigger: HTMLElement };

export function HistoryControls({ conversations, busy, onPending, limits, onReplace, locale }: { conversations: ChatConversation[]; busy: boolean; onPending: (pending: boolean) => void; limits: MediaImageLimits | undefined; onReplace: (next: ChatConversation[]) => void; locale: Locale }): React.JSX.Element {
  const repository = useRef(createHistoryRepository());
  const [pending, setPending] = useState(false);
  const busyRef = useRef(busy); busyRef.current = busy;
  const setPendingState = (value: boolean): void => { setPending(value); onPending(value); };
  const [enabled, setEnabled] = useState(false);
  const [includeImages, setIncludeImages] = useState(false);
  const [message, setMessage] = useState<StringKey | null>(null);
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const confirmationRef = useRef<Confirmation | null>(null);
  const ask = (next: Confirmation): void => { confirmationRef.current = next; setConfirmation(next); };
  const clearButtonRef = useRef<HTMLButtonElement>(null);
  const epoch = useRef(0);
  useEffect(() => {
    if (!enabled) return;
    const timer = setTimeout(() => { void repository.current.save(conversations, { includeImages }).catch(() => setMessage('chat.privacy.save_failed')); }, 300);
    return () => clearTimeout(timer);
  }, [conversations, enabled, includeImages]);
  useEffect(() => { const repo = repository.current; return () => { epoch.current++; repo.close(); onPending(false); }; }, []);
  const toggle = async (checked: boolean, trigger: HTMLElement): Promise<void> => {
    const operation = ++epoch.current;
    repository.current.setEnabled(checked);
    if (!checked) { setEnabled(false); return; }
    setPendingState(true);
    try {
      const saved = await repository.current.load({ includeImages });
      await validateImages(saved, limits);
      if (operation !== epoch.current || busyRef.current) return;
      if (saved.length && conversations.length) { ask({ kind: 'replace-saved', saved, operation, trigger }); return; }
      if (saved.length) onReplace(saved);
      setEnabled(true);
    } catch { repository.current.setEnabled(false); if (operation === epoch.current) setMessage('chat.privacy.open_failed'); } finally { if (operation === epoch.current) setPendingState(false); }
  };
  const clearAll = (): void => {
    const operation = ++epoch.current; setPendingState(true); setEnabled(false); setIncludeImages(false); repository.current.setEnabled(false);
    void repository.current.clear().then(() => { if (operation === epoch.current && !busyRef.current) { onReplace([]); setMessage('chat.privacy.cleared'); } }).catch(() => { if (operation === epoch.current) setMessage('chat.privacy.clear_failed'); }).finally(() => {
      if (operation !== epoch.current) return;
      setPendingState(false);
      // The dialog's own focus restoration may have found nothing usable while every
      // Chat control was disabled for clearing; once the button re-enables, claim it back.
      window.setTimeout(() => { if (document.activeElement === document.body) clearButtonRef.current?.focus(); }, 0);
    });
  };
  // Every answer closes the dialog first and is consumed once; the operation then runs
  // only if no newer operation (or a started generation) has made the snapshot stale.
  const answer = (confirmed: boolean): void => {
    const current = confirmationRef.current;
    confirmationRef.current = null; setConfirmation(null);
    if (current === null) return;
    if (current.kind === 'clear') { if (confirmed && !busyRef.current) clearAll(); return; }
    if (current.operation !== epoch.current) return;
    if (current.kind === 'replace-saved') {
      if (!confirmed) repository.current.setEnabled(false);
      else if (!busyRef.current) {
        try { onReplace(current.saved); setEnabled(true); }
        catch { repository.current.setEnabled(false); setMessage('chat.privacy.open_failed'); }
      }
    } else if (confirmed && !busyRef.current) {
      try { onReplace(current.imported); }
      catch { setMessage('chat.privacy.import_rejected'); }
    }
    const target = current.trigger;
    window.setTimeout(() => { const active = document.activeElement; if (target.isConnected && !target.matches(':disabled') && (active === null || active === document.body)) target.focus(); }, 0);
  };
  const dialog = (kind: Confirmation['kind'], title: StringKey, body: StringKey, confirmLabel: StringKey, testId: string): React.JSX.Element | null => confirmation?.kind === kind
    ? <ConfirmDialog open title={t(locale, title)} body={t(locale, body)} confirmLabel={t(locale, confirmLabel)} cancelLabel={t(locale, 'common.cancel')} closeLabel={t(locale, 'common.close')} tone="danger" testId={testId} onConfirm={() => answer(true)} onClose={() => answer(false)} />
    : null;
  // Dialogs render outside <details>: a collapsed disclosure must never hide an open modal.
  return <><details className="chat-privacy"><summary>{t(locale, 'chat.privacy.summary')}</summary><p>{t(locale, 'chat.privacy.body')}</p>
    <label className="toggle"><input type="checkbox" checked={enabled} disabled={busy || pending} onChange={(event) => { void toggle(event.target.checked, event.currentTarget); }} />{t(locale, 'chat.privacy.save')}</label>
    <label className="toggle"><input type="checkbox" checked={includeImages} disabled={busy || pending} onChange={(event) => setIncludeImages(event.target.checked)} />{t(locale, 'chat.privacy.include_images')}</label>
    <p>{t(locale, 'chat.privacy.note')}</p>
    <div className="chat-toolbar"><Button disabled={busy || pending} onClick={() => {
      try {
        const blob = new Blob([exportConversations(conversations, { includeImages })], { type: 'application/json' });
        const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'mlxcel-conversations.json'; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      } catch { setMessage('chat.privacy.export_failed'); }
    }}>{t(locale, 'chat.privacy.export')}</Button><Button ref={clearButtonRef} disabled={busy || pending} onClick={() => ask({ kind: 'clear' })}>{t(locale, 'chat.privacy.clear')}</Button></div>
    <label className="ds-field">{t(locale, 'chat.privacy.import')}<input type="file" accept="application/json,.json" disabled={busy || pending} onChange={(event) => {
      const trigger = event.currentTarget;
      const file = event.target.files?.[0]; event.target.value = '';
      if (!file) return;
      if (file.size > 16 * 1024 * 1024) { setMessage('chat.privacy.import_too_large'); return; }
      const operation = ++epoch.current; setPendingState(true);
      void file.text().then(async (text) => {
        const imported = importConversations(text, { includeImages });
        await validateImages(imported, limits);
        if (operation !== epoch.current || busyRef.current) return;
        ask({ kind: 'replace-import', imported, operation, trigger });
      }).catch(() => { if (operation === epoch.current) setMessage('chat.privacy.import_rejected'); }).finally(() => { if (operation === epoch.current) setPendingState(false); });
    }} /></label>
    {message ? <ErrorBanner tone="info" title={t(locale, 'chat.privacy.title')} body={t(locale, message)} /> : null}
  </details>
    {dialog('replace-saved', 'chat.privacy.replace_saved.confirm.title', 'chat.privacy.replace_saved.confirm.body', 'chat.privacy.replace_saved.confirm', 'chat-replace-saved-dialog')}
    {dialog('clear', 'chat.privacy.clear.confirm.title', 'settings.clear_history.confirm.body', 'chat.privacy.clear.confirm', 'chat-clear-history-dialog')}
    {dialog('replace-import', 'chat.privacy.replace_import.confirm.title', 'chat.privacy.replace_import.confirm.body', 'chat.privacy.replace_import.confirm', 'chat-replace-import-dialog')}
  </>;
}

async function validateImages(conversations: ChatConversation[], limits: MediaImageLimits | undefined): Promise<void> {
  for (const conversation of conversations) for (const turn of conversation.turns) {
    if (!turn.images.length) continue;
    if (!limits) throw new Error('Server image limits unavailable.');
    const files = turn.images.map((image) => {
      const binary = atob(image.dataUrl.slice(image.dataUrl.indexOf(',') + 1));
      const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
      return new File([bytes], image.name, { type: image.type });
    });
    turn.images = await loadLocalImages(files, 0, limits);
  }
}
