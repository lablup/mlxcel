// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { useEffect, useRef, useState } from 'react';
import { Button, ErrorBanner } from '../../design-system/primitives';
import { loadLocalImages, type MediaImageLimits } from './images';
import { createHistoryRepository, exportConversations, importConversations, type ChatConversation } from './history';

export function HistoryControls({ conversations, busy, onPending, limits, onReplace }: { conversations: ChatConversation[]; busy: boolean; onPending: (pending: boolean) => void; limits: MediaImageLimits | undefined; onReplace: (next: ChatConversation[]) => void }): React.JSX.Element {
  const repository = useRef(createHistoryRepository());
  const [pending, setPending] = useState(false);
  const busyRef = useRef(busy); busyRef.current = busy;
  const setPendingState = (value: boolean): void => { setPending(value); onPending(value); };
  const [enabled, setEnabled] = useState(false);
  const [includeImages, setIncludeImages] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const epoch = useRef(0);
  useEffect(() => {
    if (!enabled) return;
    const timer = setTimeout(() => { void repository.current.save(conversations, { includeImages }).catch(() => setMessage('History was not saved. Storage may be full or unavailable; export your conversation.')); }, 300);
    return () => clearTimeout(timer);
  }, [conversations, enabled, includeImages]);
  useEffect(() => { const repo = repository.current; return () => { epoch.current++; repo.close(); onPending(false); }; }, []);
  const toggle = async (checked: boolean): Promise<void> => {
    const operation = ++epoch.current;
    repository.current.setEnabled(checked);
    if (!checked) { setEnabled(false); return; }
    setPendingState(true);
    try {
      const saved = await repository.current.load({ includeImages });
      await validateImages(saved, limits);
      if (operation !== epoch.current || busyRef.current) return;
      if (saved.length && conversations.length && !window.confirm('Replace the in-memory conversations with saved history? Export current conversations first if needed.')) { repository.current.setEnabled(false); return; }
      if (saved.length) onReplace(saved);
      setEnabled(true);
    } catch { repository.current.setEnabled(false); if (operation === epoch.current) setMessage('Local history could not be opened or its images failed validation. Nothing was saved.'); } finally { if (operation === epoch.current) setPendingState(false); }
  };
  return <details className="chat-privacy"><summary>Local history and privacy</summary><p>Conversations stay in memory unless you opt in below. Signing out or refreshing requires authentication again; saved history belongs to this browser origin, not a server account. No API keys are stored.</p>
    <label className="toggle"><input type="checkbox" checked={enabled} disabled={busy || pending} onChange={(event) => { void toggle(event.target.checked); }} />Save conversations on this device (IndexedDB)</label>
    <label className="toggle"><input type="checkbox" checked={includeImages} disabled={busy || pending} onChange={(event) => setIncludeImages(event.target.checked)} />Also include attached image bytes in saved history and exports (separate consent)</label>
    <p>Turning saving off stops future writes; use Clear All to remove previously saved history. Re-enable explicitly after refresh to read it.</p>
    <div className="chat-toolbar"><Button disabled={busy || pending} onClick={() => {
      try {
        const blob = new Blob([exportConversations(conversations, { includeImages })], { type: 'application/json' });
        const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'mlxcel-conversations.json'; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      } catch { setMessage('Export exceeds the bounded history size or contains unsupported data.'); }
    }}>Export JSON</Button><Button disabled={busy || pending} onClick={() => {
      if (!window.confirm('Delete all in-memory and saved conversations for this browser origin?')) return;
      const operation = ++epoch.current; setPendingState(true); setEnabled(false); setIncludeImages(false); repository.current.setEnabled(false);
      void repository.current.clear().then(() => { if (operation === epoch.current && !busyRef.current) { onReplace([]); setMessage('All history cleared.'); } }).catch(() => { if (operation === epoch.current) setMessage('Stored history could not be cleared. Check browser storage permissions; no success is claimed.'); }).finally(() => { if (operation === epoch.current) setPendingState(false); });
    }}>Clear All</Button></div>
    <label className="ds-field">Import conversation JSON (replaces current history)<input type="file" accept="application/json,.json" disabled={busy || pending} onChange={(event) => {
      const file = event.target.files?.[0]; event.target.value = '';
      if (!file) return;
      if (file.size > 16 * 1024 * 1024) { setMessage('Import exceeds 16 MiB.'); return; }
      const operation = ++epoch.current; setPendingState(true);
      void file.text().then(async (text) => {
        const imported = importConversations(text, { includeImages });
        await validateImages(imported, limits);
        if (operation !== epoch.current || busyRef.current) return;
        if (window.confirm('Replace the current conversations with validated imported history?')) onReplace(imported);
      }).catch(() => { if (operation === epoch.current) setMessage('Import rejected: unsupported version, invalid data or exceeded limits.'); }).finally(() => { if (operation === epoch.current) setPendingState(false); });
    }} /></label>
    {message ? <ErrorBanner tone="info" title="Local history" body={message} /> : null}
  </details>;
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
