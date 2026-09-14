// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { useEffect, useRef, useState } from 'react';
import { Button, ErrorBanner, Field, Select } from '../../design-system/primitives';
import { useWebUi, useWebUiActions } from '../../state';
import type { Locale } from '../../i18n/catalog';
import type { ChatConversation, ChatTurn } from './history';
import { loadLocalImages, validateRequestImages } from './images';
import { appendFrame, buildMessages, completeTurn, MAX_PROMPT_CHARACTERS } from './stream';
import { newConversation, replaceConversations, updateConversation, useConversations, sessionGeneration } from './session';
import { Transcript } from './transcript';
import { HistoryControls } from './privacy';
import './chat.css';

export function Chat({ locale }: { locale: Locale }): React.JSX.Element {
  const snapshot = useWebUi();
  const actions = useWebUiActions();
  // Shared Settings defaults are integrated after #1846 merges; empty means server defaults.
  const defaults: Record<string, number> = {};
  const conversations = useConversations();
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [images, setImages] = useState<ChatTurn['images']>([]);
  const [error, setError] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState('');
  const [busy, setBusy] = useState(false);
  const [historyBusy, setHistoryBusy] = useState(false);
  const [imagesBusy, setImagesBusy] = useState(false);
  const imageEpoch = useRef(0);
  const composing = useRef(false);
  const active = useRef<{ controller: AbortController; turn: ChatTurn; conversation: ChatConversation; flush: () => void } | null>(null);
  const current = conversations.find((entry) => entry.id === currentId) ?? null;
  const model = snapshot.catalog.find((entry) => entry.identity.id === snapshot.selectedModelId);
  const connected = ['ready', 'streaming', 'polling'].includes(snapshot.connection);
  const canChat = connected && model?.lifecycle.state === 'ready' && model.capabilities.some((cap) => cap.task === 'chat' && cap.phase === 'provider_ready' && cap.available);
  const canImage = canChat && model?.capabilities.some((cap) => cap.task === 'vision_input' && cap.phase === 'provider_ready' && cap.available);

  useEffect(() => () => {
    imageEpoch.current++;
    const request = active.current;
    if (request !== null) {
      request.turn = { ...request.turn, status: 'interrupted', error: 'View closed. The request was aborted and was not retried.' };
      request.controller.abort();
      request.flush();
    }
  }, []);

  const create = (): void => {
    if (busy || historyBusy || imagesBusy || conversations.length >= 50) return;
    imageEpoch.current++;
    const next = newConversation();
    updateConversation(next); setCurrentId(next.id); setDraft(''); setImages([]);
  };
  const stop = (): void => {
    const request = active.current;
    if (request === null) return;
    request.turn = { ...request.turn, status: 'cancelled', error: null };
    request.controller.abort(); request.flush();
    setAnnouncement('Stopped locally; checking server activity.');
    void actions.refreshRuntime(request.turn.modelId).then((runtime) => {
      setAnnouncement(`Request aborted. Server observation refreshed at ${runtime.measurements.active_requests?.measured_at ?? 'an unknown time'}. See Activity for active requests; this is not a per-request cancellation receipt.`);
    }).catch(() => setAnnouncement('Request aborted. Backend cancellation could not be observed; inspect Activity before unloading.'));
  };
  const send = async (): Promise<void> => {
    if (active.current !== null || historyBusy || imagesBusy || !canChat || model === undefined || !draft.trim() || draft.length > MAX_PROMPT_CHARACTERS) return;
    if ((images.length || current?.turns.some((turn) => turn.images.length)) && !canImage) { setError('The selected provider does not confirm vision support. Remove images or select a vision model.'); return; }
    if (current === null && conversations.length >= 50) { setError('Conversation limit reached. Delete or export older conversations first.'); return; }
    let conversation = current ?? newConversation();
    if (conversation.turns.length >= 100) { setError('This conversation reached its 100-turn limit. Start a new conversation.'); return; }
    setCurrentId(conversation.id);
    const controller = new AbortController();
    const started = performance.now();
    const turn: ChatTurn = { id: crypto.randomUUID(), modelId: model.identity.id, inferenceId: model.identity.inference_id, modelName: model.identity.display_name, modelRevision: model.identity.revision, prompt: draft, content: '', reasoning: '', tools: [], status: 'streaming', finishReason: null, usage: null, ttftMs: null, elapsedMs: null, error: null, parameters: { ...defaults }, images: images.map((image) => ({ ...image })) };
    conversation = { ...conversation, title: conversation.turns.length === 0 && conversation.title === 'New conversation' ? draft.slice(0, 80) : conversation.title, turns: [...conversation.turns, turn], updatedAt: Date.now() };
    try {
      if (!snapshot.bootstrap) throw new Error('Limits unavailable');
      validateRequestImages(conversation.turns.flatMap((item) => item.images), snapshot.bootstrap.media_limits);
    } catch { setError('Conversation images exceed the current server count, size or decode limits. Remove attachments or start a new conversation.'); return; }
    const body = { ...turn.parameters, model: turn.inferenceId, stream: true, messages: buildMessages(conversation.systemPrompt, conversation.turns), stream_options: { include_usage: true } };
    const bodyBudget = Math.min(snapshot.bootstrap?.media_limits.max_body_bytes ?? 0, 16 * 1024 * 1024);
    if (new TextEncoder().encode(JSON.stringify(body)).byteLength > bodyBudget) { setError('The complete request exceeds the server or 16 MiB browser JSON body limit. Remove attachments or start a shorter conversation.'); return; }
    if (!updateConversation(conversation)) { setError('Conversation memory budget reached. Export and clear older history first.'); return; }
    let timer: ReturnType<typeof setTimeout> | null = null;
    const epoch = sessionGeneration();
    const request = { controller, turn, conversation, flush: (): void => {
      if (timer !== null) clearTimeout(timer);
      timer = null;
      if (sessionGeneration() !== epoch) return;
      const previous = request.conversation;
      request.conversation = { ...request.conversation, turns: request.conversation.turns.map((item) => item.id === turn.id ? request.turn : item), updatedAt: Date.now() };
      if (!updateConversation(request.conversation)) {
        controller.abort();
        request.conversation = previous;
        request.turn = { ...(previous.turns.find((item) => item.id === turn.id) ?? turn), status: 'error', error: null };
        request.conversation = { ...previous, turns: previous.turns.map((item) => item.id === turn.id ? request.turn : item) };
        updateConversation(request.conversation);
        setError('Conversation memory budget reached. No further output was retained.');
      }
    } };
    active.current = request; setBusy(true); setError(null); setDraft(''); setImages([]); request.flush(); setAnnouncement('Generating.');
    try {
      await actions.streamChatCompletions(turn.modelId, body, {
        onFrame: (frame) => {
          if (controller.signal.aborted) return;
          request.turn = appendFrame(request.turn, frame.data, performance.now() - started);
          if (timer === null) timer = setTimeout(request.flush, 50);
        },
      }, controller.signal);
      if (!controller.signal.aborted) { request.turn = completeTurn(request.turn, performance.now() - started); setAnnouncement('Response complete.'); }
    } catch {
      if (!controller.signal.aborted) {
        controller.abort();
        request.turn = { ...request.turn, status: 'error', elapsedMs: performance.now() - started, error: 'Generation failed or disconnected. Partial output is preserved. Check context limits, authentication and server availability; nothing was retried.' };
        setAnnouncement('Generation failed; partial response preserved.');
      }
    } finally {
      request.flush();
      if (active.current === request) active.current = null;
      setBusy(false);
    }
  };
  return <div className="screen-stack chat-screen">
    <section className="screen-heading"><p className="eyebrow">Local inference</p><h1 data-testid="chat-title">{locale === 'ko' ? '채팅' : 'Chat'}</h1><p>Memory-only by default. Changing the model affects the next turn, never the running request.</p></section>
    <div className="chat-toolbar"><Button onClick={create} disabled={busy || historyBusy || imagesBusy || conversations.length >= 50}>New conversation</Button><Select label="Conversation" value={currentId ?? ''} disabled={busy || historyBusy || imagesBusy} onChange={(id) => { setCurrentId(id); setDraft(''); setImages([]); }} options={[{ value: '', label: 'Choose conversation' }, ...conversations.map((item) => ({ value: item.id, label: item.title }))]} /><Select label="Model for next turn" value={snapshot.selectedModelId ?? ''} onChange={(id) => actions.selectModel(id || null)} options={[{ value: '', label: 'Select a model' }, ...snapshot.catalog.map((item) => ({ value: item.identity.id, label: `${item.identity.display_name} · ${item.lifecycle.state}` }))]} /></div>
    {!canChat ? <ErrorBanner tone="info" title="Choose a ready chat model" body="Selection never loads a model. Load one explicitly in Models. Embeddings, reranking and other tasks use their documented API, not this composer." action={<a href="#models">Open Models</a>} /> : null}
    {current ? <details><summary>Conversation settings</summary><Field label="Conversation name" value={current.title} disabled={busy || historyBusy || imagesBusy} onChange={(title) => updateConversation({ ...current, title: title.slice(0, 120) })} /><label className="ds-field">System prompt<textarea value={current.systemPrompt} maxLength={MAX_PROMPT_CHARACTERS} disabled={busy || historyBusy || imagesBusy} onChange={(event) => updateConversation({ ...current, systemPrompt: event.target.value })} /></label><p>Sampling defaults are set in <a href="#settings">Settings</a> and frozen when sending.</p><Button disabled={busy || historyBusy || imagesBusy} onClick={() => { replaceConversations(conversations.filter((item) => item.id !== current.id)); setCurrentId(null); }}>Delete conversation</Button></details> : null}
    <Transcript turns={current?.turns ?? []} onEdit={(index) => {
      if (busy || historyBusy || imagesBusy || current === null || !window.confirm('Edit this prompt and discard this response and all later turns?')) return;
      setDraft(current.turns[index].prompt); setImages(current.turns[index].images);
      updateConversation({ ...current, turns: current.turns.slice(0, index) });
    }} busy={busy || historyBusy || imagesBusy} />
    {error ? <ErrorBanner title="Chat could not continue" body={error} /> : null}
    <div className="chat-composer"><label className="ds-field">Message<textarea aria-label="Message" value={draft} maxLength={MAX_PROMPT_CHARACTERS} disabled={busy || historyBusy || imagesBusy} onChange={(event) => setDraft(event.target.value)} onCompositionStart={() => { composing.current = true; }} onCompositionEnd={() => { composing.current = false; }} onKeyDown={(event) => {
      if (event.key === 'Enter' && !event.shiftKey && !composing.current && !event.nativeEvent.isComposing && event.keyCode !== 229) { event.preventDefault(); void send(); }
    }} /></label><p>Enter sends · Shift+Enter inserts a line. {draft.length}/{MAX_PROMPT_CHARACTERS} characters.</p>
      {canImage ? <label className="ds-field">Local images<input type="file" accept="image/png,image/jpeg,image/webp" multiple disabled={busy || historyBusy || imagesBusy} onChange={(event) => {
        const files = event.target.files;
        if (files && snapshot.bootstrap) { const epoch = ++imageEpoch.current; setImagesBusy(true); void loadLocalImages(files, images.length, snapshot.bootstrap.media_limits, images).then((added) => { if (epoch === imageEpoch.current) setImages((previous) => [...previous, ...added]); }).catch(() => setError('Images must be bounded local PNG, JPEG or WebP files. No URL, SVG or HTML input is accepted.')).finally(() => { if (epoch === imageEpoch.current) setImagesBusy(false); }); }
        event.target.value = '';
      }} /></label> : <p>Image attachments require provider-confirmed vision support.</p>}
      <div className="chat-images">{images.map((image, index) => <figure key={`${image.name}-${index}`}><img src={image.dataUrl} alt={image.name} /><Button disabled={busy || historyBusy || imagesBusy} onClick={() => setImages(images.filter((_, position) => index !== position))}>Remove {image.name}</Button></figure>)}</div>
      <div className="chat-toolbar"><Button tone="primary" disabled={!canChat || busy || historyBusy || imagesBusy || !draft.trim()} onClick={() => { void send(); }}>Send</Button><Button disabled={!busy} onClick={stop}>Stop</Button></div>
    </div>
    <p role="status" aria-live="polite" aria-atomic="true">{announcement}</p>
    <HistoryControls conversations={conversations} busy={busy || imagesBusy} onPending={setHistoryBusy} limits={snapshot.bootstrap?.media_limits} onReplace={(next) => { replaceConversations(next); setCurrentId(null); }} />
  </div>;
}
