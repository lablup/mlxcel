// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import React, { useEffect, useRef, useState } from 'react';
import { isPrimaryModifier } from '../../design-system/keyboard';
import { Button, ConfirmDialog, ErrorBanner, Field, PageHeader, Select } from '../../design-system/primitives';
import { useWebUi, useWebUiActions } from '../../state';
import { t, testId, type Locale, type StringKey } from '../../i18n/catalog';
import { lifecycleLabel } from '../../provider-surfaces';
import type { ChatConversation, ChatTurn } from './history';
import { loadLocalImages, validateRequestImages } from './images';
import { appendFrame, buildMessages, completeTurn, MAX_PROMPT_CHARACTERS } from './stream';
import { consumeNewConversationRequest, newConversation, replaceConversations, updateConversation, useConversations, useNewConversationRequest, sessionGeneration } from './session';
import { Transcript } from './transcript';
import { HistoryControls } from './privacy';
import { useGenerationDefaults } from '../settings/generation-preferences';
import { resolveTurnParameters, TurnParameters, type TurnParameterDraft } from './parameters';
import './chat.css';

// Untitled conversations are auto-titled from their first prompt in either locale.
const DEFAULT_TITLES = new Set((['en', 'ko'] as const).map((code) => t(code, 'chat.conversation.default_title')));

export function Chat({ locale }: { locale: Locale }): React.JSX.Element {
  const snapshot = useWebUi();
  const actions = useWebUiActions();
  const { defaults } = useGenerationDefaults();
  const [parameterDraft, setParameterDraft] = useState<TurnParameterDraft>({});
  const conversations = useConversations();
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const [images, setImages] = useState<ChatTurn['images']>([]);
  const [error, setError] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState('');
  const [busy, setBusy] = useState(false);
  const [historyBusy, setHistoryBusy] = useState(false);
  const [imagesBusy, setImagesBusy] = useState(false);
  const [pendingEdit, setPendingEdit] = useState<{ conversationId: string; turnId: string; index: number } | null>(null);
  // Async continuations and the unmount cleanup localize with the locale current when they run.
  const localeRef = useRef(locale); localeRef.current = locale;
  const localized = (key: StringKey, values?: Record<string, string>): string => t(localeRef.current, key, values);
  const composer = useRef<HTMLTextAreaElement>(null);
  const imageEpoch = useRef(0);
  const composing = useRef(false);
  const statusEpoch = useRef(0);
  const active = useRef<{ controller: AbortController; turn: ChatTurn; conversation: ChatConversation; flush: () => void } | null>(null);
  const current = conversations.find((entry) => entry.id === currentId) ?? null;
  const model = snapshot.catalog.find((entry) => entry.identity.id === snapshot.selectedModelId);
  const connected = ['ready', 'streaming', 'polling'].includes(snapshot.connection);
  const canChat = connected && model?.lifecycle.state === 'ready' && model.capabilities.some((cap) => cap.task === 'chat' && cap.phase === 'provider_ready' && cap.available);
  const canImage = canChat && snapshot.bootstrap !== null && Object.values(snapshot.bootstrap.media_limits).every((limit) => limit > 0) && model?.capabilities.some((cap) => cap.task === 'vision_input' && cap.phase === 'provider_ready' && cap.available);

  useEffect(() => () => {
    imageEpoch.current++; statusEpoch.current++;
    const request = active.current;
    if (request !== null) {
      request.turn = { ...request.turn, status: 'interrupted', error: t(localeRef.current, 'chat.error.view_closed') };
      request.controller.abort();
      request.flush();
    }
  }, []);

  const create = (): void => {
    if (busy || historyBusy || imagesBusy || conversations.length >= 50) return;
    imageEpoch.current++;
    const next = newConversation(localized('chat.conversation.default_title'));
    updateConversation(next); setCurrentId(next.id); setDraft(''); setImages([]);
  };
  // Cmd/Ctrl+N and the palette ask through session.ts; consume first, then act, so a
  // StrictMode re-run or a later mount finds nothing. create() keeps its own guards, so
  // a request refused by the busy state or the 50-conversation cap is simply dropped.
  const newConversationRequest = useNewConversationRequest();
  const createRef = useRef(create); createRef.current = create;
  useEffect(() => { if (consumeNewConversationRequest()) createRef.current(); }, [newConversationRequest]);
  const stop = (): void => {
    const request = active.current;
    if (request === null) return;
    request.turn = { ...request.turn, status: 'cancelled', error: null };
    request.controller.abort(); request.flush();
    const epoch = ++statusEpoch.current;
    setAnnouncement(localized('chat.announce.stopped'));
    void actions.refreshRuntime(request.turn.modelId).then((runtime) => {
      if (statusEpoch.current !== epoch) return;
      setAnnouncement(localized('chat.announce.aborted_observed', { time: runtime.measurements.active_requests?.measured_at ?? localized('chat.announce.unknown_time') }));
    }).catch(() => { if (statusEpoch.current === epoch) setAnnouncement(localized('chat.announce.aborted_unobserved')); });
  };
  const send = async (): Promise<void> => {
    if (active.current !== null || historyBusy || imagesBusy || !canChat || model === undefined || !draft.trim() || draft.length > MAX_PROMPT_CHARACTERS) return;
    if ((images.length || current?.turns.some((turn) => turn.images.length)) && !canImage) { setError(localized('chat.error.vision_unsupported')); return; }
    if (current === null && conversations.length >= 50) { setError(localized('chat.error.conversation_limit')); return; }
    let parameters: Record<string, number>;
    try { parameters = resolveTurnParameters(defaults, parameterDraft); }
    catch (cause) { setError(cause instanceof Error ? cause.message : localized('chat.error.invalid_parameters')); return; }
    let conversation = current ?? newConversation(localized('chat.conversation.default_title'));
    if (conversation.turns.length >= 100) { setError(localized('chat.error.turn_limit')); return; }
    setCurrentId(conversation.id);
    const controller = new AbortController();
    const started = performance.now();
    const turn: ChatTurn = { id: crypto.randomUUID(), modelId: model.identity.id, inferenceId: model.identity.inference_id, modelName: model.identity.display_name, modelRevision: model.identity.revision, prompt: draft, content: '', reasoning: '', tools: [], status: 'streaming', finishReason: null, usage: null, ttftMs: null, elapsedMs: null, error: null, parameters, images: images.map((image) => ({ ...image })) };
    conversation = { ...conversation, title: conversation.turns.length === 0 && DEFAULT_TITLES.has(conversation.title) ? draft.slice(0, 80) : conversation.title, turns: [...conversation.turns, turn], updatedAt: Date.now() };
    try {
      if (!snapshot.bootstrap) throw new Error('Limits unavailable');
      validateRequestImages(conversation.turns.flatMap((item) => item.images), snapshot.bootstrap.media_limits);
    } catch { setError(localized('chat.error.images_exceed')); return; }
    const body = { ...turn.parameters, model: turn.inferenceId, stream: true, messages: buildMessages(conversation.systemPrompt, conversation.turns), stream_options: { include_usage: true } };
    const bodyBudget = Math.min(snapshot.bootstrap?.media_limits.max_body_bytes ?? 0, 16 * 1024 * 1024);
    if (new TextEncoder().encode(JSON.stringify(body)).byteLength > bodyBudget) { setError(localized('chat.error.body_limit')); return; }
    if (!updateConversation(conversation)) { setError(localized('chat.error.memory_budget')); return; }
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
        setError(localized('chat.error.memory_budget_stream'));
      }
    } };
    setParameterDraft({});
    active.current = request; statusEpoch.current++; setBusy(true); setError(null); setDraft(''); setImages([]); request.flush(); setAnnouncement(localized('chat.announce.generating'));
    try {
      await actions.streamChatCompletions(turn.modelId, body, {
        onFrame: (frame) => {
          if (controller.signal.aborted) return;
          request.turn = appendFrame(request.turn, frame.data, performance.now() - started);
          if (timer === null) timer = setTimeout(request.flush, 50);
        },
      }, controller.signal);
      if (!controller.signal.aborted) { request.turn = completeTurn(request.turn, performance.now() - started); setAnnouncement(localized('chat.announce.complete')); }
    } catch {
      if (!controller.signal.aborted) {
        controller.abort();
        request.turn = { ...request.turn, status: 'error', elapsedMs: performance.now() - started, error: localized('chat.error.generation_failed') };
        setAnnouncement(localized('chat.announce.failed'));
      }
    } finally {
      request.flush();
      if (active.current === request) active.current = null;
      setBusy(false);
    }
  };
  // Re-check the snapshot at answer time; a stale conversation, a stale turn at that
  // index, or a started request drops the edit.
  const confirmEdit = (): void => {
    const pending = pendingEdit;
    setPendingEdit(null);
    if (pending === null || busy || historyBusy || imagesBusy || current === null || current.id !== pending.conversationId || current.turns[pending.index]?.id !== pending.turnId) return;
    const { index } = pending;
    setDraft(current.turns[index].prompt); setImages(current.turns[index].images);
    updateConversation({ ...current, turns: current.turns.slice(0, index) });
    // The edited turn's controls are gone; continue in the composer that now holds its prompt.
    window.setTimeout(() => composer.current?.focus(), 0);
  };
  const [samplingBefore, samplingAfter = ''] = t(locale, 'chat.settings.sampling').split('{link}');
  return <div className="screen-stack chat-screen">
    <PageHeader title={t(locale, 'chat.title')} titleTestId={testId('chat.title')} description={t(locale, 'chat.intro')} actions={<Button onClick={create} disabled={busy || historyBusy || imagesBusy || conversations.length >= 50} data-testid={testId('chat.new_conversation')}>{t(locale, 'chat.new_conversation')}</Button>} />
    <div className="chat-toolbar"><Select locale={locale} label={t(locale, 'chat.conversation.label')} value={currentId ?? ''} disabled={busy || historyBusy || imagesBusy} onChange={(id) => { setCurrentId(id); setDraft(''); setImages([]); }} options={[{ value: '', label: t(locale, 'chat.conversation.choose') }, ...conversations.map((item) => ({ value: item.id, label: item.title }))]} /><Select locale={locale} label={t(locale, 'chat.model.label')} value={snapshot.selectedModelId ?? ''} onChange={(id) => actions.selectModel(id || null)} options={[{ value: '', label: t(locale, 'chat.model.choose') }, ...snapshot.catalog.map((item) => ({ value: item.identity.id, label: `${item.identity.display_name} · ${lifecycleLabel(locale, item.lifecycle.state)}` }))]} /></div>
    {!canChat ? <ErrorBanner tone="info" title={t(locale, 'chat.no_model.title')} body={t(locale, 'chat.no_model.body')} action={<a href="#models">{t(locale, 'chat.no_model.action')}</a>} /> : null}
    {current ? <details><summary>{t(locale, 'chat.settings.summary')}</summary><Field label={t(locale, 'chat.settings.name')} value={current.title} disabled={busy || historyBusy || imagesBusy} onChange={(title) => updateConversation({ ...current, title: title.slice(0, 120) })} /><label className="ds-field">{t(locale, 'chat.settings.system_prompt')}<textarea value={current.systemPrompt} maxLength={MAX_PROMPT_CHARACTERS} disabled={busy || historyBusy || imagesBusy} onChange={(event) => updateConversation({ ...current, systemPrompt: event.target.value })} /></label><p>{samplingBefore}<a href="#settings/requests">{t(locale, 'nav.settings')}</a>{samplingAfter}</p><Button disabled={busy || historyBusy || imagesBusy} onClick={() => { replaceConversations(conversations.filter((item) => item.id !== current.id)); setCurrentId(null); }}>{t(locale, 'chat.settings.delete')}</Button></details> : null}
    <TurnParameters defaults={defaults} draft={parameterDraft} onChange={setParameterDraft} locale={locale} />
    <Transcript turns={current?.turns ?? []} onEdit={(index) => {
      if (busy || historyBusy || imagesBusy || current === null) return;
      const turn = current.turns[index];
      if (!turn) return;
      setPendingEdit({ conversationId: current.id, turnId: turn.id, index });
    }} busy={busy || historyBusy || imagesBusy} locale={locale} />
    {pendingEdit !== null ? <ConfirmDialog open title={t(locale, 'chat.transcript.edit.confirm.title')} body={t(locale, 'chat.transcript.edit.confirm.body')} confirmLabel={t(locale, 'chat.transcript.edit.confirm')} cancelLabel={t(locale, 'common.cancel')} closeLabel={t(locale, 'common.close')} tone="danger" testId="chat-edit-dialog" onConfirm={confirmEdit} onClose={() => setPendingEdit(null)} /> : null}
    {error ? <ErrorBanner title={t(locale, 'chat.error.title')} body={error} /> : null}
    <div className="chat-composer"><label className="ds-field">{t(locale, 'chat.composer.label')}<textarea ref={composer} aria-label={t(locale, 'chat.composer.label')} value={draft} maxLength={MAX_PROMPT_CHARACTERS} disabled={busy || historyBusy || imagesBusy} onChange={(event) => setDraft(event.target.value)} onCompositionStart={() => { composing.current = true; }} onCompositionEnd={() => { composing.current = false; }} onKeyDown={(event) => {
      // Nothing fires mid-composition: the IME owns Enter and the modifiers until it commits.
      if (composing.current || event.nativeEvent.isComposing || event.keyCode === 229) return;
      // Cmd+N (Apple) / Ctrl+N (elsewhere) starts a conversation here too; the global handler
      // skips edit fields. Ctrl+N on Apple is the native "move down a line" caret binding, so
      // it is left untouched (no preventDefault, no create); a held key creates at most once.
      if (isPrimaryModifier(event) && !event.altKey && !event.shiftKey && event.key.toLowerCase() === 'n') {
        event.preventDefault();
        if (!event.repeat) create();
        return;
      }
      // Enter and Cmd/Ctrl+Enter send regardless of platform; Ctrl+Enter has no native binding
      // to preserve, so this stays unconditional rather than gated by isPrimaryModifier.
      if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void send(); }
    }} /></label><p>{t(locale, 'chat.composer.hint', { count: String(draft.length), max: String(MAX_PROMPT_CHARACTERS) })}</p>
      {canImage ? <label className="ds-field">{t(locale, 'chat.images.label')}<input type="file" accept="image/png,image/jpeg,image/webp" multiple disabled={busy || historyBusy || imagesBusy} onChange={(event) => {
        const files = event.target.files;
        if (files && snapshot.bootstrap) { const epoch = ++imageEpoch.current; setImagesBusy(true); void loadLocalImages(files, images.length, snapshot.bootstrap.media_limits, images).then((added) => { if (epoch === imageEpoch.current) setImages((previous) => [...previous, ...added]); }).catch(() => setError(localized('chat.error.images_invalid'))).finally(() => { if (epoch === imageEpoch.current) setImagesBusy(false); }); }
        event.target.value = '';
      }} /></label> : <p>{t(locale, 'chat.images.unsupported')}</p>}
      <div className="chat-images">{images.map((image, index) => <figure key={`${image.name}-${index}`}><img src={image.dataUrl} alt={image.name} /><Button disabled={busy || historyBusy || imagesBusy} onClick={() => setImages(images.filter((_, position) => index !== position))}>{t(locale, 'chat.images.remove', { name: image.name })}</Button></figure>)}</div>
      <div className="chat-toolbar"><Button tone="primary" disabled={!canChat || busy || historyBusy || imagesBusy || !draft.trim()} onClick={() => { void send(); }}>{t(locale, 'common.send')}</Button><Button disabled={!busy} onClick={stop}>{t(locale, 'chat.stop')}</Button></div>
    </div>
    <p role="status" aria-live="polite" aria-atomic="true">{announcement}</p>
    <HistoryControls conversations={conversations} busy={busy || imagesBusy} onPending={setHistoryBusy} limits={snapshot.bootstrap?.media_limits} onReplace={(next) => { replaceConversations(next); setCurrentId(null); }} locale={locale} />
  </div>;
}
