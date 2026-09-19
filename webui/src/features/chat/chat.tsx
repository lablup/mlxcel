// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
// Chat state and request flow. The conversation list, message list, composer, model
// picker and settings drawer are presentational; every guard stays here.
import React, { useEffect, useId, useRef, useState, useSyncExternalStore } from 'react';
import { isPrimaryModifier } from '../../design-system/keyboard';
import { Button, ConfirmDialog, Drawer, ErrorBanner, IconButton, PageHeader, Tooltip } from '../../design-system/primitives';
import { useWebUi, useWebUiActions } from '../../state';
import { t, testId, type Locale, type StringKey } from '../../i18n/catalog';
import type { ChatConversation, ChatTurn } from './history';
import { loadLocalImages, validateRequestImages } from './images';
import { appendFrame, buildMessages, completeTurn, MAX_PROMPT_CHARACTERS } from './stream';
import { consumeNewConversationRequest, newConversation, replaceConversations, updateConversation, useConversations, useNewConversationRequest, sessionGeneration } from './session';
import { MessageList, canRetry } from './message-list';
import { ConversationList, TITLE_LIMIT } from './conversation-list';
import { Composer } from './composer';
import { ModelPicker } from './model-picker';
import { SettingsDrawer } from './settings-drawer';
import { HistoryControls } from './privacy';
import { useGenerationDefaults } from '../settings/generation-preferences';
import { GENERATION_FIELDS } from '../settings/generation-defaults';
import { resolveTurnParameters, TurnParameters, type TurnParameterDraft } from './parameters';
import './chat.css';

// Untitled conversations are auto-titled from their first prompt in either locale.
const DEFAULT_TITLES = new Set((['en', 'ko'] as const).map((code) => t(code, 'chat.conversation.default_title')));
const CONVERSATION_LIMIT = 50;
// The shell's compact breakpoint: below it the conversation list moves into a Drawer.
const COMPACT_QUERY = '(max-width: 960px)';

function subscribeCompact(listener: () => void): () => void {
  if (typeof window.matchMedia !== 'function') return () => undefined;
  const query = window.matchMedia(COMPACT_QUERY);
  query.addEventListener('change', listener);
  return () => query.removeEventListener('change', listener);
}
function useCompact(): boolean {
  return useSyncExternalStore(subscribeCompact, () => typeof window.matchMedia === 'function' && window.matchMedia(COMPACT_QUERY).matches, () => false);
}

/** What one send carries: the composer's draft, or a failed turn's stored prompt for Retry. */
interface SendSource {
  readonly prompt: string;
  readonly images: ChatTurn['images'];
  readonly conversation: ChatConversation | null;
  /** The composer's draft and images are consumed on admission; Retry leaves them alone. */
  readonly fromComposer: boolean;
}

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
  const [pendingDelete, setPendingDelete] = useState<{ id: string; title: string } | null>(null);
  const [listOpen, setListOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [parametersRequest, setParametersRequest] = useState(0);
  const compact = useCompact();
  const titleId = useId();
  // Async continuations and the unmount cleanup localize with the locale current when they run.
  const localeRef = useRef(locale); localeRef.current = locale;
  const localized = (key: StringKey, values?: Record<string, string>): string => t(localeRef.current, key, values);
  const composer = useRef<HTMLTextAreaElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const imageEpoch = useRef(0);
  const composing = useRef(false);
  const statusEpoch = useRef(0);
  const active = useRef<{ controller: AbortController; turn: ChatTurn; conversation: ChatConversation; flush: () => void } | null>(null);
  const current = conversations.find((entry) => entry.id === currentId) ?? null;
  const model = snapshot.catalog.find((entry) => entry.identity.id === snapshot.selectedModelId);
  const connected = ['ready', 'streaming', 'polling'].includes(snapshot.connection);
  const canChat = connected && model?.lifecycle.state === 'ready' && model.capabilities.some((cap) => cap.task === 'chat' && cap.phase === 'provider_ready' && cap.available);
  const canImage = canChat && snapshot.bootstrap !== null && Object.values(snapshot.bootstrap.media_limits).every((limit) => limit > 0) && model?.capabilities.some((cap) => cap.task === 'vision_input' && cap.phase === 'provider_ready' && cap.available);
  const locked = busy || historyBusy || imagesBusy;
  const atLimit = conversations.length >= CONVERSATION_LIMIT;
  // The list drawer exists only at compact widths; the screen behind an open drawer is inert.
  const listDrawerOpen = compact && listOpen;

  useEffect(() => () => {
    imageEpoch.current++; statusEpoch.current++;
    const request = active.current;
    if (request !== null) {
      request.turn = { ...request.turn, status: 'interrupted', error: t(localeRef.current, 'chat.error.view_closed') };
      request.controller.abort();
      request.flush();
    }
  }, []);
  // Widening past the breakpoint removes the list drawer; like the shell's navigation
  // drawer, close it and keep focus on a visible control rather than <body>.
  useEffect(() => {
    if (compact || !listOpen) return;
    setListOpen(false);
    requestAnimationFrame(() => {
      const focused = document.activeElement;
      if (focused && focused !== document.body && !focused.closest('[aria-modal="true"]')) return;
      (listRef.current?.querySelector<HTMLElement>('[aria-current="true"]') ?? listRef.current?.querySelector<HTMLElement>('button'))?.focus();
    });
  }, [compact, listOpen]);

  const create = (): void => {
    if (locked || atLimit) return;
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
  // One click switches; the running request, a history operation or an image load blocks it.
  const select = (id: string): void => {
    if (locked) return;
    if (id !== currentId) { setCurrentId(id); setDraft(''); setImages([]); }
    setListOpen(false);
  };
  const rename = (id: string, title: string): void => {
    const item = conversations.find((entry) => entry.id === id);
    if (item === undefined || locked || !title.trim()) return;
    updateConversation({ ...item, title: title.slice(0, TITLE_LIMIT) });
  };
  // Re-check at answer time: the conversation must still exist and nothing may be running.
  const confirmDelete = (): void => {
    const pending = pendingDelete;
    setPendingDelete(null);
    if (pending === null || locked || !conversations.some((entry) => entry.id === pending.id)) return;
    replaceConversations(conversations.filter((entry) => entry.id !== pending.id));
    if (currentId === pending.id) setCurrentId(null);
    // The row and its Delete button are gone; continue from the list rather than <body>.
    window.setTimeout(() => {
      const focused = document.activeElement;
      if (focused === null || focused === document.body) (listRef.current?.querySelector<HTMLElement>('[aria-current="true"]') ?? listRef.current?.querySelector<HTMLElement>('button'))?.focus();
    }, 0);
  };
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
  const send = async (source: SendSource = { prompt: draft, images, conversation: current, fromComposer: true }): Promise<void> => {
    const { prompt, fromComposer } = source;
    if (active.current !== null || historyBusy || imagesBusy || !canChat || model === undefined || !prompt.trim() || prompt.length > MAX_PROMPT_CHARACTERS) return;
    if ((source.images.length || source.conversation?.turns.some((turn) => turn.images.length)) && !canImage) { setError(localized('chat.error.vision_unsupported')); return; }
    if (source.conversation === null && conversations.length >= CONVERSATION_LIMIT) { setError(localized('chat.error.conversation_limit')); return; }
    let parameters: Record<string, number>;
    try { parameters = resolveTurnParameters(defaults, parameterDraft); }
    catch (cause) { setError(cause instanceof Error ? cause.message : localized('chat.error.invalid_parameters')); return; }
    let conversation = source.conversation ?? newConversation(localized('chat.conversation.default_title'));
    if (conversation.turns.length >= 100) { setError(localized('chat.error.turn_limit')); return; }
    setCurrentId(conversation.id);
    const controller = new AbortController();
    const started = performance.now();
    const turn: ChatTurn = { id: crypto.randomUUID(), modelId: model.identity.id, inferenceId: model.identity.inference_id, modelName: model.identity.display_name, modelRevision: model.identity.revision, prompt, content: '', reasoning: '', tools: [], status: 'streaming', finishReason: null, usage: null, ttftMs: null, elapsedMs: null, error: null, parameters, images: source.images.map((image) => ({ ...image })), startedAt: Date.now() };
    conversation = { ...conversation, title: conversation.turns.length === 0 && DEFAULT_TITLES.has(conversation.title) ? prompt.slice(0, 80) : conversation.title, turns: [...conversation.turns, turn], updatedAt: Date.now() };
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
    active.current = request; statusEpoch.current++; setBusy(true); setError(null);
    if (fromComposer) { setDraft(''); setImages([]); }
    request.flush(); setAnnouncement(localized('chat.announce.generating'));
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
  // Retry re-sends a failed last turn's stored prompt and images through the same path,
  // in place of that turn; the slice only lands if the new request is admitted.
  const retry = (turnId: string): void => {
    if (locked || current === null) return;
    const index = current.turns.length - 1;
    const turn = current.turns[index];
    if (turn === undefined || turn.id !== turnId || !canRetry(turn, true)) return;
    void send({ prompt: turn.prompt, images: turn.images, conversation: { ...current, turns: current.turns.slice(0, index) }, fromComposer: false });
  };
  // Re-check the snapshot at answer time; a stale conversation, a stale turn at that
  // index, or a started request drops the edit.
  const confirmEdit = (): void => {
    const pending = pendingEdit;
    setPendingEdit(null);
    if (pending === null || locked || current === null || current.id !== pending.conversationId || current.turns[pending.index]?.id !== pending.turnId) return;
    const { index } = pending;
    setDraft(current.turns[index].prompt); setImages(current.turns[index].images);
    updateConversation({ ...current, turns: current.turns.slice(0, index) });
    // The edited turn's controls are gone; continue in the composer that now holds its prompt.
    window.setTimeout(() => composer.current?.focus(), 0);
  };
  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>): void => {
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
  };
  const addImages = (files: FileList): void => {
    if (!snapshot.bootstrap) return;
    const epoch = ++imageEpoch.current; setImagesBusy(true);
    void loadLocalImages(files, images.length, snapshot.bootstrap.media_limits, images).then((added) => { if (epoch === imageEpoch.current) setImages((previous) => [...previous, ...added]); }).catch(() => setError(localized('chat.error.images_invalid'))).finally(() => { if (epoch === imageEpoch.current) setImagesBusy(false); });
  };
  const overrides = GENERATION_FIELDS.filter((name) => (parameterDraft[name] ?? '').trim() !== '');
  const list = <ConversationList locale={locale} conversations={conversations} currentId={currentId} disabled={locked} atLimit={atLimit} onSelect={select} onRename={rename} onDelete={(id) => { const item = conversations.find((entry) => entry.id === id); if (item && !locked) setPendingDelete({ id, title: item.title }); }} listRef={listRef} />;
  const newButton = <Button onClick={create} disabled={locked || atLimit} data-testid={testId('chat.new_conversation')}>{t(locale, 'chat.new_conversation')}</Button>;
  const headerActions = <>
    {overrides.length ? <Button tone="ghost" className="chat-overrides" data-testid="chat-overrides" onClick={() => { setParametersRequest((value) => value + 1); setSettingsOpen(true); }}>{t(locale, 'chat.overrides', { keys: overrides.join(', ') })}</Button> : null}
    {compact ? <IconButton label={t(locale, 'chat.list.open')} icon="list" aria-haspopup="dialog" aria-expanded={listDrawerOpen} onClick={() => setListOpen(true)} data-testid="chat-list-open" /> : null}
    <IconButton label={t(locale, 'chat.settings.title')} icon="settings" aria-haspopup="dialog" aria-expanded={settingsOpen} onClick={() => setSettingsOpen(true)} data-testid="chat-settings-open" />
    {/* A disabled button does not reliably get hover, so the list also says why it is disabled. */}
    {atLimit ? <Tooltip content={t(locale, 'chat.list.limit')}>{newButton}</Tooltip> : newButton}
  </>;
  return <div className="chat-screen">
    <div className="chat-body" inert={listDrawerOpen || settingsOpen}>
      <PageHeader title={t(locale, 'chat.title')} titleTestId={testId('chat.title')} description={t(locale, 'chat.intro')} actions={headerActions} />
      <div className="chat-panes">
        {!compact ? <nav className="chat-list-pane" aria-label={t(locale, 'chat.list.label')}>{list}</nav> : null}
        <section className="chat-conversation" aria-labelledby={titleId}>
          <header className="chat-conversation-header">
            <h2 className="chat-conversation-title" id={titleId}>{current?.title ?? t(locale, 'chat.conversation.default_title')}</h2>
            <ModelPicker locale={locale} />
          </header>
          <MessageList turns={current?.turns ?? []} onEdit={(index) => {
            if (locked || current === null) return;
            const turn = current.turns[index];
            if (!turn) return;
            setPendingEdit({ conversationId: current.id, turnId: turn.id, index });
          }} onRetry={retry} busy={locked} locale={locale} />
          <div className="chat-footer">
            {error ? <ErrorBanner title={t(locale, 'chat.error.title')} body={error} /> : null}
            <Composer locale={locale} textareaRef={composer} draft={draft} onDraftChange={setDraft} images={images} onRemoveImage={(index) => setImages(images.filter((_, position) => index !== position))} canImage={Boolean(canImage)} onAddImages={addImages} disabled={locked} running={busy} canSend={canChat && !locked && draft.trim() !== ''} onSend={() => { void send(); }} onStop={stop} onKeyDown={handleKeyDown} onCompositionStart={() => { composing.current = true; }} onCompositionEnd={() => { composing.current = false; }} />
            <p className="chat-announcement" role="status" aria-live="polite" aria-atomic="true">{announcement}</p>
          </div>
        </section>
      </div>
    </div>
    {compact ? <Drawer open={listOpen} onClose={() => setListOpen(false)} title={t(locale, 'chat.list.label')} closeLabel={t(locale, 'common.close')} testId="chat-list-drawer">{list}</Drawer> : null}
    <SettingsDrawer open={settingsOpen} onClose={() => setSettingsOpen(false)} locale={locale} parametersRequest={parametersRequest} conversation={current} disabled={locked} onConversationChange={updateConversation}
      parameters={<TurnParameters defaults={defaults} draft={parameterDraft} onChange={setParameterDraft} locale={locale} />}
      history={<HistoryControls conversations={conversations} busy={busy || imagesBusy} onPending={setHistoryBusy} limits={snapshot.bootstrap?.media_limits} onReplace={(next) => { replaceConversations(next); setCurrentId(null); }} locale={locale} />} />
    {pendingEdit !== null ? <ConfirmDialog open title={t(locale, 'chat.transcript.edit.confirm.title')} body={t(locale, 'chat.transcript.edit.confirm.body')} confirmLabel={t(locale, 'chat.transcript.edit.confirm')} cancelLabel={t(locale, 'common.cancel')} closeLabel={t(locale, 'common.close')} tone="danger" testId="chat-edit-dialog" onConfirm={confirmEdit} onClose={() => setPendingEdit(null)} /> : null}
    {pendingDelete !== null ? <ConfirmDialog open title={t(locale, 'chat.list.delete.confirm.title', { title: pendingDelete.title })} body={t(locale, 'chat.list.delete.confirm.body')} confirmLabel={t(locale, 'chat.list.delete.confirm')} cancelLabel={t(locale, 'common.cancel')} closeLabel={t(locale, 'common.close')} tone="danger" testId="chat-delete-conversation-dialog" onConfirm={confirmDelete} onClose={() => setPendingDelete(null)} /> : null}
  </div>;
}
