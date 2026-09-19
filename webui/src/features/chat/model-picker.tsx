// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
// The model for the next turn, in the conversation header: Ready chat models first,
// then models on their way, then models that can be loaded. Selecting never loads;
// only the inline Load button does, after a confirmation, with the Models request.
import React, { useMemo, useRef, useState } from 'react';
import type { CatalogEntry, WebUiSnapshot } from '../../api/types';
import { Button, ConfirmDialog, Select } from '../../design-system/primitives';
import type { SelectProps } from '../../design-system/common-select';
import { t, type Locale, type StringKey } from '../../i18n/catalog';
import { lifecycleLabel } from '../../provider-surfaces';
import { useWebUi, useWebUiActions } from '../../state';
import { ConfirmAction, type Confirmation } from '../models/dialogs';
import { loadErrorMessage, submitLoad } from '../models/load-action';
import { canLoad, evictionCandidates, modelPending } from '../models/policy';
import { useLoadProfile } from '../settings/load-profiles';

const hasChat = (entry: CatalogEntry): boolean => entry.capabilities.some((cap) => cap.task === 'chat');
/** Loaded and confirmed by the provider for chat: the only models Send accepts. */
export const readyForChat = (entry: CatalogEntry): boolean => entry.lifecycle.state === 'ready' && entry.capabilities.some((cap) => cap.task === 'chat' && cap.phase === 'provider_ready' && cap.available);
const byName = (a: CatalogEntry, b: CatalogEntry): number => a.identity.display_name.localeCompare(b.identity.display_name) || a.identity.id.localeCompare(b.identity.id);

/**
 * The picker's options in display order. The shared Select has no option groups, so each
 * group is its position plus a state description on every option. Models without any chat
 * capability are left out, except the current selection, which is always listed last so
 * the trigger never goes blank.
 */
export function pickerOptions(catalog: ReadonlyArray<CatalogEntry>, selected: CatalogEntry | undefined, locale: Locale): SelectProps['options'] {
  const ready = catalog.filter(readyForChat).sort(byName);
  const moving = catalog.filter((entry) => ['loading', 'draining', 'unloading'].includes(entry.lifecycle.state) && hasChat(entry)).sort(byName);
  const idle = catalog.filter((entry) => ['unloaded', 'failed'].includes(entry.lifecycle.state) && hasChat(entry)).sort(byName);
  const option = (entry: CatalogEntry, description: string): SelectProps['options'][number] => ({ value: entry.identity.id, label: entry.identity.display_name, description });
  const options = [
    ...ready.map((entry) => option(entry, t(locale, 'chat.model.group.ready'))),
    ...moving.map((entry) => option(entry, lifecycleLabel(locale, entry.lifecycle.state))),
    ...idle.map((entry) => option(entry, t(locale, entry.lifecycle.state === 'failed' ? 'chat.model.group.failed' : 'chat.model.group.unloaded'))),
  ];
  if (selected && !options.some((item) => item.value === selected.identity.id)) options.push(option(selected, t(locale, 'chat.model.group.unavailable')));
  return selected ? options : [{ value: '', label: t(locale, 'chat.model.choose') }, ...options];
}

// What the header says about a selection that cannot chat yet.
function hintKey(state: WebUiSnapshot, model: CatalogEntry | undefined): StringKey | null {
  if (model === undefined) return 'chat.model.hint.choose';
  if (readyForChat(model)) return null;
  if (['unloaded', 'failed'].includes(model.lifecycle.state) && hasChat(model)) {
    if (canLoad(state, model)) return 'chat.model.hint.load';
    return model.lifecycle.busy || modelPending(state, model.identity.id) ? 'chat.model.hint.wait' : 'chat.model.hint.cannot_load';
  }
  if (['loading', 'draining', 'unloading'].includes(model.lifecycle.state) && hasChat(model)) return 'chat.model.hint.wait';
  return 'chat.model.hint.unavailable';
}

export function ModelPicker({ locale }: { locale: Locale }): React.JSX.Element {
  const state = useWebUi();
  const actions = useWebUiActions();
  const model = state.catalog.find((entry) => entry.identity.id === state.selectedModelId);
  const [confirming, setConfirming] = useState<CatalogEntry | null>(null);
  const [capacity, setCapacity] = useState<Confirmation | null>(null);
  const [pending, setPending] = useState(false);
  const pendingRef = useRef(false);
  // Keyed by model, so a new selection starts with a clean status line.
  const [outcome, setOutcome] = useState<{ modelId: string; error: string | null } | null>(null);
  const localeRef = useRef(locale); localeRef.current = locale;
  const { profile } = useLoadProfile((confirming ?? model)?.identity.id ?? null);
  const { profile: capacityProfile } = useLoadProfile(capacity?.kind === 'capacity' ? capacity.entry.identity.id : null);
  const offerCapacity = (entry: CatalogEntry): void => setCapacity({ kind: 'capacity', entry, instance: state.serverInstanceId });
  const stale = (modelId: string): void => setOutcome({ modelId, error: t(localeRef.current, 'models.library.stale') });
  const run = (entry: CatalogEntry, task: () => Promise<void>): void => {
    pendingRef.current = true; setPending(true); setOutcome(null);
    void task()
      .then(() => setOutcome({ modelId: entry.identity.id, error: null }))
      .catch((failure: unknown) => setOutcome({ modelId: entry.identity.id, error: loadErrorMessage(failure, localeRef.current) }))
      .finally(() => { pendingRef.current = false; setPending(false); });
  };
  // Answer-time check: the entry must still be in the catalog at the confirmed revision
  // and still loadable; otherwise nothing is sent and the header says the state changed.
  const confirmLoad = (): void => {
    const asked = confirming;
    setConfirming(null);
    if (asked === null || pendingRef.current) return;
    const entry = state.catalog.find((candidate) => candidate.identity.id === asked.identity.id);
    if (!entry || entry.identity.revision !== asked.identity.revision || !canLoad(state, entry)) { stale(asked.identity.id); return; }
    run(entry, () => submitLoad(actions, state, entry, profile, () => offerCapacity(entry)));
  };
  // The Models capacity dialog, answered the way the Models library answers it.
  const confirmCapacity = (target?: string, revision?: number): void => {
    const value = capacity;
    if (value === null || value.kind !== 'capacity' || pendingRef.current) return;
    const entry = state.catalog.find((candidate) => candidate.identity.id === value.entry.identity.id);
    if (value.instance !== state.serverInstanceId || !entry || entry.identity.revision !== value.entry.identity.revision) { stale(value.entry.identity.id); setCapacity(null); return; }
    if (target === undefined || revision === undefined || !evictionCandidates(state, entry.identity.id).some((candidate) => candidate.identity.id === target && candidate.identity.revision === revision)) { stale(entry.identity.id); return; }
    setCapacity(null);
    run(entry, () => submitLoad(actions, state, entry, capacityProfile, () => offerCapacity(entry), { id: target, revision }));
  };
  // Chat re-renders on every 50 ms stream flush; rebuild and re-sort the options only when
  // the catalog, the selection or the locale changes.
  const options = useMemo(() => pickerOptions(state.catalog, model, locale), [state.catalog, model, locale]);
  const hint = hintKey(state, model);
  // Unloaded or failed chat models get the inline Load; canLoad covers a load already
  // pending for the model, unsupported or incomplete checkpoints and a disabled action.
  const loadTarget = model !== undefined && ['unloaded', 'failed'].includes(model.lifecycle.state) && hasChat(model) ? model : undefined;
  const result = outcome !== null && model !== undefined && outcome.modelId === model.identity.id ? outcome : null;
  // The accepted request is news only until the operation settles: while the model loads
  // or the POST is still being reconciled. A later unload does not bring it back.
  const requested = result !== null && result.error === null && model !== undefined && (model.lifecycle.state === 'loading' || model.lifecycle.busy || modelPending(state, model.identity.id));
  return <div className="chat-model">
    <Select locale={locale} label={t(locale, 'chat.model.label')} value={model?.identity.id ?? ''} options={options} onChange={(id) => actions.selectModel(id || null)} testId="chat-model-picker" />
    {loadTarget ? <Button tone="primary" busy={pending} disabled={pending || !canLoad(state, loadTarget)} data-testid="chat-load" onClick={() => setConfirming(loadTarget)}>{t(locale, 'models.load')}</Button> : null}
    {/* One polite status line for hints and the accepted request; load errors are alerts. */}
    <p className="chat-model-status" role="status" data-testid="chat-model-status">{requested ? t(locale, 'chat.load.requested') : hint && !result?.error ? <>{t(locale, hint)}{hint === 'chat.model.hint.cannot_load' ? <> <a href="#models">{t(locale, 'chat.no_model.action')}</a></> : null}</> : null}</p>
    {result?.error ? <p className="chat-model-status chat-model-error" role="alert">{result.error}</p> : null}
    {confirming ? <ConfirmDialog open title={t(locale, 'chat.load.confirm.title', { model: confirming.identity.display_name })} body={[t(locale, 'chat.load.confirm.body'), Object.keys(profile).length ? t(locale, 'models.next_profile.body') : ''].filter(Boolean).join(' ')} confirmLabel={t(locale, 'models.load')} cancelLabel={t(locale, 'common.cancel')} closeLabel={t(locale, 'common.close')} testId="chat-load-dialog" onConfirm={confirmLoad} onClose={() => setConfirming(null)} /> : null}
    {capacity ? <ConfirmAction value={capacity} state={state} locale={locale} busy={pending} onClose={() => setCapacity(null)} onConfirm={confirmCapacity} /> : null}
  </div>;
}
