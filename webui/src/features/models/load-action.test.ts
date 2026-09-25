// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { WebUiHttpError } from '../../api/client';
import type { CatalogEntry, ModelActionRequest, WebUiSnapshot } from '../../api/types';
import { t } from '../../i18n/catalog';
import { LoadRefusedError, loadErrorMessage, submitLoad } from './load-action';
import { model, snapshot } from './test-fixtures';

const loadModel = vi.fn<(request: ModelActionRequest) => Promise<void>>();
const actions = { loadModel };
const target = model();
const idle: CatalogEntry = { ...target, identity: { ...target.identity, id: `mdl_${'i'.repeat(43)}`, display_name: 'idle-model', revision: 9 }, lifecycle: { ...target.lifecycle, state: 'ready' } };
const conflict = (): WebUiHttpError => new WebUiHttpError(409, { request_id: 'req_capacity', error: { code: 'conflict', message: 'capacity full', retryable: false } });
let state: WebUiSnapshot;

beforeEach(() => {
  loadModel.mockReset(); loadModel.mockResolvedValue(undefined);
  state = { ...snapshot(target), catalog: [target, idle] };
});

describe('submitLoad', () => {
  it('sends the Models request shape without a profile or eviction target', async () => {
    const onCapacity = vi.fn();
    await submitLoad(actions, state, target, {}, onCapacity);
    expect(loadModel).toHaveBeenCalledOnce();
    const request = loadModel.mock.calls[0][0];
    expect(Object.keys(request).sort()).toEqual(['action', 'expected_revision', 'idempotency_key', 'model_id']);
    expect(request).toMatchObject({ action: 'load', model_id: target.identity.id, expected_revision: target.identity.revision });
    expect(request.idempotency_key).toMatch(/^[0-9a-f-]{36}$/);
    expect(onCapacity).not.toHaveBeenCalled();
  });
  it('copies a non-empty profile and names an eviction target at its revision', async () => {
    const profile = { ctx_size: 4096, n_parallel: 2 };
    await submitLoad(actions, state, target, profile, vi.fn(), { id: idle.identity.id, revision: idle.identity.revision });
    const request = loadModel.mock.calls[0][0];
    expect(request).toEqual({ action: 'load', load_profile: { ctx_size: 4096, n_parallel: 2 }, model_id: target.identity.id, expected_revision: target.identity.revision, idempotency_key: request.idempotency_key, eviction_target_id: idle.identity.id, eviction_target_expected_revision: 9 });
    expect(request.load_profile).not.toBe(profile);
  });
  it('refuses without a request when the entry cannot load', async () => {
    const loading: CatalogEntry = { ...target, lifecycle: { ...target.lifecycle, state: 'loading' } };
    await expect(submitLoad(actions, { ...state, catalog: [loading, idle] }, loading, {}, vi.fn())).rejects.toBeInstanceOf(LoadRefusedError);
    await expect(submitLoad(actions, { ...state, connection: 'offline' }, target, {}, vi.fn())).rejects.toBeInstanceOf(LoadRefusedError);
    expect(loadModel).not.toHaveBeenCalled();
  });
  it('refuses an eviction target that is not a current candidate at that revision', async () => {
    await expect(submitLoad(actions, state, target, {}, vi.fn(), { id: idle.identity.id, revision: 8 })).rejects.toBeInstanceOf(LoadRefusedError);
    await expect(submitLoad(actions, state, target, {}, vi.fn(), { id: target.identity.id, revision: target.identity.revision })).rejects.toBeInstanceOf(LoadRefusedError);
    expect(loadModel).not.toHaveBeenCalled();
  });
  it('asks for capacity once on a conflict envelope and rethrows it', async () => {
    const failure = conflict();
    loadModel.mockRejectedValueOnce(failure);
    const onCapacity = vi.fn();
    await expect(submitLoad(actions, state, target, {}, onCapacity)).rejects.toBe(failure);
    expect(onCapacity).toHaveBeenCalledOnce();
  });
  it.each([
    ['a stale revision', new WebUiHttpError(409, { request_id: 'req_stale', error: { code: 'stale_revision', message: 'stale', retryable: false } })],
    ['a server error', new WebUiHttpError(503, null)],
    ['a transport failure', new TypeError('Failed to fetch')],
  ])('does not ask for capacity on %s', async (_name, failure) => {
    loadModel.mockRejectedValueOnce(failure);
    const onCapacity = vi.fn();
    await expect(submitLoad(actions, state, target, {}, onCapacity)).rejects.toBe(failure);
    expect(onCapacity).not.toHaveBeenCalled();
  });
});

describe('loadErrorMessage', () => {
  it('maps a refusal to the stale message and delegates everything else to the Models copy', () => {
    expect(loadErrorMessage(new LoadRefusedError(), 'ko')).toBe(t('ko', 'models.library.stale'));
    expect(loadErrorMessage(new WebUiHttpError(409, { request_id: 'req_stale', error: { code: 'stale_revision', message: 'stale', retryable: false } }), 'en')).toBe(t('en', 'models.library.stale'));
    expect(loadErrorMessage(conflict(), 'en')).toBe('capacity full');
    expect(loadErrorMessage(new TypeError('Failed to fetch'), 'en')).toBe(t('en', 'models.library.pending'));
  });
});
