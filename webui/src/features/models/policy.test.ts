// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest';
import {
  bytes,
  canChat,
  canDelete,
  canLoad,
  canUnload,
  current,
  evictionCandidates,
  inventory,
  modelPending,
  validRepo,
} from './policy';
import { model, snapshot } from './test-fixtures';

describe('authoritative Models action policy', () => {
  it('keeps unknown disk and memory distinct from zero', () => {
    expect(bytes(null, 'en')).toBe('Unknown');
    expect(bytes(0, 'en')).toBe('0 B');
    expect(bytes(1024, 'en')).toBe('1 KiB');
  });
  it.each(['offline', 'stale', 'error', 'schema-mismatch', 'unauthorized', 'forbidden'] as const)(
    'blocks mutation on %s snapshots',
    (connection) => {
      const state = { ...snapshot(), connection };
      expect(current(state)).toBe(false);
      expect(canLoad(state, model())).toBe(false);
      expect(canDelete(state, model())).toBe(false);
    },
  );
  it('never lets a user root offer deletion, even when inconsistent booleans claim eligibility', () => {
    const state = snapshot();
    for (const source of ['models_dir', 'preset', 'single_model'] as const)
      expect(canDelete(state, { ...model(), identity: { ...model().identity, source } })).toBe(false);
    expect(canDelete(state, model())).toBe(true);
    expect(
      canDelete(state, { ...model(), removal: { eligible: false, reason: 'shared path', instructions: null } }),
    ).toBe(false);
  });
  it('does not turn family support, a filename, or pre-load capability into enabled Chat', () => {
    const ready = { ...model(), lifecycle: { ...model().lifecycle, state: 'ready' as const } };
    expect(
      canChat(snapshot(ready), {
        ...ready,
        capabilities: [{ task: 'chat', phase: 'pre_load', available: true, reason: null }],
      }),
    ).toBe(false);
    expect(
      canChat(snapshot(ready), {
        ...ready,
        capabilities: [{ task: 'embedding', phase: 'provider_ready', available: true, reason: null }],
      }),
    ).toBe(false);
    expect(
      canChat(snapshot(ready), {
        ...ready,
        capabilities: [{ task: 'chat', phase: 'provider_ready', available: true, reason: null }],
      }),
    ).toBe(true);
    expect(
      canLoad(snapshot(), {
        ...model(),
        metadata: { ...model().metadata, support: { ...model().metadata.support, runnable_on_backend: false } },
      }),
    ).toBe(false);
    expect(canLoad(snapshot(), { ...model(), complete: false })).toBe(false);
  });
  it('allows explicit draining of active requests but never suggests busy eviction', () => {
    const ready = {
      ...model(),
      lifecycle: { ...model().lifecycle, state: 'ready' as const, active_requests: 2, busy: true },
    };
    expect(canUnload(snapshot(ready), ready)).toBe(true);
    expect(evictionCandidates(snapshot(ready), 'different')).toEqual([]);
    const idle = { ...ready, lifecycle: { ...ready.lifecycle, active_requests: 0, busy: false } };
    expect(evictionCandidates(snapshot(idle), 'different')).toEqual([idle]);
  });
  it('retains ambiguous POST fences and single-model read-only admission', () => {
    const state = snapshot();
    if (state.bootstrap === null) throw new Error('Missing bootstrap fixture');
    const pending = {
      ...state,
      pendingReconciliations: new Map([
        [
          'key',
          {
            kind: 'model-action' as const,
            modelId: model().identity.id,
            idempotencyKey: 'key',
            operationId: null,
            createdAt: 1,
          },
        ],
      ]),
    };
    expect(modelPending(pending, model().identity.id)).toBe(true);
    expect(canLoad(pending, model())).toBe(false);
    expect(
      canLoad(
        {
          ...state,
          bootstrap: { ...state.bootstrap, actions: { load: { state: 'read_only', reason: 'Single model' } } },
        },
        model(),
      ),
    ).toBe(false);
  });
  it('filters and sorts large CJK inventories without mutating authoritative data', () => {
    const entries = Array.from({ length: 1000 }, (_, index) => ({
      ...model(),
      identity: { ...model().identity, id: `id_${index}`, display_name: `긴-모델-이름-模型-${index}` },
    }));
    const before = entries.map((entry) => entry.identity.id);
    const result = inventory(entries, {
      query: '模型-99',
      source: 'cache',
      task: '',
      status: 'unloaded',
    });
    expect(result).toHaveLength(11);
    expect(entries.map((entry) => entry.identity.id)).toEqual(before);
  });
  it('keeps hyphen and underscore spellings of one checkpoint name as distinct rows', () => {
    const named = (name: string, id: string) => ({
      ...model(),
      identity: { ...model().identity, id, inference_id: name, display_name: name },
    });
    const hyphen = named('qwen3-0.6b-4bit', 'id_hyphen');
    const underscore = named('qwen3_0.6b_4bit', 'id_underscore');
    const search = (query: string, entries = [hyphen, underscore]) =>
      inventory(entries, { query, source: '', task: '', status: '' }).map((entry) => entry.identity.id);
    expect(search('qwen3-0.6b-4bit')).toEqual(['id_hyphen']);
    expect(search('qwen3_0.6b_4bit')).toEqual(['id_underscore']);
    const sorted = search('');
    expect([...sorted].sort()).toEqual(['id_hyphen', 'id_underscore']);
    expect(search('', [underscore, hyphen])).toEqual(sorted);
  });
  it.each([
    'https://huggingface.co/owner/repo',
    '../local',
    'owner/../repo',
    'owner/repo?token=secret',
    'owner/repo#main',
    '/tmp/model',
  ])('rejects non-repository input %s', (repo) => expect(validRepo(repo)).toBe(false));
  it('accepts public repo syntax without performing an external lookup', () =>
    expect(validRepo('mlx-community/SmolLM-135M-Instruct-4bit')).toBe(true));
});

it('matches canonical repository segment and revision bounds', async () => {
  const { validRevision } = await import('./policy');
  expect(validRepo(`${'a'.repeat(96)}/repo`)).toBe(true);
  expect(validRepo(`${'a'.repeat(97)}/repo`)).toBe(false);
  expect(validRevision('')).toBe(true);
  expect(validRevision('refs/pr/1')).toBe(true);
  for (const invalid of ['a'.repeat(129), 'refs/../main', 'refs//main', 'refs/./main', '모델', 'main?token=secret'])
    expect(validRevision(invalid)).toBe(false);
});
