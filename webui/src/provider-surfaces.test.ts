// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest';
import type { CatalogEntry, ModelLifecycleState, WebUiSnapshot } from './api/types';
import { paletteModelMatches, PALETTE_MODEL_LIMIT } from './command-palette';
import { model, snapshot } from './features/models/test-fixtures';
import { connectionFooterDetails, connectionFooterLabel, loadedModels } from './provider-surfaces';
import { initialSnapshot } from './state/reducer';

function entry(id: string, name: string, state: ModelLifecycleState): CatalogEntry {
  const base = model();
  return { ...base, identity: { ...base.identity, id, display_name: name }, lifecycle: { ...base.lifecycle, state } };
}

function withCatalog(catalog: CatalogEntry[]): WebUiSnapshot {
  return { ...snapshot(), catalog, selectedModelId: null };
}

describe('loadedModels', () => {
  it('keeps only states that hold a worker, ready first, then by display name and id', () => {
    const catalog = [
      entry('mdl_unloaded', 'aardvark', 'unloaded'),
      entry('mdl_failed', 'abacus', 'failed'),
      entry('mdl_draining', 'beta', 'draining'),
      entry('mdl_ready_z', 'zulu', 'ready'),
      entry('mdl_loading', 'alpha', 'loading'),
      entry('mdl_ready_b', 'mike', 'ready'),
      entry('mdl_ready_a', 'mike', 'ready'),
      entry('mdl_unloading', 'charlie', 'unloading'),
    ];
    expect(loadedModels(withCatalog(catalog)).map((item) => item.identity.id)).toEqual([
      'mdl_ready_a', 'mdl_ready_b', 'mdl_ready_z', 'mdl_loading', 'mdl_draining', 'mdl_unloading',
    ]);
  });

  it('ignores the browser selection and is empty when nothing is loaded', () => {
    const catalog = [entry('mdl_a', 'alpha', 'unloaded'), entry('mdl_b', 'beta', 'failed')];
    expect(loadedModels({ ...withCatalog(catalog), selectedModelId: 'mdl_a' })).toEqual([]);
    expect(loadedModels(initialSnapshot())).toEqual([]);
  });

  it('does not reorder the snapshot catalog it reads', () => {
    const catalog = [entry('mdl_b', 'beta', 'ready'), entry('mdl_a', 'alpha', 'ready')];
    const state = withCatalog(catalog);
    loadedModels(state);
    expect(state.catalog.map((item) => item.identity.id)).toEqual(['mdl_b', 'mdl_a']);
  });
});

describe('connection footer', () => {
  it('prints mode, status and version without the internal sequence', () => {
    const state = { ...snapshot(), lastSequence: 42 };
    expect(connectionFooterLabel('en', state)).toBe('model_free · ready · v0.7.0');
    expect(connectionFooterLabel('en', state)).not.toContain('42');
    expect(connectionFooterLabel('en', initialSnapshot())).toBe('Shell loaded; local API not connected');
  });

  it('returns details only after bootstrap, preferring the live instance and the newest sequence', () => {
    expect(connectionFooterDetails('en', initialSnapshot())).toBeNull();
    const state = snapshot();
    expect(connectionFooterDetails('en', { ...state, serverInstanceId: 'srv_live', lastSequence: 9, catalogSequence: 3 })).toEqual({ instance: 'srv_live', sequence: '9' });
    expect(connectionFooterDetails('en', { ...state, serverInstanceId: null, lastSequence: null, catalogSequence: 3 })).toEqual({ instance: 'srv_20260912_a', sequence: '3' });
    const fences = { ...state.resourceFences, operationsSnapshot: 5 };
    expect(connectionFooterDetails('en', { ...state, lastSequence: null, catalogSequence: null, resourceFences: fences })?.sequence).toBe('5');
  });

  it('prints unknown for absent values in the active locale', () => {
    const state = { ...snapshot(), serverInstanceId: null, lastSequence: null, catalogSequence: null };
    const bootstrap = state.bootstrap;
    if (bootstrap === null) throw new Error('Missing bootstrap fixture');
    const blank = { ...state, bootstrap: { ...bootstrap, server: { ...bootstrap.server, server_instance_id: '' } } };
    expect(connectionFooterDetails('en', blank)).toEqual({ instance: 'unknown', sequence: 'unknown' });
    expect(connectionFooterDetails('ko', blank)).toEqual({ instance: '알 수 없음', sequence: '알 수 없음' });
  });
});

describe('paletteModelMatches', () => {
  const catalog = [
    entry('mdl_qwen_unloaded', 'Qwen3-8B-4bit', 'unloaded'),
    entry('mdl_llama', 'Llama-3.1-8B', 'unloaded'),
    entry('mdl_qwen_ready', 'qwen3-0.6b-4bit', 'ready'),
    entry('mdl_qwen_loading', 'Qwen2.5-7B', 'loading'),
  ];

  it('matches display name and id substrings case-insensitively, loaded first', () => {
    expect(paletteModelMatches(catalog, 'QWEN').map((item) => item.identity.id)).toEqual(['mdl_qwen_ready', 'mdl_qwen_loading', 'mdl_qwen_unloaded']);
    expect(paletteModelMatches(catalog, '  mdl_LLAMA ').map((item) => item.identity.id)).toEqual(['mdl_llama']);
    expect(paletteModelMatches(catalog, '8b').map((item) => item.identity.id)).toEqual(['mdl_llama', 'mdl_qwen_unloaded']);
  });

  it('returns nothing for an empty query or no match', () => {
    expect(paletteModelMatches(catalog, '')).toEqual([]);
    expect(paletteModelMatches(catalog, '   ')).toEqual([]);
    expect(paletteModelMatches(catalog, 'mistral')).toEqual([]);
  });

  it('caps the hits at twenty by default and at an explicit limit', () => {
    const many = Array.from({ length: 30 }, (_, index) => entry(`mdl_${String(index).padStart(2, '0')}`, `model-${String(index).padStart(2, '0')}`, index === 29 ? 'ready' : 'unloaded'));
    const hits = paletteModelMatches(many, 'model');
    expect(hits).toHaveLength(PALETTE_MODEL_LIMIT);
    expect(PALETTE_MODEL_LIMIT).toBe(20);
    expect(hits[0].identity.id).toBe('mdl_29');
    expect(paletteModelMatches(many, 'model', 3)).toHaveLength(3);
  });
});
