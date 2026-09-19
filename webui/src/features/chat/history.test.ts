// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import { describe, expect, it, vi } from 'vitest';
import { createHistoryRepository, exportConversations, HISTORY_DATABASE, HISTORY_LIMITS,
  HistoryStorageError, HistoryValidationError, importConversations, validateConversations,
  type ChatConversation, type ChatTurn } from './history';

function fixture(): ChatConversation[] {
  return [{ id: 'conversation-1', title: 'Hello', systemPrompt: 'Be helpful.', updatedAt: 123,
    turns: [{ id: 'turn-1', modelId: 'opaque-1', modelRevision: 3, inferenceId: 'model-1', modelName: 'Model',
      prompt: 'Hi', content: 'Hello', reasoning: 'Think', tools: [], status: 'streaming',
      finishReason: null, usage: null, ttftMs: null, elapsedMs: null, error: null,
      parameters: { temperature: 0.7, seed: -1 },
      images: [{ name: 'test.png', type: 'image/png', dataUrl: 'data:image/png;base64,YQ==' }], startedAt: 1_726_000_000_000 }],
  }];
}
/** A version-1 payload as earlier builds exported and saved it: the same turns without `startedAt`. */
function versionOne(options: { includeImages?: boolean } = {}): string {
  const conversations = fixture().map((conversation) => ({ ...conversation, turns: conversation.turns.map((turn) => {
    const legacy: Partial<ChatTurn> = { ...turn, images: options.includeImages ? turn.images : [] };
    Reflect.deleteProperty(legacy, 'startedAt');
    return legacy;
  }) }));
  return JSON.stringify({ version: 1, conversations });
}
function databaseMock(quota = false) {
  let value: unknown;
  const close = vi.fn();
  const put = vi.fn((input: unknown) => { value = input; return { result: 'conversations', error: null }; });
  const get = vi.fn(() => ({ result: value, error: null }));
  const db = { objectStoreNames: { contains: () => true }, close,
    createObjectStore: vi.fn(), onversionchange: null,
    transaction: vi.fn(() => {
      const tx = { objectStore: () => ({ get, put }), abort: vi.fn(),
        error: quota ? new DOMException('full', 'QuotaExceededError') : null,
        onabort: null as null | (() => void), onerror: null as null | (() => void),
        oncomplete: null as null | (() => void) };
      queueMicrotask(() => { if (quota) tx.onabort?.(); else tx.oncomplete?.(); });
      return tx;
    }),
  };
  const requests: Array<{ result: typeof db; error: null; onsuccess: null | (() => void); onerror: null | (() => void);
    onblocked: null | (() => void); onupgradeneeded: null | (() => void) }> = [];
  const open = vi.fn(() => {
    const request = { result: db, error: null, onsuccess: null, onerror: null, onblocked: null, onupgradeneeded: null };
    requests.push(request); queueMicrotask(() => requests.at(-1)?.onsuccess?.()); return request;
  });
  const deleteDatabase = vi.fn(() => {
    const request = { onsuccess: null as null | (() => void), onerror: null, onblocked: null };
    queueMicrotask(() => { value = undefined; request.onsuccess?.(); }); return request;
  });
  return { indexedDB: { open, deleteDatabase } as unknown as IDBFactory, open, deleteDatabase, requests, db, put, get };
}

describe('bounded portable history', () => {
  it('round trips versioned history, interrupts unfinished turns, and omits images by default', () => {
    const source = fixture();
    const json = exportConversations(source);
    const result = importConversations(json);
    expect(result[0].turns[0].status).toBe('interrupted');
    expect(result[0].turns[0].images).toEqual([]);
    expect(source[0].turns[0].status).toBe('streaming');
    expect(source[0].turns[0].images).toHaveLength(1);
    expect(result[0].turns[0].parameters).toEqual({ temperature: 0.7, seed: -1 });
  });
  it('requires a bounded nonnegative immutable model revision in imported turns', () => {
    const value = fixture();
    expect(importConversations(exportConversations(value))[0].turns[0].modelRevision).toBe(3);
    for (const revision of [-1, 1.5, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      value[0].turns[0].modelRevision = revision;
      expect(() => validateConversations(value)).toThrow(HistoryValidationError);
    }
    const missing = fixture();
    Reflect.deleteProperty(missing[0].turns[0], 'modelRevision');
    expect(() => validateConversations(missing)).toThrow(HistoryValidationError);
  });
  it('requires a separate image opt-in on both export and import', () => {
    const json = exportConversations(fixture(), { includeImages: true });
    expect(importConversations(json)[0].turns[0].images).toEqual([]);
    expect(importConversations(json, { includeImages: true })[0].turns[0].images).toHaveLength(1);
  });
  it.each(['apiKey', 'authorization', '__proto__', 'constructor'])('rejects unknown or secret envelope fields: %s', (key) => {
    const json = exportConversations(fixture());
    expect(() => importConversations(json.replace('{', `{"${key}":"secret",`))).toThrow(HistoryValidationError);
  });
  it('rejects secret fields at nested boundaries and unsafe parameter keys', () => {
    const value = fixture();
    Object.assign(value[0].turns[0], { apiKey: 'secret' });
    expect(() => validateConversations(value)).toThrow(HistoryValidationError);
    const parameters = fixture();
    parameters[0].turns[0].parameters = JSON.parse('{"__proto__":1}') as Record<string, number>;
    expect(() => validateConversations(parameters)).toThrow(HistoryValidationError);
    expect(Object.prototype).not.toHaveProperty('polluted');
  });
  it('exports version 2 and keeps each turn start time through a round trip', () => {
    const json = exportConversations(fixture());
    expect(JSON.parse(json)).toMatchObject({ version: 2 });
    expect(importConversations(json)[0].turns[0].startedAt).toBe(1_726_000_000_000);
  });
  it('keeps loading version-1 payloads and back-fills an unknown start time as 0', () => {
    const result = importConversations(versionOne());
    expect(result[0].turns[0].startedAt).toBe(0);
    expect(result[0].turns[0].status).toBe('interrupted');
    expect(result[0].turns[0].modelRevision).toBe(3);
    expect(importConversations(versionOne({ includeImages: true }), { includeImages: true })[0].turns[0].images).toHaveLength(1);
    // A back-filled payload re-exports as version 2 and loads unchanged.
    expect(importConversations(exportConversations(result))[0].turns[0].startedAt).toBe(0);
  });
  it('requires startedAt in version 2 and rejects it in version 1', () => {
    const missing = JSON.parse(exportConversations(fixture())) as { version: number; conversations: Array<{ turns: Array<Record<string, unknown>> }> };
    Reflect.deleteProperty(missing.conversations[0].turns[0], 'startedAt');
    expect(() => importConversations(JSON.stringify(missing))).toThrow(HistoryValidationError);
    const early = JSON.parse(versionOne()) as { version: number; conversations: Array<{ turns: Array<Record<string, unknown>> }> };
    early.conversations[0].turns[0].startedAt = 1;
    expect(() => importConversations(JSON.stringify(early))).toThrow(HistoryValidationError);
    const unsaved = fixture(); Reflect.deleteProperty(unsaved[0].turns[0], 'startedAt');
    expect(() => validateConversations(unsaved)).toThrow(HistoryValidationError);
  });
  // 8.64e15 is the largest time a Date holds; a start time past it could never be shown.
  it.each([-1, 1.5, Infinity, 8.64e15 + 1, Number.MAX_SAFE_INTEGER + 1])('rejects a start time of %s', (startedAt) => {
    const value = fixture(); value[0].turns[0].startedAt = startedAt;
    expect(() => validateConversations(value)).toThrow(HistoryValidationError);
  });
  it('rejects invalid versions, JSON, duplicate IDs, and non-finite values', () => {
    expect(() => importConversations('{')).toThrow(HistoryValidationError);
    expect(() => importConversations('{"version":3,"conversations":[]}')).toThrow(HistoryValidationError);
    expect(() => importConversations('{"version":0,"conversations":[]}')).toThrow(HistoryValidationError);
    expect(() => validateConversations([...fixture(), ...fixture()])).toThrow(HistoryValidationError);
    const value = fixture(); value[0].turns.push(value[0].turns[0]);
    expect(() => validateConversations(value)).toThrow(HistoryValidationError);
    const numeric = fixture(); numeric[0].turns[0].parameters.temperature = NaN;
    expect(() => validateConversations(numeric)).toThrow(HistoryValidationError);
  });
  it('rejects unsupported or malformed image data even when import discards images', () => {
    for (const dataUrl of ['https://example.com/a.png', 'data:image/svg+xml;base64,YQ==', 'data:image/png;base64,Y=Q=']) {
      const value = fixture(); value[0].turns[0].images[0].dataUrl = dataUrl;
      expect(() => validateConversations(value)).toThrow(HistoryValidationError);
    }
  });
  it('enforces text, conversation, turn and image aggregate bounds', () => {
    const text = fixture(); text[0].turns[0].content = 'a'.repeat(HISTORY_LIMITS.textCharacters + 1);
    expect(() => validateConversations(text)).toThrow(HistoryValidationError);
    expect(() => validateConversations(Array.from({ length: 51 }, (_, i) => ({ ...fixture()[0], id: String(i) })))).toThrow(HistoryValidationError);
    const turns = fixture(); turns[0].turns = Array.from({ length: 201 }, (_, i) => ({ ...fixture()[0].turns[0], id: String(i) }));
    expect(() => validateConversations(turns)).toThrow(HistoryValidationError);
    const images = fixture();
    const dataUrl = `data:image/png;base64,${'AAAA'.repeat(1_500_000)}`;
    images[0].turns[0].images = Array.from({ length: 2 }, () => ({ name: 'large.png', type: 'image/png', dataUrl }));
    expect(() => validateConversations(images, { includeImages: true })).toThrow(HistoryValidationError);
  });
  it('rejects oversized JSON before attempting to parse it', () => {
    expect(() => importConversations(' '.repeat(HISTORY_LIMITS.jsonBytes + 1))).toThrow(HistoryValidationError);
  });
});

describe('opt-in IndexedDB repository', () => {
  it('does not open or write storage in default memory-only mode', async () => {
    const mock = databaseMock(); const repository = createHistoryRepository(mock);
    await repository.save(fixture()); expect(await repository.load()).toEqual([]);
    expect(mock.open).not.toHaveBeenCalled(); expect(mock.put).not.toHaveBeenCalled();
  });
  it('only persists after opt-in and restores an interrupted snapshot', async () => {
    const mock = databaseMock(); const repository = createHistoryRepository(mock);
    repository.setEnabled(true); await repository.save(fixture());
    expect(mock.open).toHaveBeenCalledWith(HISTORY_DATABASE, 1);
    expect((await repository.load())[0].turns[0].status).toBe('interrupted');
    expect((await repository.load())[0].turns[0].images).toEqual([]);
    repository.setEnabled(false); await repository.save(fixture());
    expect(mock.put).toHaveBeenCalledTimes(1); expect(mock.db.close).toHaveBeenCalledOnce();
    expect(await repository.load()).toEqual([]);
  });
  it('disabling while opening prevents any write and closes the late connection', async () => {
    const mock = databaseMock(); const repository = createHistoryRepository(mock);
    repository.setEnabled(true); const saving = repository.save(fixture()); repository.setEnabled(false);
    await expect(saving).rejects.toThrow(HistoryStorageError);
    expect(mock.put).not.toHaveBeenCalled(); expect(mock.db.close).toHaveBeenCalledOnce();
  });
  it('reports quota errors rather than silently discarding persistence failures', async () => {
    const mock = databaseMock(true); const repository = createHistoryRepository(mock);
    repository.setEnabled(true);
    await expect(repository.save(fixture())).rejects.toThrow('Local history storage is full');
  });
  it('explicit clear deletes the versioned database even when persistence is disabled', async () => {
    const mock = databaseMock(); const repository = createHistoryRepository(mock);
    await repository.clear(); expect(mock.deleteDatabase).toHaveBeenCalledWith(HISTORY_DATABASE);
    expect(mock.open).not.toHaveBeenCalled();
  });
  it('clear prevents concurrent writes until deletion completes', async () => {
    const mock = databaseMock(); const repository = createHistoryRepository(mock); repository.setEnabled(true);
    const clearing = repository.clear(); await repository.save(fixture()); await clearing;
    expect(mock.open).not.toHaveBeenCalled(); expect(mock.put).not.toHaveBeenCalled();
  });
  it('never writes through a connection that opens after a blocked-open error', async () => {
    const mock = databaseMock();
    const request: { result: typeof mock.db; onblocked?: () => void; onsuccess?: () => void } = { result: mock.db };
    mock.open.mockImplementation(() => {
      queueMicrotask(() => request.onblocked?.());
      return request as unknown as ReturnType<typeof mock.open>;
    });
    const repository = createHistoryRepository(mock); repository.setEnabled(true);
    await expect(repository.save(fixture())).rejects.toThrow('Close other WebUI tabs');
    request.onsuccess?.();
    expect(mock.put).not.toHaveBeenCalled(); expect(mock.db.close).toHaveBeenCalledOnce();
  });
  it('keeps writes paused after blocked deletion until its eventual completion', async () => {
    const mock = databaseMock();
    const request: { onblocked?: () => void; onsuccess?: () => void } = {};
    mock.deleteDatabase.mockImplementation(() => {
      queueMicrotask(() => request.onblocked?.());
      return request as unknown as ReturnType<typeof mock.deleteDatabase>;
    });
    const repository = createHistoryRepository(mock); repository.setEnabled(true);
    await expect(repository.clear()).rejects.toThrow('Close other WebUI tabs');
    await repository.save(fixture()); expect(mock.open).not.toHaveBeenCalled();
    request.onsuccess?.();
    await repository.save(fixture()); expect(mock.put).toHaveBeenCalledOnce();
  });
  it('validates persisted records instead of trusting storage contents', async () => {
    const mock = databaseMock(); mock.get.mockImplementation(() => ({ result: '{"version":9,"conversations":[]}', error: null }));
    const repository = createHistoryRepository(mock); repository.setEnabled(true);
    await expect(repository.load()).rejects.toThrow(HistoryValidationError);
  });

});

describe('imported completion invariants', () => {
  it('rejects false completion but accepts finished reasoning and inert tools', () => {
    const data = fixture(); data[0].turns[0].status = 'complete';
    expect(() => validateConversations(data)).toThrow(HistoryValidationError);
    data[0].turns[0].finishReason = 'stop'; data[0].turns[0].content = ''; data[0].turns[0].reasoning = '';
    expect(() => validateConversations(data)).toThrow(HistoryValidationError);
    data[0].turns[0].reasoning = 'Reasoning only'; expect(validateConversations(data)[0].turns[0].status).toBe('complete');
    data[0].turns[0].reasoning = ''; data[0].turns[0].tools = [{index:0,id:'call',name:'not-executed',arguments:'{}'}];
    expect(validateConversations(data)[0].turns[0].status).toBe('complete');
  });
});
