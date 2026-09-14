// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
import { describe, expect, it, vi } from 'vitest';
import { createHistoryRepository, exportConversations, HISTORY_DATABASE, HISTORY_LIMITS,
  HistoryStorageError, HistoryValidationError, importConversations, validateConversations,
  type ChatConversation } from './history';

function fixture(): ChatConversation[] {
  return [{ id: 'conversation-1', title: 'Hello', systemPrompt: 'Be helpful.', updatedAt: 123,
    turns: [{ id: 'turn-1', modelId: 'opaque-1', modelRevision: 3, inferenceId: 'model-1', modelName: 'Model',
      prompt: 'Hi', content: 'Hello', reasoning: 'Think', tools: [], status: 'streaming',
      finishReason: null, usage: null, ttftMs: null, elapsedMs: null, error: null,
      parameters: { temperature: 0.7, seed: -1 },
      images: [{ name: 'test.png', type: 'image/png', dataUrl: 'data:image/png;base64,YQ==' }] }],
  }];
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
  it('rejects invalid versions, JSON, duplicate IDs, and non-finite values', () => {
    expect(() => importConversations('{')).toThrow(HistoryValidationError);
    expect(() => importConversations('{"version":2,"conversations":[]}')).toThrow(HistoryValidationError);
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
