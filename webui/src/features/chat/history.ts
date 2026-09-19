// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.
/** Local history is memory-only until the caller explicitly enables persistence. */
export interface ChatTurn {
  id: string;
  modelId: string;
  modelRevision: number;
  inferenceId: string;
  modelName: string;
  prompt: string;
  content: string;
  reasoning: string;
  tools: Array<{ index: number; id: string; name: string; arguments: string }>;
  status: 'streaming' | 'complete' | 'cancelled' | 'interrupted' | 'error';
  finishReason: string | null;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null;
  ttftMs: number | null;
  elapsedMs: number | null;
  error: string | null;
  parameters: Record<string, number>;
  images: Array<{ name: string; type: string; dataUrl: string }>;
  /** Wall-clock send time in epoch milliseconds; 0 marks a turn imported from a version-1 payload, whose start is unknown. */
  startedAt: number;
}
export interface ChatConversation {
  id: string;
  title: string;
  systemPrompt: string;
  turns: ChatTurn[];
  updatedAt: number;
}
export interface HistoryOptions { includeImages?: boolean }
export const HISTORY_LIMITS = Object.freeze({
  conversations: 50, turnsPerConversation: 200, totalTurns: 1000,
  textCharacters: 262_144, jsonBytes: 16 * 1024 * 1024,
  imageBytes: 8 * 1024 * 1024, imagesPerTurn: 8, toolsPerTurn: 128,
});
export const HISTORY_DATABASE = 'mlxcel.webui.chat.v1';
const STORE = 'history';
const KEY = 'conversations';
// The IndexedDB schema (one object store holding one JSON string) is independent of the
// payload version inside that string. Bumping the database version would lock older
// builds out with a VersionError, so it stays at 1 while the payload moves to 2.
const DATABASE_VERSION = 1;
// Version 2 adds ChatTurn.startedAt. Version 1 payloads still import, with startedAt 0.
const PAYLOAD_VERSION = 2;
const TURN_KEYS_V1 = ['id', 'modelId', 'modelRevision', 'inferenceId', 'modelName', 'prompt', 'content',
  'reasoning', 'tools', 'status', 'finishReason', 'usage', 'ttftMs', 'elapsedMs', 'error',
  'parameters', 'images'] as const;
const TURN_KEYS_V2 = [...TURN_KEYS_V1, 'startedAt'] as const;
type PayloadVersion = 1 | 2;
const PARAMETER_KEYS = new Set([
  'temperature', 'top_p', 'top_k', 'min_p', 'max_tokens', 'seed',
  'repetition_penalty', 'presence_penalty', 'frequency_penalty',
]);
export class HistoryValidationError extends Error {
  constructor(message = 'Conversation history is invalid or exceeds the local history limits.') {
    super(message); this.name = 'HistoryValidationError';
  }
}
export class HistoryStorageError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options); this.name = 'HistoryStorageError';
  }
}
function invalid(): never { throw new HistoryValidationError(); }
function record(value: unknown, keys: readonly string[]): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return invalid();
  const proto: unknown = Object.getPrototypeOf(value);
  if (proto !== Object.prototype && proto !== null) return invalid();
  const own = Object.keys(value);
  if (own.length !== keys.length || own.some((key) => !keys.includes(key))) return invalid();
  return value as Record<string, unknown>;
}
function string(value: unknown, max: number = HISTORY_LIMITS.textCharacters): string {
  if (typeof value !== 'string' || value.length > max) return invalid();
  return value;
}
function identifier(value: unknown): string {
  const result = string(value, 512);
  if (!result || Array.from(result).some((char) => char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127)) return invalid();
  return result;
}
function number(value: unknown, integer = false): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 ||
      (integer && !Number.isSafeInteger(value))) return invalid();
  return value;
}
function nullable<T>(value: unknown, validate: (v: unknown) => T): T | null {
  return value === null ? null : validate(value);
}
function array(value: unknown, max: number): unknown[] {
  if (!Array.isArray(value) || value.length > max) return invalid();
  return value;
}
function bytes(value: string): number { return new TextEncoder().encode(value).byteLength; }
function normalize(value: unknown, options: HistoryOptions, version: PayloadVersion): ChatConversation[] {
  let totalTurns = 0;
  let totalImages = 0;
  let totalJsonBytes = 0;
  const conversationIds = new Set<string>();
  return array(value, HISTORY_LIMITS.conversations).map((raw) => {
    const item = record(raw, ['id', 'title', 'systemPrompt', 'turns', 'updatedAt']);
    const id = identifier(item.id);
    if (conversationIds.has(id)) return invalid();
    conversationIds.add(id);
    const turnIds = new Set<string>();
    const turns = array(item.turns, HISTORY_LIMITS.turnsPerConversation).map((rawTurn): ChatTurn => {
      if (++totalTurns > HISTORY_LIMITS.totalTurns) return invalid();
      const turn = record(rawTurn, version === 1 ? TURN_KEYS_V1 : TURN_KEYS_V2);
      const turnId = identifier(turn.id);
      if (turnIds.has(turnId)) return invalid();
      turnIds.add(turnId);
      const status = string(turn.status, 32);
      if (!['streaming', 'complete', 'cancelled', 'interrupted', 'error'].includes(status)) return invalid();
      const parameters: Record<string, number> = {};
      if (!turn.parameters || typeof turn.parameters !== 'object' || Array.isArray(turn.parameters) ||
          ![Object.prototype, null].includes(Object.getPrototypeOf(turn.parameters) as object | null)) return invalid();
      for (const [key, parameter] of Object.entries(turn.parameters)) {
        if (!PARAMETER_KEYS.has(key) || typeof parameter !== 'number' || !Number.isFinite(parameter)) return invalid();
        parameters[key] = parameter;
      }
      const toolIndices = new Set<number>();
      const tools = array(turn.tools, HISTORY_LIMITS.toolsPerTurn).map((rawTool) => {
        const tool = record(rawTool, ['index', 'id', 'name', 'arguments']);
        const index = number(tool.index, true);
        if (toolIndices.has(index)) return invalid();
        toolIndices.add(index);
        return { index, id: string(tool.id, 512), name: string(tool.name, 512), arguments: string(tool.arguments) };
      });
      const images = array(turn.images, HISTORY_LIMITS.imagesPerTurn).map((rawImage) => {
        const image = record(rawImage, ['name', 'type', 'dataUrl']);
        const type = string(image.type, 64);
        if (!['image/png', 'image/jpeg', 'image/webp'].includes(type)) return invalid();
        const dataUrl = string(image.dataUrl, HISTORY_LIMITS.imageBytes * 2);
        const prefix = `data:${type};base64,`;
        if (!dataUrl.startsWith(prefix)) return invalid();
        const payload = dataUrl.slice(prefix.length);
        if (!payload || payload.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/u.test(payload)) return invalid();
        totalImages += payload.length * 3 / 4 - (payload.endsWith('==') ? 2 : payload.endsWith('=') ? 1 : 0);
        if (totalImages > HISTORY_LIMITS.imageBytes) return invalid();
        return { name: string(image.name, 512), type, dataUrl };
      });
      const result: ChatTurn = {
        id: turnId, modelId: identifier(turn.modelId), modelRevision: number(turn.modelRevision, true), inferenceId: identifier(turn.inferenceId),
        modelName: string(turn.modelName, 512), prompt: string(turn.prompt), content: string(turn.content),
        reasoning: string(turn.reasoning), tools,
        status: status === 'streaming' ? 'interrupted' : status as ChatTurn['status'],
        finishReason: nullable(turn.finishReason, string),
        usage: nullable(turn.usage, (value) => {
          const usage = record(value, ['prompt_tokens', 'completion_tokens', 'total_tokens']);
          return { prompt_tokens: number(usage.prompt_tokens, true), completion_tokens: number(usage.completion_tokens, true),
            total_tokens: number(usage.total_tokens, true) };
        }),
        ttftMs: nullable(turn.ttftMs, number), elapsedMs: nullable(turn.elapsedMs, number),
        error: nullable(turn.error, string), parameters, images: options.includeImages ? images : [],
        startedAt: version === 1 ? 0 : number(turn.startedAt, true),
      };
      if (result.status === 'complete' && (!result.finishReason || (!result.content && !result.reasoning && !result.tools.length))) return invalid();
      totalJsonBytes += bytes(JSON.stringify(result));
      if (totalJsonBytes > HISTORY_LIMITS.jsonBytes) return invalid();
      return result;
    });
    return { id, title: string(item.title, 512), systemPrompt: string(item.systemPrompt), turns,
      updatedAt: number(item.updatedAt, true) };
  });
}
/** Validates in-memory conversations, which always carry the current (version 2) turn shape. */
export function validateConversations(value: unknown, options: HistoryOptions = {}): ChatConversation[] {
  return validatePayload(value, options, PAYLOAD_VERSION);
}
function validatePayload(value: unknown, options: HistoryOptions, version: PayloadVersion): ChatConversation[] {
  const result = normalize(value, options, version);
  if (bytes(JSON.stringify(result)) > HISTORY_LIMITS.jsonBytes) return invalid();
  return result;
}
export function exportConversations(value: ChatConversation[], options: HistoryOptions = {}): string {
  const json = JSON.stringify({ version: PAYLOAD_VERSION, conversations: validateConversations(value, options) });
  if (bytes(json) > HISTORY_LIMITS.jsonBytes) return invalid();
  return json;
}
export function importConversations(json: string, options: HistoryOptions = {}): ChatConversation[] {
  if (bytes(json) > HISTORY_LIMITS.jsonBytes) return invalid();
  let parsed: unknown;
  try { parsed = JSON.parse(json) as unknown; } catch { return invalid(); }
  const envelope = record(parsed, ['version', 'conversations']);
  if (envelope.version !== 1 && envelope.version !== 2) return invalid();
  return validatePayload(envelope.conversations, options, envelope.version);
}
function storageError(error: unknown): HistoryStorageError {
  const quota = error instanceof DOMException && error.name === 'QuotaExceededError';
  return new HistoryStorageError(quota
    ? 'Local history storage is full. Export or clear history, or disable persistence; the current conversation remains in memory.'
    : 'Local history storage is unavailable. The current conversation remains in memory.', { cause: error });
}
export function createHistoryRepository(options: { indexedDB?: IDBFactory } = {}) {
  let enabled = false;
  let clearing = false;
  let generation = 0;
  let connection: IDBDatabase | undefined;
  let opening: Promise<IDBDatabase> | undefined;
  const transactions = new Set<IDBTransaction>();
  const factory = () => {
    const value = options.indexedDB ?? globalThis.indexedDB;
    if (!value) throw new HistoryStorageError('This browser does not support local history storage.');
    return value;
  };
  const close = () => {
    generation++;
    for (const tx of transactions) { try { tx.abort(); } catch { /* Already completed. */ } }
    transactions.clear(); connection?.close(); connection = undefined; opening = undefined;
  };
  const open = (): Promise<IDBDatabase> => {
    if (connection) return Promise.resolve(connection);
    if (opening) return opening;
    const epoch = generation;
    opening = new Promise((resolve, reject) => {
      const request = factory().open(HISTORY_DATABASE, DATABASE_VERSION);
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(STORE)) request.result.createObjectStore(STORE);
      };
      request.onerror = () => reject(storageError(request.error));
      let blocked = false;
      request.onblocked = () => { blocked = true; reject(new HistoryStorageError('Close other WebUI tabs to access local history.')); };
      request.onsuccess = () => {
        if (blocked || !enabled || epoch !== generation) {
          request.result.close(); reject(new HistoryStorageError('Local history persistence was disabled.')); return;
        }
        connection = request.result;
        connection.onversionchange = close;
        resolve(connection);
      };
    });
    return opening.catch((error: unknown) => { if (epoch === generation) opening = undefined; throw error; });
  };
  async function transact<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T | undefined> {
    if (!enabled || clearing) return undefined;
    const epoch = generation;
    try {
      const db = await open();
      if (!enabled || clearing || generation !== epoch) return undefined;
      return await new Promise<T>((resolve, reject) => {
        const tx = db.transaction(STORE, mode);
        transactions.add(tx);
        const request = action(tx.objectStore(STORE));
        tx.oncomplete = () => { transactions.delete(tx); resolve(request.result); };
        tx.onabort = tx.onerror = () => { transactions.delete(tx); reject(storageError(tx.error ?? request.error)); };
      });
    } catch (error) { throw error instanceof HistoryStorageError ? error : storageError(error); }
  }
  return {
    setEnabled(value: boolean) { if (enabled !== value) { enabled = value; if (!value) close(); } },
    async load(historyOptions: HistoryOptions = {}): Promise<ChatConversation[]> {
      const value: unknown = await transact('readonly', (store) => store.get(KEY));
      return value === undefined ? [] : importConversations(string(value, HISTORY_LIMITS.jsonBytes), historyOptions);
    },
    async save(conversations: ChatConversation[], historyOptions: HistoryOptions = {}): Promise<void> {
      if (!enabled) return;
      const value = exportConversations(conversations, historyOptions);
      await transact('readwrite', (store) => store.put(value, KEY));
    },
    async clear(): Promise<void> {
      if (clearing) throw new HistoryStorageError('Local history is already being cleared.');
      clearing = true;
      let blocked = false;
      close();
      try {
        await new Promise<void>((resolve, reject) => {
          const request = factory().deleteDatabase(HISTORY_DATABASE);
          request.onsuccess = () => { clearing = false; resolve(); };
          request.onerror = () => { clearing = false; reject(storageError(request.error)); };
          request.onblocked = () => { blocked = true; reject(new HistoryStorageError('Close other WebUI tabs to clear local history.')); };
        });
      } catch (error) {
        // A blocked deletion cannot be cancelled; keep writes paused until its eventual completion.
        if (!blocked) clearing = false;
        throw error instanceof HistoryStorageError ? error : storageError(error);
      }
    },
    close,
  };
}
