// Copyright 2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { SseParser, type SseMessage } from './sse';
import { apiPath, encodeOpaquePathSegment, validateApiBase } from './url';
import { parseJson, validateBootstrap, validateCatalogEntry, validateCatalogList, validateErrorEnvelope, validateOperation, validateOperationAccepted, validateOperationsList, validateRuntime, validateUiEvent } from './validation';
import type { BootstrapResponse, CatalogEntry, CatalogListResponse, CatalogQuery, DownloadRequest, EventId, EventStreamHandlers, ModelActionRequest, ModelId, Operation, OperationAccepted, OperationId, OperationsListResponse, RemovalRequest, RuntimeSnapshot } from './types';

export interface WebUiApiClientOptions {
  readonly apiBase?: string;
  readonly fetchImpl?: typeof fetch;
  readonly now?: () => number;
  readonly onUnauthorized?: () => void;
}

export interface ChatStreamHandlers {
  readonly onFrame: (message: SseMessage) => void;
  readonly onDone?: () => void;
}

export interface EventReplayCursor {
  readonly lastEventId: EventId | null;
  readonly serverInstanceId: string | null;
  readonly afterSequence: number | null;
}

export class WebUiHttpError extends Error {
  constructor(readonly status: number, readonly envelope: ReturnType<typeof validateErrorEnvelope> | null, message?: string) {
    super(message ?? envelope?.error.message ?? `WebUI request failed with HTTP ${status}`);
    this.name = 'WebUiHttpError';
  }
}

const maxJsonBytes = 2 * 1024 * 1024;

export class WebUiApiClient {
  readonly apiBase: string;
  private readonly fetchImpl: typeof fetch;
  private bearerToken: string | null = null;
  private readonly onUnauthorized?: () => void;
  private readonly controllers = new Set<AbortController>();

  constructor(options: WebUiApiClientOptions = {}) {
    this.apiBase = validateApiBase(options.apiBase);
    this.fetchImpl = options.fetchImpl ?? fetch.bind(globalThis);
    this.onUnauthorized = options.onUnauthorized;
  }

  setBearerToken(token: string | null): void {
    this.bearerToken = token;
  }

  abortAll(): void {
    for (const controller of this.controllers) controller.abort();
    this.controllers.clear();
  }

  async bootstrap(signal?: AbortSignal): Promise<BootstrapResponse> {
    return this.request('/ui-api/v1/bootstrap', validateBootstrap, { method: 'GET', signal });
  }

  async catalog(query: CatalogQuery = {}, signal?: AbortSignal): Promise<CatalogListResponse> {
    return this.request('/ui-api/v1/catalog', validateCatalogList, { method: 'GET', query: { ...query }, signal });
  }

  async catalogEntry(modelId: ModelId, signal?: AbortSignal): Promise<CatalogEntry> {
    return this.request(`/ui-api/v1/catalog/${encodeOpaquePathSegment(modelId)}`, validateCatalogEntry, { method: 'GET', signal });
  }

  async refreshCatalog(idempotencyKey: string, signal?: AbortSignal): Promise<OperationAccepted> {
    return this.request('/ui-api/v1/catalog/refresh', validateOperationAccepted, { method: 'POST', body: { idempotency_key: idempotencyKey }, signal });
  }

  async modelAction(request: ModelActionRequest, signal?: AbortSignal): Promise<OperationAccepted> {
    return this.request('/ui-api/v1/model-actions', validateOperationAccepted, { method: 'POST', body: request, signal });
  }

  async download(request: DownloadRequest, signal?: AbortSignal): Promise<OperationAccepted> {
    return this.request('/ui-api/v1/downloads', validateOperationAccepted, { method: 'POST', body: request, signal });
  }

  async removeModel(request: RemovalRequest, signal?: AbortSignal): Promise<OperationAccepted> {
    return this.request('/ui-api/v1/model-removals', validateOperationAccepted, { method: 'POST', body: request, signal });
  }

  async operationsPage(query: { readonly cursor?: string } = {}, signal?: AbortSignal): Promise<OperationsListResponse> {
    return this.request('/ui-api/v1/operations', validateOperationsList, { method: 'GET', query, signal });
  }

  async operations(signal?: AbortSignal): Promise<ReadonlyArray<Operation>> {
    const items: Operation[] = [];
    let cursor: string | undefined;
    let serverInstanceId: string | null = null;
    for (;;) {
      const page = await this.operationsPage(cursor === undefined ? {} : { cursor }, signal);
      if (serverInstanceId === null) serverInstanceId = page.server_instance_id;
      else if (page.server_instance_id !== serverInstanceId) throw new Error('Operations pagination crossed a server restart; refresh the authoritative snapshot.');
      items.push(...page.items);
      if (page.pagination.next_cursor === null) return items;
      cursor = page.pagination.next_cursor;
    }
  }

  async operation(operationId: OperationId, signal?: AbortSignal): Promise<Operation> {
    return this.request(`/ui-api/v1/operations/${encodeOpaquePathSegment(operationId)}`, validateOperation, { method: 'GET', signal });
  }

  async cancelOperation(operationId: OperationId, signal?: AbortSignal): Promise<OperationAccepted> {
    return this.request(`/ui-api/v1/operations/${encodeOpaquePathSegment(operationId)}/cancel`, validateOperationAccepted, { method: 'POST', body: {}, signal });
  }

  async runtime(modelId: ModelId, signal?: AbortSignal): Promise<RuntimeSnapshot> {
    const runtime = await this.request('/ui-api/v1/runtime', validateRuntime, { method: 'GET', query: { model_id: modelId, autoload: false }, signal });
    if (runtime.model_id !== modelId) throw new Error('Runtime response model_id did not match the requested model.');
    return runtime;
  }

  async events(handlers: EventStreamHandlers, signal?: AbortSignal, cursor?: EventReplayCursor): Promise<void> {
    const hasSequenceCursor = cursor?.serverInstanceId !== undefined && cursor.serverInstanceId !== null && cursor.afterSequence !== null && cursor.afterSequence !== undefined;
    const query = hasSequenceCursor ? { server_instance_id: cursor.serverInstanceId, after_sequence: cursor.afterSequence } : undefined;
    const headers = new Headers({ Accept: 'text/event-stream' });
    if (!hasSequenceCursor && cursor?.lastEventId !== undefined && cursor.lastEventId !== null) headers.set('Last-Event-ID', cursor.lastEventId);
    await this.sse(apiPath(this.apiBase, '/ui-api/v1/events', query), { method: 'GET', signal, headers }, {
      onDone: handlers.onDone,
      onFrame: (message: SseMessage) => {
        if (message.retry !== null) handlers.onRetryAfter?.(message.retry);
        if (message.data.length > 0) handlers.onEvent(validateUiEvent(parseJson(message.data)));
      },
    });
  }

  async chatCompletions(inferenceModelId: string, body: unknown, handlers: ChatStreamHandlers, signal?: AbortSignal): Promise<void> {
    await this.sse(apiPath(this.apiBase, '/v1/chat/completions', { autoload: false }), { method: 'POST', body: withInferenceModel(body, inferenceModelId), signal, headers: { Accept: 'text/event-stream' } }, handlers);
  }

  async responses(inferenceModelId: string, body: unknown, handlers: ChatStreamHandlers, signal?: AbortSignal): Promise<void> {
    await this.sse(apiPath(this.apiBase, '/v1/responses', { autoload: false }), { method: 'POST', body: withInferenceModel(body, inferenceModelId), signal, headers: { Accept: 'text/event-stream' } }, handlers);
  }

  private async request<T>(path: string, validate: (value: unknown) => T, options: RequestOptions): Promise<T> {
    const fetched = await this.fetchWithAuth(apiPath(this.apiBase, path, options.query), options);
    try {
      if (!fetched.response.ok) await this.throwHttp(fetched.response, fetched.signal);
      return validate(parseJson(await this.readBody(fetched.response, fetched.signal)));
    } finally {
      this.controllers.delete(fetched.controller);
      fetched.cleanup();
    }
  }

  private async sse(url: string, options: RequestOptions, handlers: ChatStreamHandlers): Promise<void> {
    const fetched = await this.fetchWithAuth(url, options);
    try {
      const response = fetched.response;
      if (!response.ok) await this.throwHttp(response, fetched.signal);
      if (response.body === null) throw new Error('WebUI event stream response has no body.');
      const parser = new SseParser({ onDone: handlers.onDone, onMessage: handlers.onFrame });
      const reader = response.body.getReader();
      try {
        for (;;) {
          const read = await readWithAbort(reader, fetched.signal);
          if (read.done) break;
          parser.push(read.value);
        }
        parser.close();
        if (!parser.done && fetched.signal.aborted !== true) throw new Error('WebUI event stream ended before a DONE frame.');
      } catch (error) {
        await reader.cancel().catch(() => undefined);
        throw error;
      } finally {
        reader.releaseLock();
      }
    } finally {
      this.controllers.delete(fetched.controller);
      fetched.cleanup();
    }
  }

  private async fetchWithAuth(url: string, options: RequestOptions): Promise<FetchedResponse> {
    const controller = new AbortController();
    const { signal, cleanup } = combineSignals(controller, options.signal);
    this.controllers.add(controller);
    const headers = new Headers(options.headers);
    headers.set('Accept', headers.get('Accept') ?? 'application/json');
    if (this.bearerToken !== null) headers.set('Authorization', `Bearer ${this.bearerToken}`);
    if (options.body !== undefined) headers.set('Content-Type', 'application/json');
    try {
      const response = await this.fetchImpl(url, { method: options.method, headers, body: options.body === undefined ? undefined : JSON.stringify(options.body), signal, credentials: 'same-origin', cache: 'no-store', redirect: 'error' });
      return { response, controller, signal, cleanup };
    } catch (error) {
      this.controllers.delete(controller);
      cleanup();
      throw error;
    }
  }

  private async throwHttp(response: Response, signal: AbortSignal): Promise<never> {
    const tokenForRedaction = this.bearerToken;
    if (response.status === 401) {
      this.bearerToken = null;
      this.abortAll();
      this.onUnauthorized?.();
    }
    let envelope: ReturnType<typeof validateErrorEnvelope> | null = null;
    try {
      const text = await this.readBody(response, signal);
      if (text.length > 0) envelope = validateErrorEnvelope(parseJson(text));
    } catch {
      envelope = null;
    }
    throw new WebUiHttpError(response.status, redactEnvelope(envelope, tokenForRedaction), redactMessage(envelope?.error.message ?? `WebUI request failed with HTTP ${response.status}`, tokenForRedaction));
  }

  private async readBody(response: Response, signal: AbortSignal): Promise<string> {
    if (response.body === null) return '';
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let total = 0;
    try {
      for (;;) {
        const read = await readWithAbort(reader, signal);
        if (read.done) break;
        total += read.value.byteLength;
        if (total > maxJsonBytes) throw new Error('WebUI JSON response exceeded the configured byte limit.');
        chunks.push(read.value);
      }
    } catch (error) {
      await reader.cancel().catch(() => undefined);
      throw error;
    } finally {
      reader.releaseLock();
    }
    const merged = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return new TextDecoder().decode(merged);
  }
}

function redactEnvelope(envelope: ReturnType<typeof validateErrorEnvelope> | null, token: string | null): ReturnType<typeof validateErrorEnvelope> | null {
  if (envelope === null) return null;
  return { ...envelope, error: { ...envelope.error, message: redactMessage(envelope.error.message, token), field_errors: envelope.error.field_errors?.map((field) => ({ ...field, message: redactMessage(field.message, token) })) } };
}

function redactMessage(message: string, token: string | null): string {
  let redacted = message.replace(/Bearer\s+\S+/gi, 'Bearer [redacted]').replace(/token[=:]\s*\S+/gi, 'token=[redacted]');
  if (token !== null && token.length > 0) redacted = redacted.split(token).join('[redacted]');
  return redacted;
}

function withInferenceModel(body: unknown, inferenceModelId: string): unknown {
  if (inferenceModelId.length === 0) throw new Error('Inference model id is required for streaming inference.');
  if (typeof body !== 'object' || body === null || Array.isArray(body)) throw new Error('Streaming inference bodies must be JSON objects.');
  return { ...(body as Record<string, unknown>), model: inferenceModelId, stream: true };
}

async function readWithAbort<T>(reader: ReadableStreamDefaultReader<T>, signal: AbortSignal): Promise<ReadableStreamReadResult<T>> {
  if (signal.aborted) {
    await reader.cancel().catch(() => undefined);
    throw abortError();
  }
  let onAbort: (() => void) | null = null;
  try {
    return await Promise.race([
      reader.read(),
      new Promise<ReadableStreamReadResult<T>>((_, reject) => {
        onAbort = () => {
          void reader.cancel().catch(() => undefined);
          reject(abortError());
        };
        signal.addEventListener('abort', onAbort, { once: true });
      }),
    ]);
  } finally {
    if (onAbort !== null) signal.removeEventListener('abort', onAbort);
  }
}

function combineSignals(controller: AbortController, upstream: AbortSignal | undefined): { readonly signal: AbortSignal; readonly cleanup: () => void } {
  if (upstream === undefined) return { signal: controller.signal, cleanup: () => undefined };
  const onAbort = () => controller.abort(upstream.reason);
  if (upstream.aborted) onAbort();
  else upstream.addEventListener('abort', onAbort, { once: true });
  return { signal: controller.signal, cleanup: () => upstream.removeEventListener('abort', onAbort) };
}

function abortError(): Error {
  return typeof DOMException === 'function' ? new DOMException('The WebUI request was aborted.', 'AbortError') : new Error('The WebUI request was aborted.');
}

interface FetchedResponse {
  readonly response: Response;
  readonly controller: AbortController;
  readonly signal: AbortSignal;
  readonly cleanup: () => void;
}

interface RequestOptions {
  readonly method: 'GET' | 'POST' | 'PATCH' | 'DELETE';
  readonly query?: Readonly<Record<string, string | number | boolean | null | undefined>>;
  readonly body?: unknown;
  readonly signal?: AbortSignal;
  readonly headers?: HeadersInit;
}
