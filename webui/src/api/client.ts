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

import { SseParser } from './sse';
import { apiPath, encodeOpaquePathSegment, validateApiBase } from './url';
import { parseJson, validateBootstrap, validateCatalogEntry, validateCatalogList, validateErrorEnvelope, validateOperation, validateOperationAccepted, validateOperationsList, validateRuntime, validateUiEvent } from './validation';
import type { BootstrapResponse, CatalogEntry, CatalogListResponse, CatalogQuery, DownloadRequest, EventId, EventStreamHandlers, ModelActionRequest, ModelId, Operation, OperationAccepted, OperationId, OperationsListResponse, RemovalRequest, RuntimeSnapshot } from './types';

export interface WebUiApiClientOptions {
  readonly apiBase?: string;
  readonly fetchImpl?: typeof fetch;
  readonly now?: () => number;
  readonly onUnauthorized?: () => void;
}

export class WebUiHttpError extends Error {
  constructor(readonly status: number, readonly envelope: ReturnType<typeof validateErrorEnvelope> | null) {
    super(envelope?.error.message ?? `WebUI request failed with HTTP ${status}`);
    this.name = 'WebUiHttpError';
  }
}

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
    const path = `/ui-api/v1/catalog/${encodeOpaquePathSegment(modelId)}`;
    return this.request(path, validateCatalogEntry, { method: 'GET', signal });
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

  async operationsPage(signal?: AbortSignal): Promise<OperationsListResponse> {
    return this.request('/ui-api/v1/operations', validateOperationsList, { method: 'GET', signal });
  }

  async operations(signal?: AbortSignal): Promise<ReadonlyArray<Operation>> {
    return (await this.operationsPage(signal)).items;
  }

  async operation(operationId: OperationId, signal?: AbortSignal): Promise<Operation> {
    return this.request(`/ui-api/v1/operations/${encodeOpaquePathSegment(operationId)}`, validateOperation, { method: 'GET', signal });
  }

  async cancelOperation(operationId: OperationId, signal?: AbortSignal): Promise<OperationAccepted> {
    return this.request(`/ui-api/v1/operations/${encodeOpaquePathSegment(operationId)}/cancel`, validateOperationAccepted, { method: 'POST', body: {}, signal });
  }

  async runtime(modelId: ModelId, signal?: AbortSignal): Promise<RuntimeSnapshot> {
    return this.request('/ui-api/v1/runtime', validateRuntime, { method: 'GET', query: { model_id: modelId, autoload: false }, signal });
  }

  async events(handlers: EventStreamHandlers, signal?: AbortSignal, lastEventId?: EventId | null): Promise<void> {
    const headers = new Headers({ Accept: 'text/event-stream' });
    if (lastEventId !== undefined && lastEventId !== null) headers.set('Last-Event-ID', lastEventId);
    const fetched = await this.fetchWithAuth(apiPath(this.apiBase, '/ui-api/v1/events'), { method: 'GET', signal, headers });
    try {
      const response = fetched.response;
      if (!response.ok) await this.throwHttp(response);
      if (response.body === null) throw new Error('WebUI event stream response has no body.');
      const parser = new SseParser({
        onDone: handlers.onDone,
        onMessage: (message) => {
          if (message.retry !== null) handlers.onRetryAfter?.(message.retry);
          if (message.data.length > 0) handlers.onEvent(validateUiEvent(parseJson(message.data)));
        },
      });
      const reader = response.body.getReader();
      try {
        for (;;) {
          const read = await reader.read();
          if (read.done) break;
          parser.push(read.value);
        }
        parser.close();
        if (!parser.done && signal?.aborted !== true) throw new Error('WebUI event stream ended before a DONE frame.');
      } catch (error) {
        await reader.cancel().catch(() => undefined);
        throw error;
      } finally {
        reader.releaseLock();
      }
    } finally {
      this.controllers.delete(fetched.controller);
    }
  }

  private async request<T>(path: string, validate: (value: unknown) => T, options: RequestOptions): Promise<T> {
    const fetched = await this.fetchWithAuth(apiPath(this.apiBase, path, options.query), options);
    try {
      if (!fetched.response.ok) await this.throwHttp(fetched.response);
      return validate(parseJson(await fetched.response.text()));
    } finally {
      this.controllers.delete(fetched.controller);
    }
  }

  private async fetchWithAuth(url: string, options: RequestOptions): Promise<FetchedResponse> {
    const controller = new AbortController();
    const signal = options.signal === undefined ? controller.signal : AbortSignal.any([controller.signal, options.signal]);
    this.controllers.add(controller);
    const headers = new Headers(options.headers);
    headers.set('Accept', headers.get('Accept') ?? 'application/json');
    if (this.bearerToken !== null) headers.set('Authorization', `Bearer ${this.bearerToken}`);
    if (options.body !== undefined) headers.set('Content-Type', 'application/json');
    const response = await this.fetchImpl(url, { method: options.method, headers, body: options.body === undefined ? undefined : JSON.stringify(options.body), signal, credentials: 'same-origin', cache: 'no-store' });
    return { response, controller };
  }

  private async throwHttp(response: Response): Promise<never> {
    if (response.status === 401) {
      this.bearerToken = null;
      this.abortAll();
      this.onUnauthorized?.();
    }
    const text = await response.text().catch(() => '');
    const envelope = text.length > 0 ? validateErrorEnvelope(parseJson(text)) : null;
    throw new WebUiHttpError(response.status, envelope);
  }
}

interface FetchedResponse {
  readonly response: Response;
  readonly controller: AbortController;
}

interface RequestOptions {
  readonly method: 'GET' | 'POST' | 'PATCH' | 'DELETE';
  readonly query?: Readonly<Record<string, string | number | boolean | null | undefined>>;
  readonly body?: unknown;
  readonly signal?: AbortSignal;
  readonly headers?: HeadersInit;
}
