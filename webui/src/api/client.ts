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
import { parseJson, validateBootstrap, validateCatalogList, validateErrorEnvelope, validateOperation, validateOperationAccepted, validateRuntime, validateUiEvent, ValidationError } from './validation';
import type { BootstrapResponse, CatalogListResponse, CatalogQuery, DownloadRequest, EventStreamHandlers, ModelActionRequest, ModelId, Operation, OperationAccepted, OperationId, RemovalRequest, RuntimeSnapshot } from './types';

export interface WebUiApiClientOptions {
  readonly apiBase?: string;
  readonly fetchImpl?: typeof fetch;
  readonly now?: () => number;
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
  private readonly controllers = new Set<AbortController>();

  constructor(options: WebUiApiClientOptions = {}) {
    this.apiBase = validateApiBase(options.apiBase);
    this.fetchImpl = options.fetchImpl ?? fetch.bind(globalThis);
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

  async catalogEntry(modelId: ModelId, signal?: AbortSignal): Promise<CatalogListResponse['items'][number]> {
    const path = `/ui-api/v1/catalog/${encodeOpaquePathSegment(modelId)}`;
    return this.request(path, (value) => validateCatalogList({ schema_version: 'webui.ui-api.v1', items: [value], pagination: { limit: 1, next_cursor: null, total_known: 1 }, server_instance_id: 'entry-projection', snapshot_sequence: 0 }).items[0], { method: 'GET', signal });
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

  async operations(signal?: AbortSignal): Promise<ReadonlyArray<Operation>> {
    const value = await this.request('/ui-api/v1/operations', (entry) => entry, { method: 'GET', signal });
    const object = value as { readonly items?: unknown };
    if (!Array.isArray(object.items)) throw new ValidationError('$.items', 'expected operations list');
    return object.items.map((item, index) => validateOperation(item, `$.items[${index}]`));
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

  async events(handlers: EventStreamHandlers, signal?: AbortSignal): Promise<void> {
    const response = await this.fetchWithAuth(apiPath(this.apiBase, '/ui-api/v1/events'), { method: 'GET', signal, headers: { Accept: 'text/event-stream' } });
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
    } finally {
      reader.releaseLock();
    }
  }

  private async request<T>(path: string, validate: (value: unknown) => T, options: RequestOptions): Promise<T> {
    const response = await this.fetchWithAuth(apiPath(this.apiBase, path, options.query), options);
    if (!response.ok) await this.throwHttp(response);
    return validate(parseJson(await response.text()));
  }

  private async fetchWithAuth(url: string, options: RequestOptions): Promise<Response> {
    const controller = new AbortController();
    const signal = options.signal === undefined ? controller.signal : AbortSignal.any([controller.signal, options.signal]);
    this.controllers.add(controller);
    const headers = new Headers(options.headers);
    headers.set('Accept', headers.get('Accept') ?? 'application/json');
    if (this.bearerToken !== null) headers.set('Authorization', `Bearer ${this.bearerToken}`);
    if (options.body !== undefined) headers.set('Content-Type', 'application/json');
    try {
      return await this.fetchImpl(url, { method: options.method, headers, body: options.body === undefined ? undefined : JSON.stringify(options.body), signal, credentials: 'same-origin', cache: 'no-store' });
    } finally {
      this.controllers.delete(controller);
    }
  }

  private async throwHttp(response: Response): Promise<never> {
    const text = await response.text();
    let envelope: ReturnType<typeof validateErrorEnvelope> | null = null;
    if (text.length > 0) envelope = validateErrorEnvelope(parseJson(text));
    if (response.status === 401) {
      this.bearerToken = null;
      this.abortAll();
    }
    throw new WebUiHttpError(response.status, envelope);
  }
}

interface RequestOptions {
  readonly method: 'GET' | 'POST' | 'PATCH' | 'DELETE';
  readonly query?: Readonly<Record<string, string | number | boolean | null | undefined>>;
  readonly body?: unknown;
  readonly signal?: AbortSignal;
  readonly headers?: HeadersInit;
}
