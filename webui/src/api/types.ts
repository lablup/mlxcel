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

export type * from '../../../docs/webui/generated/ui-api';

import type {
  BootstrapResponse,
  CatalogEntry,
  CatalogListResponse,
  ErrorEnvelope,
  EventId,
  ModelId,
  Operation,
  OperationAccepted,
  OperationId,
  RuntimeSnapshot,
  SchemaVersion,
  ServerInstanceId,
  UiEvent,
} from '../../../docs/webui/generated/ui-api';

export const WEBUI_SCHEMA_VERSION: SchemaVersion = 'webui.ui-api.v1';

export type AuthSnapshot =
  | { readonly status: 'signed-out'; readonly tokenPresent: false }
  | { readonly status: 'authenticating'; readonly tokenPresent: true }
  | { readonly status: 'authenticated'; readonly tokenPresent: true };

export type ConnectionPhase =
  | 'idle'
  | 'bootstrapping'
  | 'ready'
  | 'streaming'
  | 'polling'
  | 'offline'
  | 'stale'
  | 'unauthorized'
  | 'forbidden'
  | 'schema-mismatch'
  | 'error';

export interface UiClientError {
  readonly code: string;
  readonly message: string;
  readonly retryable: boolean;
  readonly status?: number;
  readonly requestId?: string;
}

export interface PendingReconciliation {
  readonly kind: 'model-action' | 'download' | 'removal' | 'catalog-refresh';
  readonly idempotencyKey: string;
  readonly operationId: OperationId | null;
  readonly modelId?: ModelId;
  readonly createdAt: number;
}

export interface WebUiSnapshot {
  readonly schemaVersion: SchemaVersion;
  readonly auth: AuthSnapshot;
  readonly connection: ConnectionPhase;
  readonly bootstrap: BootstrapResponse | null;
  readonly catalog: ReadonlyArray<CatalogEntry>;
  readonly catalogSequence: number | null;
  readonly operations: ReadonlyMap<OperationId, Operation>;
  readonly runtimes: ReadonlyMap<ModelId, RuntimeSnapshot>;
  readonly selectedModelId: ModelId | null;
  readonly serverInstanceId: ServerInstanceId | null;
  readonly lastEventId: EventId | null;
  readonly lastSequence: number | null;
  readonly lastUpdatedAt: number | null;
  readonly lastSuccessfulAt: number | null;
  readonly error: UiClientError | null;
  readonly pendingReconciliations: ReadonlyMap<string, PendingReconciliation>;
  readonly resourceFences: ResourceFences;
}

export interface ResourceFences {
  readonly catalog: number | null;
  readonly operationsSnapshot: number | null;
  readonly operations: ReadonlyMap<OperationId, number>;
  readonly models: ReadonlyMap<ModelId, number>;
  readonly runtimes: ReadonlyMap<ModelId, number>;
}


export interface CatalogQuery {
  readonly limit?: number;
  readonly cursor?: string;
  readonly q?: string;
  readonly source?: string;
  readonly task?: string;
  readonly lifecycle?: string;
  readonly support?: boolean;
  readonly completeness?: boolean;
}

export interface EventStreamHandlers {
  readonly onEvent: (event: UiEvent) => void;
  readonly onRetryAfter?: (milliseconds: number) => void;
  readonly onDone?: () => void;
}

export interface RuntimeBoundary {
  readonly bootstrap: BootstrapResponse;
  readonly catalog?: CatalogListResponse;
  readonly operation?: Operation;
  readonly operationAccepted?: OperationAccepted;
  readonly event?: UiEvent;
  readonly error?: ErrorEnvelope;
}
