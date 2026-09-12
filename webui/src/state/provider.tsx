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

import React from 'react';
import { WebUiApiClient, WebUiHttpError, type ChatStreamHandlers } from '../api/client';
import type { DownloadRequest, ModelActionRequest, ModelId, RemovalRequest, RuntimeSnapshot, WebUiSnapshot } from '../api/types';
import { initialSnapshot, reduceWebUiSnapshot } from './reducer';
import { WebUiSynchronizer } from './sync';

export interface WebUiProviderProps {
  readonly children: React.ReactNode;
  readonly apiBase?: string;
  readonly fetchImpl?: typeof fetch;
}

export interface WebUiActions {
  readonly login: (token: string) => Promise<void>;
  readonly logout: () => void;
  readonly refresh: () => Promise<void>;
  readonly selectModel: (modelId: ModelId | null) => void;
  readonly loadModel: (request: ModelActionRequest) => Promise<void>;
  readonly unloadModel: (request: ModelActionRequest) => Promise<void>;
  readonly downloadModel: (request: DownloadRequest) => Promise<void>;
  readonly removeModel: (request: RemovalRequest) => Promise<void>;
  readonly refreshRuntime: (modelId: ModelId) => Promise<RuntimeSnapshot>;
  readonly refreshCatalog: (idempotencyKey: string) => Promise<void>;
  readonly cancelOperation: (operationId: string) => Promise<void>;
  readonly streamChatCompletions: (modelId: ModelId, body: unknown, handlers: ChatStreamHandlers, signal?: AbortSignal) => Promise<void>;
  readonly streamResponses: (modelId: ModelId, body: unknown, handlers: ChatStreamHandlers, signal?: AbortSignal) => Promise<void>;
}

const SnapshotContext = React.createContext<WebUiSnapshot | null>(null);
const ActionsContext = React.createContext<WebUiActions | null>(null);

export function WebUiProvider({ children, apiBase, fetchImpl }: WebUiProviderProps): React.JSX.Element {
  const [snapshot, dispatch] = React.useReducer(reduceWebUiSnapshot, undefined, initialSnapshot);
  const snapshotRef = React.useRef(snapshot);
  snapshotRef.current = snapshot;
  const sessionRef = React.useRef(0);
  const syncRef = React.useRef<WebUiSynchronizer | null>(null);
  const client = React.useMemo(() => new WebUiApiClient({ apiBase, fetchImpl, onUnauthorized: () => {
    sessionRef.current += 1;
    syncRef.current?.stop();
    dispatch({ type: 'logout', now: Date.now() });
  } }), [apiBase, fetchImpl]);

  React.useEffect(() => {
    const sync = new WebUiSynchronizer({ client, dispatch, getSnapshot: () => snapshotRef.current });
    syncRef.current = sync;
    return () => {
      sync.dispose();
      syncRef.current = null;
      client.abortAll();
    };
  }, [client]);

  const actions = React.useMemo<WebUiActions>(() => ({
    login: async (token: string) => {
      sessionRef.current += 1;
      const session = sessionRef.current;
      client.abortAll();
      client.setBearerToken(token);
      dispatch({ type: 'login-start' });
      try {
        const bootstrap = await client.bootstrap();
        if (sessionRef.current !== session) return;
        dispatch({ type: 'login-success', bootstrap, now: Date.now() });
        syncRef.current?.start();
      } catch (error) {
        if (sessionRef.current === session) {
          client.setBearerToken(null);
          client.abortAll();
          dispatch({ type: 'logout', now: Date.now() });
        }
        throw error;
      }
    },
    logout: () => {
      sessionRef.current += 1;
      syncRef.current?.stop();
      client.setBearerToken(null);
      client.abortAll();
      dispatch({ type: 'logout', now: Date.now() });
    },
    refresh: async () => {
      await syncRef.current?.refresh();
    },
    selectModel: (modelId: ModelId | null) => {
      client.abortAll();
      dispatch({ type: 'select-model', modelId });
      void syncRef.current?.refresh();
    },
    loadModel: async (request: ModelActionRequest) => {
      await submitOperation('model-action', request.idempotency_key, request.model_id, () => client.modelAction(request));
    },
    unloadModel: async (request: ModelActionRequest) => {
      await submitOperation('model-action', request.idempotency_key, request.model_id, () => client.modelAction(request));
    },
    downloadModel: async (request: DownloadRequest) => {
      await submitOperation('download', request.idempotency_key, undefined, () => client.download(request));
    },
    removeModel: async (request: RemovalRequest) => {
      await submitOperation('removal', request.idempotency_key, request.model_id, () => client.removeModel(request));
    },
    refreshRuntime: async (modelId: ModelId) => {
      const session = sessionRef.current;
      const runtime = await client.runtime(modelId);
      if (sessionRef.current === session) dispatch({ type: 'runtime', runtime, sequence: runtime.snapshot_sequence, now: Date.now() });
      return runtime;
    },
    refreshCatalog: async (idempotencyKey: string) => {
      await submitOperation('catalog-refresh', idempotencyKey, undefined, () => client.refreshCatalog(idempotencyKey));
    },
    cancelOperation: async (operationId: string) => {
      await client.cancelOperation(operationId);
      await syncRef.current?.refresh();
    },
    streamChatCompletions: async (modelId: ModelId, body: unknown, handlers: ChatStreamHandlers, signal?: AbortSignal) => {
      await client.chatCompletions(resolveInferenceModelId(modelId), body, handlers, signal);
    },
    streamResponses: async (modelId: ModelId, body: unknown, handlers: ChatStreamHandlers, signal?: AbortSignal) => {
      await client.responses(resolveInferenceModelId(modelId), body, handlers, signal);
    },
  }), [client]);

  async function submitOperation(kind: 'model-action' | 'download' | 'removal' | 'catalog-refresh', idempotencyKey: string, modelId: ModelId | undefined, submit: () => Promise<{ readonly operation_id: string }>): Promise<void> {
    try {
      const accepted = await submit();
      syncRef.current?.noteUnknownPost({ kind, idempotencyKey, operationId: accepted.operation_id, modelId, createdAt: Date.now() });
    } catch (error) {
      if (!(error instanceof WebUiHttpError)) {
        syncRef.current?.noteUnknownPost({ kind, idempotencyKey, operationId: null, modelId, createdAt: Date.now() });
      }
      throw error;
    }
  }

  function resolveInferenceModelId(modelId: ModelId): string {
    const entry = snapshotRef.current.catalog.find((item) => item.identity.id === modelId);
    if (entry === undefined) throw new Error('Selected model is not present in the catalog snapshot.');
    if (entry.identity.inference_id.length === 0) throw new Error('Selected model does not expose an inference model id.');
    return entry.identity.inference_id;
  }

  return <ActionsContext.Provider value={actions}><SnapshotContext.Provider value={snapshot}>{children}</SnapshotContext.Provider></ActionsContext.Provider>;
}


export function useWebUi(): WebUiSnapshot {
  const value = React.useContext(SnapshotContext);
  if (value === null) throw new Error('useWebUi must be used within WebUiProvider.');
  return value;
}

export function useWebUiActions(): WebUiActions {
  const value = React.useContext(ActionsContext);
  if (value === null) throw new Error('useWebUiActions must be used within WebUiProvider.');
  return value;
}
