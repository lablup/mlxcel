// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import type { Operation, RuntimeSnapshot, WebUiSnapshot } from '../api/types';

export const HISTORY_WINDOW_MS = 300_000;
export const HISTORY_RESOLUTION_MS = 2_000;
export const HISTORY_LIMIT = 150;
export const TERMINAL_LIMIT = 200;

export function appendRuntimeHistory(history: WebUiSnapshot['runtimeHistory'], runtime: RuntimeSnapshot, now: number): WebUiSnapshot['runtimeHistory'] {
  const retained = history.filter((sample) => sample.runtime.server_instance_id === runtime.server_instance_id && sample.runtime.model_id === runtime.model_id && sample.receivedAt > now - HISTORY_WINDOW_MS && sample.receivedAt <= now);
  const last = retained.at(-1);
  // A lifecycle event and its poll share one two-second bucket, not two samples.
  if (last !== undefined && Math.floor(last.receivedAt / HISTORY_RESOLUTION_MS) === Math.floor(now / HISTORY_RESOLUTION_MS)) retained.pop();
  return [...retained, { receivedAt: now, runtime }].slice(-HISTORY_LIMIT);
}

export function boundOperations(input: ReadonlyMap<string, Operation>, now: number): ReadonlyMap<string, Operation> {
  const sorted = [...input.values()].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  const active = sorted.filter((op) => !isTerminal(op));
  const terminal = sorted.filter((op) => isTerminal(op) && Date.parse(op.updated_at) > now - 3_600_000).slice(0, TERMINAL_LIMIT);
  return new Map([...active, ...terminal].map((op) => [op.operation_id, op]));
}

export function isTerminal(operation: Operation): boolean {
  return ['succeeded', 'failed', 'cancelled'].includes(operation.state);
}
