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

import { parseJson, validateAgainstSchema, ValidationError } from './jsonSchema';
import type { BootstrapResponse, CatalogListResponse, ErrorEnvelope, Operation, OperationAccepted, RuntimeSnapshot, UiEvent } from './types';

export { parseJson, ValidationError };

function checked<T>(schemaName: string, value: unknown, path: string): T {
  validateAgainstSchema(schemaName, value, path);
  return value as T;
}

export function validateErrorEnvelope(value: unknown, path = '$'): ErrorEnvelope {
  return checked('ErrorEnvelope', value, path);
}

export function validateBootstrap(value: unknown, path = '$'): BootstrapResponse {
  return checked('BootstrapResponse', value, path);
}

export function validateCatalogList(value: unknown, path = '$'): CatalogListResponse {
  return checked('CatalogListResponse', value, path);
}

export function validateOperation(value: unknown, path = '$'): Operation {
  return checked('Operation', value, path);
}

export function validateOperationAccepted(value: unknown, path = '$'): OperationAccepted {
  return checked('OperationAccepted', value, path);
}

export function validateRuntime(value: unknown, path = '$'): RuntimeSnapshot {
  return checked('RuntimeSnapshot', value, path);
}

export function validateUiEvent(value: unknown, path = '$'): UiEvent {
  return checked('UiEvent', value, path);
}

export function validateScalarRecord(value: unknown, path = '$'): Readonly<Record<string, string | number | boolean | null>> {
  validateAgainstSchema('RuntimeSettingsReport', { scope: 'request_only', effective: value, overridden_by_cli: [], partial_errors: [] }, path);
  return value as Readonly<Record<string, string | number | boolean | null>>;
}
