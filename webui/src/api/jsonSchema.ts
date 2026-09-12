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

import contractText from '../../../docs/webui/api.yaml?raw';

export class ValidationError extends Error {
  constructor(readonly path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = 'ValidationError';
  }
}

type JsonObject = Readonly<Record<string, unknown>>;

const contract: JsonObject = parseContract(contractText);

export function validateAgainstSchema(schemaName: string, value: unknown, path = '$'): void {
  const schemas = objectAt(contract, ['components', 'schemas'], '#/components/schemas');
  const schema = schemas[schemaName];
  if (schema === undefined) throw new ValidationError(path, `unknown schema ${schemaName}`);
  validateSchema(schema, value, path);
}

export function parseJson(text: string): unknown {
  const parsed: unknown = JSON.parse(text);
  return parsed;
}

function parseContract(text: string): JsonObject {
  const parsed = parseJson(text);
  return object(parsed, '$contract');
}

function validateSchema(schemaValue: unknown, value: unknown, path: string): void {
  const schema = object(schemaValue, `${path}#schema`);
  if (typeof schema.$ref === 'string') {
    validateSchema(resolveRef(schema.$ref), value, path);
    return;
  }
  if (schema.anyOf !== undefined) {
    if (!Array.isArray(schema.anyOf) || !schema.anyOf.some((candidate) => tryValidate(candidate, value, path))) throw new ValidationError(path, 'did not match any allowed schema');
    return;
  }
  if (schema.oneOf !== undefined) {
    if (!Array.isArray(schema.oneOf)) throw new ValidationError(path, 'schema oneOf must be an array');
    const matches = schema.oneOf.filter((candidate) => tryValidate(candidate, value, path)).length;
    if (matches !== 1) throw new ValidationError(path, `matched ${matches} oneOf schemas`);
    return;
  }
  if (schema.const !== undefined && value !== schema.const) throw new ValidationError(path, `expected constant ${String(schema.const)}`);
  if (Array.isArray(schema.enum) && !schema.enum.some((entry) => entry === value)) throw new ValidationError(path, `unexpected enum value ${String(value)}`);
  const type = schema.type;
  if (Array.isArray(type)) {
    if (!type.some((candidate) => validateType(candidate, value))) throw new ValidationError(path, `expected one of ${type.join(', ')}`);
  } else if (typeof type === 'string' && !validateType(type, value)) {
    throw new ValidationError(path, `expected ${type}`);
  }
  if (typeof value === 'string') validateString(schema, value, path);
  if (typeof value === 'number') validateNumber(schema, value, path);
  if (Array.isArray(value)) validateArray(schema, value, path);
  if (isPlainObject(value)) validateObject(schema, value, path);
}

function validateObject(schema: JsonObject, value: JsonObject, path: string): void {
  const required = arrayOfStrings(schema.required, `${path}#schema.required`, false);
  const properties = schema.properties === undefined ? {} : object(schema.properties, `${path}#schema.properties`);
  for (const key of required) if (!(key in value)) throw new ValidationError(`${path}.${key}`, 'missing required property');
  if (schema.maxProperties !== undefined && Object.keys(value).length > numberSchema(schema.maxProperties, `${path}#schema.maxProperties`)) throw new ValidationError(path, 'too many object properties');
  for (const [key, entry] of Object.entries(value)) {
    const propertySchema = properties[key];
    if (propertySchema !== undefined) {
      validateSchema(propertySchema, entry, `${path}.${key}`);
    } else if (schema.additionalProperties === false) {
      throw new ValidationError(`${path}.${key}`, 'unexpected property');
    } else if (isPlainObject(schema.additionalProperties)) {
      validateSchema(schema.additionalProperties, entry, `${path}.${key}`);
    }
  }
}

function validateArray(schema: JsonObject, value: readonly unknown[], path: string): void {
  if (schema.minItems !== undefined && value.length < numberSchema(schema.minItems, `${path}#schema.minItems`)) throw new ValidationError(path, 'array shorter than minimum');
  if (schema.maxItems !== undefined && value.length > numberSchema(schema.maxItems, `${path}#schema.maxItems`)) throw new ValidationError(path, 'array longer than maximum');
  if (schema.items !== undefined) value.forEach((entry, index) => validateSchema(schema.items, entry, `${path}[${index}]`));
}

function validateString(schema: JsonObject, value: string, path: string): void {
  if (schema.minLength !== undefined && value.length < numberSchema(schema.minLength, `${path}#schema.minLength`)) throw new ValidationError(path, 'string shorter than minimum');
  if (schema.maxLength !== undefined && value.length > numberSchema(schema.maxLength, `${path}#schema.maxLength`)) throw new ValidationError(path, 'string longer than maximum');
  if (typeof schema.pattern === 'string' && !new RegExp(schema.pattern, 'u').test(value)) throw new ValidationError(path, 'string does not match pattern');
  if (schema.format === 'date-time' && !isRfc3339DateTime(value)) throw new ValidationError(path, 'invalid RFC3339 date-time');
}

function isRfc3339DateTime(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value)) return false;
  return !Number.isNaN(Date.parse(value));
}

function validateNumber(schema: JsonObject, value: number, path: string): void {
  if (!Number.isFinite(value)) throw new ValidationError(path, 'expected finite number');
  if (schema.type === 'integer' && !Number.isInteger(value)) throw new ValidationError(path, 'expected integer');
  if (schema.minimum !== undefined && value < numberSchema(schema.minimum, `${path}#schema.minimum`)) throw new ValidationError(path, 'number below minimum');
  if (schema.maximum !== undefined && value > numberSchema(schema.maximum, `${path}#schema.maximum`)) throw new ValidationError(path, 'number above maximum');
}

function validateType(type: string, value: unknown): boolean {
  if (type === 'null') return value === null;
  if (type === 'array') return Array.isArray(value);
  if (type === 'object') return isPlainObject(value);
  if (type === 'integer') return typeof value === 'number' && Number.isInteger(value);
  if (type === 'number') return typeof value === 'number' && Number.isFinite(value);
  return typeof value === type;
}

function resolveRef(ref: string): unknown {
  if (!ref.startsWith('#/')) throw new ValidationError('$ref', `unsupported external ref ${ref}`);
  const tokens = ref.slice(2).split('/').map((part) => part.replace(/~1/g, '/').replace(/~0/g, '~'));
  return tokens.reduce<unknown>((current, token) => object(current, ref)[token], contract);
}

function tryValidate(schema: unknown, value: unknown, path: string): boolean {
  try {
    validateSchema(schema, value, path);
    return true;
  } catch (error) {
    if (error instanceof ValidationError) return false;
    throw error;
  }
}

function object(value: unknown, path: string): JsonObject {
  if (!isPlainObject(value)) throw new ValidationError(path, 'expected object');
  return value;
}

function objectAt(value: unknown, keys: readonly string[], path: string): JsonObject {
  return keys.reduce<unknown>((current, key) => object(current, path)[key], value) as JsonObject;
}

function isPlainObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function arrayOfStrings(value: unknown, path: string, required: boolean): readonly string[] {
  if (value === undefined && !required) return [];
  if (!Array.isArray(value) || !value.every((entry) => typeof entry === 'string')) throw new ValidationError(path, 'expected string array');
  return value;
}

function numberSchema(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new ValidationError(path, 'expected numeric schema bound');
  return value;
}
