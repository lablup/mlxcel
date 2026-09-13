import { describe, expect, it } from 'vitest';
import { validateCatalogEntry } from './validation';
import { validateAgainstSchema } from './jsonSchema';

type MutableJsonObject = Record<string, unknown>;

const fixtures = import.meta.glob('../../../tests/fixtures/webui/**/*.json', { eager: true, import: 'default' });

const exampleSchemas = new Map<string, string>([
  ['bootstrap.model-free.json', 'BootstrapResponse'],
  ['catalog.page.json', 'CatalogListResponse'],
  ['error.forbidden-origin.json', 'ErrorEnvelope'],
  ['error.stale-revision.json', 'ErrorEnvelope'],
  ['error.unauthorized.json', 'ErrorEnvelope'],
  ['event.1.json', 'UiEvent'],
  ['event.2.json', 'UiEvent'],
  ['event.3.json', 'UiEvent'],
  ['event.gap.json', 'UiEvent'],
  ['operation.accepted.json', 'OperationAccepted'],
  ['operation.download-running.json', 'Operation'],
  ['operation.download-succeeded.json', 'Operation'],
  ['operation.running.json', 'Operation'],
  ['operation.succeeded.json', 'Operation'],
  ['operations.list.json', 'OperationsListResponse'],
  ['operations.succeeded-list.json', 'OperationsListResponse'],
  ['request.download.json', 'DownloadRequest'],
  ['request.event-replay-query.json', 'EventReplayQuery'],
  ['request.model-action.load.json', 'ModelActionRequest'],
  ['request.removal.json', 'RemovalRequest'],
  ['runtime.snapshot.json', 'RuntimeSnapshot'],
]);

function cloneFixture(path: string): MutableJsonObject {
  return stripSchemaName(structuredClone(fixtures[path])) as MutableJsonObject;
}

function stripSchemaName(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripSchemaName);
  if (typeof value !== 'object' || value === null) return value;
  const result: MutableJsonObject = {};
  for (const [key, entry] of Object.entries(value)) if (key !== '$schemaName') result[key] = stripSchemaName(entry);
  return result;
}

function schemaFor(path: string): string {
  const name = path.split('/').at(-1) ?? path;
  const annotated = fixtures[path] as MutableJsonObject | undefined;
  if (typeof annotated?.$schemaName === 'string') return annotated.$schemaName;
  if (path.includes('/examples/error.security-')) return 'ErrorEnvelope';
  if (path.includes('/examples/')) return exampleSchemas.get(name) ?? failSchema(path);
  if (path.includes('/scenarios/')) return 'WebUiContractFixture';
  if (name === 'identity-vectors.json') return 'IdentityVectors';
  if (name === 'requirement-map.json') return 'RequirementMap';
  if (name === 'strings.json') return 'StringCatalog';
  return failSchema(path);
}

function failSchema(path: string): never {
  throw new Error(`No schema mapping for ${path}`);
}

describe('canonical WebUI contract fixtures', () => {
  it('validates every shared fixture at the JavaScript runtime boundary', () => {
    const entries = Object.entries(fixtures).sort(([left], [right]) => left.localeCompare(right));
    expect(entries).toHaveLength(43);
    for (const [path] of entries) validateAgainstSchema(schemaFor(path), cloneFixture(path), path);
  });

  it('preserves raw and resolved catalog identity and validates additive bounds', () => {
    const page = cloneFixture('../../../tests/fixtures/webui/examples/catalog.page.json');
    const entry = (page.items as MutableJsonObject[])[0];
    const metadata = entry.metadata as MutableJsonObject;
    metadata.model_type = 'qwen3';
    metadata.architecture = 'qwen3_embedding';
    metadata.declared_architectures = ['Qwen3ForSequenceClassification'];
    const parsed = validateCatalogEntry(entry);
    expect(parsed.metadata.model_type).toBe('qwen3');
    expect(parsed.metadata.architecture).toBe('qwen3_embedding');
    expect(parsed.metadata.declared_architectures).toEqual(['Qwen3ForSequenceClassification']);
    expect(parsed.removal.eligible).toBe(false);
    expect(parsed.removal.reason).not.toBeNull();
    for (const field of ['removal', 'metadata']) {
      const missing = Object.fromEntries(Object.entries(entry).filter(([key]) => key !== field));
      expect(() => validateCatalogEntry(missing)).toThrow(/missing required/);
    }
    for (const field of ['model_type', 'declared_architectures', 'unknown_reasons']) {
      const missing = Object.fromEntries(Object.entries(metadata).filter(([key]) => key !== field));
      expect(() => validateCatalogEntry({ ...entry, metadata: missing })).toThrow(/missing required/);
    }
    metadata.declared_architectures = Array(17).fill('Architecture');
    expect(() => validateCatalogEntry(entry)).toThrow();
    metadata.declared_architectures = ['a'.repeat(129)];
    expect(() => validateCatalogEntry(entry)).toThrow();
  });

  it('rejects extra properties, missing required nulls, invalid date-time and wrong discriminators', () => {
    const bootstrap = cloneFixture('../../../tests/fixtures/webui/examples/bootstrap.model-free.json');
    expect(() => validateAgainstSchema('BootstrapResponse', { ...bootstrap, leaked: true })).toThrow(/unexpected property/);
    const catalog = cloneFixture('../../../tests/fixtures/webui/examples/catalog.page.json');
    const items = catalog.items as MutableJsonObject[];
    const metadata = items[0].metadata as MutableJsonObject;
    delete metadata.architecture;
    expect(() => validateAgainstSchema('CatalogListResponse', catalog)).toThrow(/missing required property/);
    const event = cloneFixture('../../../tests/fixtures/webui/examples/event.1.json');
    event.emitted_at = '2026-09-12';
    expect(() => validateAgainstSchema('ModelRevisionEvent', event)).toThrow(/date-time/);
    const operations = cloneFixture('../../../tests/fixtures/webui/examples/operation.running.json');
    const target = operations.target as MutableJsonObject;
    target.target_kind = 'wrong';
    expect(() => validateAgainstSchema('Operation', operations)).toThrow(/oneOf|enum|constant|schema/);
  });
});
