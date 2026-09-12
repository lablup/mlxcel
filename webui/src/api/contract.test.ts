import { describe, expect, it } from 'vitest';
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
    expect(entries).toHaveLength(33);
    for (const [path] of entries) validateAgainstSchema(schemaFor(path), cloneFixture(path), path);
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
