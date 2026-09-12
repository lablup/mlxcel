import { describe, expect, it } from 'vitest';
import dflashPage from '../../../tests/fixtures/webui/examples/catalog.dflash-page.json';
import { validateAgainstSchema } from './jsonSchema';

function pageWithReason(reason: string) {
  const entry = dflashPage.items[0];
  return {
    ...dflashPage,
    items: [{
      ...entry,
      metadata: {
        ...entry.metadata,
        support: { ...entry.metadata.support, architecturally_supported_reason: reason },
        unknown_reasons: { ...entry.metadata.unknown_reasons, architecture: reason },
      },
    }],
  };
}

describe('JSON Schema Unicode code-point lengths', () => {
  it('validates whole catalog responses with BMP and astral reason boundaries', () => {
    for (const glyph of ['x', '한', '🦀']) {
      for (const count of [511, 512]) {
        expect(() => validateAgainstSchema('CatalogListResponse', pageWithReason(glyph.repeat(count)))).not.toThrow();
      }
      expect(() => validateAgainstSchema('CatalogListResponse', pageWithReason(glyph.repeat(513)))).toThrow('string longer than maximum');
    }
  });

  it('uses the same code-point count for minLength before applying patterns', () => {
    for (const glyph of ['a', '한', '🦀']) {
      expect(() => validateAgainstSchema('IdempotencyKey', glyph.repeat(7))).toThrow('string shorter than minimum');
    }
    expect(() => validateAgainstSchema('IdempotencyKey', 'a'.repeat(8))).not.toThrow();
    expect(() => validateAgainstSchema('IdempotencyKey', '🦀'.repeat(8))).toThrow('string does not match pattern');
  });
});
