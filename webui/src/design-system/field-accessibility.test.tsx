import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { createRequire } from 'node:module';
import { expect, it } from 'vitest';
import { Field } from './primitives';

interface AxeNameEngine {
  readonly utils: { getFlattenedTree(element: Element): unknown };
  readonly commons: { readonly text: { accessibleText(element: Element): string } };
}

it('keeps hints and validation errors out of the computed field name', () => {
  // Reuse the browser suite's installed axe engine, without adding a shipped dependency.
  const requireHere = createRequire(import.meta.url);
  const requireBrowserAxe = createRequire(requireHere.resolve('@axe-core/playwright'));
  const installed: unknown = requireBrowserAxe('axe-core');
  if (typeof installed !== 'object' || installed === null || !('source' in installed) || typeof installed.source !== 'string') throw new Error('Missing browser axe engine');
  window.eval(installed.source);
  const engine = (window as unknown as { axe: AxeNameEngine }).axe;
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  try {
    for (const error of [undefined, 'Outside the supported range']) {
      act(() => root.render(<Field label="Context" value="2048" hint="Server-validated context tokens" error={error} />));
      const input = host.querySelector('input');
      if (input === null) throw new Error('Missing field input');
      engine.utils.getFlattenedTree(document.documentElement);
      expect(engine.commons.text.accessibleText(input)).toBe('Context');
      const descriptions = input.getAttribute('aria-describedby')?.split(' ').map((id) => document.getElementById(id)?.textContent) ?? [];
      expect(descriptions).toEqual(error === undefined ? ['Server-validated context tokens'] : ['Server-validated context tokens', error]);
      expect(input.labels?.[0].htmlFor).toBe(input.id);
    }
  } finally { act(() => root.unmount()); host.remove(); Reflect.deleteProperty(window, 'axe'); }
});
