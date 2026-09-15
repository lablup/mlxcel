// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, it } from 'vitest';
import HighlightedCode from './code-highlight';
it('highlights supported lexical tokens while HTML remains escaped text', () => {
  const markup = renderToStaticMarkup(<HighlightedCode code={'const x = "<script>"; 42'} language="js" />);
  expect(markup).toContain('chat-code-keyword'); expect(markup).toContain('chat-code-string'); expect(markup).toContain('chat-code-number');
  expect(markup).toContain('&lt;script&gt;'); expect(markup).not.toContain('<script>');
});
it('leaves unknown languages and oversized blocks plain and preserves the full code', () => {
  expect(renderToStaticMarkup(<HighlightedCode code="const x = 42" language="unknown" />)).toBe('<code>const x = 42</code>');
  const long = 'const '.repeat(6000);
  expect(renderToStaticMarkup(<HighlightedCode code={long} language="js" />)).toBe(`<code>${long}</code>`);
});
it('caps token nodes but preserves remaining code', () => {
  const markup = renderToStaticMarkup(<HighlightedCode code={'1 '.repeat(5000)} language="js" />);
  expect((markup.match(/<span/g) || []).length).toBe(4096); expect(markup).toContain('1 '.repeat(904));
});
