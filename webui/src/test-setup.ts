// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// jsdom has no matchMedia, and Astryx components read media queries while rendering (the
// Selector's popover-or-sheet choice). This default answers as a 1440 px desktop with no
// user preferences, which is what the product's own "no matchMedia" fallbacks assume (wide
// Models inspector, non-compact shell and Chat). A test that needs another answer still
// stubs window.matchMedia itself.
const VIEWPORT_WIDTH = 1440;

function matches(query: string): boolean {
  const min = /\(\s*min-width:\s*(\d+(?:\.\d+)?)px\s*\)/.exec(query);
  const max = /\(\s*max-width:\s*(\d+(?:\.\d+)?)px\s*\)/.exec(query);
  if (!min && !max) return false;
  return (!min || VIEWPORT_WIDTH >= Number(min[1])) && (!max || VIEWPORT_WIDTH <= Number(max[1]));
}

if (typeof window !== 'undefined' && typeof window.matchMedia !== 'function') {
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    writable: true,
    value: (query: string): MediaQueryList => ({
      matches: matches(query),
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
    }),
  });
}
