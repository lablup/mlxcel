// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// jsdom has no matchMedia, and Astryx components read media queries while rendering (the
// Selector's popover-or-sheet choice). This default answers as a 1440 px desktop with no
// user preferences, which is what the product's own "no matchMedia" fallbacks assume (wide
// Models inspector, non-compact shell and Chat). A test that needs another answer still
// stubs window.matchMedia itself.
const VIEWPORT_WIDTH = 1440;

/** Whether a width media query holds at `width`: min-/max-width and the range forms AppShell uses (`width < 1024px`). */
export function matchesWidth(query: string, width: number): boolean {
  const min = /\(\s*min-width:\s*(\d+(?:\.\d+)?)px\s*\)/.exec(query);
  const max = /\(\s*max-width:\s*(\d+(?:\.\d+)?)px\s*\)/.exec(query);
  const range = /\(\s*width\s*(<=|>=|<|>)\s*(\d+(?:\.\d+)?)px\s*\)/.exec(query);
  if (!min && !max && !range) return false;
  if (range) {
    const bound = Number(range[2]);
    const holds = range[1] === '<' ? width < bound : range[1] === '<=' ? width <= bound : range[1] === '>' ? width > bound : width >= bound;
    if (!holds) return false;
  }
  return (!min || width >= Number(min[1])) && (!max || width <= Number(max[1]));
}

function matches(query: string): boolean {
  return matchesWidth(query, VIEWPORT_WIDTH);
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
