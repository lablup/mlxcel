// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { glyph, Icon, iconGlyphs, type IconName } from './icons';

let host: HTMLDivElement;
let root: Root;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });
const render = (element: React.ReactNode): SVGSVGElement => {
  act(() => root.render(element));
  const svg = host.querySelector('svg');
  if (!(svg instanceof SVGSVGElement)) throw new Error('no svg rendered');
  return svg;
};

const names = Object.keys(iconGlyphs) as IconName[];

describe('WebUI icons', () => {
  it('maps all 21 names to distinct lucide glyphs', () => {
    expect(names).toHaveLength(21);
    expect(new Set(names.map((name) => iconGlyphs[name])).size).toBe(names.length);
  });

  it.each(names)('renders %s as a decorative lucide svg that keeps the ds-icon sizing hook', (name) => {
    const svg = render(<Icon name={name} />);
    expect(svg.classList.contains('lucide')).toBe(true);
    expect(svg.classList.contains('ds-icon')).toBe(true);
    expect(svg.getAttribute('aria-hidden')).toBe('true');
    expect(svg.getAttribute('stroke')).toBe('currentColor');
    expect(svg.getAttribute('stroke-width')).toBe('2');
    expect(svg.getAttribute('viewBox')).toBe('0 0 24 24');
    expect(svg.children.length).toBeGreaterThan(0);
  });

  it('replaces the default class when the caller passes one', () => {
    const svg = render(<Icon name="close" className="custom-icon" />);
    expect(svg.classList.contains('custom-icon')).toBe(true);
    expect(svg.classList.contains('ds-icon')).toBe(false);
  });

  it('hands SideNavItem a cached lucide component that forwards its props', () => {
    expect(glyph('models')).toBe(glyph('models'));
    const Models = glyph('models');
    const svg = render(<Models className="nav-glyph" data-testid="glyph" />);
    expect(svg.classList.contains('lucide')).toBe(true);
    expect(svg.classList.contains('nav-glyph')).toBe(true);
    expect(svg.getAttribute('data-testid')).toBe('glyph');
    expect(svg.getAttribute('aria-hidden')).toBe('true');
  });

  it('pins lucide-react to the one copy @lablup/ui-common resolves', () => {
    const webui = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
    const pkg = JSON.parse(readFileSync(join(webui, 'package.json'), 'utf8')) as { dependencies: Record<string, string> };
    const lockfile = readFileSync(join(webui, 'pnpm-lock.yaml'), 'utf8');
    const snapshots = lockfile.slice(lockfile.indexOf('\nsnapshots:\n'));
    const copies = Array.from(snapshots.matchAll(/^ {2}'?lucide-react@([^:']+)'?:$/gm), (match) => match[1]);
    expect(copies).toHaveLength(1);
    expect(copies[0]?.startsWith(`${pkg.dependencies['lucide-react']}(`)).toBe(true);
  });
});
