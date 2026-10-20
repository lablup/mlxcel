// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
// @ts-expect-error A plain Node script without type declarations.
import { expectedSlideRules, SLIDE_RULES_PATH } from '../../scripts/layer-slide-rules.mjs';

const webui = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const lockfile = readFileSync(join(webui, 'pnpm-lock.yaml'), 'utf8');

describe('ui-common 0.2 dependency boundary', () => {
  it('resolves exactly one copy of Astryx core, so lab components share its theme and layer contexts', () => {
    // A second resolution of core (another version, or the same version under another peer
    // set) would give the lab Drawer its own core. The lockfile's snapshot keys list each one.
    const snapshots = lockfile.slice(lockfile.indexOf('\nsnapshots:\n'));
    const cores = Array.from(snapshots.matchAll(/^ {2}'?@astryxdesign\/core@([^:']+)'?:$/gm), (match) => match[1]);
    expect(cores).toHaveLength(1);
    expect(cores[0]).toMatch(/^0\.7\.0\(/);
    expect(lockfile).toMatch(/^ {2}'@astryxdesign\/lab>@astryxdesign\/core': 0\.7\.0$/m);
  });

  it('ships the Astryx popover slide rules as the stylesheet index.html links', async () => {
    expect(readFileSync(SLIDE_RULES_PATH, 'utf8')).toBe(await expectedSlideRules());
    const html = readFileSync(join(webui, 'index.html'), 'utf8');
    expect(html).toMatch(/<link vite-ignore id="astryx-layer-slide-rules" rel="stylesheet" href="\.\/astryx-layer-slide-rules\.css" \/>/);
    // The layer order is fixed by the first stylesheet the page loads.
    expect(html.indexOf('./layer-order.css')).toBeGreaterThan(-1);
    expect(html.indexOf('./layer-order.css')).toBeLessThan(html.indexOf('astryx-layer-slide-rules'));
  });
});
