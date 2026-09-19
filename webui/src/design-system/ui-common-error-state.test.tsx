// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Button, ErrorBanner } from './primitives';

let host: HTMLDivElement;
let root: Root;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });
const render = (element: React.ReactNode): void => { act(() => root.render(element)); };
const headings = (scope: Element): string[] => [...scope.querySelectorAll('h1, h2, h3, h4, h5, h6, [role="heading"]')].filter((node) => !['none', 'presentation'].includes(node.getAttribute('role') ?? '')).map((node) => node.textContent ?? '');
const liveRegions = (scope: Element): Element[] => [...scope.querySelectorAll('[role="alert"], [role="status"], [aria-live]')];

describe('ErrorBanner over the shared ErrorState', () => {
  it.each([['info', 'status', 'accent'], ['warning', 'alert', 'warning'], ['error', 'alert', 'danger']] as const)('announces the %s tone as %s from one live region without adding a heading', (tone, role, sharedTone) => {
    render(<section aria-labelledby="runtime-heading"><h2 id="runtime-heading">Runtime</h2><ErrorBanner tone={tone} title="Observations are stale" body="Refresh to continue" testId="banner" /></section>);
    const banner = host.querySelector('[data-testid="banner"]');
    if (!banner) throw new Error('Missing banner');
    expect(banner.getAttribute('role')).toBe(role);
    expect(liveRegions(host)).toEqual([banner]);
    expect(banner.textContent).toContain('Observations are stale');
    expect(banner.textContent).toContain('Refresh to continue');
    expect(headings(host)).toEqual(['Runtime']);
    // Delivered by the shared component: asserted last so a pre-swap run fails here.
    expect(banner.querySelector(`.error-state.error-state--${sharedTone}`)).not.toBeNull();
  });

  it('keeps anchor and icon-button actions inside the test-id element and reachable by keyboard', () => {
    const retried = vi.fn();
    render(<><ErrorBanner title="Model unavailable" body="Open the library" action={<a href="#models">Models</a>} testId="with-link" /><ErrorBanner tone="warning" title="Failed" body="Try again" action={<Button onClick={retried}>Retry</Button>} testId="with-button" /></>);
    const link = host.querySelector<HTMLAnchorElement>('[data-testid="with-link"] a[href="#models"]');
    const retry = host.querySelector<HTMLButtonElement>('[data-testid="with-button"] button');
    if (!link || !retry) throw new Error('Missing banner action');
    link.focus();
    expect(document.activeElement).toBe(link);
    retry.focus();
    expect(document.activeElement).toBe(retry);
    act(() => retry.click());
    expect(retried).toHaveBeenCalledOnce();
    expect(host.querySelectorAll('[data-testid^="with-"] .error-state')).toHaveLength(2);
  });
});
