// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Button } from './common-adapters';
import { Tabs } from './common-tabs';
import { Dialog } from './local-components';
import { LibraryStrings, loadLibraryMessages } from './common-i18n';
import { ProductTheme } from './common-theme';
import type { Locale } from '../i18n/catalog';

let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', ''); };
  HTMLDialogElement.prototype.close = function () { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); };
  host = document.createElement('div'); document.body.append(host); root = createRoot(host);
});
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });

function Sample(): React.JSX.Element {
  return <>
    <Button busy>Save</Button>
    <Tabs active="a" onChange={() => undefined} tabs={[{ id: 'a', label: 'A', panel: 'Panel' }]} />
    <Dialog open title="Title" onClose={() => undefined} testId="sample-dialog">Body</Dialog>
  </>;
}
async function renderIn(locale: Locale): Promise<void> {
  act(() => root.render(<LibraryStrings locale={locale}><Sample /></LibraryStrings>));
  await act(async () => { await loadLibraryMessages(locale); });
}
// Astryx's visually hidden loading announcement inside the busy Button.
const loading = (): string => [...(host.querySelector('button')?.querySelectorAll('[role="status"]') ?? [])].map((node) => node.textContent).join('').trim();
const strip = (): string => host.querySelector('[role="tablist"]')?.getAttribute('aria-label') ?? '';
const close = (): string => host.querySelector('[data-testid="dialog-close"]')?.getAttribute('aria-label') ?? '';

describe('ui-common and Astryx strings follow the appearance locale', () => {
  it('renders the built-in English strings without fetching a catalog', async () => {
    await expect(loadLibraryMessages('en')).resolves.toEqual({});
    await renderIn('en');
    expect(loading()).toBe('Loading');
    expect(strip()).toBe('Tabs');
    expect(close()).toBe('Close');
  });

  it('renders Astryx built-in strings in Korean once the Korean catalog chunk arrives', async () => {
    const messages = await loadLibraryMessages('ko');
    expect(messages['ko-KR']['@astryx.button.loading']).toEqual({ defaultMessage: '로드 중' });
    // ui-common's own strings ride in the same chunk.
    expect(messages['ko-KR']['uic.common.retry']?.defaultMessage).toBe('다시 시도');
    await renderIn('ko');
    expect(loading()).toBe('로드 중');
    expect(strip()).toBe('탭');
    expect(close()).toBe('닫기');
  });

  it('switches back and forth with the locale through the product theme', async () => {
    act(() => root.render(<ProductTheme family="mlxcel" scheme="light" locale="ko"><Sample /></ProductTheme>));
    await act(async () => { await loadLibraryMessages('ko'); });
    expect(loading()).toBe('로드 중');
    act(() => root.render(<ProductTheme family="mlxcel" scheme="light" locale="en"><Sample /></ProductTheme>));
    expect(loading()).toBe('Loading');
    expect(strip()).toBe('Tabs');
    act(() => root.render(<ProductTheme family="mlxcel" scheme="light" locale="ko"><Sample /></ProductTheme>));
    expect(strip()).toBe('탭');
  });

  it('keeps a product label over the catalog default', async () => {
    act(() => root.render(<LibraryStrings locale="ko"><Tabs label="설정 섹션" active="a" onChange={() => undefined} tabs={[{ id: 'a', label: 'A', panel: 'Panel' }]} /></LibraryStrings>));
    await act(async () => { await loadLibraryMessages('ko'); });
    expect(strip()).toBe('설정 섹션');
  });
});
