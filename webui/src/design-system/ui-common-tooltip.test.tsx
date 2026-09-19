// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DesignGallery } from '../gallery';
import { t } from '../i18n/catalog';
import { NativeModalContext } from './modal-context';

// Drives the real gallery call site rather than the component API, so the same
// assertions hold before and after the Tooltip props change.
let host: HTMLDivElement;
let root: Root;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() }); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView'); });

const tooltipText = t('en', 'gallery.tooltip');
const triggerText = t('en', 'gallery.hover_focus');
function trigger(): HTMLButtonElement {
  const button = [...host.querySelectorAll('button')].find((node) => node.textContent === triggerText);
  if (!button) throw new Error('Missing gallery tooltip trigger');
  return button;
}
function description(element: Element): string {
  return (element.getAttribute('aria-describedby') ?? '').split(/\s+/).filter(Boolean).map((id) => document.getElementById(id)?.textContent ?? '').join(' ').trim();
}
function hoverAndFocus(element: HTMLElement): void {
  act(() => { element.dispatchEvent(new MouseEvent('mouseover', { bubbles: true })); element.focus(); element.dispatchEvent(new FocusEvent('focusin', { bubbles: true })); });
}

describe('Tooltip adoption at the gallery call site', () => {
  it('describes the focusable trigger itself and adds no tab stop around it', () => {
    act(() => root.render(<DesignGallery locale="en" />));
    const button = trigger();
    expect(description(button)).toBe(tooltipText);
    for (let node = button.parentElement; node && !node.classList.contains('control-row'); node = node.parentElement) expect(node.tabIndex, node.className).toBeLessThan(0);
    // Delivered by the shared component: asserted last so a pre-swap run fails here.
    expect(button.closest('.tooltip__wrapper')).not.toBeNull();
  });

  it('keeps tooltip content inside a native modal instead of portalling it to body', () => {
    act(() => root.render(<NativeModalContext.Provider value={true}><DesignGallery locale="en" /></NativeModalContext.Provider>));
    const inside = trigger();
    hoverAndFocus(inside);
    expect(document.querySelector('.tooltip__content')).toBeNull();
    expect(description(inside)).toBe(tooltipText);
    expect(inside.closest('.tooltip__wrapper')).toBeNull();
    act(() => root.render(<DesignGallery locale="en" />));
    expect(trigger().closest('.tooltip__wrapper')).not.toBeNull();
  });
});
