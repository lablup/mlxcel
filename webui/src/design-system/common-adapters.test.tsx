// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act, createRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Button, IconButton, StatusBadge, ProgressBar, EmptyState, Tabs, DataTable, type ButtonProps } from './common-adapters';
import { Select } from './common-select';
import { NativeModalContext } from './modal-context';

let host: HTMLDivElement;
let root: Root;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() }); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals(); Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView'); });
const render = (element: React.ReactNode): void => { act(() => root.render(element)); };
const key = (element: Element, keyValue: string, extra: KeyboardEventInit = {}): void => {
  act(() => element.dispatchEvent(new KeyboardEvent('keydown', { key: keyValue, bubbles: true, cancelable: true, ...extra })));
};

describe('published ui-common adapters', () => {
  it('uses common buttons and forwards supported refs, events, ARIA and submit semantics', () => {
    const ref = createRef<HTMLButtonElement>();
    const clicked = vi.fn(); const submitted = vi.fn((event: React.FormEvent) => event.preventDefault());
    render(<form onSubmit={submitted}><Button ref={ref} type="submit" onClick={clicked} aria-label="Save" aria-describedby="help" aria-controls="panel" data-testid="save">Submit</Button></form>);
    expect(ref.current?.classList.contains('button')).toBe(true);
    expect(ref.current?.getAttribute('aria-label')).toBe('Save');
    expect(ref.current?.getAttribute('aria-describedby')).toBe('help');
    expect(ref.current?.getAttribute('aria-controls')).toBe('panel');
    act(() => ref.current?.click());
    expect(clicked).toHaveBeenCalledOnce(); expect(submitted).toHaveBeenCalledOnce();
    render(<Button>Not submit</Button>);
    expect(host.querySelector('button')?.type).toBe('button');
  });

  it('suppresses busy/disabled actions and names icon-only controls', () => {
    const clicked = vi.fn();
    render(<><Button busy onClick={clicked}>Wait</Button><IconButton busy icon="close" label="Close" onClick={clicked} /></>);
    for (const button of host.querySelectorAll('button')) {
      expect(button.disabled).toBe(true); expect(button.getAttribute('aria-busy')).toBe('true');
      act(() => button.click());
    }
    expect(clicked).not.toHaveBeenCalled();
    expect(host.querySelector('.ds-icon-button')?.getAttribute('aria-label')).toBe('Close');
    expect(host.querySelectorAll('.button__loading-spinner')).toHaveLength(2);
  });

  it('does not pretend unsupported native handlers are forwarded', () => {
    // @ts-expect-error alpha.19 has no onFocus forwarding; consumers must not silently lose it.
    const unsupported: ButtonProps = { onFocus: () => undefined };
    expect(unsupported).toBeDefined();
  });

  it('preserves every domain lifecycle label through common StatusTag', () => {
    const states = ['unloaded', 'loading', 'ready', 'draining', 'unloading', 'failed'] as const;
    render(<>{states.map((state) => <StatusBadge key={state} state={state}>{state}</StatusBadge>)}</>);
    expect(host.querySelectorAll('.status-tag')).toHaveLength(6);
    for (const state of states) expect(host.querySelector(`[data-state="${state}"]`)?.textContent).toBe(state);
  });

  it('keeps unknown/nonfinite progress distinct from measured zero and clamps finite values', () => {
    for (const value of [undefined, Number.NaN, Infinity]) {
      render(<ProgressBar label="Download" value={value} detail="Unknown total" />);
      expect(host.querySelector('[role="progressbar"]')?.hasAttribute('aria-valuenow')).toBe(false);
    }
    for (const [value, expected] of [[0, '0'], [-2, '0'], [44, '44'], [102, '100']] as const) {
      render(<ProgressBar label="Download" value={value} />);
      const bar = host.querySelector('[role="progressbar"]');
      expect(bar?.getAttribute('aria-valuenow')).toBe(expected);
      expect(bar?.getAttribute('aria-label')).toBe('Download');
    }
  });

  it('uses common EmptyState while retaining level-two heading and arbitrary action', () => {
    render(<EmptyState title="No models" body="Choose a repository" action={<Button>Add</Button>} testId="empty" />);
    expect(host.querySelector('.empty-state')).not.toBeNull();
    expect(host.querySelector('[role="heading"][aria-level="2"]')?.textContent).toBe('No models');
    expect(host.querySelector('[data-testid="empty"] button')?.textContent).toBe('Add');
  });

  it('namespaces duplicate tab IDs, links panels and translates keyboard callbacks', () => {
    const changed = vi.fn();
    const tabs = [{ id: 'overview', label: 'Overview', panel: 'First panel' }, { id: 'detail', label: 'Detail', panel: 'Second panel' }];
    render(<><Tabs tabs={tabs} active="overview" onChange={changed} label="First" /><Tabs tabs={tabs} active="overview" onChange={changed} label="Second" /></>);
    const ids = [...host.querySelectorAll('[id]')].map((node) => node.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const panel of host.querySelectorAll('[role="tabpanel"]')) {
      const tab = document.getElementById(panel.getAttribute('aria-labelledby') ?? '');
      expect(tab?.getAttribute('aria-controls')).toBe(panel.id);
    }
    const tab = host.querySelector('[role="tab"]');
    if (!tab) throw new Error('Missing common tab');
    key(tab, 'End'); expect(changed).toHaveBeenLastCalledWith('detail');
    changed.mockClear(); key(tab, 'ArrowRight', { isComposing: true }); expect(changed).not.toHaveBeenCalled();
    render(<Tabs tabs={[]} active="missing" onChange={changed} />);
    expect(host.querySelector('[role="tab"]')).toBeNull();
  });

  it('uses a body-portal select outside modals with localized labels and field metadata', () => {
    const changed = vi.fn();
    render(<Select locale="ko" label="모드" value="a" options={[{ value: 'a', label: 'A' }, { value: 'blocked', label: 'Blocked', disabled: true }, { value: 'b', label: 'B' }]} onChange={changed} error="오류" hint="도움말" />);
    const trigger = host.querySelector('[role="combobox"]');
    if (!trigger) throw new Error('Missing shared select trigger');
    expect(trigger.tagName).toBe('BUTTON');
    expect(trigger.getAttribute('aria-invalid')).toBe('true');
    const descriptionIds = trigger.getAttribute('aria-describedby')?.split(' ') ?? [];
    expect(descriptionIds.every((id) => document.getElementById(id))).toBe(true);
    key(trigger, 'ArrowDown', { isComposing: true }); expect(document.querySelector('[role="listbox"]')).toBeNull();
    key(trigger, 'ArrowDown'); key(trigger, 'ArrowDown'); key(trigger, 'Enter');
    expect(changed).toHaveBeenCalledWith('b');
    expect(document.querySelector('[role="listbox"]')).toBeNull();
    render(<Select locale="ko" label="모드" value="" options={[]} onChange={changed} />);
    act(() => host.querySelector<HTMLButtonElement>('[role="combobox"]')?.click());
    expect(document.body.textContent).toContain('일치하는 옵션이 없습니다.');
  });

  it('keeps native select descendants inside the modal boundary without body popups', () => {
    render(<NativeModalContext.Provider value={true}><Select label="Mode" value="a" options={[{ value: 'a', label: 'A' }]} onChange={() => undefined} busy testId="modal-select" /></NativeModalContext.Provider>);
    expect(host.querySelector('select')?.disabled).toBe(true);
    expect(host.querySelector('select')?.getAttribute('aria-busy')).toBe('true');
    expect(host.querySelector('.select__trigger')).toBeNull();
    expect(document.querySelector('.select__dropdown--portal')).toBeNull();
  });

  it('exports the common generic table with stable opaque keys and caller-owned state', () => {
    render(<DataTable rows={[{ id: 'opaque:1', value: 'First' }]} getRowKey={(row) => row.id} columns={[{ id: 'name', header: 'Name', render: (row) => row.value }]} ariaLabel="Models" />);
    expect(host.querySelector('.data-table')).not.toBeNull();
    expect(host.querySelector('[role="table"]')?.getAttribute('aria-label')).toBe('Models');
    expect(host.textContent).toContain('First');
  });
});
