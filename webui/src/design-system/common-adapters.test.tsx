// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act, createRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { createPortal } from 'react-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Button, IconButton, StatusBadge, ProgressBar, EmptyState, Tabs, DataTable, ROW_PRIMARY_CLASS, type ButtonProps, type DataTableAdapterProps } from './common-adapters';
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

  it('reapplies the heading bridge after common Tabs unmount and remount', () => {
    const changed = vi.fn();
    for (const title of ['No models', '모델 없음', 'No models again']) {
      const tabs = [{ id: 'empty', label: 'Empty', panel: <EmptyState title={title} body="Choose a repository" /> }, { id: 'other', label: 'Other', panel: 'Other panel' }];
      render(<React.StrictMode><Tabs tabs={tabs} active="other" onChange={changed} /></React.StrictMode>);
      expect(host.querySelector('.empty-state__title')).toBeNull();
      render(<React.StrictMode><Tabs tabs={tabs} active="empty" onChange={changed} /></React.StrictMode>);
      expect(host.querySelector('.empty-state__title[role="heading"][aria-level="2"]')?.textContent).toBe(title);
    }
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

describe('DataTable whole-row activation', () => {
  type Row = { id: string; name: string; primary?: 'disabled' | 'aria-disabled' | 'none' };
  const rows: Row[] = [{ id: 'a', name: 'Alpha' }, { id: 'b', name: 'Bravo' }, { id: 'c', name: 'Charlie', primary: 'disabled' }, { id: 'd', name: 'Delta', primary: 'none' }, { id: 'e', name: 'Echo', primary: 'aria-disabled' }];
  const primary = vi.fn<(id: string) => void>();
  const other = vi.fn();
  beforeEach(() => { primary.mockReset(); other.mockReset(); });
  const table = (data: Row[], extra: Partial<DataTableAdapterProps<Row>> = { activateRowPrimary: true }): React.ReactNode => (
    <DataTable
      {...extra}
      rows={data}
      getRowKey={(row) => row.id}
      ariaLabel="Rows"
      emptyState={<Button className={ROW_PRIMARY_CLASS} onClick={() => primary('empty')}>Empty action</Button>}
      loadingState={<Button className={ROW_PRIMARY_CLASS} onClick={() => primary('loading')}>Loading action</Button>}
      columns={[
        { id: 'name', header: 'Name', render: (row) => row.primary === 'none' ? row.name : row.primary === 'aria-disabled' ? <a href={`#${row.id}`} className={ROW_PRIMARY_CLASS} aria-disabled="true" onClick={(event) => { event.preventDefault(); primary(row.id); }}>{row.name}</a> : <Button className={ROW_PRIMARY_CLASS} disabled={row.primary === 'disabled'} aria-label={`Inspect ${row.name}`} onClick={() => primary(row.id)}>{row.name}</Button> },
        { id: 'detail', header: 'Detail', render: (row) => <><span className="detail">{`${row.name} detail`}</span> <Button onClick={other}>Other</Button></> },
      ]}
    />
  );
  const cell = (row: number, column = 1): HTMLElement => {
    const node = host.querySelectorAll('tbody tr')[row]?.querySelectorAll<HTMLElement>('td')[column];
    if (!node) throw new Error('Missing table cell');
    return node;
  };
  const click = (element: Element, init: MouseEventInit = {}): void => { act(() => { element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, button: 0, ...init })); }); };
  const select = (node: Node): void => { const range = document.createRange(); range.selectNodeContents(node); window.getSelection()?.removeAllRanges(); window.getSelection()?.addRange(range); };
  afterEach(() => { window.getSelection()?.removeAllRanges(); });

  it('activates the row primary control once, focused, from a plain cell', () => {
    render(table(rows));
    click(cell(1));
    expect(primary.mock.calls).toEqual([['b']]);
    expect(document.activeElement).toBe(host.querySelector('[aria-label="Inspect Bravo"]'));
    click(requireElement(cell(0).querySelector('.detail')));
    expect(primary.mock.calls).toEqual([['b'], ['a']]);
    expect(other).not.toHaveBeenCalled();
  });

  it('never double-fires the primary control and leaves other controls to themselves', () => {
    render(table(rows));
    act(() => host.querySelector<HTMLButtonElement>('[aria-label="Inspect Alpha"]')?.click());
    expect(primary.mock.calls).toEqual([['a']]);
    act(() => cell(0).querySelector('button')?.click());
    expect(other).toHaveBeenCalledOnce();
    click(cell(0), { button: 1 });
    act(() => { cell(0).addEventListener('click', (event) => event.preventDefault(), { once: true }); });
    click(cell(0));
    expect(primary.mock.calls).toEqual([['a']]);
  });

  it('treats a text selection inside the row as a selection, not an activation', () => {
    render(table(rows));
    select(requireElement(cell(0).querySelector('.detail')));
    click(cell(0));
    expect(primary).not.toHaveBeenCalled();
    window.getSelection()?.collapseToEnd();
    click(cell(0));
    expect(primary.mock.calls).toEqual([['a']]);
    select(requireElement(cell(1).querySelector('.detail')));
    click(cell(0));
    expect(primary.mock.calls).toEqual([['a'], ['a']]);
  });

  it('keeps rows with a disabled, aria-disabled or missing primary control, and the loading and empty rows, inert', () => {
    render(table(rows));
    click(cell(2)); click(cell(3)); click(cell(3, 0)); click(cell(4));
    expect(primary).not.toHaveBeenCalled();
    render(table([]));
    click(requireElement(host.querySelector('.data-table__state-cell')));
    render(table(rows, { activateRowPrimary: true, loading: true }));
    click(requireElement(host.querySelector('.data-table__state-cell')));
    click(requireElement(host.querySelector('thead th')));
    expect(primary).not.toHaveBeenCalled();
  });

  it('delegates nothing without the opt-in and keeps native row semantics with it', () => {
    render(table(rows, {}));
    expect(host.querySelector('.ds-row-activation')).toBeNull();
    click(cell(0));
    expect(primary).not.toHaveBeenCalled();
    render(table(rows));
    expect(host.querySelector('.ds-row-activation > .data-table')?.className).toBe('data-table ds-common-table');
    for (const row of host.querySelectorAll('tr')) {
      expect(row.hasAttribute('role')).toBe(false);
      expect(row.hasAttribute('tabindex')).toBe(false);
    }
    expect(host.querySelectorAll('.data-table__row--clickable')).toHaveLength(0);
  });

  it('withholds the package row props that turn rows into buttons', () => {
    // @ts-expect-error alpha.19 onRowClick renders each row as role="button"; use activateRowPrimary instead.
    const withheld: Partial<DataTableAdapterProps<Row>> = { onRowClick: () => undefined };
    expect(withheld).toBeDefined();
  });

  it('still activates the row primary control from a plain cell when a focusable ancestor also matches the nested-interactive selector', () => {
    // A focusable scroll region (tabIndex=0) wrapping the whole table, as #1918 may add, also matches
    // ROW_INTERACTIVE ([tabindex]:not([tabindex="-1"])). It sits outside the row, so it must not suppress
    // activation; only an interactive element inside the row may do that.
    render(<div tabIndex={0}>{table(rows)}</div>);
    click(cell(1));
    expect(primary.mock.calls).toEqual([['b']]);
  });

  it('ignores a click that lands on a row rendered by a portal outside the delegation wrapper', () => {
    const rowPrimary = vi.fn();
    const portalPrimary = vi.fn();
    function PortalCell(): React.ReactPortal {
      return createPortal(
        <table>
          <tbody>
            <tr>
              <td>
                <button className={ROW_PRIMARY_CLASS} onClick={portalPrimary}>
                  Portal primary
                </button>
              </td>
              <td className="portal-plain">Portal plain text</td>
            </tr>
          </tbody>
        </table>,
        document.body,
      );
    }
    render(
      <DataTable
        activateRowPrimary
        rows={[{ id: 'p', name: 'Portal' }]}
        getRowKey={(row) => row.id}
        ariaLabel="Rows"
        columns={[
          {
            id: 'name',
            header: 'Name',
            render: (row) => (
              <Button className={ROW_PRIMARY_CLASS} aria-label={`Inspect ${row.name}`} onClick={() => rowPrimary(row.id)}>
                {row.name}
              </Button>
            ),
          },
          { id: 'detail', header: 'Detail', render: () => <PortalCell /> },
        ]}
      />,
    );
    // React bubbles a portal's events to its ancestors in the React tree, not the DOM tree, so this click
    // reaches the wrapper's onClick even though the portal's own <tr> lives outside the wrapper in the DOM.
    const plain = requireElement(document.body.querySelector<HTMLElement>('.portal-plain'));
    click(plain);
    expect(rowPrimary).not.toHaveBeenCalled();
    expect(portalPrimary).not.toHaveBeenCalled();
    // Unmounting the portal's owner cleans up the DOM it placed under document.body.
    render(<div />);
    expect(document.body.querySelector('.portal-plain')).toBeNull();
  });
});

function requireElement<T extends Element>(element: T | null | undefined): T {
  if (!element) throw new Error('Missing element');
  return element;
}
