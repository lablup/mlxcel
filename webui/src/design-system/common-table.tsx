// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useCallback, useMemo, useRef } from 'react';
import * as stylex from '@stylexjs/stylex';
import { Table, pixel, useTableSortable, type TablePlugin, type TableColumn as CommonTableColumn, type TableSortState } from '@lablup/ui-common/Table';

/** Marks a row's primary control, the one whole-row activation delegates to. */
export const ROW_PRIMARY_CLASS = 'ds-row-primary';
// An anchor counts as interactive only with an href; an anchor without one is not.
const ROW_INTERACTIVE = 'a[href], button, input, select, textarea, summary, label, [contenteditable="true"], [role="button"], [role="link"], [tabindex]:not([tabindex="-1"])';

function delegateRowClick(event: React.MouseEvent<HTMLDivElement>): void {
  if (event.defaultPrevented || event.button !== 0 || !(event.target instanceof Element)) return;
  const row = event.target.closest('tbody > tr');
  if (!row || !event.currentTarget.contains(row)) return;
  // The loading and empty rows stay inert, whatever controls their state content holds.
  if (row.querySelector('[data-ds-state]')) return;
  const primary = row.querySelector<HTMLElement>(`.${ROW_PRIMARY_CLASS}`);
  if (!primary || primary.matches(':disabled, [aria-disabled="true"]')) return;
  const nested = event.target.closest(ROW_INTERACTIVE);
  if (nested && row.contains(nested)) return;
  // A drag that selected text in this row is a selection, not an activation.
  const selection = window.getSelection();
  if (selection && !selection.isCollapsed && selection.toString().trim() !== '' && (row.contains(selection.anchorNode) || row.contains(selection.focusNode))) return;
  primary.focus({ preventScroll: true });
  primary.click();
}

export type SortDirection = 'asc' | 'desc';

/** A product table column; the adapter maps it onto an Astryx Table column. */
export interface DataTableColumn<T> {
  id: string;
  header: React.ReactNode;
  render: (row: T, index: number) => React.ReactNode;
  /** Added to the column's header and body cells. */
  className?: string;
  /**
   * A fixed width in px, through Astryx's `pixel()` column width. Without it the column takes an
   * equal share of the room the fixed columns leave (or the width the product stylesheet gives it).
   */
  fixedWidth?: number;
  minWidth?: number;
  /** Kept for call-site compatibility: columns are never resizable. */
  noResize?: boolean;
  sortable?: boolean;
  sortValueAccessor?: (row: T) => string | number | boolean | null | undefined;
  sortComparator?: (a: T, b: T, direction: SortDirection) => number;
}

export type DataTableAdapterProps<T> = {
  columns: DataTableColumn<T>[];
  rows: T[];
  getRowKey: (row: T, index: number) => string;
  ariaLabel: string;
  testId?: string;
  className?: string;
  rowClassName?: (row: T, index: number) => string | undefined | false | null;
  emptyState?: React.ReactNode;
  loadingState?: React.ReactNode;
  loading?: boolean;
  /** Controlled sort: the caller owns the state and orders its rows; the header reflects it. */
  sortColumnId?: string;
  sortDirection?: SortDirection;
  /** A third activation of a sorted header clears the sort (`undefined, undefined`). */
  onSortChange?: (columnId: string | undefined, direction: SortDirection | undefined) => void;
  /**
   * Whole-row pointer activation. A click anywhere in a body row that is not on another interactive element, and that did
   * not end a text selection, focuses and clicks the row's `ROW_PRIMARY_CLASS` control, so pointer and keyboard run the same
   * handler. Rows keep their native `row` role and gain no tab stop; the primary control stays the keyboard and
   * screen-reader target, beside any other controls the row holds, and the row draws a focus outline while any of them has
   * `:focus-visible`. Rows without an enabled primary control, and the loading and empty rows, stay inert.
   */
  activateRowPrimary?: boolean;
  /**
   * The table's own scroll box becomes a named, focusable region while its content overflows it, so a keyboard user can
   * scroll it even when no row holds a control (the loading and empty states). Axe scrollable-region-focusable fails
   * without it. Not scrollable, the box gets no role and no tab stop.
   */
  overflowRegionLabel?: string;
};

type Item<T> = { key: string; index: number; row: T } & Record<string, unknown>;

// The table's scroll box. The product's own, not Astryx's (which names itself "Table" in
// English and only follows the inline axis): `useOverflowRegion` owns its role and name.
type ScrollWrapperProps = { children: React.ReactNode; htmlProps?: React.HTMLAttributes<HTMLDivElement> & { ref?: React.Ref<HTMLDivElement> }; xstyle?: stylex.StyleXStyles[]; beforeTable?: React.ReactNode; afterTable?: React.ReactNode };
const ScrollRegionContext = React.createContext<((element: HTMLElement | null) => void) | null>(null);
function TableScrollBox({ children, htmlProps, xstyle, beforeTable, afterTable }: ScrollWrapperProps): React.JSX.Element {
  const region = React.useContext(ScrollRegionContext);
  const { ref: htmlRef, className, style, ...rest } = htmlProps ?? {};
  const styled = stylex.props(...(xstyle ?? []));
  const ref = useCallback((element: HTMLDivElement | null) => {
    region?.(element);
    if (typeof htmlRef === 'function') htmlRef(element);
    else if (htmlRef) (htmlRef as React.RefObject<HTMLDivElement | null>).current = element;
  }, [region, htmlRef]);
  return <div {...rest} ref={ref} className={['ds-table-scroll', styled.className, className].filter(Boolean).join(' ')} style={{ ...styled.style, ...style }}>{beforeTable}{children}{afterTable}</div>;
}

export function DataTable<T>(props: DataTableAdapterProps<T>): React.JSX.Element {
  const { columns, rows, getRowKey, ariaLabel, testId, className, rowClassName, emptyState, loadingState, loading = false, sortColumnId, sortDirection, onSortChange, activateRowPrimary = false, overflowRegionLabel } = props;
  const region = useOverflowRegion(overflowRegionLabel === undefined ? undefined : { label: overflowRegionLabel });
  const byId = useMemo(() => new Map(columns.map((column) => [column.id, column])), [columns]);
  const sortedColumn = sortColumnId === undefined ? undefined : byId.get(sortColumnId);
  // Rows arrive in the caller's order; a sortable column's comparator repeats it, so applying
  // it here keeps a caller that relies on the table's ordering working.
  const items = useMemo(() => {
    const keyed: Item<T>[] = rows.map((row, index) => ({ key: getRowKey(row, index), index, row }));
    if (!sortedColumn || !sortDirection) return keyed;
    const sign = sortDirection === 'asc' ? 1 : -1;
    const compare = sortedColumn.sortComparator
      ? (a: Item<T>, b: Item<T>) => sortedColumn.sortComparator?.(a.row, b.row, sortDirection) ?? 0
      : sortedColumn.sortValueAccessor
        ? (a: Item<T>, b: Item<T>) => {
          const left = sortedColumn.sortValueAccessor?.(a.row) ?? null;
          const right = sortedColumn.sortValueAccessor?.(b.row) ?? null;
          if (left === right) return 0;
          if (left === null) return 1;
          if (right === null) return -1;
          return (left < right ? -1 : 1) * sign;
        }
        : null;
    if (!compare) return keyed;
    return [...keyed].sort(compare).map((item, index) => ({ ...item, index }));
  }, [rows, getRowKey, sortedColumn, sortDirection]);
  const tableColumns = useMemo<CommonTableColumn<Item<T>>[]>(() => columns.map((column) => ({
    key: column.id,
    header: column.header,
    // Header and cell alignment are Astryx's defaults. A fixed column uses Astryx's own width;
    // the others get no minimum of Astryx's own, so the product stylesheets can size them.
    width: column.fixedWidth !== undefined ? pixel(column.fixedWidth) : { type: 'proportional', value: 1, minWidth: column.minWidth ?? 0 },
    sortable: column.sortable === true,
    renderCell: (item: Item<T>) => column.render(item.row, item.index),
  })), [columns]);
  const sort: TableSortState = sortColumnId && sortDirection ? [{ sortKey: sortColumnId, direction: sortDirection === 'asc' ? 'ascending' : 'descending' }] : [];
  const sortable = useTableSortable<Item<T>>({
    sort,
    allowUnsortedState: true,
    onSortChange: (next) => {
      const entry = next[0];
      onSortChange?.(entry?.sortKey, entry ? (entry.direction === 'ascending' ? 'asc' : 'desc') : undefined);
    },
  });
  const cells = useMemo<TablePlugin<Item<T>>>(() => ({
    // Astryx writes each header's proportional width inline, which would outrank the product
    // stylesheets that size the columns (the Activity slot shares). A fixed width stays.
    transformHeaderCell: (cellProps, column) => {
      const source = byId.get(column.key);
      const extra = source?.className;
      const style = source?.fixedWidth !== undefined ? cellProps.htmlProps.style : { ...cellProps.htmlProps.style, width: undefined, minWidth: undefined };
      return { ...cellProps, htmlProps: { ...cellProps.htmlProps, style, className: [cellProps.htmlProps.className, extra].filter(Boolean).join(' ') || undefined } };
    },
    transformBodyCell: (cellProps, column) => {
      const extra = byId.get(column.key)?.className;
      return extra ? { ...cellProps, htmlProps: { ...cellProps.htmlProps, className: [cellProps.htmlProps.className, extra].filter(Boolean).join(' ') } } : cellProps;
    },
    transformBodyRow: (rowProps, item) => {
      const extra = rowClassName?.(item.row, item.index);
      return extra ? { ...rowProps, htmlProps: { ...rowProps.htmlProps, className: [rowProps.htmlProps.className, extra].filter(Boolean).join(' ') } } : rowProps;
    },
  }), [byId, rowClassName]);
  const plugins = useMemo(() => ({ cells, sortable }), [cells, sortable]);
  const state = loading ? loadingState : emptyState;
  const table = <ScrollRegionContext value={overflowRegionLabel === undefined ? null : region}>
    <Table<Item<T>>
      columns={tableColumns}
      data={loading ? [] : items}
      idKey="key"
      plugins={plugins}
      aria-label={ariaLabel}
      data-testid={testId}
      className={`ds-common-table ${className ?? ''}`.trim()}
      scrollWrapper={TableScrollBox}
      emptyState={state === undefined ? false : <span className="ds-table-state" data-ds-state>{state}</span>}
    />
  </ScrollRegionContext>;
  if (!activateRowPrimary) return table;
  return <div className="ds-row-activation" onClick={delegateRowClick}>{table}</div>;
}

/** How a scroll region is named: its own label, or the id of a visible heading. */
export type OverflowRegionName = { label: string } | { labelledBy: string };

/**
 * A callback ref for a scroll box that becomes a named, focusable region only while its content
 * overflows it on either axis, so a keyboard user can scroll it (axe scrollable-region-focusable).
 * Not scrollable, it gets no role and no tab stop. It follows the box's size and its table's, and
 * re-resolves the table when the box's content changes, so a table mounted or swapped after the
 * ref attaches is followed too. A box that stops overflowing while it has focus keeps its tab stop
 * until it loses focus: removing it would drop focus to <body>. `undefined` leaves the box alone.
 */
export function useOverflowRegion(name: OverflowRegionName | undefined): (region: HTMLElement | null) => void {
  const disconnect = useRef<(() => void) | null>(null);
  const label = name !== undefined && 'label' in name ? name.label : undefined;
  const labelledBy = name !== undefined && 'labelledBy' in name ? name.labelledBy : undefined;
  return useCallback((region: HTMLElement | null) => {
    disconnect.current?.();
    disconnect.current = null;
    if (!region || (label === undefined && labelledBy === undefined)) return;
    let blurPending = false;
    // Losing focus settles a tab stop kept for the focused box.
    const onBlur = (): void => {
      blurPending = false;
      apply(false);
    };
    const apply = (focused: boolean): void => {
      if (region.scrollWidth > region.clientWidth + 1 || region.scrollHeight > region.clientHeight + 1) {
        region.tabIndex = 0;
        region.setAttribute('role', 'region');
        if (label !== undefined) region.setAttribute('aria-label', label);
        else if (labelledBy !== undefined) region.setAttribute('aria-labelledby', labelledBy);
      } else if (focused && region.hasAttribute('tabindex')) {
        // The box has focus: keep its tab stop, role and name until focus leaves it.
        if (!blurPending) {
          blurPending = true;
          region.addEventListener('blur', onBlur, { once: true });
        }
      } else {
        region.removeAttribute('tabindex');
        region.removeAttribute('role');
        region.removeAttribute('aria-label');
        region.removeAttribute('aria-labelledby');
      }
    };
    const update = (): void => apply(document.activeElement === region);
    const removeBlur = (): void => region.removeEventListener('blur', onBlur);
    update();
    if (typeof ResizeObserver === 'undefined') {
      disconnect.current = removeBlur;
      return;
    }
    // The box resizes with the viewport, the table with its rows and their content.
    const observer = new ResizeObserver(update);
    observer.observe(region);
    let table = region.querySelector('table');
    if (table) observer.observe(table);
    // Content, a table among it, can mount, unmount or be replaced without the box resizing:
    // follow the current table and check again. The check reads layout, so it waits for the next
    // frame and runs once for every commit in it instead of forcing layout after each one.
    let frame: number | null = null;
    const settle = (): void => {
      frame = null;
      update();
    };
    const mutations =
      typeof MutationObserver === 'undefined'
        ? null
        : new MutationObserver(() => {
            const next = region.querySelector('table');
            if (next !== table) {
              if (table) observer.unobserve(table);
              if (next) observer.observe(next);
              table = next;
            }
            if (typeof requestAnimationFrame === 'undefined') update();
            else if (frame === null) frame = requestAnimationFrame(settle);
          });
    mutations?.observe(region, { childList: true, subtree: true });
    disconnect.current = () => {
      observer.disconnect();
      mutations?.disconnect();
      if (frame !== null) cancelAnimationFrame(frame);
      removeBlur();
    };
  }, [label, labelledBy]);
}
