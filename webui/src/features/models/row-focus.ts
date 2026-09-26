// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Focus that follows a row action (Load, Unload) to the control that replaces the one used.
import { useEffect, useRef, type RefObject } from 'react';
import type { CatalogEntry } from '../../api/types';

/** A row control that focus can follow an action to. */
export type RowTarget = 'load' | 'chat' | 'unload';
/**
 * After a row action, the row controls that should take focus once one can, in order of
 * preference: the first that exists and is enabled wins. `from` is the control the action started
 * from, which the action disables.
 */
// `once`: a failed action's return trip. Focus goes back to the control the user pressed if it
// is usable on the next render, and the intent is dropped either way instead of waiting.
// `page`: the page the row was last seen on, so a re-sort that moves it is told apart from the
// user turning the page away from it.
export type RowFocus = { id: string; want: readonly RowTarget[]; from: Element | null; once?: boolean; page?: number };
/** After a Load: Use in Chat for a chat model; Unload for one without chat (embedding, rerank, transcription). */
export const AFTER_LOAD: readonly RowTarget[] = ['chat', 'unload'];
export const AFTER_UNLOAD: readonly RowTarget[] = ['load'];

/** A callback ref for one row control, stable across renders. */
export type RegisterRowControl = (id: string, slot: 'inspect' | RowTarget) => (element: HTMLButtonElement | null) => void;

export type RowFocusView = {
  /** The sorted, filtered entries across every page. */
  rows: readonly CatalogEntry[];
  pageSize: number;
  visiblePage: number;
  setPage: (page: number) => void;
};

export type RowFocusHandle = {
  rowFocus: RefObject<RowFocus | null>;
  rowControls: RefObject<Map<string, HTMLButtonElement>>;
  register: RegisterRowControl;
};

export function useRowFocus({ rows, pageSize, visiblePage, setPage }: RowFocusView): RowFocusHandle {
  const rowControls = useRef(new Map<string, HTMLButtonElement>());
  const rowFocus = useRef<RowFocus | null>(null);
  // One callback ref per row control, so a re-render does not detach and reattach every row button.
  const rowRefs = useRef(new Map<string, (element: HTMLButtonElement | null) => void>());

  // Focus follows a row action to the control that replaces the one used, but only while the
  // user has not moved on: focus is still on that row's Inspect button (where it waits) or nowhere.
  useEffect(() => {
    const pending = rowFocus.current;
    if (!pending) return;
    const holder = rowControls.current.get(`${pending.id}:inspect`);
    const target = pending.want
      .map((slot) => rowControls.current.get(`${pending.id}:${slot}`))
      .find((control) => control !== undefined && !control.disabled);
    const active = document.activeElement;
    const waiting =
      active === null || active === document.body || active === holder || active === pending.from || !!active.closest('[data-dialog-focus-fallback]');
    if (!waiting) {
      rowFocus.current = null;
      return;
    }
    // The lifecycle pin re-sorts a row whose state changed, often onto another page (an unloaded
    // row leaves the Ready group at the top). Follow it there; the next render focuses it. A
    // filter that now hides the row ends the intent, and so does the user turning the page away
    // from a row that has not moved: a pointer press that does not focus the pager (Safari,
    // Firefox on macOS) leaves focus on <body>, which alone must not turn the page back. A
    // failed action's return trip (`once`) never turns the page.
    if (!pending.once) {
      const index = rows.findIndex((entry) => entry.identity.id === pending.id);
      if (index === -1) {
        rowFocus.current = null;
        return;
      }
      const rowPage = Math.floor(index / pageSize);
      if (rowPage !== visiblePage && pending.page === rowPage) {
        rowFocus.current = null;
        return;
      }
      pending.page = rowPage;
      if (rowPage !== visiblePage) {
        setPage(rowPage);
        return;
      }
    }
    if (target) {
      target.focus({ preventScroll: true });
      rowFocus.current = null;
    } else if (pending.once) rowFocus.current = null;
    else if (holder && active !== holder && !document.querySelector('dialog[open]')) holder.focus({ preventScroll: true });
  });

  const register: RegisterRowControl = (id, slot) => {
    const key = `${id}:${slot}`;
    let callback = rowRefs.current.get(key);
    if (callback === undefined) {
      callback = (element: HTMLButtonElement | null): void => {
        if (element) rowControls.current.set(key, element);
        else {
          // Detached (the row left the page or the control was swapped): forget both.
          rowControls.current.delete(key);
          rowRefs.current.delete(key);
        }
      };
      rowRefs.current.set(key, callback);
    }
    return callback;
  };

  return { rowFocus, rowControls, register };
}
