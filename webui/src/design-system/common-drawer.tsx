// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useCallback, useId, useRef } from 'react';
import { Drawer as CommonDrawer } from '@lablup/ui-common/lab';
import { IconButton } from './common-adapters';

// The approved 4185 off-canvas sheet: 320 px (520 px for `medium`), inset 12 px; CSS keeps
// the inset and the rounded sheet. `side` picks the edge it slides from.
const SHEET_WIDTHS = { narrow: 320, medium: 520 } as const;

// The product's compact off-canvas sheet over ui-common's lab Drawer, a native modal
// <dialog> (showModal): the browser keeps the page behind it inert and Astryx's layer stack
// closes the top-most layer on Escape (an IME composition's Escape included, which it
// ignores). The lab Drawer renders no header, so the adapter renders the localized title and
// close button; the close button carries data-autofocus, which the Drawer focuses on open,
// and the Drawer returns focus to the opener when it closes.
export function Drawer(props: { open: boolean; onClose: () => void; title: string; closeLabel: string; testId?: string; width?: 'narrow' | 'medium'; side?: 'start' | 'end'; children: React.ReactNode }): React.JSX.Element {
  const titleId = useId();
  const onCloseRef = useRef(props.onClose);
  onCloseRef.current = props.onClose;
  const close = useCallback(() => onCloseRef.current(), []);
  const onOpenChange = useCallback((open: boolean) => { if (!open) close(); }, [close]);
  const variant = [props.width === 'medium' ? 'ds-drawer--medium' : '', props.side === 'end' ? 'ds-drawer--end' : ''].filter(Boolean).join(' ');
  return <CommonDrawer isOpen={props.open} onOpenChange={onOpenChange} label={props.title} aria-labelledby={titleId} side={props.side ?? 'start'} width={SHEET_WIDTHS[props.width ?? 'narrow']} padding={0} data-testid={props.testId} className={variant ? `ds-drawer ${variant}` : 'ds-drawer'} onKeyDownCapture={(event: React.KeyboardEvent<HTMLDialogElement>) => { if (props.open) wrapTab(event.currentTarget, event); }}>
    <header className="ds-drawer__header">
      <h2 className="ds-drawer__title" id={titleId}>{props.title}</h2>
      <IconButton label={props.closeLabel} icon="close" onClick={close} className="ds-drawer__close" title={props.closeLabel} data-autofocus="" />
    </header>
    <div className="ds-drawer__content">{props.children}</div>
  </CommonDrawer>;
}

// The panel's tab stops as they are when a key is pressed. Usable means focusable, enabled, not
// inert or hidden, not inside a closed <details> other than as its own summary, and rendered.
// Astryx's focus trap reads its stops at key-press time too, but counts the contents of a closed
// <details> (the Models inspector's Details disclosure), which have no box to take focus.
const TAB_STOPS = 'a[href], button, input, select, textarea, summary, [tabindex]';
function tabStops(panel: HTMLElement): HTMLElement[] {
  return Array.from(panel.querySelectorAll<HTMLElement>(TAB_STOPS)).filter((element) => {
    if (element.tabIndex < 0 || element.matches(':disabled') || element.closest('[inert], [hidden]')) return false;
    const closed = element.closest('details:not([open])');
    if (closed && !(element.tagName === 'SUMMARY' && element.parentElement === closed)) return false;
    return typeof element.checkVisibility !== 'function' || element.checkVisibility({ visibilityProperty: true });
  });
}

// Wraps Tab at the panel's usable first and last stops, in the capture phase, and stops the key
// there so Astryx's trap does not act on it a second time. Any other Tab propagates: the browser
// moves focus, and a descendant (a Selector closing its list on Tab, a nested dialog's trap) still
// sees the key.
function wrapTab(panel: HTMLElement, event: React.KeyboardEvent): void {
  if (event.key !== 'Tab' || event.altKey || event.ctrlKey || event.metaKey || event.nativeEvent.isComposing || !(event.target instanceof Node) || !panel.contains(event.target)) return;
  const stops = tabStops(panel);
  if (stops.length === 0) {
    event.stopPropagation();
    event.preventDefault();
    return;
  }
  if (document.activeElement === (event.shiftKey ? stops[0] : stops[stops.length - 1])) {
    event.stopPropagation();
    event.preventDefault();
    (event.shiftKey ? stops[stops.length - 1] : stops[0]).focus();
  }
}
