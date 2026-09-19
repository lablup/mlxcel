// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useCallback, useContext, useEffect, useId, useRef } from 'react';
import { Drawer as CommonDrawer } from '@lablup/ui-common/components/Drawer';
import { Tooltip as CommonTooltip } from '@lablup/ui-common/components/Tooltip';
import { NativeModalContext } from './modal-context';

// The approved 4185 off-canvas sheet width; CSS keeps it left-anchored. Every drawer
// passes this constant inline width (the layout gate allowlists exactly it); the wider
// and end-anchored variants are modifier classes that override it in CSS.
const SHEET_WIDTH = 'min(320px, calc(100vw - 32px))';

// The product's compact off-canvas sheet over the shared Drawer. The Drawer is a
// modal aside (not a native dialog), so no NativeModalContext is provided here.
// `width` and `side` only add modifier classes: `narrow` and `start` are the sheet above;
// Chat's settings drawer and the Models inspector below 1100 px are `medium` and `end`.
export function Drawer(props: { open: boolean; onClose: () => void; title: string; closeLabel: string; testId?: string; width?: 'narrow' | 'medium'; side?: 'start' | 'end'; children: React.ReactNode }): React.JSX.Element {
  const titleId = useId();
  const onCloseRef = useRef(props.onClose);
  onCloseRef.current = props.onClose;
  // alpha.19 re-runs its open effect whenever onClose changes identity: the cleanup
  // refocuses the opener and the rerun refocuses the close button, so any parent
  // re-render while open would yank focus. Hand it one stable callback.
  const close = useCallback(() => onCloseRef.current(), []);
  const hostRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!props.open) return;
    // On the first open alpha.19 focuses its close button one frame before it makes
    // the panel visible (two frames), so that focus() is dropped. Finish the hand-off
    // once the panel can take focus, unless focus already moved into it.
    let frame = 0;
    let attempts = 0;
    const settle = (): void => {
      const panel = hostRef.current?.querySelector<HTMLElement>('.drawer');
      if (!panel || panel.contains(document.activeElement)) return;
      panel.querySelector<HTMLElement>('.drawer__close-btn')?.focus();
      if (!panel.contains(document.activeElement) && attempts++ < 10) frame = requestAnimationFrame(settle);
    };
    frame = requestAnimationFrame(settle);
    // alpha.19 listens for Escape and Tab on its panel only. A pointer press on a
    // non-focusable part of the drawer leaves focus on <body>, so an Escape from there also
    // closes it, as the native modal sheet did, and a Tab from there enters the panel at its
    // first (Shift+Tab: last) stop instead of walking the page behind it (the Models page
    // is not inert behind its inspector). Back off once defaultPrevented is set: a popup
    // that portals outside `.drawer` (like the Tooltip content) and already handles Escape
    // for itself should not also close the drawer beneath it. A native modal dialog opened
    // above the drawer (a confirmation launched from inside it) owns its keys: it closes
    // itself and returns focus to its opener in the drawer, which stays open.
    const handleKeyDown = (event: KeyboardEvent): void => {
      if ((event.key !== 'Escape' && event.key !== 'Tab') || event.isComposing || event.defaultPrevented) return;
      const panel = hostRef.current?.querySelector<HTMLElement>('.drawer');
      if (panel && event.target instanceof Node && panel.contains(event.target)) return;
      if (event.target instanceof Element && event.target.closest('dialog[open]')) return;
      if (event.key === 'Escape') {
        close();
        return;
      }
      // Only a Tab from nowhere; one from a control outside the panel (a Select list the
      // drawer portalled into <body>) keeps its own order.
      if (!panel || (event.target !== document.body && event.target !== document.documentElement) || event.altKey || event.ctrlKey || event.metaKey) return;
      const stops = tabStops(panel);
      const target = event.shiftKey ? stops.at(-1) : stops.at(0);
      if (!target) return;
      event.preventDefault();
      target.focus();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [props.open, close]);
  const variant = [props.width === 'medium' ? 'ds-drawer--medium' : '', props.side === 'end' ? 'ds-drawer--end' : ''].filter(Boolean).join(' ');
  return <div className="ds-drawer-host" ref={(element) => {
    hostRef.current = element;
    // alpha.19 has no test-id prop; tag the panel itself.
    if (props.testId) element?.querySelector('.drawer')?.setAttribute('data-testid', props.testId);
  }} onKeyDownCapture={(event) => {
    // alpha.19 closes on any Escape inside its panel, including one that only cancels an IME
    // composition in a text field. The capture phase runs before the panel's own listener.
    if (event.key === 'Escape' && (event.nativeEvent.isComposing || event.keyCode === 229)) event.stopPropagation();
    if (props.open) wrapTab(hostRef.current?.querySelector<HTMLElement>('.drawer') ?? null, event);
  }}><CommonDrawer isOpen={props.open} onClose={close} title={props.title} closeLabel={props.closeLabel} ariaLabelledBy={titleId} width={SHEET_WIDTH} className={variant ? `ds-drawer ${variant}` : 'ds-drawer'}>{props.children}</CommonDrawer></div>;
}

// The panel's tab stops as they are when a key is pressed. alpha.19 records its Tab cycle
// once, when the drawer opens, from a selector without <summary>: a control disabled after
// that (Chat's drawer controls while a response streams) left Tab stuck short of the last
// usable stop or let it leave the modal panel, and a Details disclosure (the Models
// inspector's) was never reachable. Usable means focusable, enabled, not inert or hidden,
// not inside a closed <details> other than as its own summary, and rendered.
const TAB_STOPS = 'a[href], button, input, select, textarea, summary, [tabindex]';
function tabStops(panel: HTMLElement): HTMLElement[] {
  return Array.from(panel.querySelectorAll<HTMLElement>(TAB_STOPS)).filter((element) => {
    if (element.tabIndex < 0 || element.matches(':disabled') || element.closest('[inert], [hidden]')) return false;
    const closed = element.closest('details:not([open])');
    if (closed && !(element.tagName === 'SUMMARY' && element.parentElement === closed)) return false;
    return typeof element.checkVisibility !== 'function' || element.checkVisibility({ visibilityProperty: true });
  });
}

// Wraps Tab at the panel's current first and last stops; anywhere else the browser moves
// focus in document order. Every Tab inside the panel stops here, in the capture phase, so
// the package's stale handler on the panel never runs: it would wrap early at a control it
// still believes is last (before the Details summary, or before a control enabled since).
// React onKeyDown handlers inside a drawer therefore never see Tab; none uses it.
function wrapTab(panel: HTMLElement | null, event: React.KeyboardEvent): void {
  if (event.key !== 'Tab' || event.altKey || event.ctrlKey || event.metaKey || event.nativeEvent.isComposing || !panel || !(event.target instanceof Node) || !panel.contains(event.target)) return;
  event.stopPropagation();
  const stops = tabStops(panel);
  if (stops.length === 0) {
    event.preventDefault();
    return;
  }
  if (document.activeElement !== (event.shiftKey ? stops[0] : stops[stops.length - 1])) return;
  event.preventDefault();
  (event.shiftKey ? stops[stops.length - 1] : stops[0]).focus();
}

type DescribedElement = React.ReactElement<{ 'aria-describedby'?: string }>;

export function Tooltip(props: { content: string; children: DescribedElement }): React.JSX.Element {
  const nativeModal = useContext(NativeModalContext);
  const descriptionId = useId();
  // alpha.19 sets aria-describedby on its wrapper div, and only while open, so the
  // focused control itself would lose its description. Describe the control from
  // an always-present node instead.
  const describedBy = [props.children.props['aria-describedby'], descriptionId].filter(Boolean).join(' ');
  const trigger = React.cloneElement(props.children, { 'aria-describedby': describedBy });
  // alpha.19 portals the content into body, outside a native showModal() top layer
  // (the common-select.tsx precedent), so modal descendants keep in-place markup.
  if (nativeModal) return <span className="ds-tooltip-wrap">{trigger}<span className="ds-tooltip" role="tooltip" id={descriptionId}>{props.content}</span></span>;
  // tabIndex -1: the wrapped control is already a tab stop; the wrapper must not add one.
  return <CommonTooltip content={props.content} tabIndex={-1} className="ds-tooltip-trigger">{trigger}<span id={descriptionId} hidden>{props.content}</span></CommonTooltip>;
}
