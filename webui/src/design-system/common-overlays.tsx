// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useCallback, useContext, useEffect, useId, useRef } from 'react';
import { Drawer as CommonDrawer } from '@lablup/ui-common/components/Drawer';
import { Tooltip as CommonTooltip } from '@lablup/ui-common/components/Tooltip';
import { NativeModalContext } from './modal-context';

// The approved 4185 off-canvas sheet width; CSS keeps it left-anchored.
const SHEET_WIDTH = 'min(320px, calc(100vw - 32px))';

// A right-anchored panel for details that open over a list (the Models inspector below
// 1100 px): the package's "medium" preset, capped to the viewport by CSS.
const PANEL_WIDTH = 'medium';

// Everything a Tab press can land on inside the panel, read when the key is pressed.
const TABBABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';

// Rendered only: the contents of a closed <details> and display:none controls have no boxes.
function tabbables(panel: HTMLElement): HTMLElement[] {
  return Array.from(panel.querySelectorAll<HTMLElement>(TABBABLE)).filter((element) => element.getClientRects().length > 0 && !element.closest('[inert]'));
}

// The product's compact off-canvas sheet over the shared Drawer. The Drawer is a
// modal aside (not a native dialog), so no NativeModalContext is provided here.
// `placement="end"` is the right-anchored details panel instead of the left sheet.
export function Drawer(props: { open: boolean; onClose: () => void; title: string; closeLabel: string; testId?: string; placement?: 'start' | 'end'; children: React.ReactNode }): React.JSX.Element {
  const end = props.placement === 'end';
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
    // alpha.19 listens for Escape on its panel only. A pointer press on a non-focusable
    // part of the drawer leaves focus on <body>, so also close on an Escape that starts
    // outside the panel, as the native modal sheet did. Back off once defaultPrevented
    // is set: a future popup that portals outside `.drawer` (like the Tooltip content)
    // and already handles Escape for itself should not also close the drawer beneath it.
    // A native modal <dialog> opened above the drawer (a model confirmation) sits outside
    // the panel and does not preventDefault its Escape, but it owns that key: it closes
    // itself and restores focus to its opener in the drawer, so leave the drawer open.
    // The same pointer press leaves Tab starting from <body>, which would walk the page behind
    // the modal panel; bring a Tab that starts outside the panel back to its first (or last) stop.
    const handleKeyDown = (event: KeyboardEvent): void => {
      if ((event.key !== 'Escape' && event.key !== 'Tab') || event.isComposing || event.defaultPrevented) return;
      if (event.target instanceof Element && event.target.closest('dialog')) return;
      const panel = hostRef.current?.querySelector<HTMLElement>('.drawer');
      if (panel && event.target instanceof Node && panel.contains(event.target)) return;
      if (event.key === 'Escape') {
        close();
        return;
      }
      if (!panel || event.altKey || event.ctrlKey || event.metaKey) return;
      const list = tabbables(panel);
      event.preventDefault();
      (event.shiftKey ? list[list.length - 1] : list[0])?.focus();
    };
    document.addEventListener('keydown', handleKeyDown);
    // alpha.19 builds its Tab trap list once, when the drawer opens, and its selector omits
    // <summary>. A Details disclosure was never reachable, content opened or disabled later
    // was never seen, and Tab past a stale last entry left the modal. Handle Tab here with
    // the list as it is at key-press time. Stopping propagation in the capture phase keeps
    // the package's stale handler on the same aside from also acting, so a Tab keydown inside
    // a drawer never reaches a React onKeyDown handler; nothing inside a drawer uses one for Tab.
    const drawerPanel = hostRef.current?.querySelector<HTMLElement>('.drawer');
    const handleTab = (event: KeyboardEvent): void => {
      if (event.key !== 'Tab' || event.isComposing || event.altKey || event.ctrlKey || event.metaKey || !drawerPanel) return;
      event.stopPropagation();
      const list = tabbables(drawerPanel);
      if (list.length === 0) {
        event.preventDefault();
        return;
      }
      const index = document.activeElement instanceof HTMLElement ? list.indexOf(document.activeElement) : -1;
      // Between the ends the browser moves focus in its own (document) order.
      if (event.shiftKey ? index > 0 : index !== -1 && index < list.length - 1) return;
      event.preventDefault();
      (event.shiftKey ? list[list.length - 1] : list[0]).focus();
    };
    drawerPanel?.addEventListener('keydown', handleTab, true);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('keydown', handleKeyDown);
      drawerPanel?.removeEventListener('keydown', handleTab, true);
    };
  }, [props.open, close]);
  return <div className="ds-drawer-host" ref={(element) => {
    hostRef.current = element;
    // alpha.19 has no test-id prop; tag the panel itself.
    if (props.testId) element?.querySelector('.drawer')?.setAttribute('data-testid', props.testId);
  }}><CommonDrawer isOpen={props.open} onClose={close} title={props.title} closeLabel={props.closeLabel} ariaLabelledBy={titleId} width={end ? PANEL_WIDTH : SHEET_WIDTH} className={end ? 'ds-drawer ds-drawer--end' : 'ds-drawer'}>{props.children}</CommonDrawer></div>;
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
