// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import { AppShell as CommonAppShell } from '@lablup/ui-common/AppShell';
import { SideNav, SideNavItem, SideNavRenderContext } from '@lablup/ui-common/SideNav';
import { MobileNavToggle } from '@lablup/ui-common/MobileNav';
import { glyph, Icon, type IconName } from './icons';

/**
 * Below this width the side navigation moves into the mobile drawer: ui-common AppShell's `lg`
 * breakpoint (1024 px under the neutral theme), which switches to the wide layout at equality.
 */
export const SHELL_COMPACT_QUERY = '(width < 1024px)';

export interface ShellFrameProps {
  /** The application bar, rendered in AppShell's banner landmark at every width. */
  readonly topBar: React.ReactNode;
  /** The desktop side navigation (`ShellSideNav`); AppShell renders it only above the breakpoint. */
  readonly sideNav: React.ReactNode;
  /** The compact drawer (`ShellDrawerNav`); AppShell renders it only below the breakpoint. */
  readonly drawer: React.ReactNode;
  readonly drawerOpen: boolean;
  readonly onDrawerOpenChange: (open: boolean) => void;
  readonly children: React.ReactNode;
}

/**
 * ui-common AppShell: the skip link, the banner and main landmarks, the inline side navigation
 * and the compact drawer. The page scrolls (`height="auto"`) with the bar and the side navigation
 * held in place; the product stylesheets paint them (`variant="section"` keeps AppShell's own
 * backgrounds neutral). The drawer is controlled so the shell can close it from its shortcuts.
 */
export function ShellFrame(props: ShellFrameProps): React.JSX.Element {
  return (
    <CommonAppShell
      className="app-shell"
      height="auto"
      variant="section"
      contentPadding={0}
      topNav={props.topBar}
      sideNav={props.sideNav}
      mobileNav={{ breakpoint: 'lg', hasToggle: false, isOpen: props.drawerOpen, onOpenChange: props.onDrawerOpenChange, content: props.drawer }}
    >
      {props.children}
    </CommonAppShell>
  );
}

type NavContainerProps = {
  readonly label: string;
  readonly header?: React.ReactNode;
  readonly footer?: React.ReactNode;
  readonly className?: string;
  readonly onKeyDown?: (event: React.KeyboardEvent<HTMLElement>) => void;
  readonly children: React.ReactNode;
};

/** ui-common SideNav as the desktop navigation landmark, named by `label`. */
export const ShellSideNav = React.forwardRef<HTMLElement, NavContainerProps>((props, ref) => (
  <SideNav ref={ref} aria-label={props.label} header={props.header} footer={props.footer} className={props.className} onKeyDown={props.onKeyDown}>
    {props.children}
  </SideNav>
));
ShellSideNav.displayName = 'ShellSideNav';

/**
 * The same SideNav in its drawer rendering: a ui-common MobileNav (a native modal dialog that
 * traps focus, closes on Escape and returns focus to its opener) titled by `label`.
 */
export function ShellDrawerNav(props: Omit<NavContainerProps, 'header'> & { readonly testId?: string; readonly topContent?: React.ReactNode }): React.JSX.Element {
  return (
    <SideNavRenderContext value="drawer">
      <SideNav header={props.label} topContent={props.topContent} footer={props.footer} className={props.className} onKeyDown={props.onKeyDown} data-testid={props.testId}>
        {props.children}
      </SideNav>
    </SideNavRenderContext>
  );
}

export interface ShellNavItemProps {
  readonly label: string;
  readonly icon: IconName;
  readonly href: string;
  readonly isSelected: boolean;
  readonly onSelect: () => void;
  readonly testId?: string;
}

/** A ui-common SideNavItem link; a plain activation stays in the app (no hash navigation of its own). */
export function ShellNavItem(props: ShellNavItemProps): React.JSX.Element {
  return (
    <SideNavItem
      label={props.label}
      icon={glyph(props.icon)}
      href={props.href}
      isSelected={props.isSelected}
      data-testid={props.testId}
      onClick={(event) => {
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        props.onSelect();
      }}
    />
  );
}

/** ui-common MobileNavToggle: renders only below the breakpoint and opens the AppShell drawer. */
export function ShellMenuToggle(props: { readonly label: string; readonly testId?: string; readonly className?: string }): React.JSX.Element {
  return <MobileNavToggle label={props.label} data-testid={props.testId} className={props.className}><Icon name="menu" /></MobileNavToggle>;
}
