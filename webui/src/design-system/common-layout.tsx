// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useEffect, useState } from 'react';
import { BaseCard } from '@lablup/ui-common/components/BaseCard';
import { PageHeader as CommonPageHeader } from '@lablup/ui-common/components/PageHeader';
import { PageLayout as CommonPageLayout } from '@lablup/ui-common/components/PageLayout';
import { SmoothHeight as CommonSmoothHeight } from '@lablup/ui-common/components/SmoothHeight';

// BaseCard renders a role-less div, so product cards pass their landmark role
// (an aria-label on a role-less div is an axe aria-prohibited-attr failure).
// hoverable is off: BaseCard's hover lifts the card, which reads as clickable.
export function Card(props: { children: React.ReactNode; className?: string; role?: 'article' | 'region'; ariaLabel?: string; tabIndex?: number }): React.JSX.Element {
  return <BaseCard className={`ds-card surface-card ${props.className ?? ''}`.trim()} role={props.role ?? 'article'} ariaLabel={props.ariaLabel} tabIndex={props.tabIndex} hoverable={false}>{props.children}</BaseCard>;
}

// The shell's content column. PageLayout is a width container only: "wide" caps it
// at 1400px and centers it; the shell keeps .app-shell, .app-toolbar and the grid.
export function PageLayout(props: { children: React.ReactNode; className?: string }): React.JSX.Element {
  return <CommonPageLayout variant="wide" className={`ds-page-layout ${props.className ?? ''}`.trim()}>{props.children}</CommonPageLayout>;
}

// PageHeader forwards only className, and its h1 and description take no other
// attributes. The bridge restores the product test ids and, when asked, makes the h1
// the route's dialog focus fallback: a named page heading is a better landing target
// than an unnamed layout container. It has no eyebrow slot (#1914 removes those keys).
export function PageHeader(props: { title: string; description?: string; titleTestId?: string; descriptionTestId?: string; focusFallback?: boolean; className?: string }): React.JSX.Element {
  return <div className={`ds-page-header ${props.className ?? ''}`.trim()} ref={(element) => {
    const heading = element?.querySelector<HTMLElement>('.page-header__title');
    const description = element?.querySelector<HTMLElement>('.page-header__description');
    if (heading && props.titleTestId) heading.dataset.testid = props.titleTestId;
    if (description && props.descriptionTestId) description.dataset.testid = props.descriptionTestId;
    if (heading && props.focusFallback) {
      heading.tabIndex = -1;
      heading.setAttribute('data-dialog-focus-fallback', '');
    }
  }}><CommonPageHeader title={props.title} description={props.description} /></div>;
}

// Covers the package transition (motion-standard) with margin for the last change.
const SETTLE_MS = 400;

// Animates height changes while `animate` is true (for example, while operations are
// in flight) and for one settle window after it turns false, so the final change
// still animates. SmoothHeight's active state pins an inline height with
// overflow:hidden; releasing it once settled means a steady list never clips the focus
// rings of controls at its edges.
export function SmoothHeight(props: { animate: boolean; children: React.ReactNode }): React.JSX.Element {
  const [previous, setPrevious] = useState(props.animate);
  const [settling, setSettling] = useState(false);
  if (previous !== props.animate) {
    setPrevious(props.animate);
    if (!props.animate) setSettling(true);
  }
  useEffect(() => {
    if (!settling) return;
    const timer = window.setTimeout(() => setSettling(false), SETTLE_MS);
    return () => window.clearTimeout(timer);
  }, [settling]);
  return <CommonSmoothHeight active={props.animate || settling} className="ds-smooth-height">{props.children}</CommonSmoothHeight>;
}
