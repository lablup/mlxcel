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

// A Retry button always carries a localized label: the package default is English.
type PageHeaderRetry = { onRetry?: undefined; retryLabel?: undefined } | { onRetry: () => void; retryLabel: string };

export type PageHeaderProps = {
  title: string;
  description?: string;
  titleTestId?: string;
  descriptionTestId?: string;
  className?: string;
  /** Route actions, rendered in the header's action slot beside the title. */
  actions?: React.ReactNode;
  /** The route's stale or action error, shown in the header's alert block. */
  error?: string | null;
  errorDetail?: string | null;
  /** Test id for the `.page-header__error` alert block. */
  errorTestId?: string;
} & PageHeaderRetry;

// Shell contract: every route renders exactly one PageHeader as the first child of the
// shell's PageLayout, and its h1 is always the route's dialog focus fallback
// (tabIndex=-1 plus data-dialog-focus-fallback), so restoreModalFocus always finds one
// named landing target when a dialog closes after its opener is gone. PageHeader
// forwards only className, and its h1, description and error block take no other
// attributes, so the ref bridge applies the fallback and the product test ids. It runs
// on every render because the error block mounts and unmounts with `error`. There is no
// eyebrow slot and no dismiss control (the package's dismiss label is English).
export function PageHeader(props: PageHeaderProps): React.JSX.Element {
  return <div className={`ds-page-header ${props.className ?? ''}`.trim()} ref={(element) => {
    const heading = element?.querySelector<HTMLElement>('.page-header__title');
    const description = element?.querySelector<HTMLElement>('.page-header__description');
    const error = element?.querySelector<HTMLElement>('.page-header__error');
    if (heading) {
      heading.tabIndex = -1;
      heading.setAttribute('data-dialog-focus-fallback', '');
      if (props.titleTestId) heading.dataset.testid = props.titleTestId;
    }
    if (description && props.descriptionTestId) description.dataset.testid = props.descriptionTestId;
    if (error && props.errorTestId) error.dataset.testid = props.errorTestId;
  }}><CommonPageHeader title={props.title} description={props.description} actions={props.actions} error={props.error} errorDetail={props.errorDetail} onRetry={props.onRetry} retryLabel={props.retryLabel} /></div>;
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
  // Bumped on every transition to false, even while already settling, so the effect
  // below always restarts a fresh SETTLE_MS window from the latest one. Keying the
  // effect on `settling` alone missed a false -> true -> false sequence inside the
  // window: setSettling(true) while settling was already true is a no-op value-wise,
  // so the effect would not rerun and the original timer ended the window early.
  const [settledFrom, setSettledFrom] = useState(0);
  if (previous !== props.animate) {
    setPrevious(props.animate);
    if (!props.animate) {
      setSettling(true);
      setSettledFrom((token) => token + 1);
    }
  }
  useEffect(() => {
    if (!settling) return;
    const timer = window.setTimeout(() => setSettling(false), SETTLE_MS);
    return () => window.clearTimeout(timer);
  }, [settling, settledFrom]);
  return <CommonSmoothHeight active={props.animate || settling} className="ds-smooth-height">{props.children}</CommonSmoothHeight>;
}
