// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import { ErrorState, type ErrorTone } from '@lablup/ui-common/components/ErrorState';
import { Skeleton } from '@lablup/ui-common/components/Skeleton';

type BannerTone = 'error' | 'warning' | 'info';
const errorTones: Record<BannerTone, ErrorTone> = { error: 'danger', warning: 'warning', info: 'accent' };

// Keeps the ErrorBanner name and props over the shared ErrorState, so the 23 call
// sites (and sibling issues that rely on the name) are unchanged. The action stays a
// ReactNode rendered here: ErrorState's primaryAction {label, onClick} cannot express
// an anchor or an icon-bearing Button.
export function ErrorBanner(props: { title: string; body: string; action?: React.ReactNode; tone?: BannerTone; testId?: string }): React.JSX.Element {
  const tone = props.tone ?? 'error';
  return <section className="ds-banner" data-tone={tone} role={tone === 'info' ? 'status' : 'alert'} data-testid={props.testId} ref={(element) => {
    // alpha.19 always renders role="alert" aria-live="polite" and an h2 title. The
    // section stays the only live region with the product's status/alert split, and
    // the title stays out of the document outline like the previous <strong>.
    const state = element?.querySelector('.error-state');
    state?.removeAttribute('role');
    state?.removeAttribute('aria-live');
    element?.querySelector('.error-state__title')?.setAttribute('role', 'none');
  }}><ErrorState tone={errorTones[tone]} title={props.title} message={props.body} showIcon={false} />{props.action}</section>;
}

// One announced status for a whole waiting region, with the localized label kept
// visible. SkeletonRow and SkeletonText mount a role="status" per row named "Loading"
// by default, so the shapes here are decorative Skeletons under a single status.
export function LoadingStatus(props: { label: string; rows?: number }): React.JSX.Element {
  return <div className="ds-loading" role="status">
    <p className="ds-loading__label">{props.label}</p>
    <div className="ds-loading__shapes" aria-hidden="true">{Array.from({ length: props.rows ?? 3 }, (_, index) => <Skeleton key={index} decorative width="100%" height="20px" />)}</div>
  </div>;
}
