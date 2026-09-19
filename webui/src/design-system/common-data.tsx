// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import { Badge as CommonBadge } from '@lablup/ui-common/components/Badge';
import { StatCard as CommonStatCard } from '@lablup/ui-common/components/StatCard';

// A measurement tile. Values arrive already formatted as strings:
// - formatCompactNumber is not adopted. It abbreviates to 1.2K/3.4M and otherwise
//   falls back to a locale-less toLocaleString(), while design-system/format.ts
//   threads the explicit ko-KR/en-US Locale through Intl.NumberFormat. format.ts
//   stays the single formatter.
// - animate stays off, and usePrefersReducedMotion is not adopted. The hook reads only
//   the OS prefers-reduced-motion query and cannot see the in-app Settings toggle
//   (data-reduce-motion), so the count-up would ignore it. CSS stays the only
//   reduced-motion mechanism: tokens.css and components.css also stop the shared
//   components' transitions and animations under data-reduce-motion.
// - loading renders the package's skeletons in place of the value and hint. alpha.19 mounts
//   each one as its own role="status" named "Loading" (English), so the ref bridge makes
//   them decorative; the caller owns one localized status for the whole waiting region
//   (the LoadingStatus convention).
// - sparkline is the package's trailing slot on the value line.
export function StatCard(props: { label: string; value: string; hint?: React.ReactNode; className?: string; testId?: string; loading?: boolean; sparkline?: React.ReactNode }): React.JSX.Element {
  const card = <CommonStatCard label={props.label} value={props.value} hint={props.hint} loading={props.loading} sparkline={props.sparkline} className={`ds-stat-card ${props.className ?? ''}`.trim()} testId={props.testId} />;
  if (!props.loading) return card;
  return <div className="ds-stat-card-loading" ref={(element) => {
    for (const shape of element?.querySelectorAll('.skeleton') ?? []) {
      shape.removeAttribute('role');
      shape.removeAttribute('aria-busy');
      shape.removeAttribute('aria-label');
      shape.setAttribute('aria-hidden', 'true');
    }
  }}>{card}</div>;
}

// Secondary metadata chip. Lifecycle state keeps using StatusBadge (StatusTag).
export function Badge(props: { children: React.ReactNode; tone?: 'neutral' | 'accent' }): React.JSX.Element {
  return <CommonBadge variant={props.tone === 'accent' ? 'primary' : 'default'} className="ds-badge">{props.children}</CommonBadge>;
}
