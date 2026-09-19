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
export function StatCard(props: { label: string; value: string; hint?: React.ReactNode; className?: string; testId?: string }): React.JSX.Element {
  return <CommonStatCard label={props.label} value={props.value} hint={props.hint} className={`ds-stat-card ${props.className ?? ''}`.trim()} testId={props.testId} />;
}

// Secondary metadata chip. Lifecycle state keeps using StatusBadge (StatusTag).
// `warning` flags a fact that blocks an action (an unsupported checkpoint, an unavailable task).
const badgeVariants = { neutral: 'default', accent: 'primary', warning: 'warning' } as const;
export function Badge(props: { children: React.ReactNode; tone?: keyof typeof badgeVariants; title?: string }): React.JSX.Element {
  const badge = <CommonBadge variant={badgeVariants[props.tone ?? 'neutral']} className="ds-badge">{props.children}</CommonBadge>;
  // alpha.19 Badge takes no title; the wrapper carries the full text of a truncated chip.
  return props.title ? <span className="ds-badge-wrap" title={props.title}>{badge}</span> : badge;
}
