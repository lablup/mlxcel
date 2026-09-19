// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import { BaseCard } from '@lablup/ui-common/components/BaseCard';

// BaseCard renders a role-less div, so product cards pass their landmark role
// (an aria-label on a role-less div is an axe aria-prohibited-attr failure).
// hoverable is off: BaseCard's hover lifts the card, which reads as clickable.
export function Card(props: { children: React.ReactNode; className?: string; role?: 'article' | 'region'; ariaLabel?: string; tabIndex?: number }): React.JSX.Element {
  return <BaseCard className={`ds-card surface-card ${props.className ?? ''}`.trim()} role={props.role ?? 'article'} ariaLabel={props.ariaLabel} tabIndex={props.tabIndex} hoverable={false}>{props.children}</BaseCard>;
}
