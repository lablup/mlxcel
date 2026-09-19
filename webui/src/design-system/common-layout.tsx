// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import { BaseCard } from '@lablup/ui-common/components/BaseCard';
import { PageLayout as CommonPageLayout } from '@lablup/ui-common/components/PageLayout';

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
