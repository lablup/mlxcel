// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useId, useMemo } from 'react';
import type { RuntimeSlot, RuntimeSlots } from '../../api/types';
import { DataTable, ProgressBar, StatusBadge, type DataTableColumn } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { Disclosure } from './disclosure';
import { formatCount, tokenCount } from './format';
import { SLOT_REASON_KEYS } from './reasons';

/**
 * Occupancy of one slot, in this order:
 * 1. no denominator (null or 0 request context): unknown, no bar;
 * 2. a reported prompt count: bar plus "N / D tokens";
 * 3. an idle slot: "0 / D tokens" with an empty bar. The server reports a null count
 *    when the slot holds no task (src/server/slots_state.rs), not when a measurement
 *    failed, and an empty slot holds no context. This is the one documented exception
 *    to "null is never shown as zero" (docs/webui/ux-contract.md);
 * 4. a processing slot without a count: unknown. No occupancy is derived from decoded tokens.
 */
function Occupancy({ slot, context, locale }: { slot: RuntimeSlot; context: number | null; locale: Locale }): React.JSX.Element {
  if (context === null || context <= 0) return <span className="activity-muted">{t(locale, 'activity.no_context')}</span>;
  const used = slot.prompt_tokens ?? (slot.processing ? null : 0);
  if (used === null) return <span className="activity-muted">{t(locale, 'format.unknown')}</span>;
  return <ProgressBar label={t(locale, 'activity.slot_occupancy', { id: String(slot.id) })} value={used / context * 100} detail={t(locale, 'activity.slot_progress', { used: formatCount(used, locale), total: formatCount(context, locale) })} />;
}

// A count the slot's task reported; an idle slot without a task has nothing to count.
function slotCount(value: number | null, slot: RuntimeSlot, locale: Locale): string {
  if (value !== null) return formatCount(value, locale);
  return slot.processing ? t(locale, 'format.unknown') : '';
}

function columns(locale: Locale, context: number | null): DataTableColumn<RuntimeSlot>[] {
  return [
    { id: 'slot', header: t(locale, 'activity.slot'), render: (slot) => formatCount(slot.id, locale), noResize: true, className: 'activity-slot-col-slot' },
    { id: 'state', header: t(locale, 'activity.slot_state'), render: (slot) => <StatusBadge state={slot.processing ? 'ready' : 'unloaded'}>{t(locale, slot.processing ? 'activity.processing' : 'activity.idle')}</StatusBadge>, noResize: true, className: 'activity-slot-col-state' },
    { id: 'occupancy', header: t(locale, 'activity.occupancy'), render: (slot) => <Occupancy slot={slot} context={context} locale={locale} />, noResize: true, className: 'activity-slot-occupancy activity-slot-col-occupancy' },
    { id: 'decoded', header: t(locale, 'activity.decoded'), render: (slot) => slotCount(slot.decoded_tokens, slot, locale), noResize: true, align: 'right', className: 'activity-slot-col-decoded' },
    { id: 'cached', header: t(locale, 'activity.cached'), render: (slot) => slotCount(slot.cached_prompt_tokens, slot, locale), noResize: true, align: 'right', className: 'activity-slot-col-cached' },
  ];
}

function SlotReason({ reason, available, locale }: { reason: string | null; available: boolean; locale: Locale }): React.JSX.Element | null {
  if (reason === null) return available ? null : <p className="activity-note">{t(locale, 'activity.slots_reason.unavailable')}</p>;
  const key = SLOT_REASON_KEYS.get(reason);
  if (key !== undefined) return <p className="activity-note">{t(locale, key)}</p>;
  // An unrecognized server reason: say so in the page's language, keep the raw text behind a disclosure.
  return <div className="activity-note">
    <p>{t(locale, 'activity.slots_reason.other')}</p>
    <Disclosure className="activity-disclosure" summary={t(locale, 'activity.slots_reason.raw')}>{() => <p className="activity-raw">{reason}</p>}</Disclosure>
  </div>;
}

export function SlotTable({ slots, locale }: { slots: RuntimeSlots; locale: Locale }): React.JSX.Element {
  const headingId = useId();
  const context = slots.request_context_tokens;
  const slotColumns = useMemo(() => columns(locale, context), [locale, context]);
  const unknown = t(locale, 'format.unknown');
  return <div className="activity-slots">
    <h3 id={headingId}>{t(locale, 'activity.slots')}</h3>
    <p className="activity-slot-meta">
      <span>{t(locale, 'activity.parallel', { effective: slots.effective_parallelism === null ? unknown : formatCount(slots.effective_parallelism, locale), configured: formatCount(slots.configured_parallelism, locale) })}</span>
      <span>{t(locale, 'activity.context', { tokens: tokenCount(context, locale) })}</span>
      <span>{t(locale, 'activity.pool', { tokens: tokenCount(slots.shared_pool_context_tokens, locale) })}</span>
    </p>
    <SlotReason reason={slots.reason} available={slots.available} locale={locale} />
    {/* More than eight rows scroll inside this region, not the page. It is a named tab stop so
        keyboard users can scroll it (axe scrollable-region-focusable), at any width. */}
    {slots.available ? <div className="activity-slot-scroll" role="region" aria-labelledby={headingId} tabIndex={0} data-bounded={slots.items.length > 8 || undefined}>
      <DataTable columns={slotColumns} rows={[...slots.items]} getRowKey={(slot) => String(slot.id)} ariaLabel={t(locale, 'activity.slots')} testId="activity-slot-table" emptyState={<p>{t(locale, 'activity.slots_empty')}</p>} />
    </div> : null}
  </div>;
}
