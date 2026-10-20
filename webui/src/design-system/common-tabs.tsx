// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useId } from 'react';
import { Tab, TabList } from '@lablup/ui-common/TabList';

// TabList renders the strip only; the adapter renders the active panel, linked to its tab.
// Selection follows focus (automatic activation, as the product's tabs always have): an arrow,
// Home or End key moves focus along the strip and selects the tab it lands on.
// Without a label the strip takes the catalog's tab-list name. The IME guard stays: TabList's roving arrow keys (useListFocus) do not check isComposing.
export function Tabs(props: { tabs: { id: string; label: string; panel: React.ReactNode }[]; active: string; onChange: (id: string) => void; label?: string }): React.JSX.Element {
  const namespace = useId();
  if (props.tabs.length === 0) return <section className="ds-tabs" />;
  const active = props.tabs.find((tab) => tab.id === props.active) ?? props.tabs[0];
  const tabId = (id: string): string => `${namespace}-tab-${id}`;
  const panelId = (id: string): string => `${namespace}-panel-${id}`;
  return <section className="ds-tabs" onKeyDownCapture={(event) => { if (event.nativeEvent.isComposing) event.stopPropagation(); }}>
    <TabList role="tablist" value={active.id} onChange={props.onChange} aria-label={props.label} layout="fill" overflow="visible" className="ds-tabs__list" onFocus={(event) => {
      const value = event.target instanceof Element ? event.target.closest('[role="tab"]')?.getAttribute('data-tab-value') : null;
      if (value && value !== active.id && props.tabs.some((tab) => tab.id === value)) props.onChange(value);
    }}>
      {props.tabs.map((tab) => <Tab key={tab.id} value={tab.id} label={tab.label} id={tabId(tab.id)} panelId={panelId(tab.id)} />)}
    </TabList>
    <div role="tabpanel" className="ds-tabs__panel" id={panelId(active.id)} aria-labelledby={tabId(active.id)}>{active.panel}</div>
  </section>;
}
