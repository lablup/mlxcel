// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useContext, useId } from 'react';
import { Tooltip as CommonTooltip } from '@lablup/ui-common/components/Tooltip';
import { NativeModalContext } from './modal-context';

type DescribedElement = React.ReactElement<{ 'aria-describedby'?: string }>;

export function Tooltip(props: { content: string; children: DescribedElement }): React.JSX.Element {
  const nativeModal = useContext(NativeModalContext);
  const descriptionId = useId();
  // alpha.19 sets aria-describedby on its wrapper div, and only while open, so the
  // focused control itself would lose its description. Describe the control from
  // an always-present node instead.
  const describedBy = [props.children.props['aria-describedby'], descriptionId].filter(Boolean).join(' ');
  const trigger = React.cloneElement(props.children, { 'aria-describedby': describedBy });
  // alpha.19 portals the content into body, outside a native showModal() top layer
  // (the common-select.tsx precedent), so modal descendants keep in-place markup.
  if (nativeModal) return <span className="ds-tooltip-wrap">{trigger}<span className="ds-tooltip" role="tooltip" id={descriptionId}>{props.content}</span></span>;
  // tabIndex -1: the wrapped control is already a tab stop; the wrapper must not add one.
  return <CommonTooltip content={props.content} tabIndex={-1} className="ds-tooltip-trigger">{trigger}<span id={descriptionId} hidden>{props.content}</span></CommonTooltip>;
}
