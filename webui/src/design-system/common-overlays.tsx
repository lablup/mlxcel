import React, { useId, useLayoutEffect, useRef } from 'react';
import { Tooltip as CommonTooltip } from '@lablup/ui-common/Tooltip';
import { useLayerDismissal } from '@lablup/ui-common/Layer';

/**
 * Joins a native modal <dialog> the product renders itself (Dialog, ConfirmDialog) to Astryx's
 * layer stack, so one Escape closes only the top-most layer: a confirmation opened from the
 * Drawer closes, the Drawer under it stays open. Astryx answers the Escape keydown (and
 * prevents its default, so the browser raises no cancel of its own); a close request that
 * still arrives as a native `cancel` is routed through the same check.
 */
export function useModalDialogLayer(open: boolean, dialog: React.RefObject<HTMLDialogElement | null>, dismiss: () => void): void {
  const { shouldDismissOnCloseRequest } = useLayerDismissal({ isActive: open, escapeBehavior: 'close', onDismiss: dismiss, getContainer: () => dialog.current });
  useLayoutEffect(() => {
    const element = dialog.current;
    if (!element || !open) return;
    const onCancel = (event: Event): void => {
      event.preventDefault();
      if (shouldDismissOnCloseRequest()) dismiss();
    };
    element.addEventListener('cancel', onCancel);
    return () => element.removeEventListener('cancel', onCancel);
  }, [open, dialog, dismiss, shouldDismissOnCloseRequest]);
}

type DescribedElement = React.ReactElement<{ 'aria-describedby'?: string }>;

export function Tooltip(props: { content: string; children: DescribedElement }): React.JSX.Element {
  const descriptionId = useId();
  const own = props.children.props['aria-describedby'];
  // The control is described by an always-present node, so its description does not depend
  // on the tooltip having been shown: Astryx mounts its content only on first show.
  const describedBy = [own, descriptionId].filter(Boolean).join(' ');
  const trigger = React.cloneElement(props.children, { 'aria-describedby': describedBy });
  const wrapper = useRef<HTMLSpanElement>(null);
  // Astryx adds its content's id to the trigger's aria-describedby in its own layout effect,
  // which runs before this one (a child's first). Dropping it again keeps the description
  // read once, from the node above, whether or not the tooltip is showing.
  useLayoutEffect(() => {
    const control = wrapper.current?.querySelector('[aria-describedby]');
    if (control && control.getAttribute('aria-describedby') !== describedBy) control.setAttribute('aria-describedby', describedBy);
  });
  return <span className="ds-tooltip-trigger" ref={wrapper}><CommonTooltip content={props.content}>{trigger}<span id={descriptionId} hidden>{props.content}</span></CommonTooltip></span>;
}
