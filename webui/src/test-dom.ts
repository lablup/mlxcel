// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// DOM reading helpers for tests over Astryx markup, which carries text a reader never meets:
// a Tab repeats its label in an aria-hidden copy that reserves the bold width, and every
// Button holds a visually hidden role="status" live region for its loading announcement.

/**
 * The text a sighted reader meets: textContent without aria-hidden subtrees and without the
 * visually hidden label an Astryx ProgressBar names itself with (`isLabelHidden`).
 */
export function visibleText(element: Element | null | undefined): string {
  if (!element) return '';
  const hiddenLabels = new Set<string>();
  for (const bar of element.ownerDocument.querySelectorAll('[role="progressbar"][aria-labelledby]')) for (const id of bar.getAttribute('aria-labelledby')?.split(/\s+/) ?? []) hiddenLabels.add(id);
  const walk = (current: Element): string => {
    let text = '';
    for (const node of current.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) text += node.textContent ?? '';
      else if (node instanceof Element && node.getAttribute('aria-hidden') !== 'true' && !(node.id && hiddenLabels.has(node.id) && node.closest('.ds-progress'))) text += walk(node);
    }
    return text;
  };
  return walk(element);
}

/** The element's name from aria-labelledby, aria-label or, failing both, its visible text. */
export function accessibleName(element: Element | null | undefined): string {
  if (!element) return '';
  const labelledBy = element.getAttribute('aria-labelledby');
  if (labelledBy) return labelledBy.split(/\s+/).map((id) => element.ownerDocument.getElementById(id)?.textContent?.trim() ?? '').filter(Boolean).join(' ');
  return element.getAttribute('aria-label') ?? visibleText(element).trim();
}

/** The product's own status regions, leaving out the live region inside each Astryx Button. */
export const PRODUCT_STATUS = '[role="status"]:not(button [role="status"])';
