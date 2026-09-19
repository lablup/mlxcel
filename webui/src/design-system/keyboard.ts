// Copyright 2026 Lablup Inc. Licensed under Apache-2.0.

/** The navigator surface this module reads; `userAgentData` is not yet in lib.dom.d.ts. */
interface NavigatorPlatformHint {
  readonly userAgentData?: { readonly platform?: string };
  readonly platform?: string;
  readonly userAgent?: string;
}

/**
 * True on macOS, iOS, iPadOS and iPodOS, false everywhere else, including when `navigator`
 * does not exist. A shortcut handler uses this to pick the platform's own modifier instead of
 * accepting both Cmd and Ctrl everywhere, which would otherwise steal a native text-editing
 * binding such as Ctrl+N ("move down a line") inside a macOS textarea.
 */
export function isApplePlatform(): boolean {
  if (typeof navigator === 'undefined') return false;
  const hint = navigator as NavigatorPlatformHint;
  const platform = hint.userAgentData?.platform ?? hint.platform ?? hint.userAgent ?? '';
  return /mac|iphone|ipad|ipod/i.test(platform);
}

/** The platform's primary shortcut modifier: Cmd on Apple platforms, Ctrl elsewhere. */
export function isPrimaryModifier(event: { readonly metaKey: boolean; readonly ctrlKey: boolean }): boolean {
  return isApplePlatform() ? event.metaKey : event.ctrlKey;
}
