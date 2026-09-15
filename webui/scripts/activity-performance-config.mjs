// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Pure acceptance policy: diagnostic evidence must never become a full pass.
export function performanceMode(value = 'full') {
  if (value === 'full') return { name: value, headless: false, modes: ['one-visible', 'two-visible', 'hidden'] };
  if (value === 'visible-only-headless') return { name: value, headless: true, modes: ['one-visible', 'two-visible'] };
  // Headed visible-mode overhead without the hidden acceptance. The hidden mode needs the browser
  // to report a real window-visibility change through document.hidden, which the automation
  // browser does not do; this mode keeps the observation-overhead measurement on the real headed
  // path and still declares the hidden acceptance not run.
  if (value === 'visible-only-headed') return { name: value, headless: false, modes: ['one-visible', 'two-visible'] };
  throw new Error('WEBUI_PERF_MODE must be full, visible-only-headed, or visible-only-headless.');
}
export function assertGeometry(value) {
  if (![value.innerWidth, value.innerHeight, value.visualWidth, value.visualHeight].every((n) => Number.isFinite(n) && n > 0)) throw new Error('Browser has no usable layout/visual viewport; no inference measurement may begin.');
}
export function performanceCompletion(mode, summaries) {
  // Completeness follows whether the hidden acceptance actually ran, not whether the browser was
  // headed: a headed run that skips the hidden mode is still an incomplete acceptance.
  const hiddenRan = mode.modes.includes('hidden');
  return {
    status: !hiddenRan ? 'incomplete' : summaries.some((x) => x.status === 'investigate') ? 'investigate' : 'within-target',
    scope: mode.name,
    hidden_native: hiddenRan ? 'measured' : 'not-run',
    limitation: hiddenRan ? null : mode.headless
      ? 'Visible headless diagnostic only; native hidden-tab and interactive host acceptance remain outstanding.'
      : 'Headed visible-mode observation overhead only; native hidden acceptance is not run because the automation browser reports document.hidden as false for a window it reports as minimized.',
  };
}
