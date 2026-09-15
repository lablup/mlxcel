// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Pure acceptance policy: diagnostic evidence must never become a full pass.
export function performanceMode(value = 'full') {
  if (value === 'full') return { name: value, headless: false, modes: ['one-visible', 'two-visible', 'hidden'] };
  if (value === 'visible-only-headless') return { name: value, headless: true, modes: ['one-visible', 'two-visible'] };
  throw new Error('WEBUI_PERF_MODE must be full or visible-only-headless.');
}
export function assertGeometry(value) {
  if (![value.innerWidth, value.innerHeight, value.visualWidth, value.visualHeight].every((n) => Number.isFinite(n) && n > 0)) throw new Error('Browser has no usable layout/visual viewport; no inference measurement may begin.');
}
export function performanceCompletion(mode, summaries) {
  return {
    status: mode.headless ? 'incomplete' : summaries.some((x) => x.status === 'investigate') ? 'investigate' : 'within-target',
    scope: mode.name,
    hidden_native: mode.headless ? 'not-run' : 'measured',
    limitation: mode.headless ? 'Visible headless diagnostic only; native hidden-tab and interactive host acceptance remain outstanding.' : null,
  };
}
