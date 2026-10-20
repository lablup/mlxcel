/* Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0. */
/* global window, document */
// Early theme bootstrap. index.html loads this as a classic, render-blocking
// script ahead of the stylesheet and the app module, so the document root
// carries the resolved scheme (data-theme), the family's Astryx theme name
// (data-astryx-theme, which scopes the built theme CSS) and the family before
// the first paint, and the page never flashes the wrong theme while the app
// boots. These are the values the root Astryx <Theme> writes once it mounts.
// It is an external file rather than an inline script because the served CSP
// is `script-src 'self'`.
//
// It mirrors resolveThemeSelection() in src/design-system/theme.ts and the
// stored-preference rules in src/design-system/preferences.ts, including the
// pre-#1903 flat `theme` field. src/design-system/theme-bootstrap.test.ts runs
// this file against applyAppearance() over a matrix of stored values, so a
// change on one side without the other fails the unit gate.
(function () {
  'use strict';
  var families = ['mlxcel', 'glass'];
  // ASTRYX_THEME_NAMES in src/design-system/theme.ts.
  var astryxThemes = { mlxcel: 'mlxcel', glass: 'mlxcel-glass' };
  var preferences = ['system', 'light', 'dark'];
  var family = 'mlxcel';
  var preference = 'system';
  try {
    var raw = window.localStorage.getItem('mlxcel.webui.appearance');
    var stored = raw ? JSON.parse(raw) : null;
    if (stored && typeof stored === 'object' && !Array.isArray(stored)) {
      if (families.indexOf(stored.themeFamily) !== -1) family = stored.themeFamily;
      if (preferences.indexOf(stored.colorScheme) !== -1) preference = stored.colorScheme;
      else if (stored.colorScheme === undefined && preferences.indexOf(stored.theme) !== -1) preference = stored.theme;
    }
  } catch {
    // Blocked or corrupt storage keeps the defaults, exactly as loadAppearance() does.
  }
  var scheme = preference;
  if (scheme === 'system') {
    scheme = 'light';
    try {
      if (typeof window.matchMedia === 'function' && window.matchMedia('(prefers-color-scheme: dark)').matches) scheme = 'dark';
    } catch {
      scheme = 'light';
    }
  }
  var root = document.documentElement;
  root.setAttribute('data-theme', scheme);
  root.setAttribute('data-astryx-theme', astryxThemes[family]);
  root.setAttribute('data-theme-family', family);
  root.setAttribute('data-color-scheme', preference);
})();
