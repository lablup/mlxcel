// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
/* global URL, process */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { cpSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { attributeMatches, collectSources, findThemeSelectorViolations } from './check-theme-selectors.mjs';

const webuiRoot = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
const THEME_BLOCKS = {
  path: 'src/design-system/themes/built/fixture.css',
  source: ['mlxcel', 'mlxcel-glass'].map((name) => `@scope ([data-astryx-theme="${name}"]) to ([data-astryx-theme]) { :scope { --x: 1; } }`).join('\n'),
};

function scan(path, source) {
  return findThemeSelectorViolations([THEME_BLOCKS, { path, source }]);
}

function rules(violations) {
  return violations.map((violation) => `${violation.rule}@${violation.line}`);
}

test('the checked-in WebUI tree passes', () => {
  assert.deepEqual(findThemeSelectorViolations(collectSources(webuiRoot)), []);
});

test('reintroducing a family-scoped id selector into the real tokens.css fails the gate', () => {
  const sources = collectSources(webuiRoot);
  const tokens = sources.find((source) => source.path === 'src/design-system/tokens.css');
  assert.ok(tokens, 'tokens.css is scanned');
  tokens.source += '\n:root[data-theme="mlxcel-dark"] { --mx-x: 1; }\n';
  const violations = findThemeSelectorViolations(sources);
  assert.equal(violations.length, 1);
  assert.equal(violations[0].rule, 'dead-theme-selector');
  assert.equal(violations[0].path, 'src/design-system/tokens.css');
});

test('the family-scoped ids of #1903 and the host scheme are all dead now that data-theme holds the scheme', () => {
  const legacy = [
    '[data-theme="mlxcel-light"] { --x: 1; }',
    ':root[data-theme$="-dark"] { color-scheme: dark; }',
    '@media (prefers-color-scheme: dark) {',
    '  :root:not([data-theme="glass-light"]) { --x: 1; }',
    '}',
    ':root[data-theme="system"] { --x: 1; }',
  ].join('\n');
  assert.deepEqual(rules(scan('src/design-system/tokens.css', legacy)), [
    'dead-theme-selector@1',
    'dead-theme-selector@2',
    'prefers-color-scheme@3',
    'dead-theme-selector@4',
    'dead-theme-selector@6',
  ]);
});

test('selectors that match a shipped value pass, whatever the operator', () => {
  const live = [
    ':root[data-theme="dark"] { --x: 1; }',
    ':root[data-theme=\'light\'] { --x: 1; }',
    '[data-theme^="da"] { --x: 1; }',
    '[data-theme$="ght"] { --x: 1; }',
    '[data-theme*="ar"] { --x: 1; }',
    '[data-theme~="light"] { --x: 1; }',
    '[data-theme="DARK" i] { --x: 1; }',
    '[data-theme=light] { --x: 1; }',
    '[data-theme] { --x: 1; }',
    '[data-theme-family="glass"] { --x: 1; }',
    '[data-astryx-theme="mlxcel-glass"] { --x: 1; }',
    '[data-astryx-theme|="mlxcel"] { --x: 1; }',
  ].join('\n');
  assert.deepEqual(scan('src/features/example.css', live), []);
});

test('selectors that match no shipped value fail, including case and prefix slips', () => {
  const dead = [
    '[data-theme="DARK"] { --x: 1; }',
    '[data-theme^="glass-"] { --x: 1; }',
    '[data-theme$="-dim"] { --x: 1; }',
    '[data-theme|="mlxcel"] { --x: 1; }',
    '[data-theme^=""] { --x: 1; }',
    '[data-theme-family="orange"] { --x: 1; }',
    '[data-astryx-theme="glass"] { --x: 1; }',
    '[data-astryx-theme="neutral"] { --x: 1; }',
  ].join('\n');
  assert.deepEqual(rules(scan('src/features/example.css', dead)), [1, 2, 3, 4, 5, 6, 7, 8].map((line) => `dead-theme-selector@${line}`));
});

test('attribute names match case-insensitively, as in HTML documents', () => {
  assert.deepEqual(rules(scan('src/example.css', '[DATA-THEME="glass-dark"] .x { color: red; }\n[Data-Theme="dark"] .y { color: red; }')), ['dead-theme-selector@1']);
});

test('pathological whitespace does not make the scan super-linear', () => {
  const started = Date.now();
  // Neither input matches; the old patterns backtracked cubically and quadratically on them.
  scan('src/example.ts', `import${' '.repeat(3000)};\n[data-theme="glass-dark"${' '.repeat(40000)}x`);
  assert.ok(Date.now() - started < 1000, `scan took ${Date.now() - started} ms`);
});

test('a symlinked invocation still runs the check', () => {
  const scratch = mkdtempSync(join(tmpdir(), 'theme-gate-'));
  try {
    const copy = join(scratch, 'webui');
    for (const directory of ['scripts', 'src', 'public', 'tests']) cpSync(join(webuiRoot, directory), join(copy, directory), { recursive: true });
    cpSync(join(webuiRoot, 'index.html'), join(copy, 'index.html'));
    const tokens = join(copy, 'src/design-system/tokens.css');
    writeFileSync(tokens, `${readFileSync(tokens, 'utf8')}\n:root[data-theme="glass-dark"] { --mx-x: 1; }\n`);
    const link = join(scratch, 'gate.mjs');
    symlinkSync(join(copy, 'scripts/check-theme-selectors.mjs'), link);
    assert.throws(() => execFileSync(process.execPath, [link], { stdio: 'pipe' }), (error) => error.status === 1 && String(error.stderr).includes('dead-theme-selector'));
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }
});

test('a data-color-scheme value selector is rejected because it holds the raw preference', () => {
  assert.deepEqual(rules(scan('src/example.css', '[data-color-scheme="dark"] .x { color: red; }')), ['data-color-scheme@1']);
});

test('a selector split across lines is still evaluated', () => {
  const wrapped = '.a,\n:root[data-theme\n  ="mlxcel-dark"] .b { color: red; }';
  assert.deepEqual(rules(scan('src/example.css', wrapped)), ['dead-theme-selector@2']);
});

test('the escape hatch needs a written reason on the same or the preceding line', () => {
  const withReason = '/* theme-selector-allow: print stylesheet, the printed document has no data-theme */\n@media (prefers-color-scheme: dark) { .x { color: red; } }';
  assert.deepEqual(scan('src/example.css', withReason), []);
  const sameLine = '@media (prefers-color-scheme: dark) { .x { color: red; } } /* theme-selector-allow: host OS overlay */';
  assert.deepEqual(scan('src/example.css', sameLine), []);
  const bare = '/* theme-selector-allow: */\n@media (prefers-color-scheme: dark) { .x { color: red; } }';
  assert.deepEqual(rules(scan('src/example.css', bare)), ['prefers-color-scheme@2']);
  const tooFar = '/* theme-selector-allow: a reason two lines up */\n\n@media (prefers-color-scheme: dark) { .x { color: red; } }';
  assert.deepEqual(rules(scan('src/example.css', tooFar)), ['prefers-color-scheme@3']);
});

test('app code may consult prefers-color-scheme only in the resolver and the bootstrap', () => {
  const query = "window.matchMedia('(prefers-color-scheme: dark)');";
  assert.deepEqual(scan('src/design-system/theme.ts', query), []);
  assert.deepEqual(scan('public/theme-bootstrap.js', query), []);
  assert.deepEqual(rules(scan('src/features/chat/chat.tsx', query)), ['prefers-color-scheme@1']);
  // Browser tests emulate the host scheme on purpose.
  assert.deepEqual(scan('tests/theme.spec.ts', "await client.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'dark' }] });"), []);
});

test('script literals compared with or written to data-theme or data-astryx-theme must be shipped values', () => {
  const source = [
    "if (root.dataset.theme === 'glass-dark') {}",
    "if ('mlxcel-light' !== document.documentElement.dataset.theme) {}",
    "root.dataset.theme = 'system';",
    "root.setAttribute('data-theme', 'mlxcel-dark');",
    "await expect(page.locator('html')).toHaveAttribute('data-theme', 'glass-light');",
    "root.dataset.astryxTheme = 'glass';",
    "await expect(page.locator('html')).toHaveAttribute('data-astryx-theme', 'neutral');",
    "if (root.getAttribute('data-theme') === 'dark') {}",
    "root.dataset.theme = '';",
    "await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');",
    "await expect(page.locator('html')).toHaveAttribute('data-astryx-theme', 'mlxcel-glass');",
  ].join('\n');
  assert.deepEqual(rules(scan('tests/example.spec.ts', source)), [1, 2, 3, 4, 5, 6, 7].map((line) => `dead-theme-value@${line}`));
});

test('only main.tsx imports the single theme entry and only themes/index.css imports theme files', () => {
  assert.deepEqual(scan('src/main.tsx', "import '@lablup/ui-common/styles/base.css';\nimport './design-system/themes/index.css';"), []);
  assert.deepEqual(rules(scan('src/main.tsx', "import './design-system/themes/index.css';\nimport '@lablup/ui-common/styles/themes/orange-light.css';")), ['theme-import@2']);
  assert.deepEqual(rules(scan('src/features/chat/chat.tsx', "import '../../design-system/themes/glass.css';")), ['theme-import@1']);
  assert.deepEqual(rules(scan('src/features/models/models.tsx', "import '@lablup/ui-common/styles/themes/orange-dark.css';")), ['theme-import@1']);
  assert.deepEqual(scan('src/design-system/themes/index.css', '@import "./built/mlxcel.css";\n@import "./glass.css" layer(components);'), []);
  // The built theme modules are scripts, not stylesheets, and their generated usage comments are not imports.
  assert.deepEqual(scan('src/design-system/common-theme.tsx', "import { mlxcelTheme } from './themes/built/mlxcel.js';"), []);
  assert.deepEqual(scan('src/design-system/themes/built/mlxcel.js', " *   import './mlxcel.css';"), []);
  assert.deepEqual(rules(scan('src/styles.css', '@import "./design-system/themes/glass.css";')), ['theme-import@1']);
  // theme.ts is the resolver module, not a theme stylesheet.
  assert.deepEqual(scan('src/features/chat/chat.tsx', "import { ASTRYX_THEME_NAMES } from '../../design-system/theme';"), []);
});

test('every family needs a built stylesheet scoped to its Astryx theme under themes/', () => {
  const partial = { path: 'src/design-system/themes/built/partial.css', source: '@scope ([data-astryx-theme="mlxcel"]) to ([data-astryx-theme]) { :scope { --x: 1; } }' };
  const violations = findThemeSelectorViolations([partial]);
  assert.deepEqual(violations.map((violation) => violation.message.match(/"([a-z-]+)"/)?.[1]), ['mlxcel-glass']);
  assert.ok(violations.every((violation) => violation.rule === 'missing-theme'));
  // A selector outside a stylesheet under themes/ does not count as the theme.
  assert.equal(findThemeSelectorViolations([partial, { path: 'src/features/example.css', source: '[data-astryx-theme="mlxcel-glass"] .x { color: red; }' }]).length, 1);
});

test('attribute operators follow Selectors Level 4', () => {
  assert.equal(attributeMatches('=', 'glass-dark', false, 'glass-dark'), true);
  assert.equal(attributeMatches('=', 'Glass-Dark', true, 'glass-dark'), true);
  assert.equal(attributeMatches('^=', '', false, 'glass-dark'), false);
  assert.equal(attributeMatches('$=', '-dark', false, 'glass-dark'), true);
  assert.equal(attributeMatches('*=', 'ss-d', false, 'glass-dark'), true);
  assert.equal(attributeMatches('~=', 'glass-dark', false, 'glass-dark'), true);
  assert.equal(attributeMatches('~=', 'glass dark', false, 'glass-dark'), false);
  assert.equal(attributeMatches('|=', 'glass', false, 'glass-dark'), true);
  assert.equal(attributeMatches('|=', 'glass-dark', false, 'glass-dark'), true);
  assert.equal(attributeMatches('|=', 'gla', false, 'glass-dark'), false);
});

test('the gate reads its theme values from theme.ts, not a private copy', () => {
  const gate = readFileSync(new URL('./check-theme-selectors.mjs', import.meta.url), 'utf8');
  assert.match(gate, /from '\.\.\/src\/design-system\/theme\.ts'/);
  assert.doesNotMatch(gate, /\['mlxcel', 'glass'\]|'mlxcel-glass'|\['light', 'dark'\]/);
});
