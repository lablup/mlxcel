#!/usr/bin/env node
// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
/* global process, console, URL */
/**
 * Theme-selector gate (#1903). Fails when WebUI code gates on a theme id no
 * shipped theme can produce, or on a signal that is not the applied theme.
 *
 * The app writes `data-theme` from a ThemeId, which is always
 * `<family>-<scheme>` (src/design-system/theme.ts: mlxcel-light, glass-dark,
 * and so on). A bare light or dark value therefore never matches. It is not a
 * parse error, nothing warns, and the page still renders: dark mode simply
 * stops applying. The same silent failure follows from:
 *
 *   - a `data-color-scheme` selector: that attribute carries the stored
 *     preference verbatim, so it reads `system` unless the user pinned a
 *     scheme, and the rule fires with no relationship to what is on screen;
 *   - `@media (prefers-color-scheme: ...)` in a stylesheet, or a
 *     `prefers-color-scheme` query in app code: it reflects the host OS, while
 *     the applied theme reflects the in-app choice. Only the resolver
 *     (src/design-system/theme.ts) and the pre-mount bootstrap
 *     (public/theme-bootstrap.js) consult it, to resolve `system` before
 *     `data-theme` is written;
 *   - a theme stylesheet imported anywhere but the single entry: ui-common's
 *     rule is at most one theme file alongside base.css, and components never
 *     import a theme file themselves.
 *
 * Working selectors are the explicit id (`[data-theme="glass-dark"]`), or a
 * prefix or suffix that matches at least one shipped id (`[data-theme$="-dark"]`
 * for every family). The gate evaluates every data-theme attribute selector,
 * whatever its operator, against the ids theme.ts exports, and it also checks
 * that every shipped id has a theme block under src/design-system/themes/.
 *
 * Escape hatch: a genuine host-OS concern can keep its query by carrying a
 * written reason on the same line or the line before it, for example
 *
 *     theme-selector-allow: print stylesheet, the document has no data-theme
 *
 * inside a comment. The reason is mandatory, so an exemption is a sentence
 * someone defends in review rather than a silent bypass.
 *
 * Usage:
 *   pnpm --dir webui run check:theme-selectors
 *
 * Exit code 1 on any violation; under CI each violation is also printed as a
 * GitHub ::error annotation on the offending line.
 */

import { readdirSync, readFileSync, realpathSync, statSync } from 'node:fs';
import { dirname, extname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

import { THEME_FAMILIES, THEME_IDS } from '../src/design-system/theme.ts';

const SCRIPT_EXTENSIONS = new Set(['.ts', '.tsx', '.mts', '.cts', '.js', '.jsx', '.mjs', '.cjs']);
const SCANNED_EXTENSIONS = new Set(['.css', '.html', ...SCRIPT_EXTENSIONS]);
const THEMES_DIR = 'src/design-system/themes';
const THEME_ENTRY_CSS = `${THEMES_DIR}/index.css`;
const THEME_ENTRY_IMPORTER = 'src/main.tsx';
const UI_COMMON_THEMES = '@lablup/ui-common/styles/themes/';
/** The only app files allowed to ask the host for its color scheme. */
const SYSTEM_SCHEME_RESOLVERS = new Set(['src/design-system/theme.ts', 'public/theme-bootstrap.js']);

/**
 * A reason must start with something other than a comment delimiter, so that a
 * bare marker does not read its own closing delimiter as the reason.
 */
const ALLOW_RE = /theme-selector-allow:\s*([^\s*/][^*\n]*)/;

/**
 * `[attr]`, `[attr="v"]`, `[attr^='v' i]`, `[attr=v]` for the three theme
 * attributes. Attribute names match case-insensitively, as they do in HTML
 * documents; each whitespace run has one quantifier so matching stays linear.
 */
const ATTRIBUTE_SELECTOR_RE = /\[\s*(data-theme-family|data-color-scheme|data-theme)\s*(?:([~|^$*]?=)\s*(?:"([^"]*)"|'([^']*)'|([^\s\]"'`]+))\s*(?:([iIsS])\s*)?)?\]/gi;
const MEDIA_COLOR_SCHEME_RE = /@media[^{;]*prefers-color-scheme/g;
const ANY_COLOR_SCHEME_RE = /prefers-color-scheme/g;
/** Literal values compared with, or written to, data-theme in script code. */
const SCRIPT_THEME_VALUE_RES = [
  /\.dataset\.theme\s*(?:===?|!==?)\s*(['"])([^'"\n]*)\1/g,
  /(['"])([^'"\n]*)\1\s*(?:===?|!==?)\s*[\w$.?]*\.dataset\.theme\b/g,
  /\.dataset\.theme\s*=(?!=)\s*(['"])([^'"\n]*)\1/g,
  /getAttribute\(\s*['"]data-theme['"]\s*\)\s*(?:===?|!==?)\s*(['"])([^'"\n]*)\1/g,
  /setAttribute\(\s*['"]data-theme['"]\s*,\s*(['"])([^'"\n]*)\1/g,
  /toHaveAttribute\(\s*['"]data-theme['"]\s*,\s*(['"])([^'"\n]*)\1/g,
];
const CSS_IMPORT_RE = /@import\s+(?:url\(\s*)?(['"]?)([^'")\s;]+)\1/g;
/** `import x from 'm'`, `import 'm'`, `import('m')`, `require('m')`: the first string after the keyword, lazily, so the scan is linear. */
const SCRIPT_IMPORT_RE = /\b(?:import|require)\b[^'";]*?(['"])([^'"\n]+)\1/g;

/** Evaluates one attribute selector against a candidate value, per Selectors Level 4. */
export function attributeMatches(operator, value, caseInsensitive, candidate) {
  const wanted = caseInsensitive ? value.toLowerCase() : value;
  const actual = caseInsensitive ? candidate.toLowerCase() : candidate;
  switch (operator) {
    case '=':
      return actual === wanted;
    case '^=':
      return wanted !== '' && actual.startsWith(wanted);
    case '$=':
      return wanted !== '' && actual.endsWith(wanted);
    case '*=':
      return wanted !== '' && actual.includes(wanted);
    case '~=':
      return wanted !== '' && !/\s/.test(wanted) && actual.split(/\s+/).includes(wanted);
    case '|=':
      return actual === wanted || actual.startsWith(`${wanted}-`);
    default:
      return false;
  }
}

/** Byte offset to 1-indexed line, via binary search over precomputed starts. */
function lineIndex(source) {
  const starts = [0];
  for (let index = 0; index < source.length; index += 1) {
    if (source[index] === '\n') starts.push(index + 1);
  }
  return (offset) => {
    let low = 0;
    let high = starts.length - 1;
    while (low < high) {
      const middle = (low + high + 1) >> 1;
      if (starts[middle] <= offset) low = middle;
      else high = middle - 1;
    }
    return low + 1;
  };
}

/** True when the line, or the one above it, carries a non-empty allow reason. */
function isAllowed(lines, line) {
  const own = lines[line - 1] ?? '';
  const above = lines[line - 2] ?? '';
  const match = ALLOW_RE.exec(own) ?? ALLOW_RE.exec(above);
  return Boolean(match && match[1].trim().length > 0);
}

function describeIds(ids) {
  return ids.join(', ');
}

function isThemeStylesheetSpecifier(fromPath, specifier) {
  if (specifier.startsWith(UI_COMMON_THEMES)) return true;
  if (!specifier.startsWith('.')) return false;
  const target = join(dirname(fromPath), specifier).split(sep).join('/');
  return target === THEMES_DIR || target.startsWith(`${THEMES_DIR}/`);
}

/**
 * Finds every violation across the given sources. Paths are relative to the
 * webui/ directory with forward slashes. Exported so the spec can drive it
 * without touching the filesystem.
 */
export function findThemeSelectorViolations(sources, options = {}) {
  const themeIds = options.themeIds ?? THEME_IDS;
  const families = options.families ?? THEME_FAMILIES;
  const violations = [];
  const declaredIds = new Set();
  let entryImports = 0;

  for (const { path, source } of sources) {
    const lines = source.split('\n');
    const toLine = lineIndex(source);
    const extension = extname(path);
    const isScript = SCRIPT_EXTENSIONS.has(extension);
    const report = (offset, rule, message) => {
      const line = toLine(offset);
      if (isAllowed(lines, line)) return;
      violations.push({ path, line, rule, message, text: (lines[line - 1] ?? '').trim() });
    };

    for (const match of source.matchAll(ATTRIBUTE_SELECTOR_RE)) {
      const attribute = match[1].toLowerCase();
      const operator = match[2];
      if (operator === undefined) continue;
      const value = match[3] ?? match[4] ?? match[5] ?? '';
      if (value.includes('${')) continue;
      const caseInsensitive = match[6] === 'i' || match[6] === 'I';
      if (attribute === 'data-color-scheme') {
        report(match.index, 'data-color-scheme', `${match[0]} reads the stored preference, which holds "system" unless the user pinned a scheme. Gate on the applied theme instead, for example [data-theme$="-dark"].`);
        continue;
      }
      const candidates = attribute === 'data-theme' ? themeIds : families;
      if (!candidates.some((candidate) => attributeMatches(operator, value, caseInsensitive, candidate))) {
        const kind = attribute === 'data-theme' ? 'theme id' : 'theme family';
        report(match.index, 'dead-theme-selector', `${match[0]} matches no shipped ${kind} (${describeIds(candidates)}), so the rule never applies. data-theme is always <family>-<scheme>; use [data-theme$="-dark"] or an explicit id.`);
      } else if (attribute === 'data-theme' && operator === '=' && path.startsWith(`${THEMES_DIR}/`)) {
        declaredIds.add(caseInsensitive ? value.toLowerCase() : value);
      }
    }

    if (extension === '.css') {
      for (const match of source.matchAll(MEDIA_COLOR_SCHEME_RE)) {
        report(match.index, 'prefers-color-scheme', '@media (prefers-color-scheme) follows the host OS, not the applied theme: it lands on a light surface under a dark OS and never fires for a dark theme under a light OS. Gate on [data-theme$="-dark"] or [data-theme$="-light"], or annotate a genuine OS-level concern with "theme-selector-allow: <reason>".');
      }
      for (const match of source.matchAll(CSS_IMPORT_RE)) {
        if (isThemeStylesheetSpecifier(path, match[2]) && path !== THEME_ENTRY_CSS) {
          report(match.index, 'theme-import', `${match[2]} is a theme stylesheet; only ${THEME_ENTRY_CSS} may import one.`);
        }
      }
    }

    if (isScript) {
      if ((path.startsWith('src/') || path.startsWith('public/')) && !SYSTEM_SCHEME_RESOLVERS.has(path)) {
        for (const match of source.matchAll(ANY_COLOR_SCHEME_RE)) {
          report(match.index, 'prefers-color-scheme', `prefers-color-scheme is consulted only by ${[...SYSTEM_SCHEME_RESOLVERS].join(' and ')}; read the applied theme (data-theme) or use resolveThemeSelection() instead.`);
        }
      }
      for (const pattern of SCRIPT_THEME_VALUE_RES) {
        for (const match of source.matchAll(pattern)) {
          const value = match[2];
          if (value === '' || themeIds.includes(value)) continue;
          report(match.index, 'dead-theme-value', `"${value}" is not a shipped theme id (${describeIds(themeIds)}); data-theme is always <family>-<scheme>.`);
        }
      }
      for (const match of source.matchAll(SCRIPT_IMPORT_RE)) {
        if (!isThemeStylesheetSpecifier(path, match[2])) continue;
        if (path !== THEME_ENTRY_IMPORTER) {
          report(match.index, 'theme-import', `${match[2]} is a theme stylesheet; components never import one, only ${THEME_ENTRY_IMPORTER} imports the single theme entry.`);
        } else {
          entryImports += 1;
          if (entryImports > 1) report(match.index, 'theme-import', `${THEME_ENTRY_IMPORTER} imports more than one theme stylesheet; ui-common allows at most one alongside base.css (${THEME_ENTRY_CSS}).`);
        }
      }
    }
  }

  if (options.requireCoverage !== false) {
    for (const id of themeIds) {
      if (!declaredIds.has(id)) {
        violations.push({ path: THEMES_DIR, line: 0, rule: 'missing-theme', message: `Theme id "${id}" is shipped by theme.ts but no [data-theme="${id}"] block exists under ${THEMES_DIR}/, so selecting it would render ui-common's defaults.`, text: '' });
      }
    }
  }

  return violations.sort((left, right) => (left.path === right.path ? left.line - right.line : left.path < right.path ? -1 : 1));
}

function collectFiles(root, directory, files) {
  for (const entry of readdirSync(directory)) {
    if (entry === 'node_modules' || entry.startsWith('.')) continue;
    const full = join(directory, entry);
    if (statSync(full).isDirectory()) collectFiles(root, full, files);
    else if (SCANNED_EXTENSIONS.has(extname(entry))) files.push(full);
  }
  return files;
}

/** Reads every scanned WebUI source; paths come back relative to webui/. */
export function collectSources(webuiRoot) {
  const files = [];
  for (const directory of ['src', 'public', 'tests']) collectFiles(webuiRoot, join(webuiRoot, directory), files);
  files.push(join(webuiRoot, 'index.html'));
  return files.map((full) => ({ path: relative(webuiRoot, full).split(sep).join('/'), source: readFileSync(full, 'utf8') }));
}

function main() {
  const webuiRoot = resolve(fileURLToPath(new URL('.', import.meta.url)), '..');
  const sources = collectSources(webuiRoot);
  const stylesheets = sources.filter((source) => source.path.endsWith('.css'));
  if (stylesheets.length === 0 || !sources.some((source) => source.path.startsWith(`${THEMES_DIR}/`))) {
    console.error(`Theme-selector check: no stylesheets under ${THEMES_DIR}/. Refusing to pass, since every dead gate would look absent.`);
    process.exit(1);
  }
  const violations = findThemeSelectorViolations(sources);
  if (violations.length === 0) {
    console.log(`Theme selectors: ${sources.length} WebUI file(s) checked against ${THEME_IDS.length} theme ids (${describeIds(THEME_IDS)}); no dead theme gate.`);
    process.exit(0);
  }
  console.error(`\nTheme-selector check failed: ${violations.length} violation(s).\n`);
  for (const violation of violations) {
    console.error(`  webui/${violation.path}${violation.line ? `:${violation.line}` : ''}  [${violation.rule}] ${violation.text}`);
    console.error(`    ${violation.message}\n`);
  }
  if (process.env.CI === 'true' || process.env.CI === '1') {
    for (const violation of violations) {
      console.error(`::error file=webui/${violation.path}${violation.line ? `,line=${violation.line}` : ''}::${violation.message}`);
    }
  }
  process.exit(1);
}

// Node resolves symlinks for import.meta.url, so compare real paths: a run through a symlinked path must not skip the check silently.
function realPath(path) {
  try {
    return realpathSync(path);
  } catch {
    return resolve(path);
  }
}
const invokedDirectly = import.meta.url.startsWith('file:') && Boolean(process.argv[1]) && realPath(process.argv[1]) === realPath(fileURLToPath(import.meta.url));
if (invokedDirectly) main();
