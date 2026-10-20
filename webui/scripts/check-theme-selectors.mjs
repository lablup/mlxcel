#!/usr/bin/env node
// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
/* global process, console, URL */
/**
 * Theme-selector gate (#1903). Fails when WebUI code gates on a theme value no
 * shipped theme can produce, or on a signal that is not the applied theme.
 *
 * The document root carries three theme attributes (src/design-system/theme.ts):
 *
 *   data-theme          the resolved scheme, `light` or `dark` (Astryx's own
 *                       attribute, written by the root <Theme> as well)
 *   data-astryx-theme   the family's Astryx theme name (`mlxcel`,
 *                       `mlxcel-glass`), which scopes the built theme CSS
 *   data-theme-family   the product family (`mlxcel`, `glass`)
 *
 * A selector on any other value compiles, renders and silently never applies:
 * the pre-Astryx `<family>-<scheme>` ids (`[data-theme="glass-dark"]`) are the
 * likeliest slip. The same silent failure follows from:
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
 *   - a theme stylesheet imported anywhere but the single entry
 *     (src/design-system/themes/index.css, imported only by src/main.tsx).
 *
 * The gate evaluates every attribute selector on those attributes, whatever
 * its operator, against the values theme.ts exports, and it checks that every
 * family's Astryx theme has a built stylesheet scoped to its name under
 * src/design-system/themes/.
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

import { ASTRYX_THEME_NAMES, COLOR_SCHEMES, THEME_FAMILIES } from '../src/design-system/theme.ts';

const ASTRYX_THEMES = Object.values(ASTRYX_THEME_NAMES);

const SCRIPT_EXTENSIONS = new Set(['.ts', '.tsx', '.mts', '.cts', '.js', '.jsx', '.mjs', '.cjs']);
const SCANNED_EXTENSIONS = new Set(['.css', '.html', ...SCRIPT_EXTENSIONS]);
const THEMES_DIR = 'src/design-system/themes';
const THEME_ENTRY_CSS = `${THEMES_DIR}/index.css`;
const THEME_ENTRY_IMPORTER = 'src/main.tsx';
/** `astryx theme build` output: scanned for its selectors, but its generated usage comments are not app imports. */
const BUILT_THEMES_DIR = 'src/design-system/themes/built/';
const UI_COMMON_THEMES = '@lablup/ui-common/styles/themes/';
/** The only app files allowed to ask the host for its color scheme. */
const SYSTEM_SCHEME_RESOLVERS = new Set(['src/design-system/theme.ts', 'public/theme-bootstrap.js']);

/**
 * A reason must start with something other than a comment delimiter, so that a
 * bare marker does not read its own closing delimiter as the reason.
 */
const ALLOW_RE = /theme-selector-allow:\s*([^\s*/][^*\n]*)/;

/**
 * `[attr]`, `[attr="v"]`, `[attr^='v' i]`, `[attr=v]` for the four theme
 * attributes. Attribute names match case-insensitively, as they do in HTML
 * documents; each whitespace run has one quantifier so matching stays linear.
 */
const ATTRIBUTE_SELECTOR_RE = /\[\s*(data-theme-family|data-color-scheme|data-astryx-theme|data-theme)\s*(?:([~|^$*]?=)\s*(?:"([^"]*)"|'([^']*)'|([^\s\]"'`]+))\s*(?:([iIsS])\s*)?)?\]/gi;
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
/** The same for data-astryx-theme, whose literals must be a family's Astryx theme name. */
const SCRIPT_ASTRYX_THEME_VALUE_RES = [
  /\.dataset\.astryxTheme\s*(?:===?|!==?|=(?!=))\s*(['"])([^'"\n]*)\1/g,
  /(['"])([^'"\n]*)\1\s*(?:===?|!==?)\s*[\w$.?]*\.dataset\.astryxTheme\b/g,
  /(?:getAttribute\(\s*['"]data-astryx-theme['"]\s*\)\s*(?:===?|!==?)|setAttribute\(\s*['"]data-astryx-theme['"]\s*,|toHaveAttribute\(\s*['"]data-astryx-theme['"]\s*,)\s*(['"])([^'"\n]*)\1/g,
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
  // The built theme modules (themes/built/*.js) are scripts; only stylesheets count.
  if (!specifier.startsWith('.') || !specifier.endsWith('.css')) return false;
  const target = join(dirname(fromPath), specifier).split(sep).join('/');
  return target === THEMES_DIR || target.startsWith(`${THEMES_DIR}/`);
}

/**
 * Finds every violation across the given sources. Paths are relative to the
 * webui/ directory with forward slashes. Exported so the spec can drive it
 * without touching the filesystem.
 */
export function findThemeSelectorViolations(sources, options = {}) {
  const schemes = options.schemes ?? COLOR_SCHEMES;
  const families = options.families ?? THEME_FAMILIES;
  const astryxThemes = options.astryxThemes ?? ASTRYX_THEMES;
  const candidatesFor = { 'data-theme': schemes, 'data-theme-family': families, 'data-astryx-theme': astryxThemes };
  const kindOf = { 'data-theme': 'color scheme', 'data-theme-family': 'theme family', 'data-astryx-theme': 'Astryx theme name' };
  const violations = [];
  const declaredThemes = new Set();
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
        report(match.index, 'data-color-scheme', `${match[0]} reads the stored preference, which holds "system" unless the user pinned a scheme. Gate on the applied scheme instead, for example [data-theme="dark"].`);
        continue;
      }
      const candidates = candidatesFor[attribute];
      if (!candidates.some((candidate) => attributeMatches(operator, value, caseInsensitive, candidate))) {
        report(match.index, 'dead-theme-selector', `${match[0]} matches no shipped ${kindOf[attribute]} (${describeIds(candidates)}), so the rule never applies. data-theme holds the scheme (light, dark), data-theme-family the family and data-astryx-theme the family's Astryx theme name.`);
      } else if (attribute === 'data-astryx-theme' && operator === '=' && extension === '.css' && path.startsWith(`${THEMES_DIR}/`)) {
        declaredThemes.add(caseInsensitive ? value.toLowerCase() : value);
      }
    }

    if (extension === '.css') {
      for (const match of source.matchAll(MEDIA_COLOR_SCHEME_RE)) {
        report(match.index, 'prefers-color-scheme', '@media (prefers-color-scheme) follows the host OS, not the applied theme: it lands on a light surface under a dark OS and never fires for a dark theme under a light OS. Gate on [data-theme="dark"] or [data-theme="light"], or annotate a genuine OS-level concern with "theme-selector-allow: <reason>".');
      }
      for (const match of source.matchAll(CSS_IMPORT_RE)) {
        if (isThemeStylesheetSpecifier(path, match[2]) && path !== THEME_ENTRY_CSS) {
          report(match.index, 'theme-import', `${match[2]} is a theme stylesheet; only ${THEME_ENTRY_CSS} may import one.`);
        }
      }
    }

    if (isScript && !path.startsWith(BUILT_THEMES_DIR)) {
      if ((path.startsWith('src/') || path.startsWith('public/')) && !SYSTEM_SCHEME_RESOLVERS.has(path)) {
        for (const match of source.matchAll(ANY_COLOR_SCHEME_RE)) {
          report(match.index, 'prefers-color-scheme', `prefers-color-scheme is consulted only by ${[...SYSTEM_SCHEME_RESOLVERS].join(' and ')}; read the applied theme (data-theme) or use resolveThemeSelection() instead.`);
        }
      }
      for (const [patterns, allowed, attribute] of [[SCRIPT_THEME_VALUE_RES, schemes, 'data-theme'], [SCRIPT_ASTRYX_THEME_VALUE_RES, astryxThemes, 'data-astryx-theme']]) {
        for (const pattern of patterns) {
          for (const match of source.matchAll(pattern)) {
            const value = match[2];
            if (value === '' || allowed.includes(value)) continue;
            report(match.index, 'dead-theme-value', `"${value}" is not a ${kindOf[attribute]} the app writes to ${attribute} (${describeIds(allowed)}).`);
          }
        }
      }
      for (const match of source.matchAll(SCRIPT_IMPORT_RE)) {
        if (!isThemeStylesheetSpecifier(path, match[2])) continue;
        if (path !== THEME_ENTRY_IMPORTER) {
          report(match.index, 'theme-import', `${match[2]} is a theme stylesheet; components never import one, only ${THEME_ENTRY_IMPORTER} imports the single theme entry.`);
        } else {
          entryImports += 1;
          if (entryImports > 1) report(match.index, 'theme-import', `${THEME_ENTRY_IMPORTER} imports more than one theme stylesheet; the single entry is ${THEME_ENTRY_CSS}.`);
        }
      }
    }
  }

  if (options.requireCoverage !== false) {
    for (const name of astryxThemes) {
      if (!declaredThemes.has(name)) {
        violations.push({ path: THEMES_DIR, line: 0, rule: 'missing-theme', message: `Astryx theme "${name}" is named by theme.ts but no stylesheet under ${THEMES_DIR}/ is scoped to [data-astryx-theme="${name}"], so selecting its family would render Astryx's defaults. Run \`pnpm run build:themes\`.`, text: '' });
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
    console.log(`Theme selectors: ${sources.length} WebUI file(s) checked against schemes (${describeIds(COLOR_SCHEMES)}), families (${describeIds(THEME_FAMILIES)}) and Astryx themes (${describeIds(ASTRYX_THEMES)}); no dead theme gate.`);
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
