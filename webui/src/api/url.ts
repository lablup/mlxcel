// Copyright 2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

function hasSpaceOrControls(value: string): boolean {
  return Array.from(value).some(character => { const code = character.charCodeAt(0); return code <= 32 || code >= 127 && code <= 159; });
}

function assertUnambiguousPath(path: string): void {
  if (/[\\?#]/.test(path) || hasSpaceOrControls(path) || path.includes('//')) {
    throw new Error('WebUI API base must not contain a query, hash, backslash, whitespace, controls or empty path segments.');
  }
  for (const segment of path.split('/')) {
    let decoded: string;
    try { decoded = decodeURIComponent(segment); }
    catch { throw new Error('WebUI API base contains an invalid path escape.'); }
    if (decoded === '.' || decoded === '..') throw new Error('WebUI API base must not contain dot path segments.');
    // Reject alternate separators and multiple-decoding ambiguity before URL
    // normalization can silently turn them into a different route authority.
    if (/[/\\%?#]/.test(decoded) || hasSpaceOrControls(decoded)) throw new Error('WebUI API base contains an ambiguous encoded path segment.');
  }
}

export function validateApiBase(input: string | undefined): string {
  const candidate = input ?? apiBaseFromDocument();
  if (candidate === '') return '';
  if (!candidate.startsWith('/') || candidate.startsWith('//')) {
    throw new Error('WebUI API base must be a same-origin absolute path.');
  }
  assertUnambiguousPath(candidate);
  // URL pathname encoding preserves existing escapes, unlike encodeURIComponent
  // applied to an already encoded document prefix. Root denotes no prefix.
  return new URL(candidate, 'http://webui.invalid').pathname.replace(/\/$/, '');
}

function prefixFromMount(path: string): string | null {
  const match = /^(.*)\/webui(?:\/(?:index\.html)?)?$/.exec(path);
  return match?.[1] ?? null;
}

export function apiBaseFromDocument(): string {
  if (typeof document === 'undefined') return '';
  const configured = document.querySelector<HTMLMetaElement>('meta[name="mlxcel-ui-api-base"]')?.content;
  if (configured !== undefined && configured.length > 0) return validateApiBase(configured);
  const base = document.querySelector<HTMLBaseElement>('base')?.getAttribute('href');
  if (base !== null && base !== undefined && base.length > 0) {
    // Inspect the raw path first: new URL would erase literal/escaped dot segments.
    const rawPath = base.replace(/^[a-z][a-z\d+.-]*:\/\/[^/]*/i, '');
    // A leading literal ./ (or .) is standard same-directory base syntax,
    // not an API-prefix segment. Encoded dots and parent traversal still fail.
    assertUnambiguousPath(rawPath === '.' ? '' : rawPath.replace(/^\.\//, ''));
    const parsed = new URL(base, document.URL);
    if (parsed.origin !== window.location.origin || parsed.username || parsed.password || parsed.search || parsed.hash) {
      throw new Error('WebUI document base must stay on the current origin without credentials, query or hash.');
    }
    return validateApiBase(prefixFromMount(parsed.pathname) ?? parsed.pathname);
  }
  // The bundled index has no injected base/meta. Its actual mount supplies the
  // prefix for initial login; Vite's root preview intentionally stays unprefixed.
  const path = window.location.pathname;
  const prefix = prefixFromMount(path);
  if (prefix === null) return '';
  assertUnambiguousPath(path);
  return validateApiBase(prefix);
}

export function apiPath(apiBase: string, path: string, query?: Readonly<Record<string, string | number | boolean | null | undefined>>): string {
  const isUiPath = path.startsWith('/ui-api/v1/');
  const isInferenceStreamPath = path === '/v1/chat/completions' || path === '/v1/responses';
  if (!isUiPath && !isInferenceStreamPath && path !== '/settings' && path !== '/props' && path !== '/tokenize') {
    throw new Error('WebUI client paths must stay under /ui-api/v1/ or the approved inference stream endpoints.');
  }
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query ?? {})) {
    if (value !== undefined && value !== null) {
      params.set(key, String(value));
    }
  }
  const suffix = params.size === 0 ? '' : `?${params.toString()}`;
  return `${apiBase}${path}${suffix}`;
}

export function encodeOpaquePathSegment(id: string): string {
  if (id.length === 0) {
    throw new Error('Opaque path segment must not be empty.');
  }
  return encodeURIComponent(id);
}
