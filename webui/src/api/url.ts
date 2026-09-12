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

export function validateApiBase(input: string | undefined): string {
  const candidate = input ?? apiBaseFromDocument();
  if (candidate === '') {
    return '';
  }
  if (!candidate.startsWith('/') || candidate.startsWith('//')) {
    throw new Error('WebUI API base must be a same-origin absolute path.');
  }
  if (candidate.includes('://') || candidate.includes('?') || candidate.includes('#') || candidate.includes('\\')) {
    throw new Error('WebUI API base must not contain an origin, query, hash, or backslash.');
  }
  const segments = candidate.split('/').filter((segment) => segment.length > 0);
  if (segments.some((segment) => segment === '.' || segment === '..')) {
    throw new Error('WebUI API base must not contain dot path segments.');
  }
  return `/${segments.map(encodeURIComponent).join('/')}`;
}

export function apiBaseFromDocument(): string {
  if (typeof document === 'undefined') {
    return '';
  }
  const configured = document.querySelector<HTMLMetaElement>('meta[name="mlxcel-ui-api-base"]')?.content;
  if (configured !== undefined && configured.length > 0) {
    return configured;
  }
  const base = document.querySelector<HTMLBaseElement>('base')?.getAttribute('href');
  if (base === null || base === undefined || base.length === 0) {
    return '';
  }
  const parsed = new URL(base, window.location.origin);
  if (parsed.origin !== window.location.origin || parsed.search.length > 0 || parsed.hash.length > 0) {
    throw new Error('WebUI document base must stay on the current origin.');
  }
  return parsed.pathname.replace(/\/webui\/?$/, '');
}

export function apiPath(apiBase: string, path: string, query?: Readonly<Record<string, string | number | boolean | null | undefined>>): string {
  const isUiPath = path.startsWith('/ui-api/v1/');
  const isInferenceStreamPath = path === '/v1/chat/completions' || path === '/v1/responses';
  if (!isUiPath && !isInferenceStreamPath) {
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
