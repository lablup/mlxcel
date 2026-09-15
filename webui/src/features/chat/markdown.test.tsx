// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { SafeMarkdown } from './markdown';
let host: HTMLDivElement;
let root: Root;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); });
async function render(text: string, streaming = false) { await act(async () => { root.render(<SafeMarkdown text={text} streaming={streaming} />); }); }
it('keeps hostile HTML inert, blocks every image scheme and permits only credential-free HTTP(S) links', async () => {
  await render('<img src=x onerror=alert(1)>\n![remote](https://tracker.test/image) ![inline](data:image/png;base64,eA==)\n[bad](javascript:alert) [file](file:///etc/passwd) [relative](/x) [credentials](https://user:secret@example.test) [good](https://example.test/a)');
  expect(host.querySelectorAll('img,script,iframe,svg')).toHaveLength(0);
  expect(host.textContent).toContain('<img src=x onerror=alert(1)>'); expect(host.textContent).toContain('Image omitted: remote');
  const links = host.querySelectorAll('a'); expect(links).toHaveLength(1); expect(links[0].href).toBe('https://example.test/a'); expect(links[0].rel).toBe('noopener noreferrer');
});
it('renders the documented text subset without overriding page heading hierarchy', async () => {
  await render('# Heading\n**bold** *emphasis* `code`\n- one\n- two');
  expect(host.querySelectorAll('h1,h2,h3')).toHaveLength(0); expect(host.querySelector('strong')?.textContent).toBe('Heading');
  expect(host.querySelector('em')?.textContent).toBe('emphasis'); expect(host.querySelectorAll('li')).toHaveLength(2);
});
it('leaves unfinished streaming code plain and copies via the shared ui-common button', async () => {
  const writeText = vi.fn().mockResolvedValue(undefined); Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
  await render('```js\nconst x = "<script>";', true);
  expect(host.querySelector('pre code')?.textContent).toBe('const x = "<script>";'); expect(host.querySelectorAll('pre span')).toHaveLength(0);
  const button = host.querySelector('button'); if (!button) throw new Error('Missing copy button'); expect(button.classList.contains('button')).toBe(true);
  await act(async () => button.click()); expect(writeText).toHaveBeenCalledWith('const x = "<script>";'); expect(button.textContent).toBe('Copied');
});
it('bounds display of large input and pathological unmatched delimiters', async () => {
  await render('['.repeat(100_000)); expect(host.textContent?.length).toBeLessThan(66_000); expect(host.textContent).toContain('Display truncated');
  await render('line\n'.repeat(600)); expect(host.querySelectorAll('.chat-markdown > p')).toHaveLength(513);
});
