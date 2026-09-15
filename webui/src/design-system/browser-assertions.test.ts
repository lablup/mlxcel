import { afterEach, describe, expect, it } from 'vitest';
import type { Page } from '@playwright/test';
import { expectSafeLayout } from '../../tests/browser-assertions';

// Execute the actual page-evaluation callback in jsdom; no browser or server.
const page = { evaluate: async (callback: () => unknown) => callback() } as unknown as Page;
afterEach(() => { document.body.replaceChildren(); });
describe('layout helper inline-style enforcement', () => {
  it('accepts only absent or trim-empty style attributes left by screenshot caret cleanup', async () => {
    document.body.innerHTML = '<input><textarea></textarea><input type="checkbox"><input type="file">';
    for (const element of document.querySelectorAll<HTMLElement>('input,textarea')) {
      element.style.setProperty('caret-color', 'transparent', 'important');
      element.style.setProperty('caret-color', '', '');
      expect(element.getAttribute('style')).toBe('');
      expect(element.style.length).toBe(0);
    }
    document.body.append(Object.assign(document.createElement('div'), { innerHTML: '<span style=" \n\t "></span>' }));
    await expect(expectSafeLayout(page)).resolves.toBeUndefined();
  });
  it.each(['color: red', 'width: 24px', 'not-a-declaration', '/* empty CSS, nonempty attribute */'])('still rejects unapproved nonempty inline text: %s', async (style) => {
    const input = document.createElement('input'); input.setAttribute('style', style); document.body.append(input);
    await expect(expectSafeLayout(page)).rejects.toThrow();
  });
  it('retains the narrow existing geometry allowlist and rejects extra declarations', async () => {
    document.body.innerHTML = '<div class="ds-progress"><div class="progress-bar__fill" style="width: 50%"></div></div>';
    await expect(expectSafeLayout(page)).resolves.toBeUndefined();
    document.querySelector<HTMLElement>('.progress-bar__fill')?.style.setProperty('color', 'red');
    await expect(expectSafeLayout(page)).rejects.toThrow();
  });
});
