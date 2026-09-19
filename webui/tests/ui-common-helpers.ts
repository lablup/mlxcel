// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { Page } from '@playwright/test';

type Entry = { key: string; en: string; ko: string };
const catalog = (JSON.parse(readFileSync(fileURLToPath(new URL('../../tests/fixtures/webui/strings.json', import.meta.url)), 'utf8')) as { strings: Entry[] }).strings;

// Read copy from the checked fixture so these specs follow catalog edits.
export function text(key: string, locale: 'en' | 'ko' = 'en'): string {
  const entry = catalog.find((item) => item.key === key);
  if (!entry) throw new Error(`Missing catalog string ${key}`);
  return entry[locale];
}

// A tooltip counts as shown only when a sighted user can perceive it: rendered,
// not visibility-hidden and not faded out. Playwright's toBeVisible ignores opacity.
export async function shownTooltips(page: Page): Promise<string[]> {
  return page.evaluate(() => Array.from(document.querySelectorAll<HTMLElement>('[role="tooltip"]')).filter((element) => {
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' && Number(style.opacity) > 0.5 && rect.width > 0 && rect.height > 0;
  }).map((element) => element.textContent?.trim() ?? ''));
}

export async function horizontalOverflow(page: Page): Promise<number> {
  return page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
}
