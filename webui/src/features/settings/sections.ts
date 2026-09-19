// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
/** Settings tabs in display order; each is addressed as `#settings/<section>`. */
export const SETTINGS_SECTIONS = ['appearance', 'requests', 'model', 'server'] as const;
export type SettingsSection = typeof SETTINGS_SECTIONS[number];
export const DEFAULT_SETTINGS_SECTION: SettingsSection = 'appearance';

/** A bare `#settings`, or a section this build does not know, opens the default section. */
export function settingsSectionFrom(raw: string | undefined): SettingsSection {
  return (SETTINGS_SECTIONS as readonly string[]).includes(raw ?? '') ? raw as SettingsSection : DEFAULT_SETTINGS_SECTION;
}

export function settingsSectionHash(section: SettingsSection): string {
  return `#settings/${section}`;
}
