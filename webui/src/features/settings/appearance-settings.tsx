// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// Browser-only appearance: every control writes the stored preference, which App
// applies to the document (theme family and scheme through design-system/theme.ts).
import React from 'react';
import type { AppearancePreferences, ColorSchemePreference, ContrastPreference, ThemeFamily } from '../../design-system/preferences';
import { Select, Toggle } from '../../design-system/primitives';
import { t, testId } from '../../i18n/catalog';

export function AppearanceSettings(props: { appearance: AppearancePreferences; setAppearance: (next: AppearancePreferences) => void }): React.JSX.Element {
  const { appearance } = props;
  const locale = appearance.locale;
  const set = (patch: Partial<AppearancePreferences>): void => props.setAppearance({ ...appearance, ...patch });
  return (
    <div className="settings-grid">
      <Select locale={locale} label={t(locale, 'settings.theme')} value={appearance.themeFamily} onChange={(value) => set({ themeFamily: value as ThemeFamily })} options={[{ value: 'mlxcel', label: t(locale, 'settings.theme.mlxcel') }, { value: 'glass', label: t(locale, 'settings.theme.glass') }]} testId={testId('settings.theme')} />
      <Select locale={locale} label={t(locale, 'settings.color_scheme')} value={appearance.colorScheme} onChange={(value) => set({ colorScheme: value as ColorSchemePreference })} options={[{ value: 'system', label: t(locale, 'settings.color_scheme.system') }, { value: 'light', label: t(locale, 'settings.color_scheme.light') }, { value: 'dark', label: t(locale, 'settings.color_scheme.dark') }]} testId={testId('settings.color_scheme')} />
      <Select locale={locale} label={t(locale, 'settings.material')} value={appearance.material} onChange={(value) => set({ material: value as AppearancePreferences['material'] })} options={[{ value: 'glass', label: t(locale, 'settings.material.glass') }, { value: 'tinted', label: t(locale, 'settings.material.tinted') }, { value: 'opaque', label: t(locale, 'settings.material.opaque') }]} testId={testId('settings.material')} />
      <Select locale={locale} label={t(locale, 'settings.locale')} value={appearance.locale} onChange={(value) => set({ locale: value as AppearancePreferences['locale'] })} options={[{ value: 'en', label: t(locale, 'settings.locale.en') }, { value: 'ko', label: t(locale, 'settings.locale.ko') }]} testId={testId('settings.locale')} />
      <Select locale={locale} label={t(locale, 'settings.high_contrast')} value={appearance.highContrast} onChange={(value) => set({ highContrast: value as ContrastPreference })} options={[{ value: 'system', label: t(locale, 'settings.high_contrast.system') }, { value: 'on', label: t(locale, 'settings.high_contrast.on') }, { value: 'off', label: t(locale, 'settings.high_contrast.off') }]} testId={testId('settings.high_contrast')} />
      <label className="ds-field"><span>{t(locale, 'settings.glass_intensity')}: {appearance.glassIntensity}</span><input type="range" min="0" max="100" value={appearance.glassIntensity} onChange={(event) => set({ glassIntensity: Number(event.currentTarget.value) })} data-testid={testId('settings.glass_intensity')} /></label>
      <Toggle label={t(locale, 'settings.reduce_motion')} checked={appearance.reduceMotion} onChange={(checked) => set({ reduceMotion: checked })} testId={testId('settings.reduce_motion')} />
      <Toggle label={t(locale, 'settings.reduce_transparency')} checked={appearance.reduceTransparency} onChange={(checked) => set({ reduceTransparency: checked })} testId={testId('settings.reduce_transparency')} />
    </div>
  );
}
