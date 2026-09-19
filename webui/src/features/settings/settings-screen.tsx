// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
// The Settings route: one PageHeader whose description names the active section,
// then one tab per scope. Only the active tab's panel is mounted.
import React from 'react';
import type { AppearancePreferences } from '../../design-system/preferences';
import { PageHeader, Tabs } from '../../design-system/primitives';
import { t, testId, type Locale, type StringKey } from '../../i18n/catalog';
import { AppearanceSettings } from './appearance-settings';
import { GenerationSettings } from './generation-settings';
import { SETTINGS_SECTIONS, type SettingsSection } from './sections';
import { ModelSettings, ServerSettings } from './server-settings';
import { HelpTip } from './setting-control';
import './settings.css';

const LABEL: Readonly<Record<SettingsSection, StringKey>> = { appearance: 'settings.section.appearance', requests: 'settings.section.requests', model: 'settings.section.model', server: 'settings.section.server' };
// Appearance keeps the key and test id its header description has always had.
const DESCRIPTION: Readonly<Record<SettingsSection, StringKey>> = { appearance: 'settings.browser_only', requests: 'settings.section.requests.description', model: 'settings.section.model.description', server: 'settings.section.server.description' };
const DESCRIPTION_TEST_ID: Readonly<Record<SettingsSection, string>> = { appearance: testId('settings.appearance'), requests: testId('settings.section.requests.description'), model: testId('settings.section.model.description'), server: testId('settings.section.server.description') };

function panel(section: SettingsSection, locale: Locale, appearance: AppearancePreferences, setAppearance: (next: AppearancePreferences) => void): React.ReactNode {
  if (section === 'appearance') return <AppearanceSettings appearance={appearance} setAppearance={setAppearance} />;
  if (section === 'requests') return <GenerationSettings locale={locale} />;
  if (section === 'model') return <ModelSettings locale={locale} />;
  return <ServerSettings locale={locale} />;
}

export function SettingsScreen(props: { appearance: AppearancePreferences; setAppearance: (next: AppearancePreferences) => void; section: SettingsSection; onSectionChange: (section: SettingsSection) => void }): React.JSX.Element {
  const locale = props.appearance.locale;
  return (
    <div className="screen-stack settings-screen">
      <PageHeader title={t(locale, 'settings.title')} titleTestId={testId('settings.title')} description={t(locale, DESCRIPTION[props.section])} descriptionTestId={DESCRIPTION_TEST_ID[props.section]} actions={<HelpTip label={t(locale, 'settings.help.storage')} content={t(locale, 'settings.server.privacy.body')} />} />
      <Tabs label={t(locale, 'settings.sections')} active={props.section} onChange={(id) => props.onSectionChange(id as SettingsSection)} tabs={SETTINGS_SECTIONS.map((section) => ({ id: section, label: t(locale, LABEL[section]), panel: section === props.section ? panel(section, locale, props.appearance, props.setAppearance) : null }))} />
    </div>
  );
}
