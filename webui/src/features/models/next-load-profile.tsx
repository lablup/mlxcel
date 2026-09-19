// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import { t, type Locale } from '../../i18n/catalog';

export function NextLoadProfile({ locale }: { locale: Locale }): React.JSX.Element {
  return <p data-testid="models-pending-profile">
    {t(locale, 'models.next_profile.body')}{' '}
    <a href="#settings/model">{t(locale, 'models.next_profile.edit')}</a>
  </p>;
}
