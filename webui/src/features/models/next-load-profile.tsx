// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React from 'react';
import type { Locale } from '../../i18n/catalog';

export function NextLoadProfile({ locale }: { locale: Locale }): React.JSX.Element {
  return <p data-testid="models-pending-profile">
    {locale === 'ko'
      ? '대기 중인 브라우저 프로필이며 아직 적용되지 않았습니다. 명시적 CLI/환경 설정이 우선합니다. 실제 값은 로드 후 확인하세요. '
      : 'Pending browser profile; not applied yet. Explicit CLI/environment settings take precedence. Read effective values after loading. '}
    <a href="#settings">{locale === 'ko' ? '프로필 편집' : 'Edit profile'}</a>
  </p>;
}
