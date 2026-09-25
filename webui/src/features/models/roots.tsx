// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { useState } from 'react';
import type { WebUiSnapshot } from '../../api/types';
import { Button, Dialog } from '../../design-system/primitives';
import { t, type Locale } from '../../i18n/catalog';
import { sourceLabel } from './labels';

const EXAMPLE_COMMAND = 'mlxcel-server --webui --models-dir /path/to/models --no-models-autoload';

/** The configured model roots, each with its kind and, when the server could not read it, why. */
export function RootsList({ state, locale }: { state: WebUiSnapshot; locale: Locale }): React.JSX.Element {
  return (
    <ul className="models-roots">
      {state.bootstrap?.roots.map((root, index) => (
        <li key={index}>
          <bdi>{root.display_name}</bdi> · {sourceLabel(locale, root.kind)}
          {root.error ? (
            <p role="alert">
              {t(locale, 'models.library.permission')} {root.error}
            </p>
          ) : null}
        </li>
      ))}
    </ul>
  );
}

/** How to point the server at a local models directory; a copyable example, never an executed action. */
export function RootsHelp({ locale }: { locale: Locale }): React.JSX.Element {
  const [copyState, setCopyState] = useState<'copy' | 'copied' | 'copy_failed'>('copy');
  return (
    <>
      <p>{t(locale, 'models.library.roots_help')}</p>
      <pre className="models-wrap">{EXAMPLE_COMMAND}</pre>
      <Button
        onClick={() => {
          if (!navigator.clipboard) {
            setCopyState('copy_failed');
            return;
          }
          void navigator.clipboard
            .writeText(EXAMPLE_COMMAND)
            .then(() => setCopyState('copied'))
            .catch(() => setCopyState('copy_failed'));
        }}
      >
        {t(locale, `models.library.${copyState}`)}
      </Button>
      <span role="status">{copyState !== 'copy' ? t(locale, `models.library.${copyState}`) : ''}</span>
    </>
  );
}

export function RootsDialog({ state, locale, onClose }: { state: WebUiSnapshot; locale: Locale; onClose: () => void }): React.JSX.Element {
  return (
    <Dialog open title={t(locale, 'models.library.roots')} onClose={onClose} closeLabel={t(locale, 'common.close')} testId="models-roots-dialog">
      <RootsList state={state} locale={locale} />
      <RootsHelp locale={locale} />
    </Dialog>
  );
}
