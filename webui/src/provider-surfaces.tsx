import React from 'react';
import { WebUiHttpError } from './api/client';
import type { CatalogEntry, ConnectionPhase, WebUiSnapshot } from './api/types';
import { ValidationError } from './api/validation';
import { Button, ErrorBanner, LoginView, SchemaMismatchView } from './design-system/primitives';
import type { Locale, StringKey } from './i18n/catalog';
import { t, testId } from './i18n/catalog';

export type AuthFailure = 'wrong-key' | 'offline' | 'forbidden' | 'schema' | 'generic';

export function classifyAuthFailure(error: unknown): AuthFailure {
  if (error instanceof WebUiHttpError) {
    if (error.status === 401) return 'wrong-key';
    if (error.status === 403) return 'forbidden';
    return 'generic';
  }
  if (error instanceof ValidationError || (error instanceof Error && error.name === 'ValidationError')) return 'schema';
  if (error instanceof TypeError || (error instanceof Error && /fetch|network|offline|connection/i.test(error.message))) return 'offline';
  return 'generic';
}

export function selectedModelLabel(locale: Locale, snapshot: WebUiSnapshot): string {
  const selected = selectedCatalogEntry(snapshot);
  if (selected === null) return t(locale, 'model.selected.none');
  return `${selected.identity.display_name} · ${lifecycleLabel(locale, selected.lifecycle.state)}`;
}

export function connectionFooterLabel(locale: Locale, snapshot: WebUiSnapshot): string {
  if (snapshot.bootstrap === null) return t(locale, 'connection.ready');
  return t(locale, 'connection.footer.connected', {
    mode: snapshot.bootstrap.server.mode,
    version: snapshot.bootstrap.server.build.version,
    sequence: snapshotSequence(locale, snapshot),
    status: connectionPhaseLabel(locale, snapshot.connection),
  });
}

export function connectionPhaseLabel(locale: Locale, phase: ConnectionPhase): string {
  const key: Record<ConnectionPhase, StringKey> = {
    idle: 'connection.status.idle',
    bootstrapping: 'connection.status.bootstrapping',
    ready: 'connection.status.ready',
    streaming: 'connection.status.streaming',
    polling: 'connection.status.polling',
    offline: 'connection.status.offline',
    stale: 'connection.status.stale',
    unauthorized: 'connection.status.unauthorized',
    forbidden: 'connection.status.forbidden',
    'schema-mismatch': 'connection.status.schema_mismatch',
    error: 'connection.status.error',
  };
  return t(locale, key[phase]);
}

export interface ProductConnectionSurfaceProps {
  readonly locale: Locale;
  readonly eyebrow: string;
  readonly title: string;
  readonly titleTestId: string;
  readonly snapshot: WebUiSnapshot;
  readonly authFailure: AuthFailure | null;
  readonly onLogin: (token: string) => void;
  readonly onLogout: () => void;
  readonly onRetry: () => void;
  readonly onRecoverSchema: () => void;
}

export function ProductConnectionSurface(props: ProductConnectionSurfaceProps): React.JSX.Element {
  if (props.snapshot.auth.status !== 'authenticated') {
    return <SignedOutSurface {...props} />;
  }
  if (props.snapshot.connection === 'schema-mismatch') {
    return <SchemaSurface {...props} />;
  }
  return <AuthenticatedSurface {...props} />;
}

function SignedOutSurface(props: ProductConnectionSurfaceProps): React.JSX.Element {
  if (props.authFailure === 'schema') return <SchemaSurface {...props} />;
  return (
    <div className="screen-stack">
      <section className="screen-heading connection-prompt">
        <p className="eyebrow">{props.eyebrow}</p>
        <h1 data-testid={props.titleTestId}>{props.title}</h1>
        <p data-testid={testId('connection.prompt.body')}>{t(props.locale, 'connection.prompt.body')}</p>
      </section>
      <LoginView
        title={t(props.locale, 'state.unauthorized.title')}
        body={t(props.locale, 'state.unauthorized.body')}
        tokenLabel={t(props.locale, 'login.token.label')}
        tokenHelp={t(props.locale, 'login.token.help')}
        submitLabel={t(props.locale, 'login.submit')}
        logoutLabel={t(props.locale, 'login.logout')}
        error={props.authFailure === null ? undefined : loginErrorMessage(props.locale, props.authFailure)}
        busy={props.snapshot.auth.status === 'authenticating'}
        onSubmit={props.onLogin}
        onLogout={props.onLogout}
        testId={testId('auth.login')}
      />
    </div>
  );
}

function AuthenticatedSurface(props: ProductConnectionSurfaceProps): React.JSX.Element {
  const retryable = props.snapshot.connection === 'offline' || props.snapshot.connection === 'stale' || props.snapshot.connection === 'error' || props.snapshot.connection === 'unauthorized' || props.snapshot.connection === 'forbidden';
  return (
    <div className="screen-stack">
      <section className="screen-heading connection-prompt">
        <p className="eyebrow">{props.eyebrow}</p>
        <h1 data-testid={props.titleTestId}>{props.title}</h1>
        <p data-testid={testId('connection.authenticated.body')}>{t(props.locale, 'connection.authenticated.body')}</p>
      </section>
      <ErrorBanner tone="info" title={t(props.locale, 'connection.authenticated.title')} body={connectedDetail(props.locale, props.snapshot)} testId={testId('connection.authenticated.detail')} />
      {retryable ? <ErrorBanner title={t(props.locale, 'connection.error.title')} body={connectionErrorBody(props.locale, props.snapshot.connection)} action={<Button onClick={props.onRetry}>{t(props.locale, 'common.retry')}</Button>} testId={testId('connection.error.title')} /> : null}
    </div>
  );
}

function SchemaSurface(props: ProductConnectionSurfaceProps): React.JSX.Element {
  return (
    <div className="screen-stack">
      <section className="screen-heading connection-prompt">
        <p className="eyebrow">{props.eyebrow}</p>
        <h1 data-testid={props.titleTestId}>{props.title}</h1>
      </section>
      <SchemaMismatchView title={t(props.locale, 'state.schema_mismatch.title')} body={t(props.locale, 'state.schema_mismatch.body')} actionLabel={t(props.locale, 'common.reload')} onRecover={props.onRecoverSchema} />
    </div>
  );
}

function connectedDetail(locale: Locale, snapshot: WebUiSnapshot): string {
  const bootstrap = snapshot.bootstrap;
  if (bootstrap === null) return t(locale, 'connection.prompt.detail');
  return t(locale, 'connection.authenticated.detail', {
    mode: bootstrap.server.mode,
    version: bootstrap.server.build.version,
    status: connectionPhaseLabel(locale, snapshot.connection),
    count: String(snapshot.catalog.length),
    operations: String(snapshot.operations.size),
    sequence: snapshotSequence(locale, snapshot),
  });
}

function connectionErrorBody(locale: Locale, phase: ConnectionPhase): string {
  if (phase === 'offline') return t(locale, 'state.offline.body');
  if (phase === 'stale') return t(locale, 'connection.error.stale');
  if (phase === 'forbidden') return t(locale, 'connection.error.forbidden');
  if (phase === 'unauthorized') return t(locale, 'connection.error.unauthorized');
  return t(locale, 'connection.error.generic');
}

function loginErrorMessage(locale: Locale, failure: AuthFailure): string {
  if (failure === 'wrong-key') return t(locale, 'login.error.wrong_key');
  if (failure === 'offline') return t(locale, 'login.error.offline');
  if (failure === 'forbidden') return t(locale, 'login.error.forbidden');
  if (failure === 'schema') return t(locale, 'login.error.schema');
  return t(locale, 'login.error.generic');
}

function selectedCatalogEntry(snapshot: WebUiSnapshot): CatalogEntry | null {
  if (snapshot.selectedModelId === null) return null;
  return snapshot.catalog.find((entry) => entry.identity.id === snapshot.selectedModelId) ?? null;
}

function lifecycleLabel(locale: Locale, state: CatalogEntry['lifecycle']['state']): string {
  const keys: Record<CatalogEntry['lifecycle']['state'], StringKey> = {
    unloaded: 'models.status.unloaded',
    loading: 'models.status.loading',
    ready: 'models.status.ready',
    draining: 'models.status.draining',
    unloading: 'models.status.unloading',
    failed: 'models.status.failed',
  };
  return t(locale, keys[state]);
}

function snapshotSequence(locale: Locale, snapshot: WebUiSnapshot): string {
  return String(snapshot.lastSequence ?? snapshot.catalogSequence ?? snapshot.resourceFences.operationsSnapshot ?? t(locale, 'connection.snapshot.pending'));
}
