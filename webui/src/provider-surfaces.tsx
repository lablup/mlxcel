import React from 'react';
import { WebUiHttpError } from './api/client';
import type { CatalogEntry, ConnectionPhase, ModelId, ModelLifecycleState, WebUiSnapshot } from './api/types';
import { ValidationError } from './api/validation';
import { Button, ErrorBanner, LoginView, PageHeader, SchemaMismatchView } from './design-system/primitives';
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

// Lifecycle states that hold a worker, and so memory, on the server. `failed` and
// `unloaded` never count as loaded.
const LOADED_STATES: ReadonlySet<ModelLifecycleState> = new Set<ModelLifecycleState>(['ready', 'loading', 'draining', 'unloading']);

export function isLoadedState(state: ModelLifecycleState): boolean {
  return LOADED_STATES.has(state);
}

/** Ready before the other loaded states, then loaded before the rest, then by display name and id. */
export function compareByLoadedThenName(left: CatalogEntry, right: CatalogEntry): number {
  const rank = (entry: CatalogEntry): number => (entry.lifecycle.state === 'ready' ? 0 : isLoadedState(entry.lifecycle.state) ? 1 : 2);
  return rank(left) - rank(right)
    || left.identity.display_name.localeCompare(right.identity.display_name)
    || left.identity.id.localeCompare(right.identity.id);
}

/** The models the server holds right now (not the browser's selection), in toolbar order. */
export function loadedModels(snapshot: WebUiSnapshot): ReadonlyArray<CatalogEntry> {
  return snapshot.catalog.filter((entry) => isLoadedState(entry.lifecycle.state)).sort(compareByLoadedThenName);
}

/**
 * What opening a model from a toolbar chip or the palette selects: the model while it is still in
 * the catalog, otherwise nothing, so Models opens with no inspector. The caller dispatches only
 * when this differs from the current selection, because a selection change restarts observation
 * and clears the runtime history.
 */
export function modelSelectionFor(snapshot: WebUiSnapshot, id: ModelId): ModelId | null {
  return snapshot.catalog.some((entry) => entry.identity.id === id) ? id : null;
}

export function connectionFooterLabel(locale: Locale, snapshot: WebUiSnapshot): string {
  if (snapshot.bootstrap === null) return t(locale, 'connection.ready');
  return t(locale, 'connection.footer.connected', {
    mode: snapshot.bootstrap.server.mode,
    version: snapshot.bootstrap.server.build.version,
    status: connectionPhaseLabel(locale, snapshot.connection),
  });
}

export interface ConnectionFooterDetails {
  readonly instance: string;
  readonly sequence: string;
}

/** Internal identifiers for the footer's details disclosure; null before bootstrap. */
export function connectionFooterDetails(locale: Locale, snapshot: WebUiSnapshot): ConnectionFooterDetails | null {
  if (snapshot.bootstrap === null) return null;
  const unknown = t(locale, 'format.unknown');
  const instance = snapshot.serverInstanceId ?? snapshot.bootstrap.server.server_instance_id;
  const sequence = snapshot.lastSequence ?? snapshot.catalogSequence ?? snapshot.resourceFences.operationsSnapshot;
  return { instance: instance || unknown, sequence: sequence === null ? unknown : String(sequence) };
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
      <PageHeader className="connection-prompt" title={props.title} titleTestId={props.titleTestId} description={t(props.locale, 'connection.prompt.body')} descriptionTestId={testId('connection.prompt.body')} />
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
      <PageHeader className="connection-prompt" title={props.title} titleTestId={props.titleTestId} description={t(props.locale, 'connection.authenticated.body')} descriptionTestId={testId('connection.authenticated.body')} />
      {/* No test id: the sidebar's Connection details carries the one `connection-authenticated-detail`. */}
      <ErrorBanner tone="info" title={t(props.locale, 'connection.authenticated.title')} body={connectedDetail(props.locale, props.snapshot)} />
      {retryable ? <ErrorBanner title={t(props.locale, 'connection.error.title')} body={connectionErrorBody(props.locale, props.snapshot.connection)} action={<Button onClick={props.onRetry}>{t(props.locale, 'common.retry')}</Button>} testId={testId('connection.error.title')} /> : null}
    </div>
  );
}

function SchemaSurface(props: ProductConnectionSurfaceProps): React.JSX.Element {
  return (
    <div className="screen-stack">
      <PageHeader className="connection-prompt" title={props.title} titleTestId={props.titleTestId} />
      <SchemaMismatchView title={t(props.locale, 'state.schema_mismatch.title')} body={t(props.locale, 'state.schema_mismatch.body')} actionLabel={t(props.locale, 'common.reload')} onRecover={props.onRecoverSchema} />
    </div>
  );
}

export function connectedDetail(locale: Locale, snapshot: WebUiSnapshot): string {
  const bootstrap = snapshot.bootstrap;
  if (bootstrap === null) return t(locale, 'connection.prompt.detail');
  return t(locale, 'connection.authenticated.detail', {
    mode: bootstrap.server.mode,
    version: bootstrap.server.build.version,
    status: connectionPhaseLabel(locale, snapshot.connection),
    count: snapshot.catalogSequence === null ? t(locale, 'connection.snapshot.pending') : String(snapshot.catalog.length),
    operations: snapshot.resourceFences.operationsSnapshot === null ? t(locale, 'connection.snapshot.pending') : String(snapshot.operations.size),
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

export function lifecycleLabel(locale: Locale, state: CatalogEntry['lifecycle']['state']): string {
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
