import React, { Suspense, use, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { ModelId, WebUiSnapshot } from './api/types';
import { CommandPalette } from './command-palette';
import { AppShell, globalShortcuts, type RouteId, type ShellLoadedModel } from './design-system/shell';
import { applyAppearance, DEFAULT_APPEARANCE, loadAppearance, saveAppearance, type AppearancePreferences } from './design-system/preferences';
import { getSystemColorScheme, onSystemColorSchemeChange, resolveColorScheme } from './design-system/theme';
import { ProductTheme } from './design-system/common-theme';
// The first bundle imports seam modules directly, not the primitives barrel (see primitives.tsx).
import { IconButton } from './design-system/common-adapters';
import { Dialog } from './design-system/local-components';
import { requestInspector } from './features/models/inspect-request';
import { classifyAuthFailure, connectedDetail, connectionFooterDetails, connectionFooterLabel, lifecycleLabel, loadedModels, modelSelectionFor, ProductConnectionSurface, type AuthFailure } from './provider-surfaces';
import { useWebUi, useWebUiActions } from './state';
import { t, testId } from './i18n/catalog';
import { replaceConversations, requestNewConversation } from './features/chat/session';
import { resetLibraryView } from './features/models/library-view';
import { DEFAULT_SETTINGS_SECTION, settingsSectionFrom, settingsSectionHash, type SettingsSection } from './features/settings/sections';

const routes: RouteId[] = ['models', 'chat', 'activity', 'settings', 'gallery'];

// The first bundle holds the shell and what a fresh load shows: the session-key form (the key
// lives in memory only, so every load starts signed out) and the connection surfaces. Each
// screen behind it is its own chunk, fetched the first time it renders: the Models library once
// signed in, Chat, Activity, Settings and the gallery (the initial JavaScript budget in
// scripts/webui/build_bundle.py). The shell, toolbar and connection state stay in place while a
// chunk loads. Once a chunk has arrived its screen renders synchronously, without suspending.
function splitRoute<P extends object>(load: () => Promise<React.ComponentType<P>>): { Route: (props: P) => React.JSX.Element; preload: () => Promise<React.ComponentType<P>> } {
  type Chunk = Promise<React.ComponentType<P>>;
  let pending: Chunk | null = null;
  // A chunk that failed to arrive is fetched again the next time its route renders. A fetched
  // chunk is marked fulfilled the way React reads a thenable, so use() returns it without suspending.
  const preload = (): Chunk => (pending ??= load().then((component) => { if (pending) Object.assign(pending, { status: 'fulfilled', value: component }); return component; }, (error: unknown) => { pending = null; throw error; }));
  function Route(props: P): React.JSX.Element {
    // use() runs on every render: skipping it once the chunk arrived trips React's conditional-use() error.
    const Component = use(preload());
    return <Component {...props} />;
  }
  return { Route, preload };
}
const models = splitRoute(() => import('./features/models/screen').then((module) => module.ModelsLibrary));
const chat = splitRoute(() => import('./features/chat/chat').then((module) => module.Chat));
const activity = splitRoute(() => import('./features/activity').then((module) => module.ActivityPage));
const settings = splitRoute(() => import('./features/settings/settings-screen').then((module) => module.SettingsScreen));
const gallery = splitRoute(() => import('./gallery').then((module) => module.DesignGallery));
const ModelsLibrary = models.Route;
const Chat = chat.Route;
const ActivityPage = activity.Route;
const SettingsScreen = settings.Route;
const DesignGallery = gallery.Route;

/** Loads every split route; a test awaits it once so later renders do not suspend. */
export function preloadRoutes(): Promise<unknown> {
  return Promise.all([models.preload(), chat.preload(), activity.preload(), settings.preload(), gallery.preload()]);
}

type Overlay = 'command' | 'help' | null;

// The same condition under which ChatScreen renders the real Chat, which consumes requests.
function chatAvailable(snapshot: WebUiSnapshot): boolean {
  return snapshot.auth.status === 'authenticated' && snapshot.connection !== 'schema-mismatch';
}

export interface HashLocation { readonly route: RouteId; readonly section: SettingsSection }

/** `#settings/<section>` opens that Settings tab; any other route is its bare name. Unknown hashes open Models. */
export function routeFromHash(hash: string = window.location.hash): HashLocation {
  const raw = hash.replace(/^#/, '').replace(/^\//, '');
  const [head, ...rest] = raw.split('/');
  if (head === 'settings') return { route: 'settings', section: settingsSectionFrom(rest.join('/') || undefined) };
  return { route: routes.includes(raw as RouteId) ? raw as RouteId : 'models', section: DEFAULT_SETTINGS_SECTION };
}

export function App(): React.JSX.Element {
  const snapshot = useWebUi();
  const actions = useWebUiActions();
  const [hashLocation, setHashLocation] = useState<HashLocation>(() => routeFromHash());
  const route = hashLocation.route;
  const [appearance, setAppearance] = useState<AppearancePreferences>(loadAppearance);
  const [overlay, setOverlay] = useState<Overlay>(null);
  const [authFailure, setAuthFailure] = useState<AuthFailure | null>(null);
  const authAttemptRef = useRef(0);

  useEffect(() => {
    return () => { authAttemptRef.current += 1; };
  }, []);
  // Fetch the split screens once the first paint is done, so moving to one later renders at once
  // instead of waiting for its chunk.
  useEffect(() => {
    const preload = (): void => { void preloadRoutes().catch(() => undefined); };
    if (typeof window.requestIdleCallback === 'function') {
      const handle = window.requestIdleCallback(preload, { timeout: 2000 });
      return () => window.cancelIdleCallback(handle);
    }
    const timer = window.setTimeout(preload, 500);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    const handleHash = (): void => setHashLocation(routeFromHash());
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);
  const [systemScheme, setSystemScheme] = useState(getSystemColorScheme);
  const scheme = resolveColorScheme(appearance.colorScheme, systemScheme);
  // A layout effect, so the appearance attributes are on <html> before the first paint after a
  // change. data-theme and data-astryx-theme are also what the root <Theme> writes.
  useLayoutEffect(() => {
    applyAppearance(document.documentElement, appearance, systemScheme);
    saveAppearance(appearance);
  }, [appearance, systemScheme]);
  useEffect(() => {
    // A `system` preference follows the host scheme live; data-theme is re-resolved, never set to `system`.
    if (appearance.colorScheme !== 'system') return undefined;
    setSystemScheme(getSystemColorScheme());
    return onSystemColorSchemeChange(setSystemScheme);
  }, [appearance.colorScheme]);
  useEffect(() => {
    const applyVisibility = (): void => {
      document.documentElement.dataset.documentHidden = String(document.hidden);
    };
    applyVisibility();
    document.addEventListener('visibilitychange', applyVisibility);
    return () => document.removeEventListener('visibilitychange', applyVisibility);
  }, []);
  useEffect(() => {
    if (snapshot.auth.status === 'authenticated') setAuthFailure(null);
  }, [snapshot.auth.status]);
  useEffect(() => { if (snapshot.auth.status === 'signed-out') replaceConversations([]); }, [snapshot.auth.status]);

  const navigate = (next: RouteId): void => {
    setHashLocation({ route: next, section: DEFAULT_SETTINGS_SECTION });
    window.history.replaceState(null, '', `#${next}`);
  };
  // Replace, not push: Back leaves Settings instead of stepping back through its tabs.
  const selectSettingsSection = (section: SettingsSection): void => {
    setHashLocation({ route: 'settings', section });
    window.history.replaceState(null, '', settingsSectionHash(section));
  };
  const login = (token: string): void => {
    const attempt = authAttemptRef.current + 1;
    authAttemptRef.current = attempt;
    setAuthFailure(null);
    void actions.login(token).catch((error: unknown) => {
      if (authAttemptRef.current !== attempt) return;
      if (error instanceof DOMException && error.name === 'AbortError') return;
      setAuthFailure(classifyAuthFailure(error));
    });
  };
  const logout = (): void => {
    authAttemptRef.current += 1;
    setAuthFailure(null);
    resetLibraryView();
    actions.logout();
  };
  const retry = (): void => {
    setAuthFailure(null);
    void actions.refresh();
  };
  const recoverSchema = (): void => {
    logout();
    window.location.reload();
  };
  // Chip and palette: select the model and open its inspector on Models. A model that
  // left the catalog in the meantime opens Models with no inspector. Never loads. Opening
  // the model that is already selected dispatches nothing: a selection change cancels
  // observation, refetches and clears the runtime history.
  const openModel = (id: ModelId): void => {
    const target = modelSelectionFor(snapshot, id);
    if (target !== snapshot.selectedModelId) actions.selectModel(target);
    // Below 1100 px the inspector is a drawer that opens only on request.
    if (target !== null) requestInspector();
    navigate('models');
    setOverlay(null);
  };
  // Cmd/Ctrl+N and the palette. Signed out, Chat shows the connection surface and nothing
  // would consume a request, so none is left behind.
  const newChat = (): void => {
    if (chatAvailable(snapshot)) requestNewConversation();
    navigate('chat');
    setOverlay(null);
  };
  const loaded = loadedModels(snapshot);
  // Unknown until an authenticated session holds a catalog snapshot (signed out, the window
  // before the first page, and the reset after a server restart): the toolbar must not claim
  // "No model loaded" about a server it has not read.
  const catalogKnown = snapshot.auth.status === 'authenticated' && snapshot.catalogSequence !== null;
  const shellLoaded: ShellLoadedModel[] | null = catalogKnown ? loaded.map((entry) => ({ id: entry.identity.id, name: entry.identity.display_name, state: entry.lifecycle.state, stateLabel: lifecycleLabel(appearance.locale, entry.lifecycle.state) })) : null;
  const body = renderRoute(hashLocation, appearance, setAppearance, selectSettingsSection, { snapshot, authFailure, login, logout, retry, recoverSchema });
  return (
    <ProductTheme family={appearance.themeFamily} scheme={scheme} locale={appearance.locale}>
      <AppShell locale={appearance.locale} route={route} onRouteChange={navigate} onCommand={() => setOverlay('command')} onHelp={() => setOverlay('help')} onNewChat={newChat} onOpenModel={openModel} loadedModels={shellLoaded} connection={{ label: connectionFooterLabel(appearance.locale, snapshot), state: snapshot.connection, details: connectionFooterDetails(appearance.locale, snapshot), summary: snapshot.bootstrap === null ? null : connectedDetail(appearance.locale, snapshot) }} sessionAction={snapshot.auth.tokenPresent ? <IconButton label={t(appearance.locale, 'toolbar.logout')} icon="key" onClick={logout} data-testid={testId('toolbar.logout')} /> : null} inspector={null}>
        <Suspense fallback={null}>{body}</Suspense>
      </AppShell>
      <CommandPalette open={overlay === 'command'} locale={appearance.locale} catalog={snapshot.catalog} loaded={loaded} onClose={() => setOverlay(null)} onNavigate={(next) => { navigate(next); setOverlay(null); }} onNewChat={newChat} onOpenModel={openModel} />
      <Dialog open={overlay === 'help'} title={t(appearance.locale, 'help.title')} onClose={() => setOverlay(null)} testId="help-dialog" closeLabel={t(appearance.locale, 'common.close')}>
        <p data-testid={testId('help.body')}>{t(appearance.locale, 'help.body')}</p>
        <ul className="shortcut-list" data-testid="help-shortcuts">
          {globalShortcuts.map((shortcut) => (
            <li key={shortcut.id} data-testid={`help-shortcut-${shortcut.id}`}>
              <span className="shortcut-keys">{shortcut.keys.map((key, index) => <React.Fragment key={key}>{index > 0 ? ` ${shortcut.separator} ` : null}<kbd>{key}</kbd></React.Fragment>)}</span>
              <span>{t(appearance.locale, shortcut.key)}</span>
            </li>
          ))}
        </ul>
      </Dialog>
    </ProductTheme>
  );
}

interface ProviderRouteContext {
  readonly snapshot: WebUiSnapshot;
  readonly authFailure: AuthFailure | null;
  readonly login: (token: string) => void;
  readonly logout: () => void;
  readonly retry: () => void;
  readonly recoverSchema: () => void;
}

function renderRoute(at: HashLocation, appearance: AppearancePreferences, setAppearance: (next: AppearancePreferences) => void, selectSettingsSection: (section: SettingsSection) => void, context: ProviderRouteContext): React.ReactNode {
  const { route } = at;
  if (route === 'models') return <ModelsScreen locale={appearance.locale} context={context} />;
  if (route === 'chat') return <ChatScreen locale={appearance.locale} context={context} />;
  if (route === 'activity') return <ActivityScreen locale={appearance.locale} context={context} />;
  if (route === 'settings') return <SettingsScreen appearance={appearance} setAppearance={setAppearance} section={at.section} onSectionChange={selectSettingsSection} />;
  return <DesignGallery locale={appearance.locale} />;
}

function ModelsScreen(props: { locale: AppearancePreferences['locale']; context: ProviderRouteContext }): React.JSX.Element {
  if (props.context.snapshot.auth.status === 'authenticated' && props.context.snapshot.connection !== 'schema-mismatch') return <ModelsLibrary locale={props.locale} />;
  return <ProductConnectionSurface locale={props.locale} title={t(props.locale, 'models.title')} titleTestId={testId('models.title')} snapshot={props.context.snapshot} authFailure={props.context.authFailure} onLogin={props.context.login} onLogout={props.context.logout} onRetry={props.context.retry} onRecoverSchema={props.context.recoverSchema} />;
}

function ChatScreen(props: { locale: AppearancePreferences['locale']; context: ProviderRouteContext }): React.JSX.Element {
  if (props.context.snapshot.auth.status === 'authenticated' && props.context.snapshot.connection !== 'schema-mismatch') return <Chat locale={props.locale} />;
  return <ProductConnectionSurface locale={props.locale} title={t(props.locale, 'chat.title')} titleTestId={testId('chat.title')} snapshot={props.context.snapshot} authFailure={props.context.authFailure} onLogin={props.context.login} onLogout={props.context.logout} onRetry={props.context.retry} onRecoverSchema={props.context.recoverSchema} />;
}

function ActivityScreen(props: { locale: AppearancePreferences['locale']; context: ProviderRouteContext }): React.JSX.Element {
  if (props.context.snapshot.auth.status === 'authenticated' && !['schema-mismatch', 'forbidden', 'unauthorized'].includes(props.context.snapshot.connection)) return <ActivityPage locale={props.locale} />;
  return <ProductConnectionSurface locale={props.locale} title={t(props.locale, 'activity.title')} titleTestId={testId('activity.title')} snapshot={props.context.snapshot} authFailure={props.context.authFailure} onLogin={props.context.login} onLogout={props.context.logout} onRetry={props.context.retry} onRecoverSchema={props.context.recoverSchema} />;
}

export { DEFAULT_APPEARANCE };
