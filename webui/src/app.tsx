import React, { useEffect, useRef, useState } from 'react';
import type { ModelId, WebUiSnapshot } from './api/types';
import { CommandPalette } from './command-palette';
import { AppShell, globalShortcuts, type RouteId, type ShellLoadedModel } from './design-system/shell';
import { applyAppearance, DEFAULT_APPEARANCE, loadAppearance, saveAppearance, type AppearancePreferences } from './design-system/preferences';
import { onSystemColorSchemeChange } from './design-system/theme';
import { Dialog, IconButton } from './design-system/primitives';
import { ActivityPage } from './features/activity';
import { ModelsLibrary } from './features/models/screen';
import { DesignGallery } from './gallery';
import { classifyAuthFailure, connectionFooterDetails, connectionFooterLabel, lifecycleLabel, loadedModels, modelSelectionFor, ProductConnectionSurface, type AuthFailure } from './provider-surfaces';
import { useWebUi, useWebUiActions } from './state';
import { t, testId } from './i18n/catalog';
import { Chat } from './features/chat/chat';
import { replaceConversations, requestNewConversation } from './features/chat/session';
import { SettingsScreen } from './features/settings/settings-screen';
import { DEFAULT_SETTINGS_SECTION, settingsSectionFrom, settingsSectionHash, type SettingsSection } from './features/settings/sections';

const routes: RouteId[] = ['models', 'chat', 'activity', 'settings', 'gallery'];
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

  useEffect(() => {
    const handleHash = (): void => setHashLocation(routeFromHash());
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);
  useEffect(() => {
    applyAppearance(document.documentElement, appearance);
    saveAppearance(appearance);
    // A `system` preference follows the host scheme live; data-theme is re-resolved, never set to `system`.
    if (appearance.colorScheme !== 'system') return undefined;
    return onSystemColorSchemeChange((scheme) => applyAppearance(document.documentElement, appearance, scheme));
  }, [appearance]);
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
    <>
      <AppShell locale={appearance.locale} route={route} onRouteChange={navigate} onCommand={() => setOverlay('command')} onHelp={() => setOverlay('help')} onNewChat={newChat} onOpenModel={openModel} loadedModels={shellLoaded} connection={{ label: connectionFooterLabel(appearance.locale, snapshot), state: snapshot.connection, details: connectionFooterDetails(appearance.locale, snapshot) }} sessionAction={snapshot.auth.tokenPresent ? <IconButton label={t(appearance.locale, 'toolbar.logout')} icon="key" onClick={logout} data-testid={testId('toolbar.logout')} /> : null} inspector={null}>
        {body}
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
    </>
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
