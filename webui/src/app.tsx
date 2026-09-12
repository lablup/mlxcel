import React, { useEffect, useRef, useState } from 'react';
import type { WebUiSnapshot } from './api/types';
import { AppShell, type RouteId } from './design-system/shell';
import { applyAppearance, DEFAULT_APPEARANCE, loadAppearance, saveAppearance, type AppearancePreferences, type ContrastPreference } from './design-system/preferences';
import { Button, Dialog, ErrorBanner, Field, IconButton, Select } from './design-system/primitives';
import { DesignGallery } from './gallery';
import { classifyAuthFailure, connectionFooterLabel, ProductConnectionSurface, selectedModelLabel, type AuthFailure } from './provider-surfaces';
import { useWebUi, useWebUiActions } from './state';
import { t, testId } from './i18n/catalog';

const routes: RouteId[] = ['models', 'chat', 'activity', 'settings', 'gallery'];
const paletteRoutes: RouteId[] = ['models', 'chat', 'activity', 'settings'];
type Overlay = 'command' | 'help' | null;

function routeFromHash(): RouteId {
  const raw = window.location.hash.slice(1).replace(/^\//, '') as RouteId;
  return routes.includes(raw) ? raw : 'models';
}

export function App(): React.JSX.Element {
  const snapshot = useWebUi();
  const actions = useWebUiActions();
  const [route, setRoute] = useState<RouteId>(routeFromHash);
  const [appearance, setAppearance] = useState<AppearancePreferences>(loadAppearance);
  const [overlay, setOverlay] = useState<Overlay>(null);
  const [authFailure, setAuthFailure] = useState<AuthFailure | null>(null);
  const authAttemptRef = useRef(0);

  useEffect(() => {
    return () => { authAttemptRef.current += 1; };
  }, []);

  useEffect(() => {
    const handleHash = (): void => setRoute(routeFromHash());
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);
  useEffect(() => {
    applyAppearance(document.documentElement, appearance);
    saveAppearance(appearance);
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

  const navigate = (next: RouteId): void => {
    setRoute(next);
    window.history.replaceState(null, '', `#${next}`);
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
  const body = renderRoute(route, appearance, setAppearance, { snapshot, authFailure, login, logout, retry, recoverSchema });
  return (
    <>
      <AppShell locale={appearance.locale} route={route} onRouteChange={navigate} onCommand={() => setOverlay('command')} onHelp={() => setOverlay('help')} selectedModel={selectedModelLabel(appearance.locale, snapshot)} connectionLabel={connectionFooterLabel(appearance.locale, snapshot)} connectionState={snapshot.connection} sessionAction={snapshot.auth.tokenPresent ? <IconButton label={t(appearance.locale, 'toolbar.logout')} icon="key" onClick={logout} data-testid={testId('toolbar.logout')} /> : null} inspector={null}>
        {body}
      </AppShell>
      <CommandPalette open={overlay === 'command'} locale={appearance.locale} onClose={() => setOverlay(null)} onNavigate={(next) => { navigate(next); setOverlay(null); }} />
      <Dialog open={overlay === 'help'} title={t(appearance.locale, 'help.title')} onClose={() => setOverlay(null)} testId="help-dialog" closeLabel={t(appearance.locale, 'common.close')}>
        <p data-testid={testId('help.body')}>{t(appearance.locale, 'help.body')}</p>
        <ul className="shortcut-list"><li><kbd>⌘/Ctrl</kbd> + <kbd>K</kbd> {t(appearance.locale, 'command.search')}</li><li><kbd>Esc</kbd> {t(appearance.locale, 'common.cancel')}</li><li><kbd>[</kbd> / <kbd>]</kbd> {t(appearance.locale, 'nav.models')}</li></ul>
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

function renderRoute(route: RouteId, appearance: AppearancePreferences, setAppearance: (next: AppearancePreferences) => void, context: ProviderRouteContext): React.ReactNode {
  if (route === 'models') return <ModelsScreen locale={appearance.locale} context={context} />;
  if (route === 'chat') return <ChatScreen locale={appearance.locale} context={context} />;
  if (route === 'activity') return <ActivityScreen locale={appearance.locale} context={context} />;
  if (route === 'settings') return <SettingsScreen appearance={appearance} setAppearance={setAppearance} />;
  return <DesignGallery locale={appearance.locale} />;
}

function ModelsScreen(props: { locale: AppearancePreferences['locale']; context: ProviderRouteContext }): React.JSX.Element {
  return <ProductConnectionSurface locale={props.locale} eyebrow={t(props.locale, 'routes.models.eyebrow')} title={t(props.locale, 'models.title')} titleTestId={testId('models.title')} snapshot={props.context.snapshot} authFailure={props.context.authFailure} onLogin={props.context.login} onLogout={props.context.logout} onRetry={props.context.retry} onRecoverSchema={props.context.recoverSchema} />;
}

function ChatScreen(props: { locale: AppearancePreferences['locale']; context: ProviderRouteContext }): React.JSX.Element {
  return <ProductConnectionSurface locale={props.locale} eyebrow={t(props.locale, 'routes.chat.eyebrow')} title={t(props.locale, 'chat.title')} titleTestId={testId('chat.title')} snapshot={props.context.snapshot} authFailure={props.context.authFailure} onLogin={props.context.login} onLogout={props.context.logout} onRetry={props.context.retry} onRecoverSchema={props.context.recoverSchema} />;
}

function ActivityScreen(props: { locale: AppearancePreferences['locale']; context: ProviderRouteContext }): React.JSX.Element {
  return <ProductConnectionSurface locale={props.locale} eyebrow={t(props.locale, 'routes.activity.eyebrow')} title={t(props.locale, 'activity.title')} titleTestId={testId('activity.title')} snapshot={props.context.snapshot} authFailure={props.context.authFailure} onLogin={props.context.login} onLogout={props.context.logout} onRetry={props.context.retry} onRecoverSchema={props.context.recoverSchema} />;
}

function SettingsScreen(props: { appearance: AppearancePreferences; setAppearance: (next: AppearancePreferences) => void }): React.JSX.Element {
  const set = (patch: Partial<AppearancePreferences>): void => props.setAppearance({ ...props.appearance, ...patch });
  return (
    <div className="screen-stack"><section className="screen-heading"><p className="eyebrow">{t(props.appearance.locale, 'routes.settings.eyebrow')}</p><h1 data-testid={testId('settings.title')}>{t(props.appearance.locale, 'settings.title')}</h1><p data-testid={testId('settings.appearance')}>{t(props.appearance.locale, 'settings.browser_only')}</p></section><div className="settings-grid"><Select label={t(props.appearance.locale, 'settings.theme')} value={props.appearance.theme} onChange={(value) => set({ theme: value as AppearancePreferences['theme'] })} options={[{ value: 'system', label: t(props.appearance.locale, 'settings.theme.system') }, { value: 'light', label: t(props.appearance.locale, 'settings.theme.light') }, { value: 'dark', label: t(props.appearance.locale, 'settings.theme.dark') }]} testId={testId('settings.theme')} /><Select label={t(props.appearance.locale, 'settings.material')} value={props.appearance.material} onChange={(value) => set({ material: value as AppearancePreferences['material'] })} options={[{ value: 'glass', label: t(props.appearance.locale, 'settings.material.glass') }, { value: 'tinted', label: t(props.appearance.locale, 'settings.material.tinted') }, { value: 'opaque', label: t(props.appearance.locale, 'settings.material.opaque') }]} testId={testId('settings.material')} /><Select label={t(props.appearance.locale, 'settings.locale')} value={props.appearance.locale} onChange={(value) => set({ locale: value as AppearancePreferences['locale'] })} options={[{ value: 'en', label: t(props.appearance.locale, 'settings.locale.en') }, { value: 'ko', label: t(props.appearance.locale, 'settings.locale.ko') }]} testId={testId('settings.locale')} /><Select label={t(props.appearance.locale, 'settings.high_contrast')} value={props.appearance.highContrast} onChange={(value) => set({ highContrast: value as ContrastPreference })} options={[{ value: 'system', label: t(props.appearance.locale, 'settings.high_contrast.system') }, { value: 'on', label: t(props.appearance.locale, 'settings.high_contrast.on') }, { value: 'off', label: t(props.appearance.locale, 'settings.high_contrast.off') }]} testId={testId('settings.high_contrast')} /><label className="ds-field"><span>{t(props.appearance.locale, 'settings.glass_intensity')}: {props.appearance.glassIntensity}</span><input type="range" min="0" max="100" value={props.appearance.glassIntensity} onChange={(event) => set({ glassIntensity: Number(event.currentTarget.value) })} data-testid={testId('settings.glass_intensity')} /></label><Toggle label={t(props.appearance.locale, 'settings.reduce_motion')} checked={props.appearance.reduceMotion} onChange={(checked) => set({ reduceMotion: checked })} testId={testId('settings.reduce_motion')} /><Toggle label={t(props.appearance.locale, 'settings.reduce_transparency')} checked={props.appearance.reduceTransparency} onChange={(checked) => set({ reduceTransparency: checked })} testId={testId('settings.reduce_transparency')} /></div><ErrorBanner tone="info" title={t(props.appearance.locale, 'settings.browser_only')} body={t(props.appearance.locale, 'settings.browser_only.body')} testId={testId('settings.browser_only')} /></div>
  );
}

function Toggle(props: { label: string; checked: boolean; onChange: (checked: boolean) => void; testId: string }): React.JSX.Element {
  return <label className="toggle"><input type="checkbox" checked={props.checked} onChange={(event) => props.onChange(event.currentTarget.checked)} data-testid={props.testId} /><span>{props.label}</span></label>;
}

function CommandPalette(props: { open: boolean; locale: AppearancePreferences['locale']; onClose: () => void; onNavigate: (route: RouteId) => void }): React.JSX.Element {
  const [query, setQuery] = useState('');
  const searchable = props.locale && window.location.hash.includes('gallery') ? routes : paletteRoutes;
  const items = searchable.filter((route) => route.includes(query.toLowerCase()));
  return (
    <Dialog open={props.open} title={t(props.locale, 'command.title')} onClose={props.onClose} testId="command-dialog" closeLabel={t(props.locale, 'common.close')}>
      <Field label={t(props.locale, 'command.search')} value={query} onChange={setQuery} testId={testId('command.search')} />
      {items.length === 0 ? <p data-testid={testId('command.no_results')}>{t(props.locale, 'command.no_results')}</p> : <div className="command-list">{items.map((route) => <Button key={route} onClick={() => props.onNavigate(route)}>{t(props.locale, route === 'models' ? 'nav.models' : route === 'chat' ? 'nav.chat' : route === 'activity' ? 'nav.activity' : route === 'settings' ? 'nav.settings' : 'nav.gallery')}</Button>)}</div>}
    </Dialog>
  );
}

export { DEFAULT_APPEARANCE };
