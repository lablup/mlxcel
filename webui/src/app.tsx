import React, { useEffect, useMemo, useState } from 'react';
import { AppShell, type RouteId } from './design-system/shell';
import { applyAppearance, DEFAULT_APPEARANCE, loadAppearance, saveAppearance, type AppearancePreferences } from './design-system/preferences';
import { Button, Dialog, EmptyState, ErrorBanner, Field, Inspector, Select, StatusBadge } from './design-system/primitives';
import { DesignGallery } from './gallery';
import { t, testId } from './i18n/catalog';

const routes: RouteId[] = ['models', 'chat', 'activity', 'settings', 'gallery'];


function routeFromHash(): RouteId {
  const raw = window.location.hash.slice(1).replace(/^\//, '') as RouteId;
  return routes.includes(raw) ? raw : 'models';
}

export function App(): React.JSX.Element {
  const [route, setRoute] = useState<RouteId>(routeFromHash);
  const [appearance, setAppearance] = useState<AppearancePreferences>(loadAppearance);
  const [commandOpen, setCommandOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  useEffect(() => {
    const handleHash = (): void => setRoute(routeFromHash());
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);
  useEffect(() => {
    applyAppearance(document.documentElement, appearance);
    saveAppearance(appearance);
  }, [appearance]);
  const navigate = (next: RouteId): void => {
    setRoute(next);
    window.history.replaceState(null, '', `#${next}`);
  };
  const body = useMemo(() => renderRoute(route, appearance, setAppearance), [route, appearance]);
  return (
    <>
      <AppShell locale={appearance.locale} route={route} onRouteChange={navigate} onCommand={() => setCommandOpen(true)} onHelp={() => setHelpOpen(true)} selectedModel={t(appearance.locale, 'model.selected.none')} inspector={<RouteInspector route={route} locale={appearance.locale} />}>
        {body}
      </AppShell>
      <CommandPalette open={commandOpen} locale={appearance.locale} onClose={() => setCommandOpen(false)} onNavigate={(next) => { navigate(next); setCommandOpen(false); }} />
      <Dialog open={helpOpen} title={t(appearance.locale, 'help.title')} onClose={() => setHelpOpen(false)} testId="help-dialog" closeLabel={t(appearance.locale, 'common.close')}>
        <p data-testid={testId('help.body')}>{t(appearance.locale, 'help.body')}</p>
        <ul className="shortcut-list"><li><kbd>⌘/Ctrl</kbd> + <kbd>K</kbd> {t(appearance.locale, 'command.search')}</li><li><kbd>Esc</kbd> {t(appearance.locale, 'common.cancel')}</li><li><kbd>[</kbd> / <kbd>]</kbd> {t(appearance.locale, 'nav.models')}</li></ul>
      </Dialog>
    </>
  );
}

function renderRoute(route: RouteId, appearance: AppearancePreferences, setAppearance: (next: AppearancePreferences) => void): React.ReactNode {
  if (route === 'models') return <ModelsScreen locale={appearance.locale} />;
  if (route === 'chat') return <ChatScreen locale={appearance.locale} />;
  if (route === 'activity') return <ActivityScreen locale={appearance.locale} />;
  if (route === 'settings') return <SettingsScreen appearance={appearance} setAppearance={setAppearance} />;
  return <DesignGallery locale={appearance.locale} />;
}

function ModelsScreen(props: { locale: AppearancePreferences['locale'] }): React.JSX.Element {
  return (
    <div className="screen-stack">
      <section className="screen-hero"><p className="eyebrow">{t(props.locale, 'routes.models.eyebrow')}</p><h1 data-testid={testId('models.title')}>{t(props.locale, 'models.title')}</h1><p>{t(props.locale, 'adapters.pending.body')}</p></section>
      <EmptyState title={t(props.locale, 'models.empty.title')} body={t(props.locale, 'models.empty.body')} action={<Button disabled title={t(props.locale, 'common.unavailable')}>{t(props.locale, 'common.add_model')}</Button>} />
      <ErrorBanner tone="info" title={t(props.locale, 'adapters.pending.title')} body={t(props.locale, 'adapters.pending.body')} />
    </div>
  );
}

function ChatScreen(props: { locale: AppearancePreferences['locale'] }): React.JSX.Element {
  const [message, setMessage] = useState('');
  return (
    <div className="screen-stack chat-screen"><section className="screen-hero"><p className="eyebrow">{t(props.locale, 'routes.chat.eyebrow')}</p><h1 data-testid={testId('chat.title')}>{t(props.locale, 'chat.title')}</h1><p>{t(props.locale, 'chat.pending.body')}</p></section><ErrorBanner tone="info" title={t(props.locale, 'adapters.pending.title')} body={t(props.locale, 'chat.pending.body')} /><label className="composer"><span className="sr-only">{t(props.locale, 'chat.placeholder')}</span><textarea value={message} placeholder={t(props.locale, 'chat.placeholder')} onChange={(event) => setMessage(event.currentTarget.value)} onCompositionStart={() => undefined} disabled /><Button tone="primary" disabled title={t(props.locale, 'common.unavailable')}>{t(props.locale, 'common.send')}</Button></label></div>
  );
}

function ActivityScreen(props: { locale: AppearancePreferences['locale'] }): React.JSX.Element {
  return <div className="screen-stack"><section className="screen-hero"><p className="eyebrow">{t(props.locale, 'routes.activity.eyebrow')}</p><h1 data-testid={testId('activity.title')}>{t(props.locale, 'activity.title')}</h1><p>{t(props.locale, 'activity.empty.body')}</p></section><EmptyState title={t(props.locale, 'activity.empty.title')} body={t(props.locale, 'activity.empty.body')} /><ErrorBanner tone="info" title={t(props.locale, 'adapters.pending.title')} body={t(props.locale, 'adapters.pending.body')} /></div>;
}

function SettingsScreen(props: { appearance: AppearancePreferences; setAppearance: (next: AppearancePreferences) => void }): React.JSX.Element {
  const set = (patch: Partial<AppearancePreferences>): void => props.setAppearance({ ...props.appearance, ...patch });
  return (
    <div className="screen-stack"><section className="screen-hero"><p className="eyebrow">{t(props.appearance.locale, 'routes.settings.eyebrow')}</p><h1 data-testid={testId('settings.title')}>{t(props.appearance.locale, 'settings.title')}</h1><p data-testid={testId('settings.appearance')}>{t(props.appearance.locale, 'settings.appearance')}</p></section><div className="settings-grid"><Select label={t(props.appearance.locale, 'settings.theme')} value={props.appearance.theme} onChange={(value) => set({ theme: value as AppearancePreferences['theme'] })} options={[{ value: 'system', label: t(props.appearance.locale, 'settings.theme.system') }, { value: 'light', label: t(props.appearance.locale, 'settings.theme.light') }, { value: 'dark', label: t(props.appearance.locale, 'settings.theme.dark') }]} testId={testId('settings.theme')} /><Select label={t(props.appearance.locale, 'settings.material')} value={props.appearance.material} onChange={(value) => set({ material: value as AppearancePreferences['material'] })} options={[{ value: 'glass', label: t(props.appearance.locale, 'settings.material.glass') }, { value: 'tinted', label: t(props.appearance.locale, 'settings.material.tinted') }, { value: 'opaque', label: t(props.appearance.locale, 'settings.material.opaque') }]} testId={testId('settings.material')} /><Select label={t(props.appearance.locale, 'settings.locale')} value={props.appearance.locale} onChange={(value) => set({ locale: value as AppearancePreferences['locale'] })} options={[{ value: 'en', label: t(props.appearance.locale, 'settings.locale.en') }, { value: 'ko', label: t(props.appearance.locale, 'settings.locale.ko') }]} testId={testId('settings.locale')} /><label className="ds-field"><span>{t(props.appearance.locale, 'settings.glass_intensity')}: {props.appearance.glassIntensity}</span><input type="range" min="0" max="100" value={props.appearance.glassIntensity} onChange={(event) => set({ glassIntensity: Number(event.currentTarget.value) })} data-testid={testId('settings.glass_intensity')} /></label><Toggle label={t(props.appearance.locale, 'settings.reduce_motion')} checked={props.appearance.reduceMotion} onChange={(checked) => set({ reduceMotion: checked })} testId={testId('settings.reduce_motion')} /><Toggle label={t(props.appearance.locale, 'settings.reduce_transparency')} checked={props.appearance.reduceTransparency} onChange={(checked) => set({ reduceTransparency: checked })} testId={testId('settings.reduce_transparency')} /><Toggle label={t(props.appearance.locale, 'settings.high_contrast')} checked={props.appearance.highContrast} onChange={(checked) => set({ highContrast: checked })} testId={testId('settings.high_contrast')} /></div><ErrorBanner tone="info" title={t(props.appearance.locale, 'settings.partial_success')} body={t(props.appearance.locale, 'settings.partial_success')} testId={testId('settings.partial_success')} /></div>
  );
}

function Toggle(props: { label: string; checked: boolean; onChange: (checked: boolean) => void; testId: string }): React.JSX.Element {
  return <label className="toggle"><input type="checkbox" checked={props.checked} onChange={(event) => props.onChange(event.currentTarget.checked)} data-testid={props.testId} /><span>{props.label}</span></label>;
}

function RouteInspector(props: { route: RouteId; locale: AppearancePreferences['locale'] }): React.ReactNode {
  if (props.route === 'gallery') return null;
  return <Inspector title={t(props.locale, 'gallery.data.inspector')}><p>{t(props.locale, 'adapters.pending.body')}</p><StatusBadge state="unloaded">{t(props.locale, 'models.status.unloaded')}</StatusBadge></Inspector>;
}

function CommandPalette(props: { open: boolean; locale: AppearancePreferences['locale']; onClose: () => void; onNavigate: (route: RouteId) => void }): React.JSX.Element {
  const [query, setQuery] = useState('');
  const items = routes.filter((route) => route.includes(query.toLowerCase()));
  return (
    <Dialog open={props.open} title={t(props.locale, 'command.title')} onClose={props.onClose} testId="command-dialog">
      <Field label={t(props.locale, 'command.search')} value={query} onChange={setQuery} testId={testId('command.search')} />
      {items.length === 0 ? <p data-testid={testId('command.no_results')}>{t(props.locale, 'command.no_results')}</p> : <div className="command-list">{items.map((route) => <Button key={route} onClick={() => props.onNavigate(route)}>{t(props.locale, route === 'models' ? 'nav.models' : route === 'chat' ? 'nav.chat' : route === 'activity' ? 'nav.activity' : route === 'settings' ? 'nav.settings' : 'nav.gallery')}</Button>)}</div>}
    </Dialog>
  );
}

export { DEFAULT_APPEARANCE };
