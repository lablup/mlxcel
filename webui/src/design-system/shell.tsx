import React, { useEffect, useRef, useState } from 'react';
import { Icon } from './icons';
import { IconButton, Sheet } from './primitives';
import type { StringKey, Locale } from '../i18n/catalog';
import { t, testId } from '../i18n/catalog';

export type RouteId = 'models' | 'chat' | 'activity' | 'settings' | 'gallery';

type NavItem = { id: RouteId; key: StringKey; icon: 'models' | 'chat' | 'activity' | 'settings' | 'gallery' };

export const navItems: NavItem[] = [
  { id: 'models', key: 'nav.models', icon: 'models' },
  { id: 'chat', key: 'nav.chat', icon: 'chat' },
  { id: 'activity', key: 'nav.activity', icon: 'activity' },
  { id: 'settings', key: 'nav.settings', icon: 'settings' },
  { id: 'gallery', key: 'nav.gallery', icon: 'gallery' },
];

export function AppShell(props: { locale: Locale; route: RouteId; onRouteChange: (route: RouteId) => void; onCommand: () => void; onHelp: () => void; children: React.ReactNode; inspector?: React.ReactNode; selectedModel: string }): React.JSX.Element {
  const sidebarRef = useRef<HTMLElement>(null);
  const [navOpen, setNavOpen] = useState(false);
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      const target = event.target;
      const editable = target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
      const inModal = target instanceof HTMLElement && Boolean(target.closest('dialog[open]'));
      if (event.isComposing || event.altKey || editable || inModal) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setNavOpen(false);
        props.onCommand();
      }
      if (event.key === '?') {
        event.preventDefault();
        setNavOpen(false);
        props.onHelp();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [props]);
  const moveNav = (direction: number): void => {
    const index = navItems.findIndex((item) => item.id === props.route);
    const next = navItems[(index + direction + navItems.length) % navItems.length];
    props.onRouteChange(next.id);
    requestAnimationFrame(() => sidebarRef.current?.querySelector<HTMLAnchorElement>(`a[href="#${next.id}"]`)?.focus());
  };
  const handleSidebarKey = (event: React.KeyboardEvent): void => {
    if (event.key !== '[' && event.key !== ']') return;
    if (event.altKey || event.metaKey || event.ctrlKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    moveNav(event.key === ']' ? 1 : -1);
  };
  const handleRoute = (route: RouteId): void => {
    props.onRouteChange(route);
    setNavOpen(false);
  };
  return (
    <div className="app-shell">
      <Sidebar locale={props.locale} route={props.route} onRouteChange={handleRoute} onKeyDown={handleSidebarKey} ref={sidebarRef} className="app-sidebar desktop-sidebar material-glass" />
      <Sheet open={navOpen} title={t(props.locale, 'nav.primary')} onClose={() => setNavOpen(false)} closeLabel={t(props.locale, 'common.close')} testId="mobile-nav-sheet">
        <Sidebar locale={props.locale} route={props.route} onRouteChange={handleRoute} onKeyDown={handleSidebarKey} className="app-sidebar sheet-sidebar" />
      </Sheet>
      <main className="app-main" aria-labelledby="app-title">
        <header className="app-toolbar material-glass">
          <IconButton className="mobile-menu-button" label={t(props.locale, 'toolbar.menu')} icon="menu" onClick={() => setNavOpen(true)} data-testid={testId('toolbar.menu')} />
          <div className="toolbar-title">
            <p id="app-title" data-testid={testId('app.title')}>{t(props.locale, 'app.title')}</p>
            <span data-testid={testId('app.subtitle')}>{t(props.locale, 'app.subtitle')}</span>
          </div>
          <div className="selected-model" title={props.selectedModel}>{props.selectedModel}</div>
          <div className="toolbar-actions">
            <IconButton label={t(props.locale, 'toolbar.command')} icon="command" onClick={props.onCommand} data-testid={testId('toolbar.command')} />
            <IconButton label={t(props.locale, 'toolbar.help')} icon="help" onClick={props.onHelp} data-testid={testId('toolbar.help')} />
          </div>
        </header>
        <div className="app-content-grid">
          <section className="app-content">{props.children}</section>
          {props.inspector}
        </div>
      </main>
    </div>
  );
}

const Sidebar = React.forwardRef<HTMLElement, { locale: Locale; route: RouteId; onRouteChange: (route: RouteId) => void; onKeyDown: (event: React.KeyboardEvent) => void; className: string }>((props, ref) => (
  <aside className={props.className} aria-label={t(props.locale, 'nav.primary')} onKeyDown={props.onKeyDown} ref={ref} tabIndex={-1}>
    <a className="brand-mark" href="#models" aria-label={t(props.locale, 'nav.home')} onClick={(event) => { event.preventDefault(); props.onRouteChange('models'); }}>mx</a>
    <nav className="app-nav">
      {navItems.map((item) => (
        <a key={item.id} href={`#${item.id}`} aria-label={t(props.locale, item.key)} aria-current={props.route === item.id ? 'page' : undefined} data-testid={testId(item.key)} onClick={(event) => { event.preventDefault(); props.onRouteChange(item.id); }}>
          <Icon name={item.icon} />
          <span className="app-nav-label">{t(props.locale, item.key)}</span>
        </a>
      ))}
    </nav>
    <footer>
      <span className="connection-dot" aria-hidden="true" />
      <span data-testid={testId('connection.ready')}>{t(props.locale, 'connection.ready')}</span>
    </footer>
  </aside>
));
Sidebar.displayName = 'Sidebar';
