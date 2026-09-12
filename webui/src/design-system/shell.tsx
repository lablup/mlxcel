import React, { useEffect, useRef } from 'react';
import { IconButton } from './primitives';
import type { StringKey, Locale } from '../i18n/catalog';
import { t, testId } from '../i18n/catalog';

export type RouteId = 'models' | 'chat' | 'activity' | 'settings' | 'gallery';

type NavItem = { id: RouteId; key: StringKey; icon: string };

export const navItems: NavItem[] = [
  { id: 'models', key: 'nav.models', icon: '▦' },
  { id: 'chat', key: 'nav.chat', icon: '◌' },
  { id: 'activity', key: 'nav.activity', icon: '⌁' },
  { id: 'settings', key: 'nav.settings', icon: '◎' },
  { id: 'gallery', key: 'nav.gallery', icon: '◫' },
];

export function AppShell(props: { locale: Locale; route: RouteId; onRouteChange: (route: RouteId) => void; onCommand: () => void; onHelp: () => void; children: React.ReactNode; inspector?: React.ReactNode; selectedModel: string }): React.JSX.Element {
  const sidebarRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      const target = event.target;
      const editable = target instanceof HTMLElement && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
      if (event.isComposing || editable) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        props.onCommand();
      }
      if (event.key === '?') {
        event.preventDefault();
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
    if (event.target === event.currentTarget || sidebarRef.current?.contains(event.target as Node)) {
      if (event.key === '[' || event.key === ']') {
        event.preventDefault();
        moveNav(event.key === ']' ? 1 : -1);
      }
    }
  };
  return (
    <div className="app-shell">
      <aside className="app-sidebar material-glass" aria-label={t(props.locale, 'nav.primary')} onKeyDown={handleSidebarKey} ref={sidebarRef} tabIndex={-1}>
        <a className="brand-mark" href="#models" aria-label={t(props.locale, 'nav.home')}>mx</a>
        <nav className="app-nav">
          {navItems.map((item) => (
            <a key={item.id} href={`#${item.id}`} aria-current={props.route === item.id ? 'page' : undefined} data-testid={testId(item.key)} onClick={(event) => { event.preventDefault(); props.onRouteChange(item.id); }}>
              <span aria-hidden="true">{item.icon}</span>
              <span>{t(props.locale, item.key)}</span>
            </a>
          ))}
        </nav>
        <footer>
          <span className="connection-dot" aria-hidden="true" />
          <span data-testid={testId('connection.ready')}>{t(props.locale, 'connection.ready')}</span>
        </footer>
      </aside>
      <main className="app-main" aria-labelledby="app-title">
        <header className="app-toolbar material-glass">
          <div>
            <p id="app-title" data-testid={testId('app.title')}>{t(props.locale, 'app.title')}</p>
            <span data-testid={testId('app.subtitle')}>{t(props.locale, 'app.subtitle')}</span>
          </div>
          <div className="selected-model" title={props.selectedModel}>{props.selectedModel}</div>
          <div className="toolbar-actions">
            <IconButton label={t(props.locale, 'toolbar.command')} icon="⌘K" onClick={props.onCommand} data-testid={testId('toolbar.command')} />
            <IconButton label={t(props.locale, 'toolbar.help')} icon="?" onClick={props.onHelp} data-testid={testId('toolbar.help')} />
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
