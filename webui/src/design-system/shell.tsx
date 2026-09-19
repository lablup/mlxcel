import React, { useEffect, useRef, useState } from 'react';
import { Icon } from './icons';
import { Button, Drawer, IconButton, PageLayout, StatusBadge } from './primitives';
import type { StringKey, Locale } from '../i18n/catalog';
import type { ConnectionPhase, ModelId, ModelLifecycleState } from '../api/types';
import { t, testId } from '../i18n/catalog';

export type RouteId = 'models' | 'chat' | 'activity' | 'settings' | 'gallery';

type NavItem = { id: RouteId; key: StringKey; icon: 'models' | 'chat' | 'activity' | 'settings' | 'gallery' };

export const navItems: NavItem[] = [
  { id: 'models', key: 'nav.models', icon: 'models' },
  { id: 'chat', key: 'nav.chat', icon: 'chat' },
  { id: 'activity', key: 'nav.activity', icon: 'activity' },
  { id: 'settings', key: 'nav.settings', icon: 'settings' },
];

/** A model the server holds, as the toolbar shows it. App maps catalog entries to this, so the shell imports neither the provider nor the catalog. */
export interface ShellLoadedModel {
  readonly id: ModelId;
  readonly name: string;
  readonly state: ModelLifecycleState;
  /** The localized lifecycle label for `state`. */
  readonly stateLabel: string;
}

/** The sidebar footer: the connection summary, plus internal identifiers kept in a collapsed disclosure (null before bootstrap). */
export interface ShellConnection {
  readonly label: string;
  readonly state: ConnectionPhase;
  readonly details: { readonly instance: string; readonly sequence: string } | null;
  /** The provider summary (backend, build, phase, catalog and operation counts, snapshot); shown only inside the disclosure. */
  readonly summary?: string | null;
}

/** Chips shown before the rest collapse into a "+n" chip that opens the command palette. */
export const TOOLBAR_CHIP_LIMIT = 3;

/** One keyboard shortcut as the help dialog lists it. `keys` are symbols, not translated. */
export interface GlobalShortcut {
  readonly id: 'command' | 'new-chat' | 'send' | 'escape' | 'navigate' | 'help';
  readonly keys: ReadonlyArray<string>;
  readonly separator: '+' | '/';
  readonly key: StringKey;
}

/**
 * The single source for the help dialog. Cmd/Ctrl+K, Cmd/Ctrl+N and ? are handled by
 * AppShell below; Cmd/Ctrl+Enter (and Cmd/Ctrl+N again) by the chat composer; Esc by
 * Dialog and Drawer; [ and ] by the sidebar. Add an entry here when a handler changes.
 */
export const globalShortcuts: ReadonlyArray<GlobalShortcut> = [
  { id: 'command', keys: ['⌘/Ctrl', 'K'], separator: '+', key: 'help.shortcut.command' },
  { id: 'new-chat', keys: ['⌘/Ctrl', 'N'], separator: '+', key: 'help.shortcut.new_chat' },
  { id: 'send', keys: ['⌘/Ctrl', 'Enter'], separator: '+', key: 'help.shortcut.send' },
  { id: 'escape', keys: ['Esc'], separator: '+', key: 'help.shortcut.escape' },
  { id: 'navigate', keys: ['[', ']'], separator: '/', key: 'help.shortcut.navigate' },
  { id: 'help', keys: ['?'], separator: '+', key: 'help.shortcut.help' },
];

export interface AppShellProps {
  readonly locale: Locale;
  readonly route: RouteId;
  readonly onRouteChange: (route: RouteId) => void;
  readonly onCommand: () => void;
  readonly onHelp: () => void;
  /** Cmd/Ctrl+N outside edit fields and dialogs. */
  readonly onNewChat: () => void;
  /** A loaded-model chip was activated; App selects it and opens Models. */
  readonly onOpenModel: (id: ModelId) => void;
  readonly children: React.ReactNode;
  readonly inspector?: React.ReactNode;
  /**
   * The server's loaded models in display order (never the browser's selection). `null` means
   * the shell does not know them (signed out, or no catalog snapshot yet), and the toolbar says
   * so instead of claiming that nothing is loaded; an empty array means none is loaded.
   */
  readonly loadedModels: ReadonlyArray<ShellLoadedModel> | null;
  readonly connection: ShellConnection;
  readonly sessionAction?: React.ReactNode;
}

export function AppShell(props: AppShellProps): React.JSX.Element {
  const sidebarRef = useRef<HTMLElement>(null);
  const onCommandRef = useRef(props.onCommand);
  const onHelpRef = useRef(props.onHelp);
  const onNewChatRef = useRef(props.onNewChat);
  const [navOpen, setNavOpen] = useState(false);
  useEffect(() => {
    onCommandRef.current = props.onCommand;
    onHelpRef.current = props.onHelp;
    onNewChatRef.current = props.onNewChat;
  }, [props.onCommand, props.onHelp, props.onNewChat]);
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      const target = event.target;
      const editable = target instanceof HTMLElement && (target.isContentEditable || Boolean(target.closest('[role="combobox"], [role="listbox"]')) || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName));
      // The navigation Drawer is a modal aside, not a native dialog, so match both.
      const inModal = target instanceof HTMLElement && Boolean(target.closest('dialog[open], [role="dialog"][aria-modal="true"]'));
      if (event.isComposing || event.altKey || editable || inModal) return;
      const command = event.metaKey || event.ctrlKey;
      if (command && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setNavOpen(false);
        onCommandRef.current();
      }
      // Browsers reserve Cmd/Ctrl+N for a new window and usually never deliver it to the
      // page; preventDefault still stops it wherever it does arrive. A held key still repeats
      // this keydown, so preventDefault runs every time but a new conversation is opened once.
      if (command && !event.shiftKey && event.key.toLowerCase() === 'n') {
        event.preventDefault();
        if (event.repeat) return;
        setNavOpen(false);
        onNewChatRef.current();
      }
      if (event.key === '?' || (event.key === '/' && event.shiftKey)) {
        event.preventDefault();
        setNavOpen(false);
        onHelpRef.current();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
  useEffect(() => {
    if (!navOpen || typeof window.matchMedia !== 'function') return;
    // The drawer exists only up to 960px. Widening past it would leave the drawer open
    // and its restore target (the menu button) display:none, so close it and move
    // focus to the desktop sidebar instead of letting it fall to <body>.
    const compact = window.matchMedia('(max-width: 960px)');
    const handleChange = (): void => {
      if (compact.matches) return;
      setNavOpen(false);
      requestAnimationFrame(() => {
        const active = document.activeElement;
        if (active && active !== document.body && !active.closest('[aria-modal="true"]')) return;
        const sidebar = sidebarRef.current;
        (sidebar?.querySelector<HTMLElement>('a[aria-current="page"]') ?? sidebar?.querySelector<HTMLElement>('a'))?.focus();
      });
    };
    compact.addEventListener('change', handleChange);
    return () => compact.removeEventListener('change', handleChange);
  }, [navOpen]);
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
      <Sidebar locale={props.locale} route={props.route} onRouteChange={handleRoute} onKeyDown={handleSidebarKey} ref={sidebarRef} className="app-sidebar desktop-sidebar material-glass" connection={props.connection} primary />
      <Drawer open={navOpen} title={t(props.locale, 'nav.primary')} onClose={() => setNavOpen(false)} closeLabel={t(props.locale, 'common.close')} testId="mobile-nav-sheet">
        <Sidebar locale={props.locale} route={props.route} onRouteChange={handleRoute} onKeyDown={handleSidebarKey} className="app-sidebar sheet-sidebar" connection={props.connection} />
      </Drawer>
      {/* The shared Drawer traps Tab only from inside its panel. Like the native modal sheet
          before it, keep the page behind the open drawer inert so focus cannot reach it. */}
      <main className="app-main" aria-labelledby="app-title" inert={navOpen}>
        <header className="app-toolbar material-glass">
          <IconButton className="mobile-menu-button" label={t(props.locale, 'toolbar.menu')} icon="menu" onClick={() => setNavOpen(true)} data-testid={testId('toolbar.menu')} />
          <div className="toolbar-title">
            <p id="app-title" title={t(props.locale, 'app.title')} data-testid={testId('app.title')}>{t(props.locale, 'app.title')}</p>
            <span title={t(props.locale, 'app.subtitle')} data-testid={testId('app.subtitle')}>{t(props.locale, 'app.subtitle')}</span>
          </div>
          <LoadedModels locale={props.locale} models={props.loadedModels} onOpenModel={props.onOpenModel} onMore={props.onCommand} />
          <div className="toolbar-actions">
            {props.sessionAction}
            <IconButton label={t(props.locale, 'toolbar.command')} icon="command" onClick={props.onCommand} data-testid={testId('toolbar.command')} />
            <IconButton label={t(props.locale, 'toolbar.help')} icon="help" onClick={props.onHelp} data-testid={testId('toolbar.help')} />
          </div>
        </header>
        <div className={`app-content-grid ${props.inspector ? 'has-inspector' : ''}`.trim()}>
          <PageLayout className="app-content">{props.children}</PageLayout>
          {props.inspector}
        </div>
      </main>
    </div>
  );
}

// The server's loaded models: a count and up to TOOLBAR_CHIP_LIMIT chips, then a "+n"
// chip that opens the command palette, whose empty query lists every loaded model. A chip
// only selects and inspects; it never loads or unloads. Its accessible name is its
// visible text (name and lifecycle), so it takes no aria-label (WCAG 2.5.3). Unknown
// (null) and none loaded (empty) are different claims, so each has its own text.
function LoadedModels(props: { locale: Locale; models: ReadonlyArray<ShellLoadedModel> | null; onOpenModel: (id: ModelId) => void; onMore: () => void }): React.JSX.Element {
  const models = props.models ?? [];
  const shown = models.slice(0, TOOLBAR_CHIP_LIMIT);
  const hidden = models.length - shown.length;
  const status = props.models === null ? 'toolbar.loaded.unknown' : models.length === 0 ? 'toolbar.loaded.none' : null;
  return (
    <div className="toolbar-loaded" role="group" aria-label={t(props.locale, 'toolbar.loaded.label')} data-testid={testId('toolbar.loaded.label')} onFocus={revealFocusedChip}>
      {status !== null ? <span className="toolbar-loaded-count" data-testid={testId(status)}>{t(props.locale, status)}</span> : <>
        <span className="toolbar-loaded-count" data-testid={testId('toolbar.loaded.count')}>{t(props.locale, 'toolbar.loaded.count', { count: String(models.length) })}</span>
        {shown.map((model) => (
          <Button key={model.id} tone="ghost" className="toolbar-chip" title={model.name} data-testid="toolbar-loaded-chip" onClick={() => props.onOpenModel(model.id)}>
            <span className="truncate">{model.name}</span>
            <StatusBadge state={model.state}>{model.stateLabel}</StatusBadge>
          </Button>
        ))}
        {hidden > 0 ? <Button tone="ghost" className="toolbar-chip toolbar-chip-more" aria-label={t(props.locale, 'toolbar.loaded.more_label', { count: String(hidden) })} data-testid={testId('toolbar.loaded.more')} onClick={props.onMore}>{t(props.locale, 'toolbar.loaded.more', { count: String(hidden) })}</Button> : null}
      </>}
    </div>
  );
}

// Below 960 px the chip row scrolls sideways. Chromium's own focus scrolling leaves an element
// that is already partly visible where it is, so a keyboard-focused chip half past the row's
// edge stays clipped; scroll it fully into view. Pointer focus is left alone, so a chip never
// moves between mousedown and mouseup.
function revealFocusedChip(event: React.FocusEvent<HTMLDivElement>): void {
  const target = event.target;
  if (target instanceof HTMLElement && typeof target.scrollIntoView === 'function' && target.matches(':focus-visible')) target.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

// `primary` marks the desktop copy; the off-canvas sheet renders a second one, and a test id must name one element.
const Sidebar = React.forwardRef<HTMLElement, { locale: Locale; route: RouteId; onRouteChange: (route: RouteId) => void; onKeyDown: (event: React.KeyboardEvent) => void; className: string; connection: ShellConnection; primary?: boolean }>((props, ref) => (
  <aside className={props.className} aria-label={t(props.locale, 'nav.primary')} onKeyDown={props.onKeyDown} ref={ref} tabIndex={-1}>
    <a className="brand-mark" href="#models" aria-label={t(props.locale, 'nav.home')} onClick={(event) => { event.preventDefault(); props.onRouteChange('models'); }}><span className="brand-symbol" aria-hidden="true">mx</span><span className="brand-name">mlxcel</span></a>
    <nav className="app-nav">
      {navItems.map((item) => (
        <a key={item.id} href={`#${item.id}`} aria-label={t(props.locale, item.key)} aria-current={props.route === item.id ? 'page' : undefined} data-testid={testId(item.key)} onClick={(event) => { event.preventDefault(); props.onRouteChange(item.id); }}>
          <Icon name={item.icon} />
          <span className="app-nav-label">{t(props.locale, item.key)}</span>
        </a>
      ))}
    </nav>
    <footer>
      <div className="connection-status">
        <span className="connection-dot" data-state={props.connection.state} aria-hidden="true" />
        <span data-testid={testId('connection.ready')}>{props.connection.label}</span>
      </div>
      {/* Internal identifiers stay out of the summary line, collapsed until asked for. */}
      {props.connection.details ? (
        <details className="connection-details" data-testid={testId('connection.footer.details')}>
          <summary>{t(props.locale, 'connection.footer.details')}</summary>
          {props.connection.summary ? <p data-testid={props.primary ? testId('connection.authenticated.detail') : undefined}>{props.connection.summary}</p> : null}
          <p data-testid={testId('connection.footer.instance')}>{t(props.locale, 'connection.footer.instance', { id: props.connection.details.instance })}</p>
          <p data-testid={testId('connection.footer.sequence')}>{t(props.locale, 'connection.footer.sequence', { sequence: props.connection.details.sequence })}</p>
        </details>
      ) : null}
    </footer>
  </aside>
));
Sidebar.displayName = 'Sidebar';
