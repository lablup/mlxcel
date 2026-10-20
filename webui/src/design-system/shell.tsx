import React, { useEffect, useRef, useState } from 'react';
// The first bundle imports seam modules directly, not the primitives barrel (see primitives.tsx).
import { Button, IconButton, StatusBadge } from './common-adapters';
import { SHELL_COMPACT_QUERY, ShellDrawerNav, ShellFrame, ShellMenuToggle, ShellNavItem, ShellSideNav } from './common-app-shell';
import { PageLayout } from './common-layout';
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
      // The navigation drawer and the product dialogs are native modal dialogs; match the ARIA form too.
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
    // The drawer exists only below the AppShell breakpoint. Widening past it unmounts the drawer
    // (its modal dialog closes and focus falls to <body>) and mounts the desktop side navigation:
    // close the state and move focus to the current route's link there.
    const compact = window.matchMedia(SHELL_COMPACT_QUERY);
    const handleChange = (): void => {
      if (compact.matches) return;
      setNavOpen(false);
      let frames = 0;
      const settle = (): void => {
        const sidebar = sidebarRef.current;
        if (!sidebar && frames++ < 120) {
          requestAnimationFrame(settle);
          return;
        }
        const active = document.activeElement;
        if (active && active !== document.body && !active.closest('dialog')) return;
        (sidebar?.querySelector<HTMLElement>('a[aria-current="page"]') ?? sidebar?.querySelector<HTMLElement>('a'))?.focus();
      };
      requestAnimationFrame(settle);
    };
    compact.addEventListener('change', handleChange);
    return () => compact.removeEventListener('change', handleChange);
  }, [navOpen]);
  const handleNavKey = (event: React.KeyboardEvent<HTMLElement>): void => {
    if (event.key !== '[' && event.key !== ']') return;
    if (event.altKey || event.metaKey || event.ctrlKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    const container = event.currentTarget;
    const index = navItems.findIndex((item) => item.id === props.route);
    const next = navItems[(index + (event.key === ']' ? 1 : -1) + navItems.length) % navItems.length];
    props.onRouteChange(next.id);
    requestAnimationFrame(() => container.querySelector<HTMLAnchorElement>(`a[href="#${next.id}"]`)?.focus());
  };
  const handleRoute = (route: RouteId): void => {
    props.onRouteChange(route);
    setNavOpen(false);
  };
  const items = navItems.map((item) => (
    <ShellNavItem key={item.id} label={t(props.locale, item.key)} icon={item.icon} href={`#${item.id}`} isSelected={props.route === item.id} testId={testId(item.key)} onSelect={() => handleRoute(item.id)} />
  ));
  const topBar = (
    <div className="app-toolbar material-glass">
      <ShellMenuToggle label={t(props.locale, 'toolbar.menu')} testId={testId('toolbar.menu')} className="ds-icon-button mobile-menu-button" />
      <BrandMark locale={props.locale} className="brand-mark toolbar-brand" onHome={() => handleRoute('models')} />
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
    </div>
  );
  return (
    <ShellFrame
      topBar={topBar}
      sideNav={<ShellSideNav ref={sidebarRef} label={t(props.locale, 'nav.primary')} footer={<ConnectionFooter locale={props.locale} connection={props.connection} />} className="app-sidebar desktop-sidebar material-glass" onKeyDown={handleNavKey}>{items}</ShellSideNav>}
      drawer={<ShellDrawerNav label={t(props.locale, 'nav.primary')} topContent={<BrandMark locale={props.locale} onHome={() => handleRoute('models')} />} footer={<ConnectionFooter locale={props.locale} connection={props.connection} />} className="app-sidebar sheet-sidebar" testId="mobile-nav-sheet" onKeyDown={handleNavKey}>{items}</ShellDrawerNav>}
      drawerOpen={navOpen}
      onDrawerOpenChange={setNavOpen}
    >
      {/* AppShell's main landmark is a role="main" div, so a route's <header> (the PageHeader) would
          map to a banner; an unnamed <section> scopes it back to a plain header and adds no landmark. */}
      <section className={`app-content-grid ${props.inspector ? 'has-inspector' : ''}`.trim()}>
        <PageLayout className="app-content">{props.children}</PageLayout>
        {props.inspector}
      </section>
    </ShellFrame>
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
            <span className="toolbar-chip-content">
              <span className="truncate">{model.name}</span>
              <StatusBadge state={model.state} appearance="inline">{model.stateLabel}</StatusBadge>
            </span>
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

// The product mark, a link home: in the bar beside the side navigation, and at the top of the
// compact drawer (the bar hides its copy there). Neither copy carries a test id.
function BrandMark(props: { locale: Locale; onHome: () => void; className?: string }): React.JSX.Element {
  return <a className={props.className ?? 'brand-mark'} href="#models" aria-label={t(props.locale, 'nav.home')} onClick={(event) => { event.preventDefault(); props.onHome(); }}><span className="brand-symbol" aria-hidden="true">mx</span><span className="brand-name">mlxcel</span></a>;
}

// AppShell renders the side navigation above its breakpoint and the drawer below it, never both,
// so the one footer on the page carries the test ids.
function ConnectionFooter(props: { locale: Locale; connection: ShellConnection }): React.JSX.Element {
  return (
    <footer className="app-sidebar-footer">
      <div className="connection-status">
        <span className="connection-dot" data-state={props.connection.state} aria-hidden="true" />
        <span data-testid={testId('connection.ready')}>{props.connection.label}</span>
      </div>
      {/* Internal identifiers stay out of the summary line, collapsed until asked for. */}
      {props.connection.details ? (
        <details className="connection-details" data-testid={testId('connection.footer.details')}>
          <summary>{t(props.locale, 'connection.footer.details')}</summary>
          {props.connection.summary ? <p data-testid={testId('connection.authenticated.detail')}>{props.connection.summary}</p> : null}
          <p data-testid={testId('connection.footer.instance')}>{t(props.locale, 'connection.footer.instance', { id: props.connection.details.instance })}</p>
          <p data-testid={testId('connection.footer.sequence')}>{t(props.locale, 'connection.footer.sequence', { sequence: props.connection.details.sequence })}</p>
        </details>
      ) : null}
    </footer>
  );
}
