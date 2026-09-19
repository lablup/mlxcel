import React from 'react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import stringsFixture from '../../tests/fixtures/webui/strings.json';
import { App } from './app';
import { LoginView, SchemaMismatchView } from './design-system/primitives';
import { DEFAULT_APPEARANCE, applyAppearance, loadAppearance, saveAppearance } from './design-system/preferences';
import { globalShortcuts } from './design-system/shell';
import { consumeNewConversationRequest } from './features/chat/session';
import { entries, t } from './i18n/catalog';
import { WebUiProvider } from './state';

let root: Root | null = null;
let host: HTMLDivElement | null = null;

beforeEach(() => {
  HTMLDialogElement.prototype.showModal = vi.fn(function showModal(this: HTMLDialogElement) { this.setAttribute('open', ''); });
  HTMLDialogElement.prototype.close = vi.fn(function close(this: HTMLDialogElement) { this.removeAttribute('open'); this.dispatchEvent(new Event('close')); });
  localStorage.clear();
  window.history.replaceState(null, '', '/webui/#models');
  host = document.createElement('div');
  host.id = 'root';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(() => {
  vi.restoreAllMocks();
  act(() => root?.unmount());
  host?.remove();
  document.documentElement.removeAttribute('style');
  document.documentElement.dataset.theme = '';
  root = null;
  host = null;
});

function neverFetch(): Promise<Response> {
  throw new Error('Unexpected WebUI API request before explicit login.');
}

function renderApp(): void {
  act(() => root?.render(<WebUiProvider fetchImpl={neverFetch as typeof fetch}><App /></WebUiProvider>));
}

function keydown(key: string, init: KeyboardEventInit = {}): KeyboardEvent {
  const target = document.activeElement instanceof HTMLElement ? document.activeElement : window;
  const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(event);
  return event;
}

function goTo(route: string): void {
  act(() => { window.history.replaceState(null, '', `#${route}`); window.dispatchEvent(new HashChangeEvent('hashchange')); });
}

const settle = async (): Promise<void> => { await act(async () => { await new Promise((resolve) => window.setTimeout(resolve, 10)); }); };
const element = (selector: string): HTMLElement => {
  const found = document.querySelector<HTMLElement>(selector);
  if (!found) throw new Error(`Missing ${selector}`);
  return found;
};

describe('mlxcel WebUI shell', () => {
  it('keeps hash navigation client-side', () => {
    const url = new URL('http://127.0.0.1:8080/private/webui/#models');
    expect(url.pathname).toBe('/private/webui/');
    expect(url.hash).toBe('#models');
  });

  it('opens the command palette with Cmd/Ctrl+K and restores focus on Escape', async () => {
    renderApp();
    const command = document.querySelector<HTMLButtonElement>('[data-testid="toolbar-command"]');
    command?.focus();
    act(() => keydown('k', { metaKey: true }));
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(true);
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="command-dialog"] [data-testid="dialog-close"]')?.click());
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    expect(document.activeElement).toBe(command);
  });

  it('keeps overlays mutually exclusive and suppresses shortcuts inside modals', async () => {
    renderApp();
    act(() => keydown('k', { metaKey: true }));
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(true);
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    act(() => keydown('?'));
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(true);
    expect(document.querySelector('[data-testid="help-dialog"]')?.hasAttribute('open')).toBe(false);
    act(() => document.querySelector<HTMLButtonElement>('[data-testid="command-dialog"] [data-testid="dialog-close"]')?.click());
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    document.body.focus();
    act(() => keydown('?'));
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    expect(document.querySelector('[data-testid="help-dialog"]')?.hasAttribute('open')).toBe(true);
    act(() => keydown('k', { metaKey: true }));
    expect(document.querySelector('[data-testid="help-dialog"]')?.hasAttribute('open')).toBe(true);
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(false);
  });

  it('moves navigation with brackets only when the sidebar owns focus', () => {
    renderApp();
    const navModels = document.querySelector<HTMLAnchorElement>('[data-testid="nav-models"]');
    navModels?.focus();
    act(() => navModels?.dispatchEvent(new KeyboardEvent('keydown', { key: ']', bubbles: true })));
    expect(window.location.hash).toBe('#chat');
    document.querySelector<HTMLButtonElement>('[data-testid="toolbar-command"]')?.focus();
    act(() => keydown(']'));
    expect(window.location.hash).toBe('#chat');
  });

  it('does not open global overlays from editable fields, IME composition, or Alt chords', () => {
    renderApp();
    act(() => document.querySelector<HTMLAnchorElement>('[data-testid="nav-settings"]')?.click());
    const select = document.querySelector<HTMLButtonElement>('[data-testid="settings-theme"] [role="combobox"]');
    select?.focus();
    act(() => keydown('k', { metaKey: true }));
    expect(document.querySelector('[data-testid="command-dialog"]')?.hasAttribute('open')).toBe(false);
    act(() => keydown('?', { isComposing: true }));
    act(() => keydown('?', { altKey: true }));
    expect(document.querySelector('[data-testid="help-dialog"]')?.hasAttribute('open')).toBe(false);
  });

  it('applies appearance through data attributes without inline style', () => {
    applyAppearance(document.documentElement, { ...DEFAULT_APPEARANCE, glassIntensity: 71, reduceTransparency: true, highContrast: 'on' });
    expect(document.documentElement.dataset.glassIntensity).toBe('71');
    expect(document.documentElement.dataset.material).toBe('opaque');
    expect(document.documentElement.dataset.highContrast).toBe('on');
    expect(document.documentElement.getAttribute('style')).toBeNull();
  });

  it('keeps appearance usable when storage is invalid or unavailable', () => {
    localStorage.setItem('mlxcel.webui.appearance', '[]');
    expect(loadAppearance()).toEqual(DEFAULT_APPEARANCE);
    localStorage.setItem('mlxcel.webui.appearance', '{');
    expect(loadAppearance()).toEqual(DEFAULT_APPEARANCE);
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new DOMException('blocked', 'SecurityError'); });
    expect(loadAppearance()).toEqual(DEFAULT_APPEARANCE);
    vi.restoreAllMocks();
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new DOMException('full', 'QuotaExceededError'); });
    expect(() => saveAppearance(DEFAULT_APPEARANCE)).not.toThrow();
  });


  it('submits LoginView tokens without persisting them and clears recovery state', () => {
    const submitted: string[] = [];
    act(() => root?.render(<><LoginView title="Login" body="Use local key" tokenLabel="Session key" tokenHelp="Memory only" submitLabel="Connect" logoutLabel="Clear" onSubmit={(token) => submitted.push(token)} onLogout={() => submitted.push('logout')} testId="login-view" /><SchemaMismatchView title="Mismatch" body="Update required" actionLabel="Reload" onRecover={() => submitted.push('recover')} /></>));
    const form = document.querySelector<HTMLFormElement>('[data-testid="login-view"]');
    const input = document.querySelector<HTMLInputElement>('input[type="password"]');
    expect(form).not.toBeNull();
    expect(input).not.toBeNull();
    if (!input || !form) return;
    expect(form.getAttribute('autocomplete')).toBe('off');
    expect(input.getAttribute('autocomplete')).toBe('off');
    expect(input.getAttribute('spellcheck')).toBe('false');
    expect(input.getAttribute('autocapitalize')).toBe('none');
    expect(input.getAttribute('autocorrect')).toBe('off');
    const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    act(() => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
    expect(submitted).toEqual([]);
    act(() => { valueSetter?.call(input, 'secret-token'); input.dispatchEvent(new Event('input', { bubbles: true })); });
    act(() => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
    expect(submitted).toContain('secret-token');
    expect(input?.value).toBe('');
    expect(localStorage.getItem('secret-token')).toBeNull();
    act(() => document.querySelectorAll('button')[1]?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(submitted).toContain('logout');
    act(() => document.querySelectorAll('button')[2]?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(submitted).toContain('recover');
    act(() => root?.render(<LoginView title="Login" body="Use local key" tokenLabel="Session key" tokenHelp="Memory only" submitLabel="Connect" error="Denied" busy onSubmit={(token) => submitted.push(token)} testId="login-view" />));
    const busyInput = document.querySelector<HTMLInputElement>('input[type="password"]');
    expect(busyInput?.disabled).toBe(true);
    expect(busyInput?.getAttribute('aria-invalid')).toBe('true');
    act(() => document.querySelector<HTMLFormElement>('[data-testid="login-view"]')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
    expect(submitted.filter((value) => value === '')).toHaveLength(0);
    expect(submitted.filter((value) => value === 'secret-token')).toHaveLength(1);
    expect(document.body.textContent).toContain('Denied');
  });

  it('blocks programmatic LoginView submits while busy even with a buffered token', () => {
    const submitted: string[] = [];
    const renderLogin = (busy: boolean) => root?.render(<LoginView title="Login" body="Use local key" tokenLabel="Session key" tokenHelp="Memory only" submitLabel="Connect" busy={busy} onSubmit={(token) => submitted.push(token)} testId="login-view" />);
    act(() => renderLogin(false));
    const input = document.querySelector<HTMLInputElement>('input[type="password"]');
    const form = document.querySelector<HTMLFormElement>('[data-testid="login-view"]');
    expect(input).not.toBeNull();
    expect(form).not.toBeNull();
    if (!input || !form) return;
    const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    act(() => { valueSetter?.call(input, 'busy-token'); input.dispatchEvent(new Event('input', { bubbles: true })); });
    act(() => renderLogin(true));
    act(() => form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
    expect(submitted).toEqual([]);
  });

  it('keeps the checked string fixture synchronized with typed keys', () => {
    const fixture = stringsFixture.strings.map((entry) => entry.key).sort();
    const typed = entries.map((entry) => entry.key).sort();
    expect(fixture).toEqual(typed);
    for (const entry of stringsFixture.strings) {
      expect(entry.en).toBeTruthy();
      expect(entry.ko).toBeTruthy();
      expect(entry.test_id).toMatch(/^[a-z0-9-]+$/);
    }
  });

  it.each([['metaKey'], ['ctrlKey']] as const)('navigates to Chat with %s+N from Models, Activity and Settings', (modifier) => {
    renderApp();
    for (const route of ['models', 'activity', 'settings']) {
      goTo(route);
      act(() => { (document.activeElement as HTMLElement | null)?.blur(); });
      let event: KeyboardEvent | undefined;
      act(() => { event = keydown('n', { [modifier]: true }); });
      expect(event?.defaultPrevented).toBe(true);
      expect(window.location.hash).toBe('#chat');
      // Signed out, Chat renders the connection surface; no request is left pending for a later mount.
      expect(consumeNewConversationRequest()).toBe(false);
    }
  });

  it('ignores a held key repeat for the global Cmd/Ctrl+N shortcut, but still prevents its default action', () => {
    renderApp();
    goTo('models');
    act(() => { (document.activeElement as HTMLElement | null)?.blur(); });
    let event: KeyboardEvent | undefined;
    act(() => { event = keydown('n', { metaKey: true, repeat: true }); });
    expect(event?.defaultPrevented).toBe(true);
    expect(window.location.hash).toBe('#models');
    expect(consumeNewConversationRequest()).toBe(false);
    act(() => { event = keydown('n', { metaKey: true }); });
    expect(event?.defaultPrevented).toBe(true);
    expect(window.location.hash).toBe('#chat');
  });

  it('suppresses Cmd/Ctrl+N in edit fields, during IME composition, with Alt or Shift, and inside dialogs', async () => {
    renderApp();
    goTo('settings');
    element('[data-testid="settings-theme"] [role="combobox"]').focus();
    let event: KeyboardEvent | undefined;
    act(() => { event = keydown('n', { metaKey: true }); });
    expect(event?.defaultPrevented).toBe(false);
    expect(window.location.hash).toBe('#settings');
    act(() => { (document.activeElement as HTMLElement | null)?.blur(); });
    expect(document.activeElement).toBe(document.body);
    for (const init of [{ ctrlKey: true, isComposing: true }, { ctrlKey: true, altKey: true }, { metaKey: true, shiftKey: true }]) {
      act(() => { event = keydown('n', init); });
      expect(event?.defaultPrevented).toBe(false);
      expect(window.location.hash).toBe('#settings');
    }
    act(() => { keydown('k', { metaKey: true }); });
    await settle();
    expect(element('[data-testid="command-dialog"]').hasAttribute('open')).toBe(true);
    element('[data-testid="command-dialog"] [data-testid="dialog-close"]').focus();
    act(() => { event = keydown('n', { metaKey: true }); });
    expect(event?.defaultPrevented).toBe(false);
    expect(window.location.hash).toBe('#settings');
  });

  it.each([['en'], ['ko']] as const)('lists exactly the globalShortcuts entries in the %s help dialog, in order', async (locale) => {
    localStorage.setItem('mlxcel.webui.appearance', JSON.stringify({ ...DEFAULT_APPEARANCE, locale }));
    renderApp();
    document.body.focus();
    act(() => { keydown('?'); });
    await settle();
    const items = Array.from(document.querySelectorAll<HTMLLIElement>('[data-testid="help-shortcuts"] > li'));
    expect(globalShortcuts).toHaveLength(6);
    expect(items).toHaveLength(globalShortcuts.length);
    expect(items.map((item) => item.dataset.testid)).toEqual(globalShortcuts.map((shortcut) => `help-shortcut-${shortcut.id}`));
    expect(globalShortcuts.map((shortcut) => shortcut.id)).toEqual(['command', 'new-chat', 'send', 'escape', 'navigate', 'help']);
    items.forEach((item, index) => {
      const shortcut = globalShortcuts[index];
      expect(Array.from(item.querySelectorAll('kbd')).map((kbd) => kbd.textContent)).toEqual([...shortcut.keys]);
      expect(item.textContent).toContain(t(locale, shortcut.key));
    });
    expect(items[4].textContent).toContain('[ / ]');
    expect(items[0].textContent).toContain('⌘/Ctrl + K');
  });

  it('keeps gallery out of the command palette and finds commands by localized label or route id', async () => {
    localStorage.setItem('mlxcel.webui.appearance', JSON.stringify({ ...DEFAULT_APPEARANCE, locale: 'ko' }));
    window.history.replaceState(null, '', '/webui/#gallery');
    renderApp();
    act(() => { element('[data-testid="toolbar-command"]').click(); });
    await settle();
    const dialog = element('[data-testid="command-dialog"]');
    const labels = (): string[] => Array.from(dialog.querySelectorAll<HTMLButtonElement>('.command-list button')).map((button) => button.textContent ?? '');
    expect(labels()).toEqual(['모델', '대화', '활동', '설정', '새 대화']);
    expect(dialog.textContent).not.toContain(t('ko', 'nav.gallery'));
    const search = element('[data-testid="command-search"]') as HTMLInputElement;
    const type = (value: string): void => act(() => { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(search, value); search.dispatchEvent(new Event('input', { bubbles: true })); });
    type('설정');
    expect(labels()).toEqual(['설정']);
    type('SETT');
    expect(labels()).toEqual(['설정']);
    type('gallery');
    expect(labels()).toEqual([]);
    expect(dialog.querySelector('[data-testid="command-no-results"]')).not.toBeNull();
    act(() => { element('[data-testid="command-dialog"] [data-testid="dialog-close"]').click(); });
    await settle();
    act(() => { element('[data-testid="toolbar-command"]').click(); });
    await settle();
    expect(search.value).toBe('');
    expect(labels()).toHaveLength(5);
  });

  it.each([['models', 'models-title'], ['chat', 'chat-title'], ['activity', 'activity-title'], ['settings', 'settings-title'], ['gallery', 'gallery-title']])('gives the signed-out %s route exactly one dialog focus fallback, its PageHeader title', (route, titleId) => {
    window.history.replaceState(null, '', `/webui/#${route}`);
    renderApp();
    const fallbacks = document.querySelectorAll<HTMLElement>('[data-dialog-focus-fallback]');
    expect(fallbacks).toHaveLength(1);
    expect(fallbacks[0].dataset.testid).toBe(titleId);
    expect(fallbacks[0].matches('.app-content .ds-page-header h1.page-header__title')).toBe(true);
    expect(fallbacks[0].tabIndex).toBe(-1);
  });

  it('lands focus on the route PageHeader title inside the shell PageLayout when a dialog closes after its trigger was detached', async () => {
    renderApp();
    const trigger = document.createElement('button');
    trigger.textContent = 'Temporary trigger';
    document.body.append(trigger);
    trigger.focus();
    act(() => { trigger.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', metaKey: true, bubbles: true, cancelable: true })); });
    await settle();
    expect(element('[data-testid="command-dialog"]').hasAttribute('open')).toBe(true);
    trigger.remove();
    act(() => { element('[data-testid="command-dialog"] [data-testid="dialog-close"]').click(); });
    await settle();
    const title = element('[data-testid="models-title"]');
    expect(document.activeElement).toBe(title);
    expect(title.hasAttribute('data-dialog-focus-fallback')).toBe(true);
    expect(title.closest('.app-content.page-layout')).not.toBeNull();
  });
});
