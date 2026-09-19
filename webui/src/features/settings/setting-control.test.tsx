// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import React, { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { parseSettingInput, type SettingKind, type SettingSpec } from '../../api/settings';
import { groupLiveSettings, liveSettingGroup, stagedDefault } from './live-settings';
import { SettingControl, settingLabel } from './setting-control';

let host: HTMLDivElement;
let root: Root;
beforeEach(() => { vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true); Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() }); host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView'); });

const spec = (name: string, type: SettingKind, extra: Partial<SettingSpec> = {}): SettingSpec => ({ name, type, default: null, mutable: true, allowed: null, help: `${name} help`, ...extra });

/** Renders one control with a live draft, the way LiveSettings owns it, and exposes the draft. */
function renderControl(setting: SettingSpec, value: unknown, initial?: string): { draft: () => string | undefined } {
  let current: string | undefined = initial;
  function Host(): React.JSX.Element {
    const [draft, setDraft] = useState<string | undefined>(initial);
    current = draft;
    return <SettingControl spec={setting} value={value} draft={draft} onChange={setDraft} locale="en" />;
  }
  act(() => root.render(<Host />));
  return { draft: () => current };
}
const setValue = (element: HTMLInputElement | HTMLTextAreaElement, value: string): void => act(() => {
  const prototype = element instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, 'value')?.set?.call(element, value);
  element.dispatchEvent(new Event('input', { bubbles: true }));
});

describe('SettingControl maps each schema kind to its control', () => {
  it('renders a checkbox Toggle for bool and drafts true/false text', () => {
    const control = renderControl(spec('flag', 'bool'), false);
    const box = host.querySelector<HTMLInputElement>('input[type="checkbox"]');
    expect(box?.checked).toBe(false);
    act(() => box?.click());
    expect(control.draft()).toBe('true');
    expect(parseSettingInput(spec('flag', 'bool'), control.draft() ?? '')).toBe(true);
  });

  it('renders a Select with exactly the allowed values for an enum string', () => {
    renderControl(spec('diffusion_sampler', 'str', { allowed: ['entropy-bound', 'confidence-threshold'] }), 'entropy-bound');
    expect(host.querySelector('[data-testid="setting-diffusion_sampler"] [role="combobox"]')).not.toBeNull();
    expect(host.querySelector('input, textarea')).toBeNull();
    act(() => host.querySelector<HTMLElement>('[role="combobox"]')?.click());
    const options = [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent);
    expect(options).toEqual(['entropy-bound', 'confidence-threshold']);
  });

  it('keeps an enum value outside allowed visible as a disabled option', () => {
    renderControl(spec('diffusion_sampler', 'str', { allowed: ['entropy-bound'] }), 'legacy');
    act(() => host.querySelector<HTMLElement>('[role="combobox"]')?.click());
    const extra = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((option) => option.textContent?.startsWith('legacy'));
    expect(extra?.getAttribute('aria-disabled')).toBe('true');
  });

  it('renders a text input for a free string', () => {
    renderControl(spec('name', 'str'), 'abc');
    const input = host.querySelector<HTMLInputElement>('input');
    expect(input?.type).toBe('text');
    expect(input?.value).toBe('abc');
  });

  it('renders number inputs with step 1 for int and step any for float, showing f32 values short', () => {
    renderControl(spec('default_top_k', 'int'), 40);
    expect(host.querySelector('input')?.getAttribute('type')).toBe('number');
    expect(host.querySelector('input')?.getAttribute('step')).toBe('1');
    act(() => root.unmount()); root = createRoot(host);
    renderControl(spec('default_temperature', 'float'), 0.800000011920929);
    expect(host.querySelector('input')?.getAttribute('step')).toBe('any');
    expect(host.querySelector('input')?.value).toBe('0.8');
  });

  it.each(['array', 'object', 'object_or_null'] as const)('renders a JSON textarea for %s and reports invalid JSON on blur', (type) => {
    renderControl(spec('payload', type), type === 'array' ? ['a'] : { a: 1 });
    const area = host.querySelector('textarea');
    expect(area).not.toBeNull();
    if (!area) return;
    setValue(area, '{oops');
    act(() => { area.focus(); area.blur(); });
    expect(host.textContent).toContain('Enter valid JSON');
    expect(area.getAttribute('aria-invalid')).toBe('true');
  });

  it('opens a null value with the unset toggle on and the inner control empty, and never shows the literal null', () => {
    renderControl(spec('default_seed', 'int_or_null'), null);
    const [number, unset] = [host.querySelector<HTMLInputElement>('input[type="number"]'), host.querySelector<HTMLInputElement>('[data-testid="setting-default_seed-unset"]')];
    expect(unset?.checked).toBe(true);
    expect(number?.value).toBe('');
    expect(number?.disabled).toBe(true);
    expect(host.innerHTML).not.toMatch(/\bnull\b/);
  });

  it('writes null into the draft when the unset toggle turns on, and restores the value when it turns off', () => {
    const control = renderControl(spec('max_denoising_steps', 'int_or_null'), 12);
    const unset = (): HTMLInputElement | null => host.querySelector<HTMLInputElement>('[data-testid="setting-max_denoising_steps-unset"]');
    expect(unset()?.checked).toBe(false);
    act(() => unset()?.click());
    expect(control.draft()).toBe('null');
    expect(parseSettingInput(spec('max_denoising_steps', 'int_or_null'), control.draft() ?? '')).toBeNull();
    expect(host.querySelector<HTMLInputElement>('input[type="number"]')?.disabled).toBe(true);
    act(() => unset()?.click());
    expect(control.draft()).toBeUndefined();
    expect(host.querySelector<HTMLInputElement>('input[type="number"]')?.value).toBe('12');
  });

  it('labels from the catalog with the raw key as secondary text, and falls back to the raw key alone', () => {
    expect(settingLabel('en', 'default_temperature')).toEqual({ label: 'Temperature', key: 'default_temperature' });
    expect(settingLabel('ko', 'default_temperature').label).toBe('온도');
    expect(settingLabel('en', 'some_future_knob')).toEqual({ label: 'some_future_knob', key: null });
    renderControl(spec('default_temperature', 'float'), 1);
    expect(host.querySelector('label span')?.textContent).toBe('Temperature');
    expect(host.textContent).toContain('default_temperature');
  });
});

describe('live setting groups', () => {
  it('groups by prefix in Sampling, DRY, Diffusion, Template, Other order without assuming a fixed list', () => {
    expect(['default_temperature', 'default_dry_base', 'diffusion_sampler', 'chat_template_kwargs', 'lang_bias_config', 'logit_bias_x', 'my_template', 'timeout_seconds', 'max_denoising_steps'].map(liveSettingGroup)).toEqual(['sampling', 'dry', 'diffusion', 'template', 'template', 'template', 'template', 'other', 'other']);
    const names = Array.from({ length: 256 }, (_, index) => `knob_${index}`);
    const grouped = groupLiveSettings([spec('timeout_seconds', 'int'), spec('default_dry_base', 'float'), spec('default_seed', 'int_or_null'), ...names.map((name) => spec(name, 'str'))]);
    expect(grouped.map((entry) => entry.group)).toEqual(['sampling', 'dry', 'other']);
    expect(grouped[2].specs).toHaveLength(257);
  });

  it('stages reset defaults as the text a person would type, or exact JSON when that text would not read back', () => {
    expect(stagedDefault(spec('default_temperature', 'float', { default: 0.800000011920929 }))).toBe('0.8');
    expect(stagedDefault(spec('default_seed', 'int_or_null', { default: null }))).toBe('null');
    expect(stagedDefault(spec('breakers', 'array', { default: ['\n'] }))).toBe('["\\n"]');
    const precise = 0.123456789;
    const staged = stagedDefault(spec('x', 'float', { default: precise }));
    expect(Math.fround(Number(staged))).toBe(Math.fround(precise));
  });
});
