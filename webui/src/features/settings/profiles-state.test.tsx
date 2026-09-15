import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it } from 'vitest';
import { useLoadProfile } from './load-profiles';
import { useGenerationDefaults } from './generation-preferences';
const id = `mdl_${'a'.repeat(43)}`;
it('keeps reusable and model scopes independent, with safe reset and frozen request defaults', () => {
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
  let profiles: ReturnType<typeof useLoadProfile> | undefined;
  let generation: ReturnType<typeof useGenerationDefaults> | undefined;
  function Probe(): React.JSX.Element { profiles = useLoadProfile(id); generation = useGenerationDefaults(); return <span />; }
  try {
    act(() => root.render(<Probe />));
    act(() => profiles?.save({ ctx_size: 2048 }, 'reusable'));
    act(() => profiles?.save({ ctx_size: 4096 }, 'model'));
    expect(profiles?.profile.ctx_size).toBe(4096); expect(profiles?.reusable.ctx_size).toBe(2048);
    act(() => profiles?.reset('reusable'));
    expect(profiles?.profile.ctx_size).toBe(4096); expect(profiles?.reusable).toEqual({});
    act(() => profiles?.save({ ctx_size: 8192 }, 'reusable'));
    act(() => profiles?.reset('model'));
    expect(profiles?.profile.ctx_size).toBe(8192); expect(profiles?.modelProfile).toEqual({});
    const before = localStorage.getItem('mlxcel.webui.load-profiles.v1');
    act(() => generation?.setDefaults({ temperature: 0.75 }));
    expect(localStorage.getItem('mlxcel.webui.load-profiles.v1')).toBe(before);
    expect([...Array(localStorage.length).keys()].map((index) => localStorage.key(index))).not.toContain('generation-defaults');
    expect(generation?.defaults).toEqual({ temperature: 0.75 });
    act(() => generation?.reset());
  } finally { act(() => root.unmount()); host.remove(); }
});
describe('browser preferences failure', () => {
  it('rejects imported secrets before replacing any saved profile', () => {
    const host = document.createElement('div'); const root = createRoot(host); let profiles: ReturnType<typeof useLoadProfile> | undefined;
    function Probe(): React.JSX.Element { profiles = useLoadProfile(null); return <span />; }
    try { act(() => root.render(<Probe />)); const previous = profiles?.exportJson(); expect(() => profiles?.importJson('{"version":1,"reusable":{"token":"secret"},"models":{}}')).toThrow(); expect(profiles?.exportJson()).toBe(previous); } finally { act(() => root.unmount()); }
  });
});
