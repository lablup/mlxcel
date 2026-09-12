import { describe, expect, it } from 'vitest';

describe('mlxcel WebUI scaffold', () => {
  it('keeps hash navigation client-side', () => {
    const url = new URL('http://127.0.0.1:8080/private/webui/#models');
    expect(url.pathname).toBe('/private/webui/');
    expect(url.hash).toBe('#models');
  });
});
