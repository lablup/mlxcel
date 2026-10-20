import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    fs: { allow: ['..'] },
  },
  test: {
    environment: 'jsdom',
    // ui-common components import their own stylesheets; inline the package so Vitest handles them.
    server: { deps: { inline: [/@lablup\/ui-common/] } },
    globals: true,
    setupFiles: ['src/test-setup.ts'],
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
});
