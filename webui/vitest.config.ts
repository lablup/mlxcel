import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    fs: { allow: ['..'] },
  },
  test: {
    environment: 'jsdom',
    server: { deps: { inline: ['@lablup/ui-common'] } },
    globals: true,
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
});
