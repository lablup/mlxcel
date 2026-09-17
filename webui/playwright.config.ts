import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testIgnore: ['**/csp.spec.ts', '**/chat-real.spec.ts', '**/router-real.spec.ts', '**/engines.spec.ts', '**/performance.spec.ts'],
  fullyParallel: false,
  reporter: 'list',
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  use: {
    baseURL: 'http://127.0.0.1:4173',
  },
  webServer: {
    command: 'MLXCEL_WEBUI_OUT_DIR=.playwright-dist pnpm run build && pnpm exec vite preview --outDir .playwright-dist --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
