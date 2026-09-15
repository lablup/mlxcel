// Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  testMatch: ['engines.spec.ts', 'performance.spec.ts'],
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  timeout: 45_000,
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', testMatch: ['engines.spec.ts'], use: { ...devices['Desktop Safari'] } },
    // Linux headed WebKit is a supported Playwright mode when the CI command is wrapped in Xvfb.
    // https://playwright.dev/docs/ci#running-headed
    {
      name: 'webkit-headed-performance',
      testMatch: ['performance.spec.ts'],
      use: { ...devices['Desktop Safari'], headless: false },
    },
  ],
  use: { baseURL: 'http://127.0.0.1:4173', trace: 'off', screenshot: 'off', video: 'off' },
  webServer: {
    command: 'MLXCEL_WEBUI_OUT_DIR=.playwright-dist pnpm run build && pnpm exec vite preview --outDir .playwright-dist --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
